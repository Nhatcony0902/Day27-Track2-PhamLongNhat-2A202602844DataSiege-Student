"""
Data Siege defense.

Strategy
--------
* Cost: exactly ONE metered call per event (the one tool that sees that event),
  never a second. A 200-event private run costs ~300 credits against a 320
  budget. Coverage beats a small overage (see _can_spend), so the free
  budget guard only stops spending at a 25% overage.
* Statistics: ctx.baseline publishes each clean metric's bound as mean +/- 3 sigma.
  - two-sided metrics (row_count, mean_amount) give mean and sigma directly:
    mean = (lo+hi)/2, sigma = (hi-lo)/6.
  - one-sided metrics only publish the upper bound. We learn the clean mean
    online (from events we judged clean) and back out sigma = (max - mean)/3.
  Every metric becomes a z-score; we alert at z >= Z_ALERT, well inside the
  published 3-sigma bound so near-normal ("subtle") faults still get caught.
  Sigma is taken from the published calibration, never from a short online
  window: that window systematically underestimates the spread and floods
  false alarms.
* Metrics with no published bound (std_amount) are learned online and need
  stronger evidence (Z_LEARNED).
* Structural signals alert unconditionally: contract violations, freshness
  beyond the contract's own declared SLA, an upstream edge seen on earlier
  runs of the same job (or declared in the run's inputs) missing now,
  an output with no downstream consumers.
"""
import sys
from api import Verdict

# z on baseline-calibrated metrics. A missed fault costs 0.5/n_faulty, a false
# alarm 0.3/n_clean; with ~1 fault in 4 events, alerting pays once P(fault)
# exceeds ~18%. An offline simulation (clean values from practice/public,
# injected faults of 1.5-4 sigma) puts the optimum at z ~ 1.75-2.0; 2.0 keeps
# the per-metric false-alarm rate moderate in case real noise is heavier.
Z_ALERT = 2.0
Z_LEARNED = 3.5     # z on metrics with no published baseline
CHI_COMBINED = 13.0 # sum of z^2 across a batch's noise-like metrics
WARMUP = 6          # clean samples before a learned mean is trusted
MAX_OVERAGE = 0.25  # past this, assume something is wrong and stop spending

DEBUG = False  # stderr is an unread pipe in the harness: keep output tiny


def _log(*a):
    if DEBUG:
        print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# clean-stream model
# --------------------------------------------------------------------------

def _num(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    return float(x)


def _hist(ctx, key):
    return ctx.state.setdefault("hist", {}).setdefault(key, [])


def _learn(ctx, key, value):
    v = _num(value)
    if v is None:
        return
    h = _hist(ctx, key)
    h.append(v)
    if len(h) > 300:
        del h[0]


def _mean_sd(vals):
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
    return m, var ** 0.5


def _bl(ctx, k):
    return _num(ctx.baseline.get(k))


def z_two_sided(ctx, value, lo_key, hi_key):
    v, lo, hi = _num(value), _bl(ctx, lo_key), _bl(ctx, hi_key)
    if v is None or lo is None or hi is None or hi <= lo:
        return 0.0
    return abs(v - (lo + hi) / 2.0) / ((hi - lo) / 6.0)


def z_upper(ctx, key, value, max_key, two_sided=False):
    """Metric with a published clean upper bound max = mean + 3 sigma."""
    v, mx = _num(value), _bl(ctx, max_key)
    if v is None or mx is None:
        return 0.0
    h = _hist(ctx, key)
    if len(h) < WARMUP:
        # no trusted mean yet: only the published bound itself
        return 3.0 * v / mx if mx > 0 and v > mx else 0.0
    mean = sum(h) / len(h)
    if mean >= mx:
        return 3.0 * v / mx if mx > 0 and v > mx else 0.0
    sd = (mx - mean) / 3.0
    z = (v - mean) / sd
    return abs(z) if two_sided else max(z, 0.0)


def z_learned(ctx, key, value):
    v = _num(value)
    h = _hist(ctx, key)
    if v is None or len(h) < max(WARMUP, 10):
        return 0.0
    m, sd = _mean_sd(h)
    return abs(v - m) / sd if sd > 0 else 0.0


def decide(zs, hard=(), learned=(), chi_keys=()):
    if hard:
        return True, 1.0, "; ".join(hard)
    for k in learned:
        if zs.get(k, 0.0) >= Z_LEARNED:
            return True, 0.7, f"{k} z={zs[k]:.2f} (learned)"
    base = {k: z for k, z in zs.items() if k not in learned}
    if base:
        k, top = max(base.items(), key=lambda kv: kv[1])
        if top >= Z_ALERT:
            return True, min(1.0, top / 6.0), f"{k} z={top:.2f}"
    chi = sum(zs.get(k, 0.0) ** 2 for k in chi_keys)
    if chi_keys and chi >= CHI_COMBINED:
        return True, 0.6, f"combined chi2={chi:.1f}"
    top = max(zs.values()) if zs else 0.0
    return False, max(0.0, 1.0 - top / Z_ALERT), f"max z={top:.2f}"


def _can_spend(ctx, cost):
    """Overage is charged linearly (0.2 * overage, capped at 1.0): one extra
    2-credit call costs 0.2*2/budget of score, while catching one fault is
    worth 0.5/n_faulty. With faults at ~1 in 4 events the expected value of
    a call stays positive even past the budget, so only stop once the
    projected overage would exceed MAX_OVERAGE."""
    rem = _num(ctx.tools.budget_remaining())
    spent = _num(ctx.tools.spend_so_far())
    if rem is None or spent is None:
        return True
    budget = rem + spent
    return budget <= 0 or (spent + cost - budget) / budget <= MAX_OVERAGE


def _ok(r):
    return isinstance(r, dict) and "error" not in r


def _skip(pillar, why):
    return Verdict(alert=False, confidence=0.0, reason=why, pillar=pillar)


# --------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------

def register(ctx):
    ctx.on("data_batch", check_data_batch)
    ctx.on("contract_checkpoint", check_contract_checkpoint)
    ctx.on("lineage_run", check_lineage_run)
    ctx.on("feature_materialization", check_feature_materialization)
    ctx.on("embedding_batch", check_embedding_batch)


def check_data_batch(payload, ctx):
    if not _can_spend(ctx, 1.0):
        return _skip("checks", "budget guard")
    r = ctx.tools.batch_profile(payload.get("batch_id"))
    if not _ok(r):
        return _skip("checks", f"tool error: {r}")

    nr = r.get("null_rate")
    null_rate = nr.get("customer_id") if isinstance(nr, dict) else nr
    zs = {
        "row_count": z_two_sided(ctx, r.get("row_count"), "row_count_min", "row_count_max"),
        "mean_amount": z_two_sided(ctx, r.get("mean_amount"), "mean_amount_min", "mean_amount_max"),
        "null_rate": z_upper(ctx, "db.null", null_rate, "null_rate_max"),
        "staleness": z_upper(ctx, "db.stale", r.get("staleness_min"), "staleness_min_max"),
        "std_amount": z_learned(ctx, "db.std", r.get("std_amount")),
    }
    alert, conf, reason = decide(zs, learned=("std_amount",),
                                 chi_keys=("row_count", "mean_amount", "std_amount"))
    if not alert:
        _learn(ctx, "db.null", null_rate)
        _learn(ctx, "db.stale", r.get("staleness_min"))
        _learn(ctx, "db.std", r.get("std_amount"))
    _log("db", reason)
    return Verdict(alert=alert, confidence=conf, reason=reason, pillar="checks")


def check_contract_checkpoint(payload, ctx):
    if not _can_spend(ctx, 1.5):
        return _skip("contracts", "budget guard")
    r = ctx.tools.contract_diff(payload.get("contract_id"), payload.get("checkpoint_batch_id"))
    if not _ok(r):
        return _skip("contracts", f"tool error: {r}")

    hard = []
    violations = r.get("violations") or []
    if violations:
        hard.append("violations: " + ",".join(map(str, violations)))
    delay = _num(r.get("freshness_delay_min"))
    sla = payload.get("declared_sla")
    sla_fresh = _num(sla.get("freshness_min")) if isinstance(sla, dict) else None
    if delay is not None and sla_fresh is not None and delay > sla_fresh:
        hard.append(f"freshness {delay} > declared SLA {sla_fresh}")

    zs = {"freshness_delay": z_upper(ctx, "cc.fresh", delay, "freshness_delay_max_min")}
    alert, conf, reason = decide(zs, hard)
    if not alert:
        _learn(ctx, "cc.fresh", delay)
    _log("cc", reason)
    return Verdict(alert=alert, confidence=conf, reason=reason, pillar="contracts")


def _names(x):
    out = set()
    if isinstance(x, (list, tuple)):
        for it in x:
            if isinstance(it, str):
                out.add(it)
            elif isinstance(it, dict) and isinstance(it.get("name"), str):
                out.add(it["name"])
    elif isinstance(x, str):
        out.add(x)
    return out


def check_lineage_run(payload, ctx):
    if not _can_spend(ctx, 1.0):
        return _skip("lineage", "budget guard")
    r = ctx.tools.lineage_graph_slice(payload.get("run_id"))
    if not _ok(r):
        return _skip("lineage", f"tool error: {r}")

    hard = []
    job = payload.get("job") if isinstance(payload.get("job"), str) else "_default"
    actual_up = _names(r.get("actual_upstream"))
    down = _num(r.get("actual_downstream_count"))
    jobs = ctx.state.setdefault("lineage_jobs", {})
    hist = jobs.get(job)

    expected_up = _names(payload.get("inputs"))
    if hist:
        expected_up |= hist["up"]
    missing = expected_up - actual_up
    if missing:
        hard.append("missing upstream: " + ",".join(sorted(missing)))
    if down is not None:
        if down <= 0:
            hard.append("orphaned output (0 downstream)")
        elif hist and hist["down_min"] is not None and down < hist["down_min"]:
            hard.append(f"downstream count {down} < usual {hist['down_min']}")

    zs = {"duration": z_upper(ctx, "ln.dur", r.get("duration_ms"), "lineage_duration_ms_max",
                              two_sided=True)}
    alert, conf, reason = decide(zs, hard)
    if not alert:
        _learn(ctx, "ln.dur", r.get("duration_ms"))
        if hist is None:
            jobs[job] = {"up": set(actual_up), "down_min": down}
        else:
            hist["up"] |= actual_up
            if down is not None:
                hist["down_min"] = down if hist["down_min"] is None else min(hist["down_min"], down)
    _log("ln", reason)
    return Verdict(alert=alert, confidence=conf, reason=reason, pillar="lineage")


def check_feature_materialization(payload, ctx):
    if not _can_spend(ctx, 2.0):
        return _skip("ai_infra", "budget guard")
    r = ctx.tools.feature_drift(payload.get("feature_view"), payload.get("batch_id"))
    if not _ok(r):
        return _skip("ai_infra", f"tool error: {r}")

    shift = _num(r.get("mean_shift_sigma"))
    if shift is None:
        try:
            shift = abs(r["serve_mean"] - r["train_mean"]) / r["train_std"]
        except Exception:
            shift = None
    zs = {"mean_shift_sigma": z_upper(ctx, "fm.shift", shift, "feature_mean_shift_sigma_max")}
    alert, conf, reason = decide(zs)
    if not alert:
        _learn(ctx, "fm.shift", shift)
    _log("fm", reason)
    return Verdict(alert=alert, confidence=conf, reason=reason, pillar="ai_infra")


def check_embedding_batch(payload, ctx):
    if not _can_spend(ctx, 2.0):
        return _skip("ai_infra", "budget guard")
    r = ctx.tools.embedding_drift(payload.get("corpus"), payload.get("chunk_batch_id"))
    if not _ok(r):
        return _skip("ai_infra", f"tool error: {r}")

    zs = {
        "centroid_shift": z_upper(ctx, "eb.shift", r.get("centroid_shift"),
                                  "embedding_centroid_shift_max"),
        "doc_age": z_upper(ctx, "eb.age", r.get("avg_doc_age_days"),
                           "corpus_avg_doc_age_days_max"),
    }
    alert, conf, reason = decide(zs)
    if not alert:
        _learn(ctx, "eb.shift", r.get("centroid_shift"))
        _learn(ctx, "eb.age", r.get("avg_doc_age_days"))
    _log("eb", reason)
    return Verdict(alert=alert, confidence=conf, reason=reason, pillar="ai_infra")
