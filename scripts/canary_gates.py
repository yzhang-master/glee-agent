"""Deterministic promotion and rollback gates for live canary reports.

The functions in this module are deliberately pure and read-only.  They take
one experiment object produced by :mod:`scripts.canary_report`, return a new
gate object, and never mutate the raw report.  Keeping the evaluator separate
also lets the collector's JSON schema remain backward compatible.

The frozen 2026-08-25 rules leave two statistical details unspecified.  This
implementation resolves them conservatively:

* standardized means use pooled arm counts as fixed common-support weights;
* normalized payoff/revenue confidence bounds use the worst-case variance
  ``1/4`` for an outcome bounded to ``[0, 1]`` and a one-sided 95% normal
  bound.  This is wider than a plug-in variance bound when outcomes are away
  from one half, but aggregated reports do not retain a second moment from
  which a faithful non-parametric bootstrap could be reconstructed.

Conversion intervals are one-sided 95% Wilson score bounds.  The persuasion
zero-sale risk-difference upper bound is the conservative Newcombe component
bound ``WilsonUpper(treatment) - WilsonLower(control)``.

Integration is intentionally one line after a raw report is built::

    report["gates"] = evaluate_report_gates(report)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any


RULE_VERSION = "2026-08-25-frozen-v1"
ONE_SIDED_95_Z = 1.6448536269514722


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _count(value: Any) -> int:
    number = _number(value)
    if number is None or number < 0 or not number.is_integer():
        return 0
    return int(number)


def _bounded_mean(value: Any) -> float | None:
    number = _number(value)
    if number is None or not 0 <= number <= 1:
        return None
    return number


def _check(value: Any, operator: str, threshold: float | int) -> dict:
    number = _number(value)
    if operator == ">=":
        passed = number is not None and number >= threshold
    elif operator == ">":
        passed = number is not None and number > threshold
    elif operator == "<=":
        passed = number is not None and number <= threshold
    else:  # pragma: no cover - only module constants select operators
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "value": value if number is not None else None,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def _wilson_bounds(successes: int, trials: int) -> tuple[float | None, float | None]:
    """Return one-sided 95% Wilson lower and upper component bounds."""
    if trials <= 0 or successes < 0 or successes > trials:
        return None, None
    p = successes / trials
    z = ONE_SIDED_95_Z
    z2 = z * z
    denominator = 1 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))
        / denominator
    )
    return max(0.0, center - radius), min(1.0, center + radius)


def _binomial_summary(successes: Any, trials: Any) -> dict:
    n = _count(trials)
    k = _count(successes)
    if k > n:
        return {
            "successes": k,
            "trials": n,
            "rate": None,
            "lower_95_one_sided": None,
            "upper_95_one_sided": None,
        }
    lower, upper = _wilson_bounds(k, n)
    return {
        "successes": k,
        "trials": n,
        "rate": k / n if n else None,
        "lower_95_one_sided": lower,
        "upper_95_one_sided": upper,
    }


def _difference_summary(
    treatment_mean: Any,
    treatment_n: Any,
    control_mean: Any,
    control_n: Any,
) -> dict:
    """Difference of bounded means with a worst-case-variance normal bound."""
    mt = _bounded_mean(treatment_mean)
    mc = _bounded_mean(control_mean)
    nt = _count(treatment_n)
    nc = _count(control_n)
    if mt is None or mc is None or not nt or not nc:
        return {
            "available": False,
            "treatment_mean": mt,
            "control_mean": mc,
            "difference": None,
            "lower_95_one_sided": None,
            "upper_95_one_sided": None,
            "method": "bounded_normal_worst_case_variance",
        }
    difference = mt - mc
    standard_error_upper = math.sqrt(0.25 / nt + 0.25 / nc)
    margin = ONE_SIDED_95_Z * standard_error_upper
    return {
        "available": True,
        "treatment_mean": mt,
        "control_mean": mc,
        "difference": difference,
        "standard_error_upper": standard_error_upper,
        "lower_95_one_sided": difference - margin,
        "upper_95_one_sided": difference + margin,
        "method": "bounded_normal_worst_case_variance",
    }


def _group_key(cell: Mapping[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    values: list[Any] = []
    for field in fields:
        value = cell.get(field, "unknown")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, sort_keys=True, separators=(",", ":"))
        values.append(value)
    return tuple(values)


def _group_label(fields: tuple[str, ...], key: tuple[Any, ...]) -> dict:
    return dict(zip(fields, key, strict=True))


def _aggregate_cells(metric: Mapping[str, Any], fields: tuple[str, ...]) -> dict:
    groups: dict[tuple[Any, ...], dict[str, float | int]] = defaultdict(
        lambda: {"resolved": 0, "sum": 0.0, "source_cells": 0}
    )
    cells = metric.get("cells", {})
    if not isinstance(cells, Mapping):
        return {}
    for entry in cells.values():
        if not isinstance(entry, Mapping):
            continue
        cell = entry.get("cell")
        cell = cell if isinstance(cell, Mapping) else {}
        n = _count(entry.get("resolved"))
        mean = _bounded_mean(
            entry.get("mean_normalized_payoff", entry.get("mean_revenue_share"))
        )
        if not n or mean is None:
            continue
        group = groups[_group_key(cell, fields)]
        group["resolved"] += n
        group["sum"] += n * mean
        group["source_cells"] += 1
    return dict(groups)


def _standardized_difference(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
    fields: tuple[str, ...],
) -> dict:
    """Compare bounded outcomes on common support with pooled fixed weights."""
    tg = _aggregate_cells(treatment, fields)
    cg = _aggregate_cells(control, fields)
    common = sorted(set(tg) & set(cg), key=repr)
    grouped_t = sum(int(group["resolved"]) for group in tg.values())
    grouped_c = sum(int(group["resolved"]) for group in cg.values())
    # An invalid or unkeyed row must not disappear and make common support
    # look better than it is.  Fall back only for old synthetic inputs that do
    # not carry the arm-level resolved count.
    total_t = max(_count(treatment.get("resolved")), grouped_t)
    total_c = max(_count(control.get("resolved")), grouped_c)
    common_t = sum(int(tg[key]["resolved"]) for key in common)
    common_c = sum(int(cg[key]["resolved"]) for key in common)
    pooled = common_t + common_c

    support = {
        "dimensions": list(fields),
        "treatment_groups": len(tg),
        "control_groups": len(cg),
        "common_groups": len(common),
        "treatment_resolved": total_t,
        "control_resolved": total_c,
        "treatment_grouped_resolved": grouped_t,
        "control_grouped_resolved": grouped_c,
        "treatment_common_resolved": common_t,
        "control_common_resolved": common_c,
        "treatment_common_fraction": common_t / total_t if total_t else None,
        "control_common_fraction": common_c / total_c if total_c else None,
        "treatment_only_groups": len(set(tg) - set(cg)),
        "control_only_groups": len(set(cg) - set(tg)),
    }
    if not common or not pooled:
        return {
            "available": False,
            "treatment_mean": None,
            "control_mean": None,
            "difference": None,
            "lower_95_one_sided": None,
            "upper_95_one_sided": None,
            "method": "pooled_common_support_bounded_normal",
            "support": support,
            "cells": [],
        }

    treatment_mean = 0.0
    control_mean = 0.0
    variance_upper = 0.0
    cells: list[dict] = []
    for key in common:
        nt = int(tg[key]["resolved"])
        nc = int(cg[key]["resolved"])
        mt = float(tg[key]["sum"]) / nt
        mc = float(cg[key]["sum"]) / nc
        weight = (nt + nc) / pooled
        treatment_mean += weight * mt
        control_mean += weight * mc
        # Every normalized terminal outcome is in [0, 1], hence variance is
        # at most 1/4.  Counts are treated as independent within each arm.
        variance_upper += weight * weight * (0.25 / nt + 0.25 / nc)
        cells.append(
            {
                "cell": _group_label(fields, key),
                "treatment_resolved": nt,
                "control_resolved": nc,
                "treatment_mean": mt,
                "control_mean": mc,
                "pooled_weight": weight,
            }
        )
    difference = treatment_mean - control_mean
    standard_error_upper = math.sqrt(variance_upper)
    margin = ONE_SIDED_95_Z * standard_error_upper
    return {
        "available": True,
        "treatment_mean": treatment_mean,
        "control_mean": control_mean,
        "difference": difference,
        "standard_error_upper": standard_error_upper,
        "lower_95_one_sided": difference - margin,
        "upper_95_one_sided": difference + margin,
        "method": "pooled_common_support_bounded_normal",
        "support": support,
        "cells": cells,
    }


def _arm_rate_guardrails(experiment: Mapping[str, Any]) -> dict:
    """Compare treatment/control censor and logged-error event rates.

    Censoring is exact in the arm outcome metrics.  The health schema does not
    retain event IDs, so ``result_errors`` and ``invalid_results`` can overlap;
    summing them is a conservative upper bound on the error-event count.
    """
    metrics = experiment.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}

    censor: dict[str, dict] = {}
    for arm in ("treatment", "control"):
        metric = metrics.get(arm, {})
        metric = metric if isinstance(metric, Mapping) else {}
        censored = _count(metric.get("censored"))
        observed = _count(metric.get("resolved")) + censored
        censor[arm] = {
            "censored": censored,
            "observed": observed,
            "rate": censored / observed if observed else None,
        }
    censor_excess = (
        censor["treatment"]["rate"] - censor["control"]["rate"]
        if censor["treatment"]["rate"] is not None
        and censor["control"]["rate"] is not None
        else None
    )

    assignment = experiment.get("assignment", {})
    assignment = assignment if isinstance(assignment, Mapping) else {}
    agents = experiment.get("agents", {})
    agents = agents if isinstance(agents, Mapping) else {}
    errors: dict[str, dict] = {}
    for arm in ("treatment", "control"):
        labels = assignment.get(f"{arm}_agents", [])
        labels = labels if isinstance(labels, (list, tuple)) else []
        events = 0
        failures = 0
        for label in labels:
            agent = agents.get(label, {})
            agent = agent if isinstance(agent, Mapping) else {}
            health = agent.get("health", {})
            health = health if isinstance(health, Mapping) else {}
            turns = _count(health.get("turns"))
            results = _count(health.get("result_events"))
            events += turns + results
            failures += _count(health.get("turn_errors"))
            # This sum is intentionally an upper bound because an invalid
            # result can also have a persisted error string.
            failures += _count(health.get("result_errors"))
            failures += _count(health.get("invalid_results"))
        failures = min(failures, events) if events else failures
        errors[arm] = {
            "failures_upper_bound": failures,
            "events": events,
            "rate_upper_bound": failures / events if events else None,
        }
    error_excess = (
        errors["treatment"]["rate_upper_bound"]
        - errors["control"]["rate_upper_bound"]
        if errors["treatment"]["rate_upper_bound"] is not None
        and errors["control"]["rate_upper_bound"] is not None
        else None
    )
    return {
        "censoring": {
            **censor,
            "treatment_excess": censor_excess,
            "check": _check(censor_excess, "<=", 0.03),
        },
        "errors": {
            **errors,
            "treatment_excess_upper_bound": error_excess,
            "check": _check(error_excess, "<=", 0.01),
            "method": "logged_event_union_upper_bound",
        },
    }


def _support_checks(standardized: Mapping[str, Any]) -> dict:
    support = standardized.get("support", {})
    support = support if isinstance(support, Mapping) else {}
    return {
        "treatment_common_coverage": _check(
            support.get("treatment_common_fraction"), ">=", 0.90
        ),
        "control_common_coverage": _check(
            support.get("control_common_fraction"), ">=", 0.90
        ),
    }


def _health_guardrails(experiment: Mapping[str, Any]) -> dict:
    totals = defaultdict(int)
    agents = experiment.get("agents", {})
    if isinstance(agents, Mapping):
        for agent in agents.values():
            if not isinstance(agent, Mapping):
                continue
            health = agent.get("health", {})
            routing = agent.get("routing", {})
            health = health if isinstance(health, Mapping) else {}
            routing = routing if isinstance(routing, Mapping) else {}
            for key in (
                "turn_errors",
                "corrections",
                "result_errors",
                "invalid_results",
                "http_503",
                "duplicate_turns",
            ):
                totals[key] += _count(health.get(key))
            for key in (
                "checked",
                "assigned_matches",
                "affected",
                "affected_assigned_matches",
                "replay_errors",
                "affected_wrong_variant",
                "affected_unknown",
                "direction_violations",
            ):
                totals[key] += _count(routing.get(key))

    metrics = experiment.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    deterministic_checked = 0
    deterministic_matches = 0
    for arm in ("treatment", "control"):
        metric = metrics.get(arm, {})
        if not isinstance(metric, Mapping):
            continue
        deterministic_checked += _count(metric.get("deterministic_route_checked"))
        deterministic_matches += _count(metric.get("deterministic_route_matches"))

    # 503s are transport failures and can overlap turn/result error totals.
    # Aggregation cannot prove which invalid result belonged to which 503, so
    # these remain explicit manual-review warnings rather than silently being
    # cleared or causing an irreversible automatic rollback.
    non_transport_errors = max(
        totals["turn_errors"] + totals["result_errors"] - totals["http_503"], 0
    )
    hard_failures = {
        "non_transport_errors": non_transport_errors,
        "corrections": totals["corrections"],
        "routing_mismatches": max(
            totals["checked"] - totals["assigned_matches"], 0
        ),
        "affected_routing_mismatches": max(
            totals["affected"] - totals["affected_assigned_matches"], 0
        ),
        "replay_errors": totals["replay_errors"],
        "wrong_variant": totals["affected_wrong_variant"],
        "unknown_variant": totals["affected_unknown"],
        "direction_violations": totals["direction_violations"],
        "deterministic_route_mismatches": max(
            deterministic_checked - deterministic_matches, 0
        ),
    }
    warnings = {
        "http_503": totals["http_503"],
        "invalid_results": totals["invalid_results"],
        "duplicate_turns": totals["duplicate_turns"],
    }
    return {
        "passed": not any(hard_failures.values()),
        "manual_review_required": bool(
            warnings["http_503"] or warnings["invalid_results"]
        ),
        "hard_failures": hard_failures,
        "warnings": warnings,
    }


def _bargaining_gate(experiment: Mapping[str, Any]) -> dict:
    metrics = experiment.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    treatment = metrics.get("treatment", {})
    control = metrics.get("control", {})
    treatment = treatment if isinstance(treatment, Mapping) else {}
    control = control if isinstance(control, Mapping) else {}
    guardrails = _health_guardrails(experiment)
    arm_rates = _arm_rate_guardrails(experiment)

    sample_checks = {
        "treatment_affected": _check(treatment.get("affected_games"), ">=", 300),
        "treatment_resolved": _check(treatment.get("resolved"), ">=", 300),
        "control_affected": _check(control.get("affected_games"), ">=", 900),
        "control_resolved": _check(control.get("resolved"), ">=", 900),
    }
    horizon_metrics: dict[str, dict] = {}
    treatment_horizons = treatment.get("horizon_strata", {})
    control_horizons = control.get("horizon_strata", {})
    treatment_horizons = (
        treatment_horizons if isinstance(treatment_horizons, Mapping) else {}
    )
    control_horizons = control_horizons if isinstance(control_horizons, Mapping) else {}
    for horizon in ("finite", "unlimited"):
        th = treatment_horizons.get(horizon, {})
        ch = control_horizons.get(horizon, {})
        th = th if isinstance(th, Mapping) else {}
        ch = ch if isinstance(ch, Mapping) else {}
        horizon_metrics[horizon] = {
            "treatment_resolved": _check(th.get("resolved"), ">=", 100),
            # The frozen phrase "100 per horizon" did not name an arm.  Both
            # arms are required here; this is the conservative interpretation.
            "control_resolved": _check(ch.get("resolved"), ">=", 100),
        }

    conversion: dict[str, dict] = {}
    conversion_sources = {
        "overall": (treatment, 0.104),
        "finite": (treatment_horizons.get("finite", {}), 0.137),
        "unlimited": (treatment_horizons.get("unlimited", {}), 0.073),
    }
    for name, (source, floor) in conversion_sources.items():
        source = source if isinstance(source, Mapping) else {}
        summary = _binomial_summary(
            source.get("direct_converted"), source.get("direct_resolved")
        )
        summary["historical_floor"] = floor
        summary["passed"] = (
            summary["lower_95_one_sided"] is not None
            and summary["lower_95_one_sided"] > floor
        )
        conversion[name] = summary

    standardized = _standardized_difference(
        treatment,
        control,
        ("role", "horizon", "phase", "max_rounds"),
    )
    payoff_checks = {
        "lift": _check(standardized.get("difference"), ">=", 0.020),
        "lower_bound": _check(
            standardized.get("lower_95_one_sided"), ">", 0.0
        ),
    }
    support_checks = _support_checks(standardized)
    affected_unstandardized = _difference_summary(
        treatment.get("mean_normalized_payoff"),
        treatment.get("resolved"),
        control.get("mean_normalized_payoff"),
        control.get("resolved"),
    )

    rollback_triggers: list[str] = []
    if not guardrails["passed"]:
        rollback_triggers.append("health_or_routing_guardrail")
    treatment_direct_n = _count(treatment.get("direct_resolved"))
    treatment_direct_rate = _number(treatment.get("direct_conversion_rate"))
    if (
        treatment_direct_n >= 100
        and treatment_direct_rate is not None
        and treatment_direct_rate <= 0.08
    ):
        rollback_triggers.append("direct_conversion_at_or_below_8pct_after_100")
    treatment_resolved = _count(treatment.get("resolved"))
    treatment_payoff = _number(treatment.get("mean_normalized_payoff"))
    if (
        treatment_resolved >= 100
        and treatment_payoff is not None
        and treatment_payoff <= 0.27
    ):
        rollback_triggers.append("normalized_payoff_at_or_below_0.27_after_100")

    promotion_ready = (
        all(check["passed"] for check in sample_checks.values())
        and all(
            check["passed"]
            for horizon in horizon_metrics.values()
            for check in horizon.values()
        )
        and all(item["passed"] for item in conversion.values())
        and all(check["passed"] for check in payoff_checks.values())
        and all(check["passed"] for check in support_checks.values())
        and arm_rates["censoring"]["check"]["passed"]
        and arm_rates["errors"]["check"]["passed"]
        and standardized["available"]
        and guardrails["passed"]
    )
    if rollback_triggers:
        decision = "rollback"
    elif promotion_ready and guardrails["manual_review_required"]:
        decision = "manual_review"
    elif promotion_ready:
        decision = "promote"
    else:
        decision = "continue"

    return {
        "experiment": experiment.get("name", "barg_dis_anchor"),
        "family": "bargaining",
        "rule_version": RULE_VERSION,
        "decision": decision,
        "promotion_ready": promotion_ready,
        "rollback_triggers": rollback_triggers,
        "guardrails": guardrails,
        "arm_rate_guardrails": arm_rates,
        "sample_checks": sample_checks,
        "horizon_checks": horizon_metrics,
        "statistics": {
            "direct_conversion": conversion,
            "standardized_payoff": standardized,
            "payoff_checks": payoff_checks,
            "support_checks": support_checks,
            # This is useful but is not true all-enrolled ITT: the current raw
            # bargaining metric contains only games with an exact divergence.
            "affected_unstandardized_payoff": affected_unstandardized,
            "itt": {
                "available": False,
                "reason": (
                    "the raw report does not retain terminal payoffs for "
                    "unaffected enrolled bargaining games"
                ),
            },
        },
        "notes": [
            "finite and unlimited minimums are required in both arms",
            "at least 90% of resolved observations in each arm must be on common support",
        ],
    }


def _persuasion_gate(experiment: Mapping[str, Any]) -> dict:
    metrics = experiment.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    treatment = metrics.get("treatment", {})
    control = metrics.get("control", {})
    treatment = treatment if isinstance(treatment, Mapping) else {}
    control = control if isinstance(control, Mapping) else {}
    guardrails = _health_guardrails(experiment)
    arm_rates = _arm_rate_guardrails(experiment)

    sample_checks = {
        "treatment_completed": _check(treatment.get("resolved"), ">=", 1000),
        "control_completed": _check(control.get("resolved"), ">=", 1000),
    }

    treatment_p = treatment.get("p_strata", {})
    control_p = control.get("p_strata", {})
    treatment_p = treatment_p if isinstance(treatment_p, Mapping) else {}
    control_p = control_p if isinstance(control_p, Mapping) else {}
    p_checks: dict[str, dict] = {}
    for p in sorted(set(treatment_p) | set(control_p), key=str):
        tmetric = treatment_p.get(p, {})
        cmetric = control_p.get(p, {})
        tmetric = tmetric if isinstance(tmetric, Mapping) else {}
        cmetric = cmetric if isinstance(cmetric, Mapping) else {}
        p_checks[str(p)] = {
            "treatment_resolved": _check(tmetric.get("resolved"), ">=", 150),
            "control_resolved": _check(cmetric.get("resolved"), ">=", 150),
        }

    standardized = _standardized_difference(
        treatment,
        control,
        (
            "p",
            "message_type",
            "price",
            "total_rounds",
            "opponent_type",
        ),
    )
    revenue_checks = {
        "lift": _check(standardized.get("difference"), ">=", 0.025),
        "lower_bound": _check(
            standardized.get("lower_95_one_sided"), ">", 0.0
        ),
    }
    support_checks = _support_checks(standardized)
    temporal = _standardized_difference(
        treatment,
        control,
        (
            "p",
            "message_type",
            "price",
            "total_rounds",
            "opponent_type",
            "start_block_15m",
        ),
    )

    t_zero = _binomial_summary(
        treatment.get("zero_sales"), treatment.get("resolved")
    )
    c_zero = _binomial_summary(control.get("zero_sales"), control.get("resolved"))
    risk_upper = (
        t_zero["upper_95_one_sided"] - c_zero["lower_95_one_sided"]
        if t_zero["upper_95_one_sided"] is not None
        and c_zero["lower_95_one_sided"] is not None
        else None
    )
    zero_sale = {
        "treatment": t_zero,
        "control": c_zero,
        "risk_difference": (
            t_zero["rate"] - c_zero["rate"]
            if t_zero["rate"] is not None and c_zero["rate"] is not None
            else None
        ),
        "upper_95_one_sided": risk_upper,
        "method": "newcombe_wilson_component",
        "check": _check(risk_upper, "<=", 0.02),
    }

    itt_revenue = _difference_summary(
        treatment.get("mean_revenue_share"),
        treatment.get("resolved"),
        control.get("mean_revenue_share"),
        control.get("resolved"),
    )
    p_ready = bool(p_checks) and all(
        check["passed"] for pair in p_checks.values() for check in pair.values()
    )
    promotion_ready = (
        all(check["passed"] for check in sample_checks.values())
        and p_ready
        and standardized["available"]
        and all(check["passed"] for check in revenue_checks.values())
        and all(check["passed"] for check in support_checks.values())
        and zero_sale["check"]["passed"]
        and arm_rates["censoring"]["check"]["passed"]
        and arm_rates["errors"]["check"]["passed"]
        and guardrails["passed"]
    )

    rollback_triggers: list[str] = []
    if not guardrails["passed"]:
        rollback_triggers.append("health_or_routing_guardrail")
    if rollback_triggers:
        decision = "rollback"
    elif promotion_ready and guardrails["manual_review_required"]:
        decision = "manual_review"
    elif promotion_ready:
        decision = "promote"
    else:
        decision = "continue"

    return {
        "experiment": experiment.get("name", "pers_blind_lie"),
        "family": "persuasion",
        "rule_version": RULE_VERSION,
        "decision": decision,
        "promotion_ready": promotion_ready,
        "rollback_triggers": rollback_triggers,
        "guardrails": guardrails,
        "arm_rate_guardrails": arm_rates,
        "sample_checks": sample_checks,
        "p_strata_checks": p_checks,
        "statistics": {
            "standardized_revenue": standardized,
            "revenue_checks": revenue_checks,
            "support_checks": support_checks,
            "temporal_support_diagnostic": temporal["support"],
            "zero_sale_noninferiority": zero_sale,
            "itt_revenue": {"available": itt_revenue["available"], **itt_revenue},
        },
        "notes": [
            "150 completed games are required in each arm for every observed p stratum",
            "time blocks are diagnostics, not configuration-standardization dimensions",
            "at least 90% of resolved observations in each arm must be on common support",
            "no efficacy rollback boundary was frozen for persuasion",
        ],
    }


def evaluate_gate(experiment: Mapping[str, Any]) -> dict | None:
    """Evaluate a supported experiment without modifying its raw metrics."""
    name = experiment.get("name")
    family = experiment.get("family")
    if name == "barg_dis_anchor" or family == "bargaining":
        return _bargaining_gate(experiment)
    if name == "pers_blind_lie" or family == "persuasion":
        return _persuasion_gate(experiment)
    return None


def evaluate_report_gates(report: Mapping[str, Any]) -> dict[str, dict]:
    """Return gates keyed by experiment name; unsupported families are omitted."""
    gates: dict[str, dict] = {}
    experiments = report.get("experiments", [])
    if not isinstance(experiments, list):
        return gates
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            continue
        gate = evaluate_gate(experiment)
        if gate is not None:
            gates[str(experiment.get("name", gate["family"]))] = gate
    return gates
