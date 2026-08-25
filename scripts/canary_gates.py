"""Deterministic promotion and rollback gates for live canary reports.

The functions in this module are deliberately pure and read-only.  They take
one experiment object produced by :mod:`scripts.canary_report`, return a new
gate object, and never mutate the raw report.  Keeping the evaluator separate
also lets the collector's JSON schema remain backward compatible.

The amended-v2 design uses pre-cut fixed configuration weights, deadline-zero
all-enrolled ITT outcomes, observed second moments, one-sided Wilson bounds,
and Newcombe/MOVER risk-difference bounds.  Missing or malformed evidence is
never converted to a favorable zero.  Until a manifest-backed, within-game
switchback exists, fixed-label evidence is explicitly capped at ``screen_pass``
and is not described as confirmatory.

Integration is intentionally one line after a raw report is built::

    report["gates"] = evaluate_report_gates(report)
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any


RULE_VERSION = "2026-08-25-amended-v2"
ONE_SIDED_95_Z = 1.6448536269514722

AMENDMENT_PROVENANCE = {
    "original_v1_commit": "3bbd78b10a06024819b8888680f467237523f1ff",
    "original_v1_committed_at_utc": "2026-08-25T17:05:29Z",
    "support_amendment_commit": "843cf92a65ba23850c8c531b54db1d12d3e1260d",
    "support_amendment_at_utc": "2026-08-25T17:15:16Z",
    "v2_design_frozen_at_utc": "2026-08-25T17:44:20Z",
    "frozen_before_any_live_outcomes": False,
    "live_outcomes_inspected": True,
    "prospective_confirmatory_cutoff": {
        "max_persisted_event_ts": 1787681087.9577537,
        "max_persisted_event_utc": "2026-08-25T18:04:47.957754+00:00",
        "reporter_parent_git_head": "5fe69ce9863ee3053932bf99eaa82feee612ce7b",
        "reporter_source_sha256": (
            "cad077340a8372eb6c3f397e82e00ac1e2c2550f633c4c22ef8198e53bbcaf9c"
        ),
        "bargaining": {
            "affected": {
                "treatment": {"games": 56, "resolved": 55, "censored": 1},
                "control": {"games": 99, "resolved": 96, "censored": 3},
            },
            "all_enrolled_itt": {
                "treatment": {
                    "games": 368,
                    "matured": 303,
                    "pending_maturation": 65,
                    "resolved": 300,
                    "censored": 3,
                    "invalid_terminals": 0,
                },
                "control": {
                    "games": 744,
                    "matured": 658,
                    "pending_maturation": 86,
                    "resolved": 650,
                    "censored": 8,
                    "invalid_terminals": 0,
                },
            },
        },
        "persuasion": {
            "affected": {
                "treatment": {"games": 106, "resolved": 104, "censored": 15},
                "control": {"games": 94, "resolved": 102, "censored": 10},
            },
            "all_enrolled_itt": {
                "treatment": {
                    "games": 119,
                    "matured": 77,
                    "pending_maturation": 42,
                    "resolved": 76,
                    "censored": 1,
                    "invalid_terminals": 0,
                },
                "control": {
                    "games": 112,
                    "matured": 88,
                    "pending_maturation": 24,
                    "resolved": 88,
                    "censored": 0,
                    "invalid_terminals": 0,
                },
            },
        },
        "prior_rows_status": "pilot_or_exploratory_screen_only",
        "confirmatory_requirement": (
            "first assignment and enrollment must be strictly after the max "
            "persisted event timestamp, with a frozen prefix identity"
        ),
    },
    "claim_scope": "exploratory efficacy screen and safety monitoring",
    "rationale": (
        "methodological hardening after sparse-support, censoring, and "
        "post-outcome-weighting false-positive audits; thresholds use fixed "
        "pre-cut reference distributions rather than observed canary effects"
    ),
}

BARG_ITT_WEIGHTS = {
    ("player_1", "finite"): 0.248379196081,
    ("player_1", "unlimited"): 0.245209623973,
    ("player_2", "finite"): 0.254430197378,
    ("player_2", "unlimited"): 0.251980982567,
}
BARG_AFFECTED_WEIGHTS = {
    ("player_1", "finite", "offer"): 0.403147699758,
    ("player_1", "unlimited", "offer"): 0.380145278450,
    ("player_2", "finite", "decision"): 0.010895883777,
    ("player_2", "finite", "offer"): 0.100484261501,
    ("player_2", "unlimited", "offer"): 0.105326876513,
}
BARG_OPPORTUNITY_PREVALENCE = 0.119003025501

PERS_P_WEIGHTS = {"0.333333": 592 / 1762, "0.5": 597 / 1762, "0.8": 573 / 1762}
_PERS_CELL_COUNTS = {
    ("0.333333", "binary", 100.0, 20): 95,
    ("0.333333", "binary", 10000.0, 20): 100,
    ("0.333333", "binary", 1000000.0, 20): 107,
    ("0.333333", "text", 100.0, 20): 103,
    ("0.333333", "text", 10000.0, 20): 84,
    ("0.333333", "text", 1000000.0, 20): 103,
    ("0.5", "binary", 100.0, 20): 104,
    ("0.5", "binary", 10000.0, 20): 88,
    ("0.5", "binary", 1000000.0, 20): 109,
    ("0.5", "text", 100.0, 20): 95,
    ("0.5", "text", 10000.0, 20): 105,
    ("0.5", "text", 1000000.0, 20): 96,
    ("0.8", "binary", 100.0, 20): 92,
    ("0.8", "binary", 10000.0, 20): 97,
    ("0.8", "binary", 1000000.0, 20): 95,
    ("0.8", "text", 100.0, 20): 88,
    ("0.8", "text", 10000.0, 20): 89,
    ("0.8", "text", 1000000.0, 20): 112,
}
PERS_CELL_WEIGHTS = {key: count / 1762 for key, count in _PERS_CELL_COUNTS.items()}


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


def _strict_count(value: Any) -> int | None:
    """Parse a required count without turning absent evidence into zero."""
    if type(value) is not int or value < 0:
        return None
    return value


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
    n = _strict_count(trials)
    k = _strict_count(successes)
    valid = n is not None and k is not None and n > 0 and k <= n
    if not valid:
        return {
            "successes": k,
            "trials": n,
            "rate": None,
            "lower_95_one_sided": None,
            "upper_95_one_sided": None,
            "valid": False,
        }
    assert n is not None and k is not None
    lower, upper = _wilson_bounds(k, n)
    return {
        "successes": k,
        "trials": n,
        "rate": k / n if n else None,
        "lower_95_one_sided": lower,
        "upper_95_one_sided": upper,
        "valid": True,
    }


def _newcombe_difference(
    treatment_successes: Any,
    treatment_trials: Any,
    control_successes: Any,
    control_trials: Any,
) -> dict:
    """One-sided Newcombe/MOVER upper bound for two independent risks."""
    treatment = _binomial_summary(treatment_successes, treatment_trials)
    control = _binomial_summary(control_successes, control_trials)
    if not treatment["valid"] or not control["valid"]:
        return {
            "available": False,
            "treatment": treatment,
            "control": control,
            "difference": None,
            "lower_95_one_sided": None,
            "upper_95_one_sided": None,
            "method": "newcombe_mover_wilson",
        }
    pt = treatment["rate"]
    pc = control["rate"]
    assert pt is not None and pc is not None
    upper_t = treatment["upper_95_one_sided"]
    lower_c = control["lower_95_one_sided"]
    lower_t = treatment["lower_95_one_sided"]
    upper_c = control["upper_95_one_sided"]
    assert None not in (upper_t, lower_c, lower_t, upper_c)
    difference = pt - pc
    upper = difference + math.sqrt((upper_t - pt) ** 2 + (pc - lower_c) ** 2)
    lower = difference - math.sqrt((pt - lower_t) ** 2 + (upper_c - pc) ** 2)
    return {
        "available": True,
        "treatment": treatment,
        "control": control,
        "difference": difference,
        "lower_95_one_sided": lower,
        "upper_95_one_sided": upper,
        "method": "newcombe_mover_wilson",
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


def _aggregate_difference(
    treatment: Mapping[str, Any], control: Mapping[str, Any], *, outcome: str
) -> dict:
    tm = _entry_moments(treatment, outcome)
    cm = _entry_moments(control, outcome)
    if not tm["valid"] or not cm["valid"]:
        return {
            "available": False,
            "difference": None,
            "lower_95_one_sided": None,
            "method": "raw_all_enrolled_observed_variance",
        }
    difference = tm["mean"] - cm["mean"]
    variance = 0.0
    fallback = False
    for arm in (tm, cm):
        if arm["n"] >= 2 and arm["sum_sq"] is not None:
            sample_variance = max(
                (arm["sum_sq"] - arm["sum"] ** 2 / arm["n"])
                / (arm["n"] - 1),
                0.0,
            )
        else:
            sample_variance = 0.25
            fallback = True
        variance += sample_variance / arm["n"]
    standard_error = math.sqrt(variance)
    return {
        "available": True,
        "treatment_mean": tm["mean"],
        "control_mean": cm["mean"],
        "difference": difference,
        "standard_error": standard_error,
        "lower_95_one_sided": difference - ONE_SIDED_95_Z * standard_error,
        "method": "raw_all_enrolled_observed_variance",
        "worst_case_variance_fallback": fallback,
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


def _canonical_fixed_value(field: str, value: Any) -> Any:
    if field == "p":
        number = _number(value)
        if number is None:
            return "invalid"
        for expected, key in ((1 / 3, "0.333333"), (0.5, "0.5"), (0.8, "0.8")):
            if math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12):
                return key
        return f"unsupported:{number:.17g}"
    if field == "price":
        number = _number(value)
        return number if number is not None else "invalid"
    return value


def _fixed_key(cell: Mapping[str, Any], fields: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(
        _canonical_fixed_value(field, cell.get(field)) for field in fields
    )


def _entry_moments(entry: Mapping[str, Any], outcome: str) -> dict:
    if outcome == "normalized_payoff":
        mean_key = "mean_normalized_payoff"
        sum_key = "normalized_payoff_sum"
        square_key = "normalized_payoff_sum_squares"
    elif outcome == "revenue_share":
        mean_key = "mean_revenue_share"
        sum_key = "revenue_share_sum"
        square_key = "revenue_share_sum_squares"
    else:
        mean_key = "mean_normalized_outcome"
        sum_key = "normalized_outcome_sum"
        square_key = "normalized_outcome_sum_squares"
    n = _strict_count(entry.get("resolved" if outcome != "itt" else "matured"))
    mean = _bounded_mean(entry.get(mean_key))
    if n is None or n <= 0 or mean is None:
        return {"valid": False, "n": n, "mean": mean, "sum": None, "sum_sq": None}
    total = _number(entry.get(sum_key))
    sum_sq = _number(entry.get(square_key))
    valid = total is not None and 0 <= total <= n and abs(total / n - mean) <= 1e-9
    if sum_sq is not None:
        lower = total * total / n if total is not None else math.inf
        valid = valid and lower - 1e-9 <= sum_sq <= total + 1e-9
    else:
        valid = False
    return {
        "valid": valid,
        "n": n,
        "mean": mean,
        "sum": total,
        "sum_sq": sum_sq,
    }


def _aggregate_fixed_cells(
    metric: Mapping[str, Any],
    fields: tuple[str, ...],
    outcome: str,
) -> tuple[dict[tuple[Any, ...], dict], list[str]]:
    groups: dict[tuple[Any, ...], dict] = {}
    failures: list[str] = []
    cells = metric.get("cells")
    if not isinstance(cells, Mapping):
        return {}, ["cells missing or malformed"]
    for raw_key, entry in cells.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("cell"), Mapping):
            failures.append(f"cell {raw_key!s} malformed")
            continue
        key = _fixed_key(entry["cell"], fields)
        moments = _entry_moments(entry, outcome)
        if not moments["valid"]:
            failures.append(f"cell {raw_key!s} has invalid moments")
            continue
        group = groups.setdefault(
            key,
            {"n": 0, "sum": 0.0, "sum_sq": 0.0, "has_sum_sq": True},
        )
        group["n"] += moments["n"]
        group["sum"] += moments["sum"]
        if moments["sum_sq"] is None:
            group["has_sum_sq"] = False
        else:
            group["sum_sq"] += moments["sum_sq"]
    return groups, failures


def _fixed_standardized_difference(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
    fields: tuple[str, ...],
    weights: Mapping[tuple[Any, ...], float],
    outcome: str,
) -> dict:
    """Fixed-weight contrast with no renormalization and conservative missing mass."""
    tg, t_failures = _aggregate_fixed_cells(treatment, fields, outcome)
    cg, c_failures = _aggregate_fixed_cells(control, fields, outcome)
    treatment_extra = sorted(set(tg) - set(weights), key=repr)
    control_extra = sorted(set(cg) - set(weights), key=repr)
    if treatment_extra:
        t_failures.append(f"unexpected fixed cells: {treatment_extra!r}")
    if control_extra:
        c_failures.append(f"unexpected fixed cells: {control_extra!r}")
    common_mass = 0.0
    treatment_mean = 0.0
    control_mean = 0.0
    variance = 0.0
    observed_variance_cells = 0
    fallback_variance_cells = 0
    cells: list[dict] = []
    common_t = 0
    common_c = 0
    for key, weight in weights.items():
        tm = tg.get(key)
        cm = cg.get(key)
        supported = tm is not None and cm is not None and tm["n"] > 0 and cm["n"] > 0
        row = {
            "cell": _group_label(fields, key),
            "weight": weight,
            "supported": supported,
            "treatment_resolved": tm["n"] if tm is not None else 0,
            "control_resolved": cm["n"] if cm is not None else 0,
        }
        if supported:
            mt = tm["sum"] / tm["n"]
            mc = cm["sum"] / cm["n"]
            treatment_mean += weight * mt
            control_mean += weight * mc
            common_mass += weight
            common_t += tm["n"]
            common_c += cm["n"]
            row.update({"treatment_mean": mt, "control_mean": mc})
            for arm in (tm, cm):
                if arm["has_sum_sq"] and arm["n"] >= 2:
                    sample_variance = max(
                        (arm["sum_sq"] - arm["sum"] ** 2 / arm["n"])
                        / (arm["n"] - 1),
                        0.0,
                    )
                    observed_variance_cells += 1
                else:
                    sample_variance = 0.25
                    fallback_variance_cells += 1
                variance += weight * weight * sample_variance / arm["n"]
        cells.append(row)
    missing_mass = max(1.0 - common_mass, 0.0)
    observed_difference = treatment_mean - control_mean
    standard_error = math.sqrt(variance) if common_mass > 0 else None
    sampling_lower = (
        observed_difference - ONE_SIDED_95_Z * standard_error
        if standard_error is not None
        else None
    )
    conservative_point = observed_difference - missing_mass
    conservative_lower = (
        sampling_lower - missing_mass if sampling_lower is not None else None
    )
    total_t = _strict_count(
        treatment.get("resolved" if outcome != "itt" else "matured")
    )
    total_c = _strict_count(control.get("resolved" if outcome != "itt" else "matured"))
    return {
        "available": common_mass > 0 and not t_failures and not c_failures,
        "dimensions": list(fields),
        "treatment_mean_on_fixed_scale": treatment_mean,
        "control_mean_on_fixed_scale": control_mean,
        "difference": observed_difference,
        "missing_mass_conservative_difference": conservative_point,
        "standard_error": standard_error,
        "lower_95_one_sided": sampling_lower,
        "missing_mass_conservative_lower_95_one_sided": conservative_lower,
        "method": "fixed_weight_observed_variance_with_bounded_fallback",
        "variance": {
            "observed_arm_cells": observed_variance_cells,
            "worst_case_fallback_arm_cells": fallback_variance_cells,
            "clustering_limitation": (
                "cell aggregates do not identify within-agent or temporal clustering"
            ),
        },
        "support": {
            "dimensions": list(fields),
            "fixed_mass": sum(weights.values()),
            "common_mass": common_mass,
            "missing_mass": missing_mass,
            "treatment_common_observations": common_t,
            "control_common_observations": common_c,
            "treatment_total_observations": total_t,
            "control_total_observations": total_c,
            "treatment_count_coverage": common_t / total_t if total_t else None,
            "control_count_coverage": common_c / total_c if total_c else None,
        },
        "integrity_failures": {
            "treatment": t_failures,
            "control": c_failures,
        },
        "cells": cells,
    }


def _arm_rate_guardrails(experiment: Mapping[str, Any]) -> dict:
    """Compare treatment/control censor and logged-error event rates.

    The health schema does not retain event IDs.  The union lower bound is
    ``turn_errors + max(result_errors, invalid_results)`` and its upper bound
    is ``turn_errors + min(result_events, result_errors + invalid_results)``.
    Promotion uses the worst-case treatment-upper minus control-lower excess.
    """
    metrics = experiment.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}

    censor: dict[str, dict] = {}
    for arm in ("treatment", "control"):
        metric = metrics.get(arm, {})
        metric = metric if isinstance(metric, Mapping) else {}
        censored = _strict_count(metric.get("censored"))
        resolved = _strict_count(metric.get("resolved"))
        observed = resolved + censored if resolved is not None and censored is not None else None
        censor[arm] = {
            "censored": censored,
            "observed": observed,
            "rate": censored / observed if censored is not None and observed else None,
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
        lower_failures = 0
        upper_failures = 0
        valid = bool(labels)
        integrity_failures: list[str] = []
        for label in labels:
            agent = agents.get(label, {})
            agent = agent if isinstance(agent, Mapping) else {}
            health = agent.get("health", {})
            health = health if isinstance(health, Mapping) else {}
            required = {
                key: _strict_count(health.get(key))
                for key in (
                    "turns",
                    "result_events",
                    "turn_errors",
                    "result_errors",
                    "invalid_results",
                )
            }
            if any(value is None for value in required.values()):
                valid = False
                integrity_failures.append(f"{label}: missing/malformed health count")
                continue
            turns = required["turns"]
            results = required["result_events"]
            turn_errors = required["turn_errors"]
            result_errors = required["result_errors"]
            invalid_results = required["invalid_results"]
            assert None not in (
                turns, results, turn_errors, result_errors, invalid_results
            )
            if (
                turn_errors > turns
                or result_errors > results
                or invalid_results > results
            ):
                valid = False
                integrity_failures.append(f"{label}: health count exceeds denominator")
                continue
            events += turns + results
            lower_failures += turn_errors + max(result_errors, invalid_results)
            upper_failures += turn_errors + min(
                results, result_errors + invalid_results
            )
        valid = valid and events > 0 and upper_failures <= events
        absolute = _binomial_summary(upper_failures, events) if valid else _binomial_summary(None, None)
        errors[arm] = {
            "failures_lower_bound": lower_failures if valid else None,
            "failures_upper_bound": upper_failures if valid else None,
            "events": events,
            "rate_lower_bound": lower_failures / events if valid else None,
            "rate_upper_bound": upper_failures / events if valid else None,
            "absolute_wilson": absolute,
            "absolute_upper_at_most_0.01": bool(
                absolute["upper_95_one_sided"] is not None
                and absolute["upper_95_one_sided"] <= 0.01
            ),
            "valid": valid,
            "integrity_failures": integrity_failures,
        }
    error_excess = (
        errors["treatment"]["rate_upper_bound"]
        - errors["control"]["rate_lower_bound"]
        if errors["treatment"]["rate_upper_bound"] is not None
        and errors["control"]["rate_lower_bound"] is not None
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
            "absolute_check": bool(
                errors["treatment"]["absolute_upper_at_most_0.01"]
                and errors["control"]["absolute_upper_at_most_0.01"]
            ),
            "method": "logged_event_union_bounds",
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


def _validate_moment_bounds(total: Any, sum_squares: Any, n: int) -> bool:
    total_number = _number(total)
    squares_number = _number(sum_squares)
    if total_number is None or squares_number is None or not 0 <= total_number <= n:
        return False
    lower = total_number * total_number / n if n else 0.0
    return lower - 1e-9 <= squares_number <= total_number + 1e-9


def _validate_affected_arm(metric: Mapping[str, Any], family: str) -> list[str]:
    failures: list[str] = []
    total_key = "affected_games" if family == "bargaining" else "blind_seller_games"
    required = {
        key: _strict_count(metric.get(key))
        for key in (total_key, "resolved", "censored")
    }
    if any(value is None for value in required.values()):
        return ["required affected count missing/malformed"]
    total = required[total_key]
    resolved = required["resolved"]
    censored = required["censored"]
    assert total is not None and resolved is not None and censored is not None
    if total != resolved + censored:
        failures.append("affected total != resolved + censored")
    if family == "bargaining":
        games = _strict_count(metric.get("games"))
        if games is None or games != total:
            failures.append("bargaining games != affected_games")
    if family == "bargaining":
        valid = _strict_count(metric.get("normalized_payoff_valid"))
        invalid = _strict_count(metric.get("normalized_payoff_invalid"))
        if valid is None or invalid is None or valid + invalid != resolved:
            failures.append("normalized payoff validity counts inconsistent")
        if not _validate_moment_bounds(
            metric.get("normalized_payoff_sum"),
            metric.get("normalized_payoff_sum_squares"),
            resolved,
        ):
            failures.append("normalized payoff moments invalid")
    else:
        valid = _strict_count(metric.get("revenue_share_valid"))
        invalid = _strict_count(metric.get("revenue_share_invalid"))
        zero = _strict_count(metric.get("zero_sales"))
        if valid is None or invalid is None or valid + invalid != resolved:
            failures.append("revenue validity counts inconsistent")
        if zero is None or zero > resolved:
            failures.append("zero_sales missing/malformed/inconsistent")
        if not _validate_moment_bounds(
            metric.get("revenue_share_sum"),
            metric.get("revenue_share_sum_squares"),
            resolved,
        ):
            failures.append("revenue moments invalid")
    cells = metric.get("cells")
    if not isinstance(cells, Mapping):
        failures.append("affected cells missing/malformed")
    else:
        cell_resolved = 0
        cell_censored = 0
        cell_sum = 0.0
        cell_sum_sq = 0.0
        cells_valid = True
        sum_key = (
            "normalized_payoff_sum" if family == "bargaining" else "revenue_share_sum"
        )
        square_key = (
            "normalized_payoff_sum_squares"
            if family == "bargaining"
            else "revenue_share_sum_squares"
        )
        for entry in cells.values():
            if not isinstance(entry, Mapping):
                cells_valid = False
                continue
            n = _strict_count(entry.get("resolved"))
            c = _strict_count(entry.get("censored"))
            total_value = _number(entry.get(sum_key))
            square_value = _number(entry.get(square_key))
            if None in (n, c, total_value, square_value):
                cells_valid = False
                continue
            cell_resolved += n
            cell_censored += c
            cell_sum += total_value
            cell_sum_sq += square_value
        if not cells_valid:
            failures.append("affected cell aggregate malformed")
        if cell_resolved != resolved:
            failures.append("affected cell resolved counts are not additive")
        if cell_censored != censored:
            failures.append("affected cell censored counts are not additive")
        top_sum = _number(metric.get(sum_key))
        top_sq = _number(metric.get(square_key))
        if top_sum is None or not math.isclose(cell_sum, top_sum, abs_tol=1e-9):
            failures.append("affected cell sums are not additive")
        if top_sq is None or not math.isclose(cell_sum_sq, top_sq, abs_tol=1e-9):
            failures.append("affected cell sums of squares are not additive")
    return failures


def _validate_itt_arm(metric: Mapping[str, Any], maturity_lag_s: int) -> list[str]:
    failures: list[str] = []
    required_keys = (
        "games",
        "matured",
        "pending_maturation",
        "resolved",
        "censored",
        "deadline_censored",
        "timely_valid_terminals",
        "deadline_zeroes",
        "zero_sales",
    )
    counts = {key: _strict_count(metric.get(key)) for key in required_keys}
    if any(value is None for value in counts.values()):
        return ["required ITT count missing/malformed"]
    if _strict_count(metric.get("maturity_lag_s")) != maturity_lag_s:
        failures.append("maturity lag does not match amended-v2")
    values = {key: int(value) for key, value in counts.items() if value is not None}
    if values["games"] != values["matured"] + values["pending_maturation"]:
        failures.append("ITT games != matured + pending")
    invalid = _strict_count(metric.get("invalid_terminals"))
    if invalid is None:
        failures.append("ITT invalid terminal count missing/malformed")
        invalid = 0
    if values["matured"] != values["resolved"] + values["censored"] + invalid:
        failures.append("ITT matured != resolved + deadline-censored + invalid")
    if values["resolved"] != values["timely_valid_terminals"]:
        failures.append("ITT resolved != timely valid terminals")
    if values["censored"] != values["deadline_censored"]:
        failures.append("ITT censored != deadline_censored")
    if values["deadline_zeroes"] != values["censored"] + invalid:
        failures.append("ITT deadline zeroes != censored + invalid")
    if invalid:
        failures.append("ITT contains invalid timely terminals")
    if values["zero_sales"] > values["matured"]:
        failures.append("ITT zero sales exceed matured")
    if not _validate_moment_bounds(
        metric.get("normalized_outcome_sum"),
        metric.get("normalized_outcome_sum_squares"),
        values["matured"],
    ):
        failures.append("ITT moments invalid")
    cells = metric.get("cells")
    if not isinstance(cells, Mapping):
        failures.append("ITT cells missing/malformed")
    else:
        for key in ("games", "matured", "resolved", "censored"):
            subtotal = 0
            valid = True
            for entry in cells.values():
                count = _strict_count(entry.get(key)) if isinstance(entry, Mapping) else None
                if count is None:
                    valid = False
                    break
                subtotal += count
            if not valid or subtotal != values[key]:
                failures.append(f"ITT cell {key} counts are not additive")
        total_sum = 0.0
        total_sq = 0.0
        moments_valid = True
        for entry in cells.values():
            raw_sum = _number(entry.get("normalized_outcome_sum"))
            raw_sq = _number(entry.get("normalized_outcome_sum_squares"))
            if raw_sum is None or raw_sq is None:
                moments_valid = False
                continue
            total_sum += raw_sum
            total_sq += raw_sq
        top_sum = _number(metric.get("normalized_outcome_sum"))
        top_sq = _number(metric.get("normalized_outcome_sum_squares"))
        if (
            not moments_valid
            or top_sum is None
            or not math.isclose(total_sum, top_sum, abs_tol=1e-9)
        ):
            failures.append("ITT cell sums are not additive")
        if (
            not moments_valid
            or top_sq is None
            or not math.isclose(total_sq, top_sq, abs_tol=1e-9)
        ):
            failures.append("ITT cell sums of squares are not additive")
    p_strata = metric.get("p_strata")
    if isinstance(p_strata, Mapping) and p_strata:
        for key in ("games", "matured", "resolved", "censored"):
            subtotal = 0
            valid = True
            for entry in p_strata.values():
                count = _strict_count(entry.get(key)) if isinstance(entry, Mapping) else None
                if count is None:
                    valid = False
                    break
                subtotal += count
            if not valid or subtotal != values[key]:
                failures.append(f"ITT p-stratum {key} counts are not additive")
    return failures


def _sum_itt_stratum(metric: Mapping[str, Any], field: str, value: Any) -> dict:
    matured = 0
    censored = 0
    valid = True
    cells = metric.get("cells")
    if not isinstance(cells, Mapping):
        return {"matured": None, "censored": None, "valid": False}
    for entry in cells.values():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("cell"), Mapping):
            valid = False
            continue
        if _canonical_fixed_value(field, entry["cell"].get(field)) != value:
            continue
        n = _strict_count(entry.get("matured"))
        k = _strict_count(entry.get("censored"))
        if n is None or k is None or k > n:
            valid = False
            continue
        matured += n
        censored += k
    return {"matured": matured, "censored": censored, "valid": valid and matured > 0}


def _censor_safety(
    itt: Mapping[str, Any],
    *,
    family: str,
) -> dict:
    treatment = itt.get("treatment", {})
    control = itt.get("control", {})
    treatment = treatment if isinstance(treatment, Mapping) else {}
    control = control if isinstance(control, Mapping) else {}
    if family == "bargaining":
        strata = {"overall": (None, None, 0.03), "finite": ("horizon", "finite", 0.03), "unlimited": ("horizon", "unlimited", 0.03)}
    else:
        strata = {
            "overall": (None, None, 0.06),
            "0.333333": ("p", "0.333333", 0.06),
            "0.5": ("p", "0.5", 0.07),
            "0.8": ("p", "0.8", 0.06),
        }
    output: dict[str, dict] = {}
    for name, (field, value, absolute_ceiling) in strata.items():
        samples: dict[str, dict] = {}
        for arm, metric in (("treatment", treatment), ("control", control)):
            if field is None:
                n = _strict_count(metric.get("matured"))
                k = _strict_count(metric.get("censored"))
                sample_valid = n is not None and k is not None and n > 0 and k <= n
            else:
                aggregate = _sum_itt_stratum(metric, field, value)
                n = aggregate["matured"]
                k = aggregate["censored"]
                sample_valid = bool(aggregate["valid"])
            summary = _binomial_summary(k, n) if sample_valid else _binomial_summary(None, None)
            summary["absolute_ceiling"] = absolute_ceiling
            summary["absolute_pass"] = bool(
                summary["upper_95_one_sided"] is not None
                and summary["upper_95_one_sided"] <= absolute_ceiling
            )
            samples[arm] = summary
        excess = _newcombe_difference(
            samples["treatment"]["successes"],
            samples["treatment"]["trials"],
            samples["control"]["successes"],
            samples["control"]["trials"],
        )
        excess["ceiling"] = 0.03
        excess["pass"] = bool(
            excess["upper_95_one_sided"] is not None
            and excess["upper_95_one_sided"] <= 0.03
        )
        output[name] = {
            **samples,
            "treatment_minus_control": excess,
            "pass": bool(
                samples["treatment"]["absolute_pass"]
                and samples["control"]["absolute_pass"]
                and excess["pass"]
            ),
        }
    return {"strata": output, "pass": all(item["pass"] for item in output.values())}


def _scheduled_censor_harm(
    censor_safety: Mapping[str, Any], *, treatment_min: int, control_min: int
) -> list[str]:
    strata = censor_safety.get("strata", {})
    strata = strata if isinstance(strata, Mapping) else {}
    overall = strata.get("overall", {})
    overall = overall if isinstance(overall, Mapping) else {}
    treatment = overall.get("treatment", {})
    control = overall.get("control", {})
    treatment_n = _strict_count(treatment.get("trials")) if isinstance(treatment, Mapping) else None
    control_n = _strict_count(control.get("trials")) if isinstance(control, Mapping) else None
    if (
        treatment_n is None
        or control_n is None
        or treatment_n < treatment_min
        or control_n < control_min
    ):
        return []
    reasons: list[str] = []
    for name, entry in strata.items():
        if not isinstance(entry, Mapping):
            continue
        for arm in ("treatment", "control"):
            summary = entry.get(arm, {})
            if not isinstance(summary, Mapping):
                continue
            lower = _number(summary.get("lower_95_one_sided"))
            ceiling = _number(summary.get("absolute_ceiling"))
            if lower is not None and ceiling is not None and lower > ceiling:
                reasons.append(f"{name} {arm} censor Wilson lower exceeds ceiling")
        contrast = entry.get("treatment_minus_control", {})
        if isinstance(contrast, Mapping):
            lower = _number(contrast.get("lower_95_one_sided"))
            if lower is not None and lower > 0.03:
                reasons.append(f"{name} censor excess MOVER lower exceeds 0.03")
    return reasons


def _causal_confirmation(experiment: Mapping[str, Any], family: str) -> dict:
    evidence = experiment.get("randomization_evidence")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    blocks = evidence.get("agent_blocks")
    blocks = blocks if isinstance(blocks, Mapping) else {}
    labels = ("main", "test_a", "test_b", "test_c")
    representation = 50 if family == "bargaining" else 150
    control_representation = 150
    label_checks: dict[str, dict] = {}
    for label in labels:
        entry = blocks.get(label, {})
        entry = entry if isinstance(entry, Mapping) else {}
        treatment_blocks = _strict_count(entry.get("treatment_blocks"))
        control_blocks = _strict_count(entry.get("control_blocks"))
        treatment_n = _strict_count(entry.get("treatment_n"))
        control_n = _strict_count(entry.get("control_n"))
        label_checks[label] = {
            "treatment_blocks_at_least_30": bool(
                treatment_blocks is not None and treatment_blocks >= 30
            ),
            "control_blocks_at_least_30": bool(
                control_blocks is not None and control_blocks >= 30
            ),
            "treatment_representation": bool(
                treatment_n is not None and treatment_n >= representation
            ),
            "control_representation": bool(
                control_n is not None and control_n >= control_representation
            ),
        }
    design_checks = {
        "manifest_backed": evidence.get("manifest_backed") is True,
        "approved_manifest_identity": evidence.get("approved_manifest_identity") is True,
        "within_game_randomization": evidence.get("within_game_randomization") is True,
        "mirrored_simultaneous_periods": evidence.get("mirrored_simultaneous_periods") is True,
        "strictly_post_v2_checkpoint": evidence.get("strictly_post_v2_checkpoint") is True,
    }
    passed = all(design_checks.values()) and all(
        check for entry in label_checks.values() for check in entry.values()
    )
    return {
        "pass": passed,
        "design_checks": design_checks,
        "labels": label_checks,
        "status_cap_without_pass": "screen_pass",
    }


def _health_guardrails(experiment: Mapping[str, Any]) -> dict:
    totals = defaultdict(int)
    agents = experiment.get("agents", {})
    assignment = experiment.get("assignment", {})
    assignment = assignment if isinstance(assignment, Mapping) else {}
    expected_agents = {
        str(label)
        for arm in ("treatment_agents", "control_agents")
        for label in (
            assignment.get(arm, [])
            if isinstance(assignment.get(arm, []), (list, tuple))
            else []
        )
    }
    evidence_failures: list[str] = []
    if isinstance(agents, Mapping):
        for label in expected_agents:
            agent = agents.get(label)
            if not isinstance(agent, Mapping):
                evidence_failures.append(f"{label}: agent evidence missing")
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
                value = _strict_count(health.get(key))
                if value is None:
                    evidence_failures.append(f"{label}: health.{key} missing/malformed")
                else:
                    totals[key] += value
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
                value = _strict_count(routing.get(key))
                if value is None:
                    evidence_failures.append(f"{label}: routing.{key} missing/malformed")
                else:
                    totals[key] += value
            checked = _strict_count(routing.get("checked"))
            matches = _strict_count(routing.get("assigned_matches"))
            affected = _strict_count(routing.get("affected"))
            affected_matches = _strict_count(routing.get("affected_assigned_matches"))
            if checked is not None and matches is not None and matches > checked:
                evidence_failures.append(f"{label}: routing matches exceed checked")
            if (
                affected is not None
                and affected_matches is not None
                and affected_matches > affected
            ):
                evidence_failures.append(f"{label}: affected matches exceed affected")
    else:
        evidence_failures.append("agents evidence missing/malformed")
    if not expected_agents:
        evidence_failures.append("assignment labels missing")

    metrics = experiment.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    deterministic_checked = 0
    deterministic_matches = 0
    for arm in ("treatment", "control"):
        metric = metrics.get(arm, {})
        if not isinstance(metric, Mapping):
            evidence_failures.append(f"{arm}: metrics missing/malformed")
            continue
        if experiment.get("family") == "persuasion":
            checked = _strict_count(metric.get("deterministic_route_checked"))
            matches = _strict_count(metric.get("deterministic_route_matches"))
            if checked is None or matches is None or matches > checked:
                evidence_failures.append(f"{arm}: deterministic route evidence invalid")
            else:
                deterministic_checked += checked
                deterministic_matches += matches

    # 503s are transport failures and can overlap turn/result error totals.
    # Aggregation cannot prove which invalid result belonged to which 503, so
    # these remain explicit manual-review warnings rather than silently being
    # cleared or causing an irreversible automatic rollback.
    non_transport_errors = max(
        totals["turn_errors"] + totals["result_errors"] - totals["http_503"], 0
    )
    hard_failures = {
        "missing_or_inconsistent_evidence": len(evidence_failures),
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
        "integrity_failures": evidence_failures,
    }


def _bargaining_gate(experiment: Mapping[str, Any]) -> dict:
    metrics = experiment.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    treatment = metrics.get("treatment", {})
    control = metrics.get("control", {})
    treatment = treatment if isinstance(treatment, Mapping) else {}
    control = control if isinstance(control, Mapping) else {}
    itt = experiment.get("itt", {})
    itt = itt if isinstance(itt, Mapping) else {}
    itt_treatment = itt.get("treatment", {})
    itt_control = itt.get("control", {})
    itt_treatment = itt_treatment if isinstance(itt_treatment, Mapping) else {}
    itt_control = itt_control if isinstance(itt_control, Mapping) else {}
    guardrails = _health_guardrails(experiment)
    arm_rates = _arm_rate_guardrails(experiment)

    sample_checks = {
        "treatment_affected": _check(treatment.get("affected_games"), ">=", 300),
        "treatment_resolved": _check(treatment.get("resolved"), ">=", 300),
        "control_affected": _check(control.get("affected_games"), ">=", 900),
        "control_resolved": _check(control.get("resolved"), ">=", 900),
        "treatment_all_enrolled_matured": _check(
            itt_treatment.get("matured"), ">=", 2521
        ),
        "control_all_enrolled_matured": _check(
            itt_control.get("matured"), ">=", 7563
        ),
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

    standardized = _fixed_standardized_difference(
        treatment,
        control,
        ("role", "horizon", "phase"),
        BARG_AFFECTED_WEIGHTS,
        "normalized_payoff",
    )
    itt_standardized = _fixed_standardized_difference(
        itt_treatment,
        itt_control,
        ("role", "horizon"),
        BARG_ITT_WEIGHTS,
        "itt",
    )
    raw_itt = _aggregate_difference(itt_treatment, itt_control, outcome="itt")
    payoff_checks = {
        "affected_lift": _check(
            standardized.get("missing_mass_conservative_difference"), ">=", 0.020
        ),
        "affected_lower_bound": _check(
            standardized.get("missing_mass_conservative_lower_95_one_sided"),
            ">",
            0.0,
        ),
        "itt_fixed_point_nonnegative": _check(
            itt_standardized.get("missing_mass_conservative_difference"),
            ">=",
            0.0,
        ),
        "itt_fixed_lower_bound": _check(
            itt_standardized.get("missing_mass_conservative_lower_95_one_sided"),
            ">",
            -0.010,
        ),
        "itt_raw_point_nonnegative": _check(raw_itt.get("difference"), ">=", 0.0),
    }
    support_checks = {
        "affected_common_reference_mass": _check(
            standardized["support"]["common_mass"], ">=", 0.90
        ),
        "itt_common_reference_mass": _check(
            itt_standardized["support"]["common_mass"], ">=", 0.90
        ),
    }
    censor_safety = _censor_safety(itt, family="bargaining")
    causal = _causal_confirmation(experiment, "bargaining")
    integrity_failures = {
        "treatment_affected": _validate_affected_arm(treatment, "bargaining"),
        "control_affected": _validate_affected_arm(control, "bargaining"),
        "treatment_itt": _validate_itt_arm(itt_treatment, 1200),
        "control_itt": _validate_itt_arm(itt_control, 1200),
    }
    unknown_assignments = _strict_count(
        (itt.get("integrity", {}) if isinstance(itt.get("integrity"), Mapping) else {}).get(
            "unknown_assignment_games"
        )
    )
    if unknown_assignments is None or unknown_assignments:
        integrity_failures["assignment"] = [
            "unknown or missing manifest-backed assignment evidence"
        ]
    data_integrity_pass = not any(integrity_failures.values())
    affected_unstandardized = _difference_summary(
        treatment.get("mean_normalized_payoff"),
        treatment.get("resolved"),
        control.get("mean_normalized_payoff"),
        control.get("resolved"),
    )

    rollback_triggers: list[str] = []
    if not guardrails["passed"]:
        rollback_triggers.append("health_or_routing_guardrail")
    treatment_direct = _strict_count(treatment.get("direct_resolved"))
    control_direct = _strict_count(control.get("direct_resolved"))
    treatment_direct_rate = _number(treatment.get("direct_conversion_rate"))
    if (
        treatment_direct is not None
        and control_direct is not None
        and treatment_direct >= 100
        and control_direct >= 300
        and treatment_direct_rate is not None
        and treatment_direct_rate <= 0.08
    ):
        rollback_triggers.append("scheduled affected direct conversion <= 0.08")
    itt_treatment_n = _strict_count(itt_treatment.get("matured"))
    itt_control_n = _strict_count(itt_control.get("matured"))
    deadline_payoff = _bounded_mean(itt_treatment.get("mean_normalized_outcome"))
    if (
        itt_treatment_n is not None
        and itt_control_n is not None
        and itt_treatment_n >= 100
        and itt_control_n >= 300
        and deadline_payoff is not None
        and deadline_payoff <= 0.27
    ):
        rollback_triggers.append("scheduled treatment deadline payoff <= 0.27")
    rollback_triggers.extend(
        _scheduled_censor_harm(censor_safety, treatment_min=50, control_min=150)
    )
    screen_ready = (
        all(check["passed"] for check in sample_checks.values())
        and all(
            check["passed"]
            for horizon in horizon_metrics.values()
            for check in horizon.values()
        )
        and all(item["passed"] for item in conversion.values())
        and all(check["passed"] for check in payoff_checks.values())
        and all(check["passed"] for check in support_checks.values())
        and arm_rates["errors"]["check"]["passed"]
        and arm_rates["errors"]["absolute_check"]
        and censor_safety["pass"]
        and standardized["available"]
        and itt_standardized["available"]
        and data_integrity_pass
        and guardrails["passed"]
    )
    promotion_ready = screen_ready and causal["pass"]
    if rollback_triggers:
        decision = "rollback"
    elif screen_ready and guardrails["manual_review_required"]:
        decision = "manual_review"
    elif promotion_ready:
        decision = "promote"
    elif screen_ready:
        decision = "screen_pass"
    else:
        decision = "continue"

    return {
        "experiment": experiment.get("name", "barg_dis_anchor"),
        "family": "bargaining",
        "rule_version": RULE_VERSION,
        "amendment_provenance": dict(AMENDMENT_PROVENANCE),
        "decision": decision,
        "screen_ready": screen_ready,
        "promotion_ready": promotion_ready,
        "rollback_triggers": rollback_triggers,
        "guardrails": guardrails,
        "arm_rate_guardrails": arm_rates,
        "censor_safety": censor_safety,
        "causal_confirmation": causal,
        "data_integrity": {
            "passed": data_integrity_pass,
            "failures": integrity_failures,
        },
        "sample_checks": sample_checks,
        "horizon_checks": horizon_metrics,
        "statistics": {
            "direct_conversion": conversion,
            "standardized_payoff": standardized,
            "itt_payoff": itt_standardized,
            "raw_itt_payoff": raw_itt,
            "payoff_checks": payoff_checks,
            "support_checks": support_checks,
            "affected_unstandardized_payoff": affected_unstandardized,
            "itt": itt_standardized,
        },
        "interim": {
            "efficacy_repeated_look_binding": False,
            "note": "no invented repeated-look efficacy or optional-stopping claim",
        },
        "notes": [
            "finite and unlimited minimums are required in both arms",
            "fixed reference weights are never renormalized",
            "missing reference mass is charged its worst-case bounded-outcome contrast",
            "fixed-label evidence cannot exceed screen_pass",
        ],
    }


def _persuasion_gate(experiment: Mapping[str, Any]) -> dict:
    metrics = experiment.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    treatment = metrics.get("treatment", {})
    control = metrics.get("control", {})
    treatment = treatment if isinstance(treatment, Mapping) else {}
    control = control if isinstance(control, Mapping) else {}
    itt = experiment.get("itt", {})
    itt = itt if isinstance(itt, Mapping) else {}
    itt_treatment = itt.get("treatment", {})
    itt_control = itt.get("control", {})
    itt_treatment = itt_treatment if isinstance(itt_treatment, Mapping) else {}
    itt_control = itt_control if isinstance(itt_control, Mapping) else {}
    guardrails = _health_guardrails(experiment)
    arm_rates = _arm_rate_guardrails(experiment)

    sample_checks = {
        "treatment_completed": _check(itt_treatment.get("matured"), ">=", 1000),
        "control_completed": _check(itt_control.get("matured"), ">=", 1000),
    }

    treatment_p = itt_treatment.get("p_strata", {})
    control_p = itt_control.get("p_strata", {})
    treatment_p = treatment_p if isinstance(treatment_p, Mapping) else {}
    control_p = control_p if isinstance(control_p, Mapping) else {}
    p_checks: dict[str, dict] = {}
    for p in PERS_P_WEIGHTS:
        tmetric = treatment_p.get(p, {})
        cmetric = control_p.get(p, {})
        tmetric = tmetric if isinstance(tmetric, Mapping) else {}
        cmetric = cmetric if isinstance(cmetric, Mapping) else {}
        p_checks[str(p)] = {
            "treatment_matured": _check(tmetric.get("matured"), ">=", 150),
            "control_matured": _check(cmetric.get("matured"), ">=", 150),
        }

    standardized = _fixed_standardized_difference(
        treatment,
        control,
        (
            "p",
            "message_type",
            "price",
            "total_rounds",
        ),
        PERS_CELL_WEIGHTS,
        "revenue_share",
    )
    itt_standardized = _fixed_standardized_difference(
        itt_treatment,
        itt_control,
        ("p", "message_type", "price", "total_rounds"),
        PERS_CELL_WEIGHTS,
        "itt",
    )
    raw_itt = _aggregate_difference(itt_treatment, itt_control, outcome="itt")
    affected_revenue_diagnostic = {
        "affected_lift": _check(
            standardized.get("missing_mass_conservative_difference"), ">=", 0.025
        ),
        "affected_lower_bound": _check(
            standardized.get("missing_mass_conservative_lower_95_one_sided"),
            ">",
            0.0,
        ),
    }
    revenue_checks = {
        "itt_lift": _check(
            itt_standardized.get("missing_mass_conservative_difference"),
            ">=",
            0.025,
        ),
        "itt_lower_bound": _check(
            itt_standardized.get("missing_mass_conservative_lower_95_one_sided"),
            ">",
            0.0,
        ),
        "raw_itt_point_nonnegative": _check(raw_itt.get("difference"), ">=", 0.0),
    }
    support_checks = {
        "affected_common_reference_mass": _check(
            standardized["support"]["common_mass"], ">=", 0.90
        ),
        "itt_common_reference_mass": _check(
            itt_standardized["support"]["common_mass"], ">=", 0.90
        ),
    }
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

    zero_difference = _newcombe_difference(
        itt_treatment.get("zero_sales"),
        itt_treatment.get("matured"),
        itt_control.get("zero_sales"),
        itt_control.get("matured"),
    )
    zero_sale = {
        **zero_difference,
        "check": _check(zero_difference.get("upper_95_one_sided"), "<=", 0.02),
    }
    censor_safety = _censor_safety(itt, family="persuasion")
    causal = _causal_confirmation(experiment, "persuasion")
    integrity_failures = {
        "treatment_affected": _validate_affected_arm(treatment, "persuasion"),
        "control_affected": _validate_affected_arm(control, "persuasion"),
        "treatment_itt": _validate_itt_arm(itt_treatment, 1800),
        "control_itt": _validate_itt_arm(itt_control, 1800),
    }
    unknown_assignments = _strict_count(
        (itt.get("integrity", {}) if isinstance(itt.get("integrity"), Mapping) else {}).get(
            "unknown_assignment_games"
        )
    )
    if unknown_assignments is None or unknown_assignments:
        integrity_failures["assignment"] = [
            "unknown or missing manifest-backed assignment evidence"
        ]
    data_integrity_pass = not any(integrity_failures.values())
    p_ready = bool(p_checks) and all(
        check["passed"] for pair in p_checks.values() for check in pair.values()
    )
    screen_ready = (
        all(check["passed"] for check in sample_checks.values())
        and p_ready
        and standardized["available"]
        and itt_standardized["available"]
        and all(check["passed"] for check in revenue_checks.values())
        and all(check["passed"] for check in support_checks.values())
        and zero_sale["check"]["passed"]
        and arm_rates["errors"]["check"]["passed"]
        and arm_rates["errors"]["absolute_check"]
        and censor_safety["pass"]
        and data_integrity_pass
        and guardrails["passed"]
    )
    promotion_ready = screen_ready and causal["pass"]

    rollback_triggers: list[str] = []
    if not guardrails["passed"]:
        rollback_triggers.append("health_or_routing_guardrail")
    rollback_triggers.extend(
        _scheduled_censor_harm(censor_safety, treatment_min=150, control_min=150)
    )
    if rollback_triggers:
        decision = "rollback"
    elif screen_ready and guardrails["manual_review_required"]:
        decision = "manual_review"
    elif promotion_ready:
        decision = "promote"
    elif screen_ready:
        decision = "screen_pass"
    else:
        decision = "continue"

    return {
        "experiment": experiment.get("name", "pers_blind_lie"),
        "family": "persuasion",
        "rule_version": RULE_VERSION,
        "amendment_provenance": dict(AMENDMENT_PROVENANCE),
        "decision": decision,
        "screen_ready": screen_ready,
        "promotion_ready": promotion_ready,
        "rollback_triggers": rollback_triggers,
        "guardrails": guardrails,
        "arm_rate_guardrails": arm_rates,
        "censor_safety": censor_safety,
        "causal_confirmation": causal,
        "data_integrity": {
            "passed": data_integrity_pass,
            "failures": integrity_failures,
        },
        "sample_checks": sample_checks,
        "p_strata_checks": p_checks,
        "statistics": {
            "standardized_revenue": standardized,
            "itt_standardized_revenue": itt_standardized,
            "revenue_checks": revenue_checks,
            "affected_revenue_diagnostic": affected_revenue_diagnostic,
            "support_checks": support_checks,
            "temporal_support_diagnostic": temporal["support"],
            "zero_sale_noninferiority": zero_sale,
            "itt_revenue": itt_standardized,
            "raw_itt_revenue": raw_itt,
        },
        "interim": {
            "efficacy_repeated_look_binding": False,
            "note": "no invented repeated-look efficacy or optional-stopping claim",
        },
        "notes": [
            "150 matured games are required in each arm for each fixed p stratum",
            "time blocks are diagnostics, not configuration-standardization dimensions",
            "fixed reference weights are never renormalized",
            "missing reference mass is charged its worst-case bounded-outcome contrast",
            "fixed-label evidence cannot exceed screen_pass",
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
