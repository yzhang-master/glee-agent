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
from copy import deepcopy
from typing import Any


RULE_VERSION = "2026-08-25-amended-v2"
ONE_SIDED_95_Z = 1.6448536269514722
_MAX_EXACT_COUNT = 2**53 - 1

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

_CONFIRMATION_PLAN_SHA256 = (
    "b002b688d02df3233b7dd4f21a5595cf149b4cc8dd501a0bfc2ee5bccd11d745"
)
_CONFIRMATION_PLAN_ID = "confirmation-v2-20260825-2100z"
_CONFIRMATION_STRATEGY_SHA256 = (
    "631ef69862d572644ba855174a411f80a220b11ed5c20e30b43ffc31f1303388"
)
_CONFIRMATION_ACTIVATED_AT = 1787691600
_CONFIRMATION_EXPIRES_AT = 1787950800
_CONFIRMATION_TARGET_SHA256 = {
    "data/targets.json": (
        "1d24a579ca2b611e3b30af4ddf7af5b84ad13e7198fa55b93a2f5e6617e65e25"
    ),
    "data/live_targets.json": (
        "3dcaff69f17175648e4b46499859bf183bba03b1321364de329d01bed0e618a3"
    ),
}
_CONFIRMATION_RULES = {
    "bargaining": {
        "family": "bargaining",
        "rule_id": "barg-anchor-confirm-v2",
        "knob": "barg_dis_anchor",
        "control": 0.58,
        "treatment": 0.5,
        "treatment_probability": 0.25,
    },
    "persuasion": {
        "family": "persuasion",
        "rule_id": "pers-blind-confirm-v2",
        "knob": "pers_blind_lie",
        "control": 1.0,
        "treatment": 0.4,
        "treatment_probability": 0.5,
    },
}

# Promotion accepts only the independently frozen scheduled-look declaration
# pinned here.  Reporter-supplied strings and ``status=verified`` assertions
# are not sufficient without this exact pre-outcome declaration identity.
_CONFIRMATION_LOOK_DECLARATION = {
    "schema_version": 1,
    "path": "data/canary_analysis_plan.json",
    "sha256": "39943a3877adafae71f6bdacfab13a02f0065dc1b955ef6184fbb14dfe20e260",
    "bytes": 5950,
    "look_id": "final-confirmatory-expiry-plus-persuasion-maturity-v1",
    "declared_at_ts": 1787689408,
    "analysis_as_of_ts": 1787952600,
    "enrollment_cutoff_exclusive": 1787950800,
    "prefix_procedure_version": "stable-filtered-jsonl-snapshot-v2",
    "prefix_capture_not_before": 1787952900,
    "prefix_output_path_template": (
        "data/canary-confirmation-prefix/{declaration_sha256}.json"
    ),
}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return out if math.isfinite(out) else None


def _count(value: Any) -> int:
    number = _number(value)
    if (
        number is None
        or not 0 <= number <= _MAX_EXACT_COUNT
        or not number.is_integer()
    ):
        return 0
    return int(number)


def _strict_count(value: Any) -> int | None:
    """Parse a required count without turning absent evidence into zero."""
    # Statistical paths convert counts to binary64. Reject values outside its
    # exact integer range before a division, interval, or moment calculation.
    if type(value) is not int or not 0 <= value <= _MAX_EXACT_COUNT:
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
        passed = number is not None and (
            number >= threshold
            or math.isclose(number, threshold, rel_tol=1e-12, abs_tol=1e-12)
        )
    elif operator == ">":
        passed = number is not None and number > threshold
    elif operator == "<=":
        passed = number is not None and (
            number <= threshold
            or math.isclose(number, threshold, rel_tol=1e-12, abs_tol=1e-12)
        )
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
            try:
                value = json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            except (TypeError, ValueError, OverflowError, RecursionError):
                value = "invalid"
        elif type(value) is int and not 0 <= value <= _MAX_EXACT_COUNT:
            value = "invalid"
        elif type(value) is float and not math.isfinite(value):
            value = "invalid"
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
        number = _number(value) if type(value) in (int, float) else None
        if number is None:
            return "invalid"
        for expected, key in ((1 / 3, "0.333333"), (0.5, "0.5"), (0.8, "0.8")):
            if math.isclose(number, expected, rel_tol=0.0, abs_tol=1e-12):
                return key
        return f"unsupported:{number:.17g}"
    if field == "price":
        number = _number(value) if type(value) in (int, float) else None
        return number if number is not None else "invalid"
    if field == "total_rounds":
        return value if type(value) is int and value == 20 else "invalid"
    if field == "message_type":
        return (
            value
            if type(value) is str and value in {"binary", "text"}
            else "invalid"
        )
    if field == "role":
        return (
            value
            if type(value) is str and value in {"player_1", "player_2"}
            else "invalid"
        )
    if field == "horizon":
        return (
            value
            if type(value) is str and value in {"finite", "unlimited"}
            else "invalid"
        )
    if field == "phase":
        return (
            value
            if type(value) is str and value in {"offer", "decision"}
            else "invalid"
        )
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
    # The frozen decimal weights can differ from exactly one by machine-scale
    # rounding.  Missing support is relative to that frozen mass, not a new
    # synthetic 1e-12 cell.  This preserves the no-renormalization contract
    # while allowing mathematically exact inclusive threshold equality.
    fixed_mass = sum(weights.values())
    missing_mass = max(fixed_mass - common_mass, 0.0)
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
            "fixed_mass": fixed_mass,
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


def _aggregate_fixed_risk_cells(
    metric: Mapping[str, Any],
    fields: tuple[str, ...],
) -> tuple[dict[tuple[Any, ...], dict[str, int]], list[str]]:
    """Aggregate zero-sale counts without accepting malformed evidence as zero."""
    groups: dict[tuple[Any, ...], dict[str, int]] = {}
    failures: list[str] = []
    cells = metric.get("cells")
    if not isinstance(cells, Mapping):
        return {}, ["cells missing or malformed"]
    for raw_key, entry in cells.items():
        if not isinstance(entry, Mapping) or not isinstance(entry.get("cell"), Mapping):
            failures.append(f"cell {raw_key!s} malformed")
            continue
        matured = _strict_count(entry.get("matured"))
        zero_sales = _strict_count(entry.get("zero_sales"))
        if (
            matured is None
            or matured <= 0
            or zero_sales is None
            or zero_sales > matured
        ):
            failures.append(f"cell {raw_key!s} has invalid zero-sale counts")
            continue
        key = _fixed_key(entry["cell"], fields)
        group = groups.setdefault(key, {"matured": 0, "zero_sales": 0})
        group["matured"] += matured
        group["zero_sales"] += zero_sales
    return groups, failures


def _fixed_weight_risk_difference(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
    fields: tuple[str, ...],
    weights: Mapping[tuple[Any, ...], float],
) -> dict:
    """Frozen-weight MOVER contrast for stratified zero-sale risks.

    Each cell uses a Wilson component interval.  The fixed-weight MOVER
    excursions are combined in quadrature.  Weights are never renormalized;
    unsupported reference mass is charged the worst possible treatment-minus-
    control risk contrast.
    """
    treatment_groups, treatment_failures = _aggregate_fixed_risk_cells(
        treatment, fields
    )
    control_groups, control_failures = _aggregate_fixed_risk_cells(control, fields)
    treatment_extra = sorted(set(treatment_groups) - set(weights), key=repr)
    control_extra = sorted(set(control_groups) - set(weights), key=repr)
    if treatment_extra:
        treatment_failures.append(f"unexpected fixed cells: {treatment_extra!r}")
    if control_extra:
        control_failures.append(f"unexpected fixed cells: {control_extra!r}")

    treatment_risk = 0.0
    control_risk = 0.0
    upper_excursion_sq = 0.0
    lower_excursion_sq = 0.0
    common_mass = 0.0
    cells: list[dict] = []
    for key, weight in weights.items():
        treatment_group = treatment_groups.get(key)
        control_group = control_groups.get(key)
        supported = treatment_group is not None and control_group is not None
        row = {
            "cell": _group_label(fields, key),
            "weight": weight,
            "supported": supported,
        }
        if supported:
            treatment_summary = _binomial_summary(
                treatment_group["zero_sales"], treatment_group["matured"]
            )
            control_summary = _binomial_summary(
                control_group["zero_sales"], control_group["matured"]
            )
            supported = bool(treatment_summary["valid"] and control_summary["valid"])
            row["supported"] = supported
            row["treatment"] = treatment_summary
            row["control"] = control_summary
            if supported:
                pt = treatment_summary["rate"]
                pc = control_summary["rate"]
                upper_t = treatment_summary["upper_95_one_sided"]
                lower_t = treatment_summary["lower_95_one_sided"]
                upper_c = control_summary["upper_95_one_sided"]
                lower_c = control_summary["lower_95_one_sided"]
                assert None not in (pt, pc, upper_t, lower_t, upper_c, lower_c)
                treatment_risk += weight * pt
                control_risk += weight * pc
                common_mass += weight
                upper_excursion_sq += weight * weight * (
                    (upper_t - pt) ** 2 + (pc - lower_c) ** 2
                )
                lower_excursion_sq += weight * weight * (
                    (pt - lower_t) ** 2 + (upper_c - pc) ** 2
                )
        cells.append(row)

    fixed_mass = sum(weights.values())
    missing_mass = max(fixed_mass - common_mass, 0.0)
    difference = treatment_risk - control_risk
    upper = (
        difference + math.sqrt(upper_excursion_sq) + missing_mass
        if common_mass > 0
        else None
    )
    lower = (
        difference - math.sqrt(lower_excursion_sq) - missing_mass
        if common_mass > 0
        else None
    )
    return {
        "available": bool(
            common_mass > 0 and not treatment_failures and not control_failures
        ),
        "treatment_risk_on_fixed_scale": treatment_risk,
        "control_risk_on_fixed_scale": control_risk,
        "difference": difference if common_mass > 0 else None,
        "lower_95_one_sided": lower,
        "upper_95_one_sided": upper,
        "missing_mass_conservative_upper_95_one_sided": upper,
        "method": "fixed_weight_stratified_newcombe_mover_wilson",
        "support": {
            "dimensions": list(fields),
            "fixed_mass": fixed_mass,
            "common_mass": common_mass,
            "missing_mass": missing_mass,
        },
        "integrity_failures": {
            "treatment": treatment_failures,
            "control": control_failures,
        },
        "cells": cells,
    }


def _prospective_binding_view(
    experiment: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict]:
    """Select the reporter's approved prospective cohort, fail closed if malformed.

    Top-level metrics intentionally remain a mixed/legacy diagnostic for schema
    compatibility.  Once the reporter exposes ``analysis_cohorts``, no binding
    decision may silently fall back to those mixed aggregates.
    """
    cohorts = experiment.get("analysis_cohorts")
    if not isinstance(cohorts, Mapping):
        analysis_as_of = _number(experiment.get("analysis_as_of_ts"))
        prospective_required = bool(
            analysis_as_of is not None
            and analysis_as_of >= _CONFIRMATION_ACTIVATED_AT
        )
        return experiment, {
            "present": False,
            "source": "top_level_legacy_or_screen_only",
            "contract_valid": not prospective_required,
            "binding_eligible": False,
            "integrity_pass": True,
            "passed": not prospective_required,
            "failures": (
                ["prospective analysis cohort missing after activation"]
                if prospective_required
                else []
            ),
        }

    failures: list[str] = []
    if set(cohorts) != {"legacy", "prospective", "outside_confirmation"}:
        failures.append("analysis cohort set does not match the frozen contract")
    prospective = cohorts.get("prospective")
    prospective = prospective if isinstance(prospective, Mapping) else {}
    required_top = {
        "selection",
        "binding_eligible",
        "enrolled_games",
        "excluded_games",
        "variants",
        "metrics",
        "itt",
        "affected_turns",
        "health",
        "routing",
        "integrity",
    }
    if set(prospective) != required_top:
        failures.append("prospective cohort fields do not match the frozen contract")
    if prospective.get("selection") != "approved_manifest_confirmation_only":
        failures.append("prospective cohort selection is not approved confirmation")

    enrolled_games = _strict_count(prospective.get("enrolled_games"))
    excluded_games = _strict_count(prospective.get("excluded_games"))
    if enrolled_games is None:
        failures.append("prospective enrolled_games must be a literal nonnegative integer")
    if excluded_games is None:
        failures.append("prospective excluded_games must be a literal nonnegative integer")

    variants = prospective.get("variants")
    variants = variants if isinstance(variants, Mapping) else {}
    variant_fields = {
        "games",
        "resolved",
        "censored",
        "affected_games",
        "affected_turns",
        "direction_violations",
    }
    variant_games: dict[str, int] = {}
    variant_counts: dict[str, dict[str, int]] = {}
    if set(variants) != {"treatment", "control"}:
        failures.append("prospective variants arms missing or malformed")
    for arm in ("treatment", "control"):
        leaf = variants.get(arm)
        if not isinstance(leaf, Mapping) or set(leaf) != variant_fields:
            failures.append(f"prospective {arm} variant leaf malformed")
            continue
        counts = {field: _strict_count(leaf.get(field)) for field in variant_fields}
        if any(value is None for value in counts.values()):
            failures.append(f"prospective {arm} variant counts malformed")
            continue
        arm_counts = {
            field: int(value)
            for field, value in counts.items()
            if value is not None
        }
        games = arm_counts["games"]
        resolved = arm_counts["resolved"]
        censored = arm_counts["censored"]
        affected_games = arm_counts["affected_games"]
        affected_turns = arm_counts["affected_turns"]
        direction_violations = arm_counts["direction_violations"]
        variant_games[arm] = games
        variant_counts[arm] = arm_counts
        if (
            resolved + censored != games
            or affected_games > games
            or affected_games > affected_turns
            or direction_violations > affected_turns
        ):
            failures.append(f"prospective {arm} variant counts are not additive")
    if enrolled_games is not None and len(variant_games) == 2:
        if sum(variant_games.values()) != enrolled_games:
            failures.append("prospective variant games do not add to enrolled_games")

    metrics = prospective.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    itt = prospective.get("itt")
    itt = itt if isinstance(itt, Mapping) else {}
    health = prospective.get("health")
    health = health if isinstance(health, Mapping) else {}
    routing = prospective.get("routing")
    routing = routing if isinstance(routing, Mapping) else {}
    integrity = prospective.get("integrity")
    integrity = integrity if isinstance(integrity, Mapping) else {}

    if set(metrics) != {"treatment", "control"} or not all(
        isinstance(metrics.get(arm), Mapping) for arm in ("treatment", "control")
    ):
        failures.append("prospective metrics arms missing or malformed")
    if set(itt) != {"treatment", "control", "integrity"} or not all(
        isinstance(itt.get(arm), Mapping) for arm in ("treatment", "control")
    ):
        failures.append("prospective ITT arms missing or malformed")
    itt_games: dict[str, int] = {}
    for arm in ("treatment", "control"):
        leaf = itt.get(arm)
        games = _strict_count(leaf.get("games")) if isinstance(leaf, Mapping) else None
        if games is None:
            failures.append(f"prospective {arm} ITT games malformed")
        else:
            itt_games[arm] = games
    itt_total = sum(itt_games.values()) if len(itt_games) == 2 else None
    if itt_total is not None:
        if itt_total <= 0:
            failures.append("prospective ITT population must be nonempty")
        if enrolled_games is not None and itt_total > enrolled_games:
            failures.append("prospective ITT population exceeds enrolled_games")
    family = experiment.get("family")
    if len(variant_games) == 2 and len(itt_games) == 2:
        for arm in ("treatment", "control"):
            if family == "bargaining" and itt_games[arm] != variant_games[arm]:
                failures.append(
                    f"prospective {arm} bargaining ITT games do not match variant games"
                )
            elif family != "bargaining" and itt_games[arm] > variant_games[arm]:
                failures.append(
                    f"prospective {arm} ITT games exceed variant games"
                )
    if family == "persuasion" and len(itt_games) == 2:
        for arm in ("treatment", "control"):
            metric = metrics.get(arm)
            eligible_games = (
                _strict_count(metric.get("blind_seller_games"))
                if isinstance(metric, Mapping)
                else None
            )
            if eligible_games is None or eligible_games != itt_games[arm]:
                failures.append(
                    f"prospective {arm} persuasion metric population does not match ITT"
                )

    health_fields = {
        "turn_events",
        "turn_errors",
        "corrections",
        "result_events",
        "result_errors",
        # ``invalid_results`` remains the total; the three components below
        # separate agent-caused invalidity from reporter-side evidence faults
        # and must reconcile exactly against it.
        "invalid_results",
        "invalid_moves",
        "invalid_terminals",
        "provenance_faults",
        "http_503",
    }
    if set(health) != {"treatment", "control"}:
        failures.append("prospective health arms missing or malformed")
    health_counts: dict[str, dict[str, int]] = {}
    for arm in ("treatment", "control"):
        leaf = health.get(arm)
        if not isinstance(leaf, Mapping) or set(leaf) != health_fields or any(
            _strict_count(leaf.get(field)) is None for field in health_fields
        ):
            failures.append(f"prospective {arm} health leaf malformed")
        else:
            turns = _strict_count(leaf.get("turn_events"))
            results = _strict_count(leaf.get("result_events"))
            turn_errors = _strict_count(leaf.get("turn_errors"))
            result_errors = _strict_count(leaf.get("result_errors"))
            invalid_results = _strict_count(leaf.get("invalid_results"))
            http_503 = _strict_count(leaf.get("http_503"))
            assert None not in (
                turns,
                results,
                turn_errors,
                result_errors,
                invalid_results,
                http_503,
            )
            health_counts[arm] = {
                "turn_events": int(turns),
                "result_events": int(results),
            }
            invalid_moves = _strict_count(leaf.get("invalid_moves"))
            invalid_terminals = _strict_count(leaf.get("invalid_terminals"))
            provenance_faults = _strict_count(leaf.get("provenance_faults"))
            assert None not in (
                invalid_moves,
                invalid_terminals,
                provenance_faults,
            )
            if (
                turn_errors > turns
                or result_errors > results
                or invalid_results > results
                or http_503 > turns + results
            ):
                failures.append(
                    f"prospective {arm} health count exceeds its denominator"
                )
            component_sum = invalid_moves + invalid_terminals + provenance_faults
            # The sources may co-occur on one event, so the total is bounded by
            # the largest single component below and the sum above.
            if (
                invalid_results > component_sum
                or invalid_results
                < max(invalid_moves, invalid_terminals, provenance_faults)
            ):
                failures.append(
                    f"prospective {arm} invalid result components do not "
                    "reconcile with the reported total"
                )
            population = variant_counts.get(arm)
            if population is not None:
                if turns < population["games"]:
                    failures.append(
                        f"prospective {arm} health does not cover variant games"
                    )
                if results < population["resolved"]:
                    failures.append(
                        f"prospective {arm} result health does not cover resolved games"
                    )
                if population["games"] == 0 and (turns or results):
                    failures.append(
                        f"prospective {arm} health has events without variant games"
                    )

    routing_fields = {
        "checked",
        "assigned_matches",
        "replay_errors",
        "assignment_integrity_errors",
        "duplicate_causal_turn_conflicts",
        "affected",
        "affected_assigned_matches",
        "affected_wrong_variant",
        "affected_unknown",
        "direction_violations",
    }
    routing_integrity_fields = {
        "unknown_assignment_turns",
        "unknown_assignment_games",
        "unapproved_prospective_games",
        "duplicate_causal_turn_conflicts",
    }
    if set(routing) != {"treatment", "control", "integrity"}:
        failures.append("prospective routing arms missing or malformed")
    for arm in ("treatment", "control"):
        leaf = routing.get(arm)
        if not isinstance(leaf, Mapping) or set(leaf) != routing_fields or any(
            _strict_count(leaf.get(field)) is None for field in routing_fields
        ):
            failures.append(f"prospective {arm} routing leaf malformed")
        else:
            values = {
                field: _strict_count(leaf.get(field)) for field in routing_fields
            }
            assert all(value is not None for value in values.values())
            checked = int(values["checked"])
            assigned = int(values["assigned_matches"])
            affected = int(values["affected"])
            affected_assigned = int(values["affected_assigned_matches"])
            affected_wrong = int(values["affected_wrong_variant"])
            affected_unknown = int(values["affected_unknown"])
            direction_violations = int(values["direction_violations"])
            if (
                assigned > checked
                or affected > checked
                or affected_assigned > affected
                or affected_wrong + affected_unknown
                != affected - affected_assigned
            ):
                failures.append(
                    f"prospective {arm} routing counts are not additive"
                )
            population = variant_counts.get(arm)
            if population is not None:
                if checked < population["games"]:
                    failures.append(
                        f"prospective {arm} routing does not cover variant games"
                    )
                if affected != population["affected_turns"]:
                    failures.append(
                        f"prospective {arm} routing affected count mismatches variants"
                    )
                if direction_violations != population["direction_violations"]:
                    failures.append(
                        f"prospective {arm} routing direction count mismatches variants"
                    )
                if population["games"] == 0 and any(
                    int(value) for value in values.values()
                ):
                    failures.append(
                        f"prospective {arm} routing has events without variant games"
                    )
            arm_health = health_counts.get(arm)
            if arm_health is not None and checked > arm_health["turn_events"]:
                failures.append(
                    f"prospective {arm} routing exceeds logged turn health"
                )
    routing_integrity = routing.get("integrity")
    if (
        not isinstance(routing_integrity, Mapping)
        or set(routing_integrity) != routing_integrity_fields
        or any(
            _strict_count(routing_integrity.get(field)) is None
            for field in routing_integrity_fields
        )
    ):
        failures.append("prospective routing integrity malformed")

    if set(integrity) != {"pass", "assignment_failures", "clock_valid"}:
        failures.append("prospective cohort integrity fields malformed")
    assignment_failures = _strict_count(integrity.get("assignment_failures"))
    if type(integrity.get("pass")) is not bool:
        failures.append("prospective cohort integrity pass must be literal boolean")
    if type(integrity.get("clock_valid")) is not bool:
        failures.append("prospective cohort clock validity must be literal boolean")
    if assignment_failures is None:
        failures.append("prospective cohort assignment failures malformed")
    if type(prospective.get("binding_eligible")) is not bool:
        failures.append("prospective binding eligibility must be literal boolean")
    if not isinstance(prospective.get("affected_turns"), list):
        failures.append("prospective affected rows missing or malformed")

    binding_eligible = prospective.get("binding_eligible") is True
    integrity_pass = bool(
        integrity.get("pass") is True
        and integrity.get("clock_valid") is True
        and assignment_failures == 0
        and isinstance(routing_integrity, Mapping)
        and not any(
            _strict_count(routing_integrity.get(field)) or 0
            for field in routing_integrity_fields
        )
    )
    schema_contract_valid = not failures
    expected_binding_eligible = bool(
        schema_contract_valid
        and integrity_pass
        and itt_total is not None
        and itt_total > 0
    )
    binding_inconsistent = bool(
        type(prospective.get("binding_eligible")) is bool
        and binding_eligible is not expected_binding_eligible
    )
    if binding_inconsistent:
        failures.append(
            "prospective binding_eligible is inconsistent with valid nonempty data"
        )
    if schema_contract_valid and not integrity_pass:
        failures.append("prospective cohort assignment or clock integrity failed")
    contract_valid = bool(schema_contract_valid and not binding_inconsistent)

    view = dict(experiment)
    view["metrics"] = metrics
    view["itt"] = itt
    view["affected_turns"] = prospective.get("affected_turns", [])
    view["_binding_cohort_health"] = health
    view["_binding_cohort_routing"] = routing
    info = {
        "present": True,
        "source": "analysis_cohorts.prospective",
        "contract_valid": contract_valid,
        "binding_eligible": binding_eligible,
        "integrity_pass": integrity_pass,
        "passed": not failures,
        "failures": failures,
    }
    view["_binding_cohort_integrity"] = info
    return view, info


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

    arm_health = experiment.get("arm_health")
    arm_health = arm_health if isinstance(arm_health, Mapping) else {}
    binding_health = experiment.get("_binding_cohort_health")
    binding_health_present = isinstance(binding_health, Mapping)
    binding_health = binding_health if binding_health_present else {}
    analysis_as_of = _number(experiment.get("analysis_as_of_ts"))
    reporter_arm_health_present = "arm_health" in experiment
    reporter_integrity = arm_health.get("integrity")
    reporter_integrity = (
        reporter_integrity if isinstance(reporter_integrity, Mapping) else {}
    )
    reporter_integrity_counts = {
        key: _strict_count(reporter_integrity.get(key))
        for key in (
            "unknown_turn_events",
            "unknown_result_events",
            "unassigned_or_missing_after_activation",
        )
    }
    reporter_contract_valid = bool(
        type(arm_health.get("schema_version")) is int
        and arm_health.get("schema_version") == 1
        and arm_health.get("attribution")
        == "immutable first-game assignment plus raw event occurrences"
        and _number(arm_health.get("prospective_activation"))
        == _CONFIRMATION_ACTIVATED_AT
        and isinstance(arm_health.get("cohorts"), Mapping)
        and analysis_as_of is not None
        and all(value is not None for value in reporter_integrity_counts.values())
        and not any(reporter_integrity_counts.values())
        and arm_health.get("integrity_pass") is True
    )
    use_reporter_arm_health = reporter_arm_health_present and reporter_contract_valid
    health_cohort = (
        "prospective"
        if analysis_as_of is not None
        and analysis_as_of >= _CONFIRMATION_ACTIVATED_AT
        else "legacy"
    )
    assignment = experiment.get("assignment", {})
    assignment = assignment if isinstance(assignment, Mapping) else {}
    agents = experiment.get("agents", {})
    agents = agents if isinstance(agents, Mapping) else {}
    errors: dict[str, dict] = {}
    for arm in ("treatment", "control"):
        events = 0
        lower_failures = 0
        upper_failures = 0
        integrity_failures: list[str] = []
        if binding_health_present:
            health = binding_health.get(arm, {})
            health = health if isinstance(health, Mapping) else {}
            required = {
                key: _strict_count(health.get(key))
                for key in (
                    "turn_events",
                    "result_events",
                    "turn_errors",
                    "result_errors",
                    "invalid_results",
                )
            }
            if any(value is None for value in required.values()):
                valid = False
                integrity_failures.append(
                    f"prospective:{arm}: missing/malformed cohort health count"
                )
            else:
                turns = required["turn_events"]
                results = required["result_events"]
                turn_errors = required["turn_errors"]
                result_errors = required["result_errors"]
                invalid_results = required["invalid_results"]
                assert None not in (
                    turns,
                    results,
                    turn_errors,
                    result_errors,
                    invalid_results,
                )
                valid = bool(
                    turn_errors <= turns
                    and result_errors <= results
                    and invalid_results <= results
                )
                if not valid:
                    integrity_failures.append(
                        f"prospective:{arm}: cohort health count exceeds denominator"
                    )
                else:
                    events = turns + results
                    lower_failures = turn_errors + max(
                        result_errors, invalid_results
                    )
                    upper_failures = turn_errors + min(
                        results, result_errors + invalid_results
                    )
        elif reporter_arm_health_present and not reporter_contract_valid:
            valid = False
            integrity_failures.append("reporter arm-health contract malformed")
        elif use_reporter_arm_health:
            cohorts = arm_health.get("cohorts", {})
            cohort = (
                cohorts.get(health_cohort, {})
                if isinstance(cohorts, Mapping)
                else {}
            )
            health = cohort.get(arm, {}) if isinstance(cohort, Mapping) else {}
            health = health if isinstance(health, Mapping) else {}
            required = {
                key: _strict_count(health.get(key))
                for key in (
                    "turn_events",
                    "result_events",
                    "turn_errors",
                    "result_errors",
                    "invalid_results",
                )
            }
            if any(value is None for value in required.values()):
                valid = False
                integrity_failures.append(
                    f"{health_cohort}:{arm}: missing/malformed arm health count"
                )
            else:
                turns = required["turn_events"]
                results = required["result_events"]
                turn_errors = required["turn_errors"]
                result_errors = required["result_errors"]
                invalid_results = required["invalid_results"]
                assert None not in (
                    turns, results, turn_errors, result_errors, invalid_results
                )
                valid = bool(
                    turn_errors <= turns
                    and result_errors <= results
                    and invalid_results <= results
                )
                if not valid:
                    integrity_failures.append(
                        f"{health_cohort}:{arm}: health count exceeds denominator"
                    )
                else:
                    events = turns + results
                    lower_failures = turn_errors + max(
                        result_errors, invalid_results
                    )
                    upper_failures = turn_errors + min(
                        results, result_errors + invalid_results
                    )
        else:
            labels = assignment.get(f"{arm}_agents", [])
            labels = labels if isinstance(labels, (list, tuple)) else []
            valid = bool(labels)
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
                    integrity_failures.append(
                        f"{label}: missing/malformed health count"
                    )
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
                    integrity_failures.append(
                        f"{label}: health count exceeds denominator"
                    )
                    continue
                events += turns + results
                lower_failures += turn_errors + max(result_errors, invalid_results)
                upper_failures += turn_errors + min(
                    results, result_errors + invalid_results
                )
        valid = valid and events > 0 and upper_failures <= events
        absolute = (
            _binomial_summary(upper_failures, events)
            if valid
            else _binomial_summary(None, None)
        )
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
            "attribution": (
                "analysis_cohorts.prospective"
                if binding_health_present
                else (
                    f"reporter_game_arm:{health_cohort}"
                    if use_reporter_arm_health
                    else (
                        "invalid_reporter_arm_health"
                        if reporter_arm_health_present
                        else "legacy_static_agent_labels"
                    )
                )
            ),
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


def _reported_rate_matches_counts(value: Any, successes: int, trials: int) -> bool:
    if successes > trials:
        return False
    if trials == 0:
        return value is None
    rate = _bounded_mean(value)
    return rate is not None and math.isclose(
        rate, successes / trials, rel_tol=0.0, abs_tol=1e-12
    )


def _validate_affected_arm(metric: Mapping[str, Any], family: str) -> list[str]:
    failures: list[str] = []
    total_key = "affected_games" if family == "bargaining" else "blind_seller_games"
    valid_key = (
        "normalized_payoff_valid"
        if family == "bargaining"
        else "revenue_share_valid"
    )
    invalid_key = (
        "normalized_payoff_invalid"
        if family == "bargaining"
        else "revenue_share_invalid"
    )
    sum_key = (
        "normalized_payoff_sum"
        if family == "bargaining"
        else "revenue_share_sum"
    )
    square_key = (
        "normalized_payoff_sum_squares"
        if family == "bargaining"
        else "revenue_share_sum_squares"
    )
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
    valid = _strict_count(metric.get(valid_key))
    invalid = _strict_count(metric.get(invalid_key))
    if valid is None or invalid is None or valid + invalid != resolved:
        failures.append("affected outcome validity counts inconsistent")
    if invalid is not None and invalid != 0:
        failures.append("affected outcomes contain invalid normalized values")
    if not _validate_moment_bounds(
        metric.get(sum_key), metric.get(square_key), resolved
    ):
        failures.append("affected outcome moments invalid")
    if family == "bargaining":
        games = _strict_count(metric.get("games"))
        if games is None or games != total:
            failures.append("bargaining games != affected_games")
        direct_resolved = _strict_count(metric.get("direct_resolved"))
        direct_converted = _strict_count(metric.get("direct_converted"))
        if direct_resolved is None or direct_converted is None:
            failures.append("direct conversion counts missing/malformed")
        else:
            if direct_resolved > resolved or direct_converted > direct_resolved:
                failures.append("direct conversion counts exceed bounds")
            if not _reported_rate_matches_counts(
                metric.get("direct_conversion_rate"),
                direct_converted,
                direct_resolved,
            ):
                failures.append("direct conversion rate inconsistent with counts")

        horizons = metric.get("horizon_strata")
        if not isinstance(horizons, Mapping):
            failures.append("bargaining horizon strata missing/malformed")
        else:
            expected_horizons = {"finite", "unlimited"}
            extras = set(horizons) - expected_horizons
            if extras:
                failures.append(f"unexpected bargaining horizon strata: {sorted(extras)!r}")
            horizon_totals = {
                key: 0
                for key in (
                    "games",
                    "resolved",
                    "censored",
                    "direct_resolved",
                    "direct_converted",
                )
            }
            for horizon in sorted(expected_horizons):
                entry = horizons.get(horizon)
                if not isinstance(entry, Mapping):
                    failures.append(f"{horizon} horizon missing/malformed")
                    continue
                counts = {
                    key: _strict_count(entry.get(key)) for key in horizon_totals
                }
                if any(value is None for value in counts.values()):
                    failures.append(f"{horizon} horizon count missing/malformed")
                    continue
                values = {key: int(value) for key, value in counts.items()}
                if values["games"] != values["resolved"] + values["censored"]:
                    failures.append(
                        f"{horizon} horizon games != resolved + censored"
                    )
                if (
                    values["direct_resolved"] > values["resolved"]
                    or values["direct_converted"] > values["direct_resolved"]
                ):
                    failures.append(f"{horizon} direct conversion counts exceed bounds")
                if not _reported_rate_matches_counts(
                    entry.get("direct_conversion_rate"),
                    values["direct_converted"],
                    values["direct_resolved"],
                ):
                    failures.append(
                        f"{horizon} direct conversion rate inconsistent with counts"
                    )
                for key, value in values.items():
                    horizon_totals[key] += value
            expected_totals = {
                "games": games,
                "resolved": resolved,
                "censored": censored,
                "direct_resolved": direct_resolved,
                "direct_converted": direct_converted,
            }
            for key, expected in expected_totals.items():
                if expected is None or horizon_totals[key] != expected:
                    failures.append(f"bargaining horizon {key} counts are not additive")
    else:
        zero = _strict_count(metric.get("zero_sales"))
        if zero is None or zero > resolved:
            failures.append("zero_sales missing/malformed/inconsistent")
    cells = metric.get("cells")
    if not isinstance(cells, Mapping):
        failures.append("affected cells missing/malformed")
    else:
        cell_resolved = 0
        cell_censored = 0
        cell_valid = 0
        cell_invalid = 0
        cell_sum = 0.0
        cell_sum_sq = 0.0
        cells_valid = True
        for entry in cells.values():
            if not isinstance(entry, Mapping):
                cells_valid = False
                continue
            n = _strict_count(entry.get("resolved"))
            c = _strict_count(entry.get("censored"))
            entry_total_key = (
                "games" if family == "bargaining" else "blind_seller_games"
            )
            entry_total = _strict_count(entry.get(entry_total_key))
            entry_valid = _strict_count(entry.get(valid_key))
            entry_invalid = _strict_count(entry.get(invalid_key))
            total_value = _number(entry.get(sum_key))
            square_value = _number(entry.get(square_key))
            if None in (
                n,
                c,
                entry_total,
                entry_valid,
                entry_invalid,
                total_value,
                square_value,
            ):
                cells_valid = False
                continue
            assert n is not None and c is not None
            assert entry_total is not None
            assert entry_valid is not None and entry_invalid is not None
            if entry_total != n + c or entry_valid + entry_invalid != n:
                cells_valid = False
            if entry_invalid != 0:
                cells_valid = False
            if not _validate_moment_bounds(total_value, square_value, n):
                cells_valid = False
            cell_resolved += n
            cell_censored += c
            cell_valid += entry_valid
            cell_invalid += entry_invalid
            cell_sum += total_value
            cell_sum_sq += square_value
        if not cells_valid:
            failures.append("affected cell aggregate malformed")
        if cell_resolved != resolved:
            failures.append("affected cell resolved counts are not additive")
        if cell_censored != censored:
            failures.append("affected cell censored counts are not additive")
        if valid is None or cell_valid != valid:
            failures.append("affected cell valid counts are not additive")
        if invalid is None or cell_invalid != invalid:
            failures.append("affected cell invalid counts are not additive")
        top_sum = _number(metric.get(sum_key))
        top_sq = _number(metric.get(square_key))
        if top_sum is None or not math.isclose(cell_sum, top_sum, abs_tol=1e-9):
            failures.append("affected cell sums are not additive")
        if top_sq is None or not math.isclose(cell_sum_sq, top_sq, abs_tol=1e-9):
            failures.append("affected cell sums of squares are not additive")
    return failures


def _validate_itt_arm(
    metric: Mapping[str, Any], maturity_lag_s: int, *, family: str
) -> list[str]:
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
        "late_terminals",
        "terminal_conflicts",
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
    if values["terminal_conflicts"]:
        failures.append("ITT contains conflicting timely terminals")
    if values["zero_sales"] > values["matured"]:
        failures.append("ITT zero sales exceed matured")
    if values["zero_sales"] < values["deadline_zeroes"]:
        failures.append("ITT zero sales are fewer than forced deadline zeroes")
    if not _validate_moment_bounds(
        metric.get("normalized_outcome_sum"),
        metric.get("normalized_outcome_sum_squares"),
        values["matured"],
    ):
        failures.append("ITT moments invalid")

    linked_count_fields = (
        "games",
        "matured",
        "pending_maturation",
        "resolved",
        "censored",
        "deadline_censored",
        "timely_valid_terminals",
        "deadline_zeroes",
        "late_terminals",
        "invalid_terminals",
        "terminal_conflicts",
        "zero_sales",
    )
    linked_number_fields = (
        "normalized_outcome_sum",
        "normalized_outcome_sum_squares",
    )

    def linked_values(entry: Mapping[str, Any], label: str) -> dict[str, int | float] | None:
        linked_counts = {
            key: _strict_count(entry.get(key)) for key in linked_count_fields
        }
        linked_numbers = {key: _number(entry.get(key)) for key in linked_number_fields}
        if any(value is None for value in (*linked_counts.values(), *linked_numbers.values())):
            failures.append(f"{label} linked statistic missing/malformed")
            return None
        output: dict[str, int | float] = {
            key: int(value) for key, value in linked_counts.items() if value is not None
        }
        output.update(
            {key: float(value) for key, value in linked_numbers.items() if value is not None}
        )
        matured = int(output["matured"])
        resolved = int(output["resolved"])
        censored = int(output["censored"])
        invalid_terminals = int(output["invalid_terminals"])
        zero_sales = int(output["zero_sales"])
        pending = int(output["pending_maturation"])
        deadline_censored = int(output["deadline_censored"])
        timely_valid_terminals = int(output["timely_valid_terminals"])
        deadline_zeroes = int(output["deadline_zeroes"])
        if int(output["games"]) != matured + pending:
            failures.append(f"{label} games != matured + pending")
        if matured != resolved + censored + invalid_terminals:
            failures.append(f"{label} matured decomposition inconsistent")
        if resolved != timely_valid_terminals:
            failures.append(f"{label} resolved != timely valid terminals")
        if censored != deadline_censored:
            failures.append(f"{label} censored != deadline_censored")
        if deadline_zeroes != censored + invalid_terminals:
            failures.append(f"{label} deadline zero decomposition inconsistent")
        if zero_sales > matured:
            failures.append(f"{label} zero_sales exceed matured")
        if zero_sales < deadline_zeroes:
            failures.append(f"{label} zero_sales fewer than deadline zeroes")
        if not _validate_moment_bounds(
            output["normalized_outcome_sum"],
            output["normalized_outcome_sum_squares"],
            matured,
        ):
            failures.append(f"{label} moments invalid")
        return output

    top_linked: dict[str, int | float] = {
        "games": values["games"],
        "matured": values["matured"],
        "pending_maturation": values["pending_maturation"],
        "resolved": values["resolved"],
        "censored": values["censored"],
        "deadline_censored": values["deadline_censored"],
        "timely_valid_terminals": values["timely_valid_terminals"],
        "deadline_zeroes": values["deadline_zeroes"],
        "late_terminals": values["late_terminals"],
        "invalid_terminals": invalid,
        "terminal_conflicts": values["terminal_conflicts"],
        "zero_sales": values["zero_sales"],
        "normalized_outcome_sum": _number(metric.get("normalized_outcome_sum"))
        or 0.0,
        "normalized_outcome_sum_squares": _number(
            metric.get("normalized_outcome_sum_squares")
        )
        or 0.0,
    }

    def equal_stat(left: int | float, right: int | float, field: str) -> bool:
        if field in linked_count_fields:
            return left == right
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-9)

    cells = metric.get("cells")
    cell_totals = {key: 0 for key in linked_count_fields}
    cell_totals.update({key: 0.0 for key in linked_number_fields})
    cells_by_p: dict[str, dict[str, int | float]] = {}
    if not isinstance(cells, Mapping):
        failures.append("ITT cells missing/malformed")
    else:
        for raw_key, entry in cells.items():
            if not isinstance(entry, Mapping) or not isinstance(entry.get("cell"), Mapping):
                failures.append(f"ITT cell {raw_key!s} missing/malformed")
                continue
            linked = linked_values(entry, f"ITT cell {raw_key!s}")
            if linked is None:
                continue
            for field, value in linked.items():
                cell_totals[field] += value
            if family == "persuasion":
                fixed_cell_key = _fixed_key(
                    entry["cell"],
                    ("p", "message_type", "price", "total_rounds"),
                )
                if fixed_cell_key not in PERS_CELL_WEIGHTS:
                    failures.append(
                        f"ITT cell {raw_key!s} is not a literal frozen configuration"
                    )
                p_key = fixed_cell_key[0]
                p_totals = cells_by_p.setdefault(
                    str(p_key),
                    {
                        **{key: 0 for key in linked_count_fields},
                        **{key: 0.0 for key in linked_number_fields},
                    },
                )
                for field, value in linked.items():
                    p_totals[field] += value
        for field, expected in top_linked.items():
            if not equal_stat(cell_totals[field], expected, field):
                failures.append(f"ITT cell {field} statistics are not additive")

    p_strata = metric.get("p_strata")
    if family == "persuasion":
        expected_p = set(PERS_P_WEIGHTS)
        if not isinstance(p_strata, Mapping):
            failures.append("ITT persuasion p strata missing/malformed")
        else:
            actual_p = set(p_strata)
            if actual_p != expected_p:
                failures.append(
                    "ITT persuasion p strata do not match the frozen expected set"
                )
            p_totals = {key: 0 for key in linked_count_fields}
            p_totals.update({key: 0.0 for key in linked_number_fields})
            for p_key in sorted(expected_p):
                entry = p_strata.get(p_key)
                if not isinstance(entry, Mapping):
                    failures.append(f"ITT p stratum {p_key} missing/malformed")
                    continue
                linked = linked_values(entry, f"ITT p stratum {p_key}")
                if linked is None:
                    continue
                for field, value in linked.items():
                    p_totals[field] += value
                cell_linked = cells_by_p.get(p_key)
                if cell_linked is None:
                    failures.append(f"ITT p stratum {p_key} has no linked cells")
                    continue
                for field, value in linked.items():
                    if not equal_stat(cell_linked[field], value, field):
                        failures.append(
                            f"ITT p stratum {p_key} {field} does not match cells"
                        )
            for field, expected in top_linked.items():
                if not equal_stat(p_totals[field], expected, field):
                    failures.append(
                        f"ITT p-stratum {field} statistics are not additive"
                    )
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
        strata = {
            "overall": (None, None, 0.03),
            "finite": ("horizon", "finite", 0.03),
            "unlimited": ("horizon", "unlimited", 0.03),
        }
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
    """Verify the reporter's prospective confirmation contract.

    Legacy ``randomization_evidence`` booleans are intentionally ignored.  A
    caller cannot promote by asserting conclusions: the reporter must expose
    linked sufficient statistics and exact frozen identities that this
    evaluator can recompute against the experiment's ITT population.
    """
    evidence = experiment.get("prospective_confirmation")
    evidence = evidence if isinstance(evidence, Mapping) else {}
    contract = evidence.get("contract")
    contract = contract if isinstance(contract, Mapping) else {}
    linked_rows = evidence.get("linked_itt_rows")
    linked_rows = linked_rows if isinstance(linked_rows, Mapping) else {}
    blocks = evidence.get("labels")
    blocks = blocks if isinstance(blocks, Mapping) else {}
    prefix = evidence.get("immutable_prefix")
    prefix = prefix if isinstance(prefix, Mapping) else {}
    scheduled_look = evidence.get("scheduled_look")
    scheduled_look = scheduled_look if isinstance(scheduled_look, Mapping) else {}
    reporter_verification = evidence.get("reporter_verification")
    reporter_verification = (
        reporter_verification
        if isinstance(reporter_verification, Mapping)
        else {}
    )
    expected_rule = _CONFIRMATION_RULES.get(family, {})
    labels = ("main", "test_a", "test_b", "test_c")
    treatment_representation = 50 if family == "bargaining" else 150
    control_representation = 150

    def exact_literal(actual: Any, expected: Any) -> bool:
        if isinstance(expected, Mapping):
            return bool(
                isinstance(actual, Mapping)
                and set(actual) == set(expected)
                and all(exact_literal(actual[key], value) for key, value in expected.items())
            )
        return type(actual) is type(expected) and actual == expected

    def sha256_literal(value: Any) -> bool:
        return bool(
            isinstance(value, str)
            and len(value) == 64
            and value == value.lower()
            and all(character in "0123456789abcdef" for character in value)
        )

    expected_contract = {
        "artifact_sha256": _CONFIRMATION_PLAN_SHA256,
        "plan_id": _CONFIRMATION_PLAN_ID,
        "strategy_aggregate_sha256": _CONFIRMATION_STRATEGY_SHA256,
        "activated_at": _CONFIRMATION_ACTIVATED_AT,
        "expires_at": _CONFIRMATION_EXPIRES_AT,
        "target_sha256": _CONFIRMATION_TARGET_SHA256,
        "rule": expected_rule,
    }

    itt = experiment.get("itt")
    itt = itt if isinstance(itt, Mapping) else {}
    metrics = experiment.get("metrics")
    metrics = metrics if isinstance(metrics, Mapping) else {}
    linked_counts = {
        arm: _strict_count(linked_rows.get(arm)) for arm in ("treatment", "control")
    }
    itt_counts: dict[str, int | None] = {}
    population_counts: dict[str, int | None] = {}
    population_fields = (
        ("treatment_affected_rows", "control_affected_rows")
        if family == "bargaining"
        else ("treatment_matured_rows", "control_matured_rows")
    )
    for arm in ("treatment", "control"):
        metric = itt.get(arm)
        metric = metric if isinstance(metric, Mapping) else {}
        itt_counts[arm] = _strict_count(metric.get("games"))
        if family == "bargaining":
            affected_metric = metrics.get(arm)
            affected_metric = (
                affected_metric if isinstance(affected_metric, Mapping) else {}
            )
            population_counts[arm] = _strict_count(
                affected_metric.get("affected_games")
            )
        else:
            population_counts[arm] = _strict_count(metric.get("matured"))

    label_checks: dict[str, dict] = {}
    label_row_totals = {"treatment": 0, "control": 0}
    label_population_totals = {"treatment": 0, "control": 0}
    common_blocks_total = 0
    for label in labels:
        entry = blocks.get(label, {})
        entry = entry if isinstance(entry, Mapping) else {}
        treatment_rows = _strict_count(entry.get("treatment_rows"))
        control_rows = _strict_count(entry.get("control_rows"))
        treatment_blocks = _strict_count(entry.get("treatment_blocks"))
        control_blocks = _strict_count(entry.get("control_blocks"))
        same_blocks = _strict_count(entry.get("same_30m_blocks"))
        treatment_population = _strict_count(entry.get(population_fields[0]))
        control_population = _strict_count(entry.get(population_fields[1]))
        expected_label_fields = {
            "treatment_rows",
            "control_rows",
            population_fields[0],
            population_fields[1],
            "treatment_blocks",
            "control_blocks",
            "same_30m_blocks",
        }
        counts_valid = all(
            value is not None
            for value in (
                treatment_rows,
                control_rows,
                treatment_population,
                control_population,
                treatment_blocks,
                control_blocks,
                same_blocks,
            )
        ) and set(entry) == expected_label_fields
        if counts_valid:
            assert None not in (
                treatment_rows,
                control_rows,
                treatment_population,
                control_population,
                treatment_blocks,
                control_blocks,
                same_blocks,
            )
            label_row_totals["treatment"] += treatment_rows
            label_row_totals["control"] += control_rows
            label_population_totals["treatment"] += treatment_population
            label_population_totals["control"] += control_population
            common_blocks_total += same_blocks
        label_checks[label] = {
            "counts_valid": counts_valid,
            "treatment_representation": bool(
                treatment_population is not None
                and treatment_population >= treatment_representation
            ),
            "control_representation": bool(
                control_population is not None
                and control_population >= control_representation
            ),
            "population_is_subset_of_all_enrolled": bool(
                treatment_population is not None
                and treatment_rows is not None
                and treatment_population <= treatment_rows
                and control_population is not None
                and control_rows is not None
                and control_population <= control_rows
            ),
            "both_arms_have_blocks": bool(
                treatment_blocks is not None
                and treatment_blocks > 0
                and control_blocks is not None
                and control_blocks > 0
                and treatment_rows is not None
                and treatment_blocks <= treatment_rows
                and control_rows is not None
                and control_blocks <= control_rows
            ),
            "same_block_representation": bool(
                same_blocks is not None
                and same_blocks > 0
                and treatment_blocks is not None
                and control_blocks is not None
                and same_blocks <= min(treatment_blocks, control_blocks)
            ),
        }

    prospective_rows = _strict_count(evidence.get("prospective_rows"))
    approved_rows = _strict_count(evidence.get("approved_rows"))
    common_blocks_reported = _strict_count(
        evidence.get("common_agent_30m_blocks")
    )
    linked_total = (
        linked_counts["treatment"] + linked_counts["control"]
        if linked_counts["treatment"] is not None
        and linked_counts["control"] is not None
        else None
    )
    first_enrollment_ts = _number(evidence.get("first_enrollment_ts"))
    last_enrollment_ts = _number(evidence.get("last_enrollment_ts"))
    analysis_as_of_ts = _number(experiment.get("analysis_as_of_ts"))

    prefix_bytes = _strict_count(prefix.get("bytes"))
    prefix_records = _strict_count(prefix.get("records"))
    prefix_last_event_ts = _number(prefix.get("last_event_ts"))
    prefix_sha256 = prefix.get("sha256")
    scheduled_at_ts = _number(scheduled_look.get("scheduled_at_ts"))
    declared_at_ts = _number(scheduled_look.get("declared_at_ts"))
    scheduled_analysis_ts = _number(scheduled_look.get("analysis_as_of_ts"))
    declaration_pin = _CONFIRMATION_LOOK_DECLARATION
    declaration_pin_available = bool(
        declaration_pin.get("schema_version") == 1
        and isinstance(declaration_pin.get("path"), str)
        and declaration_pin.get("path")
        and sha256_literal(declaration_pin.get("sha256"))
        and _strict_count(declaration_pin.get("bytes")) not in (None, 0)
        and isinstance(declaration_pin.get("look_id"), str)
        and declaration_pin.get("look_id")
        and _number(declaration_pin.get("declared_at_ts")) is not None
        and _number(declaration_pin.get("analysis_as_of_ts")) is not None
        and declaration_pin.get("enrollment_cutoff_exclusive")
        == _CONFIRMATION_EXPIRES_AT
        and declaration_pin.get("prefix_procedure_version")
        == "stable-filtered-jsonl-snapshot-v2"
        and declaration_pin.get("prefix_capture_not_before")
        == _number(declaration_pin.get("analysis_as_of_ts")) + 300
        and declaration_pin.get("prefix_output_path_template")
        == "data/canary-confirmation-prefix/{declaration_sha256}.json"
    )

    design_checks = {
        "structured_reporter_contract": bool(
            type(evidence.get("schema_version")) is int
            and evidence.get("schema_version") == 2
            and evidence.get("producer")
            == "scripts.canary_report:prospective-confirmation-v2"
        ),
        "exact_plan_strategy_targets_and_family_rule": exact_literal(
            contract, expected_contract
        ),
        "exact_itt_row_linkage": bool(
            linked_total is not None
            and linked_total > 0
            and all(
                linked_counts[arm] == itt_counts[arm]
                for arm in ("treatment", "control")
            )
            and label_row_totals == linked_counts
        ),
        "exact_population_row_linkage": bool(
            all(population_counts[arm] is not None for arm in ("treatment", "control"))
            and label_population_totals == population_counts
        ),
        "all_prospective_rows_approved_and_linked": bool(
            linked_total is not None
            and prospective_rows == linked_total
            and approved_rows == linked_total
        ),
        "strictly_within_frozen_activation_window": bool(
            first_enrollment_ts is not None
            and last_enrollment_ts is not None
            and _CONFIRMATION_ACTIVATED_AT
            <= first_enrollment_ts
            <= last_enrollment_ts
            < _CONFIRMATION_EXPIRES_AT
        ),
        "all_four_labels_exactly_reported": set(blocks) == set(labels),
        "same_30m_blocks_all_labels": bool(
            common_blocks_reported == common_blocks_total
            and common_blocks_total >= 30
            and all(
                item["counts_valid"]
                and item["treatment_representation"]
                and item["control_representation"]
                and item["both_arms_have_blocks"]
                and item["same_block_representation"]
                for item in label_checks.values()
            )
        ),
        "immutable_prefix_identity": bool(
            prefix.get("status") == "verified"
            and prefix.get("producer")
            == "scripts.canary_report:immutable-prefix-v1"
            and prefix.get("algorithm") == "sha256"
            and sha256_literal(prefix_sha256)
            and prefix_bytes is not None
            and prefix_bytes > 0
            and prefix_records is not None
            and linked_total is not None
            and prefix_records >= linked_total
            and prefix_last_event_ts is not None
            and analysis_as_of_ts is not None
            # The scheduled analysis time bounds the physical prefix; it is
            # not itself an event timestamp.  Sparse logs normally end before
            # that boundary, but must still cover every enrolled observation.
            and last_enrollment_ts is not None
            and last_enrollment_ts <= prefix_last_event_ts <= analysis_as_of_ts
        ),
        "pinned_scheduled_look_declaration": bool(
            declaration_pin_available
            and exact_literal(
                scheduled_look.get("declaration_artifact"), declaration_pin
            )
            and scheduled_look.get("declaration_sha256")
            == declaration_pin.get("sha256")
            and scheduled_look.get("look_id") == declaration_pin.get("look_id")
            and declared_at_ts == _number(declaration_pin.get("declared_at_ts"))
        ),
        "reporter_recomputed_prefix_and_declaration": bool(
            type(reporter_verification.get("schema_version")) is int
            and reporter_verification.get("schema_version") == 1
            and reporter_verification.get("producer")
            == "scripts.canary_report:confirmation-verifier-v1"
            and reporter_verification.get("prefix_recomputed_from_sources") is True
            and reporter_verification.get("declaration_recomputed_from_artifact")
            is True
            and reporter_verification.get("prefix_sha256") == prefix_sha256
            and reporter_verification.get("declaration_sha256")
            == declaration_pin.get("sha256")
            and set(reporter_verification)
            == {
                "schema_version",
                "producer",
                "prefix_recomputed_from_sources",
                "declaration_recomputed_from_artifact",
                "prefix_sha256",
                "declaration_sha256",
            }
        ),
        "predeclared_scheduled_look_linked_to_prefix": bool(
            declaration_pin_available
            and scheduled_look.get("status") == "verified"
            and scheduled_look.get("plan_id") == _CONFIRMATION_PLAN_ID
            and isinstance(scheduled_look.get("look_id"), str)
            and bool(scheduled_look.get("look_id"))
            and sha256_literal(scheduled_look.get("declaration_sha256"))
            and declared_at_ts is not None
            and declared_at_ts < _CONFIRMATION_ACTIVATED_AT
            and scheduled_at_ts is not None
            and scheduled_at_ts
            == _number(declaration_pin.get("analysis_as_of_ts"))
            and scheduled_analysis_ts is not None
            and analysis_as_of_ts is not None
            and scheduled_analysis_ts == analysis_as_of_ts
            and analysis_as_of_ts
            == _number(declaration_pin.get("analysis_as_of_ts"))
            and scheduled_look.get("prefix_sha256") == prefix_sha256
        ),
    }
    passed = all(design_checks.values()) and all(
        check for entry in label_checks.values() for check in entry.values()
    )
    return {
        "pass": passed,
        "expected_contract": deepcopy(expected_contract),
        "expected_declaration_artifact": deepcopy(declaration_pin),
        "required_label_population_fields": {
            "treatment": population_fields[0],
            "control": population_fields[1],
        },
        "design_checks": design_checks,
        "labels": label_checks,
        "linked_itt_rows": linked_counts,
        "legacy_randomization_evidence_ignored": "randomization_evidence"
        in experiment,
        "pending_confirmation_input": (
            "reporter has not supplied verified immutable-prefix and scheduled-look "
            "evidence"
            if not (
                design_checks["immutable_prefix_identity"]
                and design_checks["pinned_scheduled_look_declaration"]
                and design_checks[
                    "reporter_recomputed_prefix_and_declaration"
                ]
                and design_checks["predeclared_scheduled_look_linked_to_prefix"]
            )
            else None
        ),
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
    binding_health = experiment.get("_binding_cohort_health")
    binding_routing = experiment.get("_binding_cohort_routing")
    if isinstance(binding_health, Mapping) and isinstance(binding_routing, Mapping):
        for arm in ("treatment", "control"):
            health = binding_health.get(arm)
            routing = binding_routing.get(arm)
            health = health if isinstance(health, Mapping) else {}
            routing = routing if isinstance(routing, Mapping) else {}
            for key in (
                "turn_errors",
                "corrections",
                "result_errors",
                "invalid_results",
                "http_503",
            ):
                value = _strict_count(health.get(key))
                if value is None:
                    evidence_failures.append(
                        f"prospective {arm}: health.{key} missing/malformed"
                    )
                else:
                    totals[key] += value
            for key in (
                "checked",
                "assigned_matches",
                "affected",
                "affected_assigned_matches",
                "replay_errors",
                "assignment_integrity_errors",
                "duplicate_causal_turn_conflicts",
                "affected_wrong_variant",
                "affected_unknown",
                "direction_violations",
            ):
                value = _strict_count(routing.get(key))
                if value is None:
                    evidence_failures.append(
                        f"prospective {arm}: routing.{key} missing/malformed"
                    )
                else:
                    totals[key] += value
            checked = _strict_count(routing.get("checked"))
            matches = _strict_count(routing.get("assigned_matches"))
            affected = _strict_count(routing.get("affected"))
            affected_matches = _strict_count(
                routing.get("affected_assigned_matches")
            )
            if checked is not None and matches is not None and matches > checked:
                evidence_failures.append(
                    f"prospective {arm}: routing matches exceed checked"
                )
            if (
                affected is not None
                and affected_matches is not None
                and affected_matches > affected
            ):
                evidence_failures.append(
                    f"prospective {arm}: affected matches exceed affected"
                )
        routing_integrity = binding_routing.get("integrity")
        routing_integrity = (
            routing_integrity if isinstance(routing_integrity, Mapping) else {}
        )
        for key in (
            "unknown_assignment_turns",
            "unknown_assignment_games",
            "unapproved_prospective_games",
            "duplicate_causal_turn_conflicts",
        ):
            value = _strict_count(routing_integrity.get(key))
            if value is None:
                evidence_failures.append(
                    f"prospective routing integrity.{key} missing/malformed"
                )
            elif value:
                evidence_failures.append(
                    f"prospective routing integrity.{key} is nonzero"
                )
    elif isinstance(agents, Mapping):
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
    if not expected_agents and not isinstance(binding_health, Mapping):
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
        "assignment_integrity_errors": totals["assignment_integrity_errors"],
        "duplicate_causal_turn_conflicts": totals[
            "duplicate_causal_turn_conflicts"
        ],
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
    observed_hard_failures = {
        key: value
        for key, value in hard_failures.items()
        if key != "missing_or_inconsistent_evidence"
    }
    return {
        "passed": not any(hard_failures.values()),
        "evidence_complete": not evidence_failures,
        "observed_harm": any(observed_hard_failures.values()),
        "manual_review_required": bool(
            warnings["http_503"] or warnings["invalid_results"]
        ),
        "hard_failures": hard_failures,
        "warnings": warnings,
        "integrity_failures": evidence_failures,
    }


def _bargaining_gate(experiment: Mapping[str, Any]) -> dict:
    experiment, binding_cohort = _prospective_binding_view(experiment)
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
        "itt_raw_lower_bound": _check(
            raw_itt.get("lower_95_one_sided"), ">", -0.010
        ),
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
        "treatment_itt": _validate_itt_arm(
            itt_treatment, 1200, family="bargaining"
        ),
        "control_itt": _validate_itt_arm(
            itt_control, 1200, family="bargaining"
        ),
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
    if not binding_cohort["passed"]:
        integrity_failures["analysis_cohort"] = list(
            binding_cohort["failures"]
        )
    data_integrity_pass = not any(integrity_failures.values())
    affected_unstandardized = _difference_summary(
        treatment.get("mean_normalized_payoff"),
        treatment.get("resolved"),
        control.get("mean_normalized_payoff"),
        control.get("resolved"),
    )

    rollback_triggers: list[str] = []
    if guardrails["observed_harm"]:
        rollback_triggers.append("health_or_routing_guardrail")
    treatment_direct = _strict_count(treatment.get("direct_resolved"))
    treatment_converted = _strict_count(treatment.get("direct_converted"))
    control_direct = _strict_count(control.get("direct_resolved"))
    treatment_direct_rate = (
        treatment_converted / treatment_direct
        if treatment_direct is not None
        and treatment_direct > 0
        and treatment_converted is not None
        and treatment_converted <= treatment_direct
        else None
    )
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
    deadline_sum = _number(itt_treatment.get("normalized_outcome_sum"))
    deadline_payoff = (
        deadline_sum / itt_treatment_n
        if itt_treatment_n is not None
        and itt_treatment_n > 0
        and deadline_sum is not None
        and _validate_moment_bounds(
            deadline_sum,
            itt_treatment.get("normalized_outcome_sum_squares"),
            itt_treatment_n,
        )
        else None
    )
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
    binding_rollback = bool(
        rollback_triggers
        and data_integrity_pass
        and guardrails["evidence_complete"]
    )
    if binding_rollback:
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
        "amendment_provenance": deepcopy(AMENDMENT_PROVENANCE),
        "decision": decision,
        "screen_ready": screen_ready,
        "promotion_ready": promotion_ready,
        "rollback_triggers": rollback_triggers,
        "binding_rollback": binding_rollback,
        "binding_analysis_cohort": binding_cohort,
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
    experiment, binding_cohort = _prospective_binding_view(experiment)
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
    fixed_zero_difference = _fixed_weight_risk_difference(
        itt_treatment,
        itt_control,
        ("p", "message_type", "price", "total_rounds"),
        PERS_CELL_WEIGHTS,
    )
    raw_zero_check = _check(
        zero_difference.get("upper_95_one_sided"), "<=", 0.02
    )
    fixed_zero_check = _check(
        fixed_zero_difference.get("missing_mass_conservative_upper_95_one_sided"),
        "<=",
        0.02,
    )
    zero_upper_bounds = (
        zero_difference.get("upper_95_one_sided"),
        fixed_zero_difference.get("missing_mass_conservative_upper_95_one_sided"),
    )
    binding_zero_upper = (
        max(float(value) for value in zero_upper_bounds)
        if all(_number(value) is not None for value in zero_upper_bounds)
        else None
    )
    zero_sale = {
        **zero_difference,
        "raw_newcombe": zero_difference,
        "fixed_weight": fixed_zero_difference,
        "raw_check": raw_zero_check,
        "fixed_weight_check": fixed_zero_check,
        "check": _check(binding_zero_upper, "<=", 0.02),
    }
    censor_safety = _censor_safety(itt, family="persuasion")
    causal = _causal_confirmation(experiment, "persuasion")
    integrity_failures = {
        "treatment_affected": _validate_affected_arm(treatment, "persuasion"),
        "control_affected": _validate_affected_arm(control, "persuasion"),
        "treatment_itt": _validate_itt_arm(
            itt_treatment, 1800, family="persuasion"
        ),
        "control_itt": _validate_itt_arm(
            itt_control, 1800, family="persuasion"
        ),
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
    if not binding_cohort["passed"]:
        integrity_failures["analysis_cohort"] = list(
            binding_cohort["failures"]
        )
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
    if guardrails["observed_harm"]:
        rollback_triggers.append("health_or_routing_guardrail")
    rollback_triggers.extend(
        _scheduled_censor_harm(censor_safety, treatment_min=150, control_min=150)
    )
    binding_rollback = bool(
        rollback_triggers
        and data_integrity_pass
        and guardrails["evidence_complete"]
    )
    if binding_rollback:
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
        "amendment_provenance": deepcopy(AMENDMENT_PROVENANCE),
        "decision": decision,
        "screen_ready": screen_ready,
        "promotion_ready": promotion_ready,
        "rollback_triggers": rollback_triggers,
        "binding_rollback": binding_rollback,
        "binding_analysis_cohort": binding_cohort,
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
