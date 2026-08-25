"""Regression tests for deterministic bargaining and persuasion gates."""

from __future__ import annotations

import copy

import pytest

from scripts.canary_gates import (
    BARG_AFFECTED_WEIGHTS,
    BARG_ITT_WEIGHTS,
    PERS_CELL_WEIGHTS,
    PERS_P_WEIGHTS,
    _newcombe_difference,
    evaluate_gate,
    evaluate_report_gates,
)


def _agent(*, route_miss=0, errors=0, http_503=0, invalid=0):
    checked = 10
    return {
        "health": {
            "turns": 1000,
            "turn_errors": errors,
            "corrections": 0,
            "result_events": 1000,
            "result_errors": 0,
            "invalid_results": invalid,
            "http_503": http_503,
            "duplicate_turns": 0,
        },
        "routing": {
            "checked": checked,
            "assigned_matches": checked - route_miss,
            "affected": checked,
            "affected_assigned_matches": checked - route_miss,
            "replay_errors": 0,
            "affected_wrong_variant": 0,
            "affected_unknown": route_miss,
            "direction_violations": 0,
        },
    }


def _moments(n, mean, *, prefix):
    return {
        f"{prefix}_sum": n * mean,
        f"{prefix}_sum_squares": n * mean * mean,
    }


def _barg_arm(n, mean):
    scale = n // 300
    base_counts = [121, 114, 3, 30, 32]
    cells = {}
    finite_n = 0
    unlimited_n = 0
    for index, ((role, horizon, phase), count) in enumerate(
        zip(BARG_AFFECTED_WEIGHTS, base_counts, strict=True)
    ):
        count *= scale
        finite_n += count if horizon == "finite" else 0
        unlimited_n += count if horizon == "unlimited" else 0
        cells[str(index)] = {
            "cell": {
                "role": role,
                "horizon": horizon,
                "phase": phase,
                "max_rounds": "6" if horizon == "finite" else "unlimited",
            },
            "games": count,
            "resolved": count,
            "censored": 0,
            "normalized_payoff_valid": count,
            "normalized_payoff_invalid": 0,
            "mean_normalized_payoff": mean,
            **_moments(count, mean, prefix="normalized_payoff"),
        }
    converted_finite = round(finite_n * 0.40)
    converted_unlimited = round(unlimited_n * 0.40)
    return {
        "games": n,
        "affected_games": n,
        "resolved": n,
        "censored": 0,
        "normalized_payoff_valid": n,
        "normalized_payoff_invalid": 0,
        "mean_normalized_payoff": mean,
        **_moments(n, mean, prefix="normalized_payoff"),
        "direct_resolved": n,
        "direct_converted": converted_finite + converted_unlimited,
        "direct_conversion_rate": (converted_finite + converted_unlimited) / n,
        "horizon_strata": {
            "finite": {
                "resolved": finite_n,
                "direct_resolved": finite_n,
                "direct_converted": converted_finite,
            },
            "unlimited": {
                "resolved": unlimited_n,
                "direct_resolved": unlimited_n,
                "direct_converted": converted_unlimited,
            },
        },
        "cells": cells,
    }


def _itt_leaf(n, mean, *, censored=0):
    resolved = n - censored
    total = resolved * mean
    return {
        "games": n,
        "matured": n,
        "pending_maturation": 0,
        "resolved": resolved,
        "censored": censored,
        "deadline_censored": censored,
        "timely_valid_terminals": resolved,
        "deadline_zeroes": censored,
        "late_terminals": 0,
        "invalid_terminals": 0,
        "normalized_outcome_sum": total,
        "normalized_outcome_sum_squares": resolved * mean * mean,
        "mean_normalized_outcome": total / n,
        "sample_variance_normalized_outcome": None,
        "zero_sales": censored,
        "zero_sales_rate": censored / n,
    }


def _barg_itt(n, mean):
    counts = [round(n * weight) for weight in BARG_ITT_WEIGHTS.values()]
    counts[-1] += n - sum(counts)
    metric = {
        **_itt_leaf(n, mean),
        "population": "all strictly enrolled bargaining games",
        "maturity_lag_s": 1200,
        "cells": {},
        "p_strata": {},
    }
    for index, ((role, horizon), count) in enumerate(
        zip(BARG_ITT_WEIGHTS, counts, strict=True)
    ):
        metric["cells"][str(index)] = {
            "cell": {"role": role, "horizon": horizon},
            **_itt_leaf(count, mean),
        }
    return metric


def _barg_experiment():
    return {
        "name": "barg_dis_anchor",
        "family": "bargaining",
        "assignment": {
            "treatment_agents": ["test_b"],
            "control_agents": ["main"],
        },
        "agents": {"test_b": _agent(), "main": _agent()},
        "metrics": {
            "treatment": _barg_arm(300, 0.60),
            "control": _barg_arm(900, 0.40),
        },
        "itt": {
            "treatment": _barg_itt(2521, 0.45),
            "control": _barg_itt(7563, 0.40),
            "integrity": {"unknown_assignment_games": 0},
        },
    }


def _pers_cells(revenue, *, scale=1):
    cells = {}
    for index, ((p, message, price, rounds), weight) in enumerate(
        PERS_CELL_WEIGHTS.items()
    ):
        count = round(weight * 1762) * scale
        cells[str(index)] = {
            "cell": {
                "p": 1 / 3 if p == "0.333333" else float(p),
                "message_type": message,
                "price": price,
                "total_rounds": rounds,
                "opponent_type": "hidden",
                "start_block_15m": 0,
            },
            "blind_seller_games": count,
            "resolved": count,
            "censored": 0,
            "revenue_share_valid": count,
            "revenue_share_invalid": 0,
            "mean_revenue_share": revenue,
            "zero_sales": 0,
            **_moments(count, revenue, prefix="revenue_share"),
        }
    return cells


def _pers_arm(revenue, zero_sales):
    n = 1762
    cells = _pers_cells(revenue)
    return {
        "blind_seller_games": n,
        "resolved": n,
        "censored": 0,
        "revenue_share_valid": n,
        "revenue_share_invalid": 0,
        "mean_revenue_share": revenue,
        **_moments(n, revenue, prefix="revenue_share"),
        "zero_sales": zero_sales,
        "p_strata": {},
        "cells": cells,
        "deterministic_route_checked": 20,
        "deterministic_route_matches": 20,
    }


def _pers_itt(revenue, zero_sales, *, scale=1, censor_fraction=0.0):
    n = 1762 * scale
    censored = round(n * censor_fraction)
    metric = {
        **_itt_leaf(n, revenue, censored=censored),
        "population": "all strictly enrolled explicit blind-seller games",
        "maturity_lag_s": 1800,
        "cells": {},
        "p_strata": {},
    }
    # Keep the requested zero-sale sufficient statistic independent of the
    # constant-payoff moments used by these synthetic fixtures.
    metric["zero_sales"] = zero_sales if not censor_fraction else censored
    metric["zero_sales_rate"] = metric["zero_sales"] / n
    for index, ((p, message, price, rounds), weight) in enumerate(
        PERS_CELL_WEIGHTS.items()
    ):
        count = round(weight * 1762) * scale
        cell_censored = round(count * censor_fraction)
        entry = _itt_leaf(count, revenue, censored=cell_censored)
        metric["cells"][str(index)] = {
            "cell": {
                "p": 1 / 3 if p == "0.333333" else float(p),
                "message_type": message,
                "price": price,
                "total_rounds": rounds,
            },
            **entry,
        }
    # Correct any rounding residue in the last fixed cell.
    last = metric["cells"][str(len(metric["cells"]) - 1)]
    residue = n - sum(entry["games"] for entry in metric["cells"].values())
    for key in ("games", "matured", "resolved", "timely_valid_terminals"):
        last[key] += residue
    last["normalized_outcome_sum"] += residue * revenue
    last["normalized_outcome_sum_squares"] += residue * revenue * revenue
    last["mean_normalized_outcome"] = (
        last["normalized_outcome_sum"] / last["matured"]
    )
    for p, weight in PERS_P_WEIGHTS.items():
        count = round(weight * n)
        p_censored = round(count * censor_fraction)
        metric["p_strata"][p] = _itt_leaf(count, revenue, censored=p_censored)
    p_last = metric["p_strata"]["0.8"]
    p_residue = n - sum(entry["games"] for entry in metric["p_strata"].values())
    for key in ("games", "matured", "resolved", "timely_valid_terminals"):
        p_last[key] += p_residue
    p_last["normalized_outcome_sum"] += p_residue * revenue
    p_last["normalized_outcome_sum_squares"] += p_residue * revenue * revenue
    p_last["mean_normalized_outcome"] = p_last["normalized_outcome_sum"] / p_last["matured"]
    return metric


def _pers_experiment():
    return {
        "name": "pers_blind_lie",
        "family": "persuasion",
        "assignment": {
            "treatment_agents": ["test_a"],
            "control_agents": ["main"],
        },
        "agents": {"test_a": _agent(), "main": _agent()},
        "metrics": {
            "treatment": _pers_arm(0.60, 100),
            "control": _pers_arm(0.50, 150),
        },
        "itt": {
            "treatment": _pers_itt(0.60, 100),
            "control": _pers_itt(0.50, 150),
            "integrity": {"unknown_assignment_games": 0},
        },
    }


def test_bargaining_fixed_label_evidence_is_capped_at_screen_pass():
    gate = evaluate_gate(_barg_experiment())

    assert gate is not None
    assert gate["decision"] == "screen_pass"
    assert gate["screen_ready"] is True
    assert gate["promotion_ready"] is False
    assert gate["statistics"]["direct_conversion"]["overall"]["passed"]
    assert gate["statistics"]["direct_conversion"]["finite"]["passed"]
    assert gate["statistics"]["direct_conversion"]["unlimited"]["passed"]
    standardized = gate["statistics"]["standardized_payoff"]
    assert standardized["difference"] == pytest.approx(0.20)
    assert standardized["lower_95_one_sided"] > 0
    assert standardized["support"]["common_mass"] == pytest.approx(1)
    assert gate["statistics"]["itt"]["available"] is True
    assert gate["causal_confirmation"]["pass"] is False


@pytest.mark.parametrize(
    ("rate", "payoff", "trigger"),
    [
        (0.08, 0.40, "scheduled affected direct conversion <= 0.08"),
        (0.20, 0.27, "scheduled treatment deadline payoff <= 0.27"),
    ],
)
def test_bargaining_early_rollback_is_armed_at_100(rate, payoff, trigger):
    experiment = _barg_experiment()
    treatment = experiment["metrics"]["treatment"]
    treatment["affected_games"] = 100
    treatment["resolved"] = 100
    treatment["direct_resolved"] = 100
    treatment["direct_converted"] = round(100 * rate)
    treatment["direct_conversion_rate"] = rate
    treatment["mean_normalized_payoff"] = payoff
    if payoff <= 0.27:
        experiment["itt"]["treatment"]["mean_normalized_outcome"] = payoff

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert gate["decision"] == "rollback"
    assert trigger in gate["rollback_triggers"]


def test_hard_routing_failure_rolls_back_but_transport_requires_review():
    broken = _barg_experiment()
    broken["agents"]["test_b"] = _agent(route_miss=1)
    gate = evaluate_gate(broken)
    assert gate is not None
    assert gate["decision"] == "rollback"
    assert gate["guardrails"]["hard_failures"]["routing_mismatches"] == 1

    transport = _barg_experiment()
    transport["agents"]["main"] = _agent(http_503=1, invalid=1)
    gate = evaluate_gate(transport)
    assert gate is not None
    assert gate["decision"] == "manual_review"
    assert gate["screen_ready"] is True
    assert gate["guardrails"]["manual_review_required"] is True


def test_persuasion_fixed_label_evidence_is_capped_at_screen_pass():
    gate = evaluate_gate(_pers_experiment())

    assert gate is not None
    assert gate["decision"] == "screen_pass"
    assert gate["screen_ready"] is True
    assert gate["promotion_ready"] is False
    revenue = gate["statistics"]["standardized_revenue"]
    assert revenue["difference"] == pytest.approx(0.10)
    assert revenue["lower_95_one_sided"] > 0
    zero = gate["statistics"]["zero_sale_noninferiority"]
    assert zero["difference"] < 0
    assert zero["upper_95_one_sided"] <= 0.02
    assert all(
        check["passed"]
        for pair in gate["p_strata_checks"].values()
        for check in pair.values()
    )
    assert "start_block_15m" not in revenue["support"]["dimensions"]


def test_persuasion_time_blocks_do_not_fragment_configuration_support():
    experiment = _pers_experiment()
    for cell in experiment["metrics"]["control"]["cells"].values():
        cell["cell"]["start_block_15m"] = 99

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert gate["decision"] == "screen_pass"
    support = gate["statistics"]["standardized_revenue"]["support"]
    assert support["common_mass"] == pytest.approx(1)
    temporal = gate["statistics"]["temporal_support_diagnostic"]
    assert temporal["common_groups"] == 0


def test_persuasion_waits_for_each_arm_in_every_observed_p_stratum():
    experiment = _pers_experiment()
    experiment["itt"]["treatment"]["p_strata"]["0.333333"]["matured"] = 149

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert gate["decision"] == "continue"
    assert not gate["p_strata_checks"]["0.333333"]["treatment_matured"]["passed"]


def test_persuasion_zero_sale_upper_bound_can_block_promotion():
    experiment = _pers_experiment()
    experiment["itt"]["treatment"]["zero_sales"] = 400
    experiment["itt"]["control"]["zero_sales"] = 100

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert gate["decision"] == "continue"
    assert not gate["statistics"]["zero_sale_noninferiority"]["check"]["passed"]


def test_bargaining_tiny_favorable_common_support_cannot_promote():
    experiment = _barg_experiment()
    treatment = experiment["metrics"]["treatment"]
    control = experiment["metrics"]["control"]
    treatment["cells"] = {"common": treatment["cells"]["0"]}
    control["cells"] = {"common": control["cells"]["0"]}
    for entry, mean in ((treatment["cells"]["common"], 0.9), (control["cells"]["common"], 0.1)):
        n = entry["resolved"]
        entry["mean_normalized_payoff"] = mean
        entry.update(_moments(n, mean, prefix="normalized_payoff"))

    gate = evaluate_gate(experiment)

    assert gate is not None
    standardized = gate["statistics"]["standardized_payoff"]
    assert standardized["difference"] > 0
    assert gate["decision"] == "continue"
    assert standardized["support"]["common_mass"] < 0.90
    assert not gate["statistics"]["support_checks"][
        "affected_common_reference_mass"
    ]["passed"]


def test_persuasion_tiny_favorable_common_support_cannot_promote():
    experiment = _pers_experiment()
    treatment = experiment["metrics"]["treatment"]
    control = experiment["metrics"]["control"]
    treatment["cells"] = {"common": treatment["cells"]["0"]}
    control["cells"] = {"common": control["cells"]["0"]}
    for entry, mean in ((treatment["cells"]["common"], 0.9), (control["cells"]["common"], 0.1)):
        n = entry["resolved"]
        entry["mean_revenue_share"] = mean
        entry.update(_moments(n, mean, prefix="revenue_share"))

    gate = evaluate_gate(experiment)

    assert gate is not None
    standardized = gate["statistics"]["standardized_revenue"]
    assert standardized["difference"] > 0
    assert gate["decision"] == "continue"
    assert standardized["support"]["common_mass"] < 0.90
    assert not gate["statistics"]["support_checks"][
        "affected_common_reference_mass"
    ]["passed"]


def test_error_union_uses_treatment_upper_minus_control_lower():
    error = _pers_experiment()
    for label in ("test_a", "main"):
        health = error["agents"][label]["health"]
        health["turns"] = 100
        health["result_events"] = 100
    error["agents"]["test_a"]["health"]["invalid_results"] = 5
    error["agents"]["main"]["health"]["result_errors"] = 2
    error["agents"]["main"]["health"]["invalid_results"] = 2

    gate = evaluate_gate(error)

    assert gate is not None
    error_gate = gate["arm_rate_guardrails"]["errors"]
    assert error_gate["treatment"]["rate_upper_bound"] == pytest.approx(5 / 200)
    assert error_gate["control"]["rate_lower_bound"] == pytest.approx(2 / 200)
    assert error_gate["treatment_excess_upper_bound"] == pytest.approx(0.015)
    assert not error_gate["check"]["passed"]


def test_bilateral_heavy_censoring_fails_absolute_maturation_gate():
    experiment = _pers_experiment()
    experiment["itt"]["treatment"] = _pers_itt(
        0.60, 15858, scale=10, censor_fraction=0.9
    )
    experiment["itt"]["control"] = _pers_itt(
        0.50, 15858, scale=10, censor_fraction=0.9
    )

    gate = evaluate_gate(experiment)

    assert gate is not None
    overall = gate["censor_safety"]["strata"]["overall"]
    assert not overall["treatment"]["absolute_pass"]
    assert not overall["control"]["absolute_pass"]
    assert gate["decision"] == "rollback"


@pytest.mark.parametrize("bad", [None, "0", -1])
def test_missing_or_malformed_zero_sales_never_becomes_favorable_zero(bad):
    experiment = _pers_experiment()
    if bad is None:
        experiment["itt"]["treatment"].pop("zero_sales")
    else:
        experiment["itt"]["treatment"]["zero_sales"] = bad

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert gate["decision"] != "screen_pass"
    assert not gate["data_integrity"]["passed"]
    assert not gate["statistics"]["zero_sale_noninferiority"]["available"]


def _set_itt_mean(metric, mean):
    for entry in [metric, *metric["cells"].values(), *metric["p_strata"].values()]:
        n = entry["matured"]
        entry["normalized_outcome_sum"] = n * mean
        entry["normalized_outcome_sum_squares"] = n * mean * mean
        entry["mean_normalized_outcome"] = mean


def test_adverse_true_itt_blocks_favorable_resolved_overlap():
    experiment = _pers_experiment()
    _set_itt_mean(experiment["itt"]["treatment"], 0.49)
    _set_itt_mean(experiment["itt"]["control"], 0.50)
    # The affected complete-case overlap still looks strongly favorable.
    assert experiment["metrics"]["treatment"]["mean_revenue_share"] == 0.60

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert gate["statistics"]["standardized_revenue"]["difference"] > 0
    assert gate["statistics"]["raw_itt_revenue"]["difference"] < 0
    assert not gate["statistics"]["revenue_checks"][
        "raw_itt_point_nonnegative"
    ]["passed"]
    assert gate["decision"] == "continue"


def test_expected_p_set_is_fixed_even_if_stratum_disappears_in_both_arms():
    experiment = _pers_experiment()
    for arm in ("treatment", "control"):
        experiment["itt"][arm]["p_strata"].pop("0.8")
        experiment["itt"][arm]["cells"] = {
            key: value
            for key, value in experiment["itt"][arm]["cells"].items()
            if value["cell"]["p"] != 0.8
        }

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert set(gate["p_strata_checks"]) == {"0.333333", "0.5", "0.8"}
    assert not gate["p_strata_checks"]["0.8"]["treatment_matured"]["passed"]
    assert gate["decision"] == "continue"


def test_newcombe_mover_upper_uses_quadrature_not_component_subtraction():
    summary = _newcombe_difference(100, 1000, 150, 1000)
    pt = summary["treatment"]["rate"]
    pc = summary["control"]["rate"]
    ut = summary["treatment"]["upper_95_one_sided"]
    lc = summary["control"]["lower_95_one_sided"]
    expected = (pt - pc) + ((ut - pt) ** 2 + (pc - lc) ** 2) ** 0.5

    assert summary["upper_95_one_sided"] == pytest.approx(expected)
    assert summary["upper_95_one_sided"] < ut - lc


def test_report_evaluation_is_deterministic_and_does_not_mutate_raw_report():
    report = {
        "generated_at": "frozen",
        "experiments": [_barg_experiment(), _pers_experiment(), {"name": "neg"}],
    }
    original = copy.deepcopy(report)

    first = evaluate_report_gates(report)
    second = evaluate_report_gates(report)

    assert first == second
    assert report == original
    assert set(first) == {"barg_dis_anchor", "pers_blind_lie"}
