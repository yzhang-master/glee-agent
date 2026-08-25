"""Regression tests for deterministic bargaining and persuasion gates."""

from __future__ import annotations

import copy

import pytest

from scripts.canary_gates import evaluate_gate, evaluate_report_gates


def _agent(*, route_miss=0, errors=0, http_503=0, invalid=0):
    checked = 10
    return {
        "health": {
            "turns": 100,
            "turn_errors": errors,
            "corrections": 0,
            "result_events": 100,
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


def _barg_arm(n, mean, conversions, finite_n, unlimited_n):
    finite_conversions, unlimited_conversions = conversions
    cells = {}
    for horizon, count, payoff in (
        ("finite", finite_n, mean),
        ("unlimited", unlimited_n, mean),
    ):
        cells[horizon] = {
            "cell": {
                "role": "player_1",
                "horizon": horizon,
                "phase": "offer",
                "max_rounds": "6" if horizon == "finite" else "unlimited",
                "my_delta": 0.9,
            },
            "resolved": count,
            "mean_normalized_payoff": payoff,
        }
    return {
        "affected_games": n,
        "resolved": n,
        "censored": 0,
        "mean_normalized_payoff": mean,
        "direct_resolved": n,
        "direct_converted": finite_conversions + unlimited_conversions,
        "direct_conversion_rate": (finite_conversions + unlimited_conversions) / n,
        "horizon_strata": {
            "finite": {
                "resolved": finite_n,
                "direct_resolved": finite_n,
                "direct_converted": finite_conversions,
            },
            "unlimited": {
                "resolved": unlimited_n,
                "direct_resolved": unlimited_n,
                "direct_converted": unlimited_conversions,
            },
        },
        "cells": cells,
    }


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
            "treatment": _barg_arm(300, 0.60, (60, 60), 150, 150),
            "control": _barg_arm(900, 0.40, (150, 150), 450, 450),
        },
    }


def _pers_arm(n, revenue, zero_sales):
    cells = {}
    p_strata = {}
    for p in ("0.25", "0.75"):
        cells[p] = {
            "cell": {
                "p": float(p),
                "message_type": "binary",
                "price": 100.0,
                "total_rounds": 10,
                "opponent_type": "hidden",
                "start_block_15m": 0,
            },
            "resolved": n // 2,
            "mean_revenue_share": revenue,
        }
        p_strata[p] = {"resolved": n // 2}
    return {
        "blind_seller_games": n,
        "resolved": n,
        "censored": 0,
        "mean_revenue_share": revenue,
        "zero_sales": zero_sales,
        "p_strata": p_strata,
        "cells": cells,
        "deterministic_route_checked": 20,
        "deterministic_route_matches": 20,
    }


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
            "treatment": _pers_arm(1000, 0.60, 100),
            "control": _pers_arm(1000, 0.50, 150),
        },
    }


def test_bargaining_promotes_only_after_all_frozen_gates_pass():
    gate = evaluate_gate(_barg_experiment())

    assert gate is not None
    assert gate["decision"] == "promote"
    assert gate["promotion_ready"] is True
    assert gate["statistics"]["direct_conversion"]["overall"]["passed"]
    assert gate["statistics"]["direct_conversion"]["finite"]["passed"]
    assert gate["statistics"]["direct_conversion"]["unlimited"]["passed"]
    standardized = gate["statistics"]["standardized_payoff"]
    assert standardized["difference"] == pytest.approx(0.20)
    assert standardized["lower_95_one_sided"] > 0
    assert standardized["support"]["common_groups"] == 2
    assert gate["statistics"]["itt"]["available"] is False


@pytest.mark.parametrize(
    ("rate", "payoff", "trigger"),
    [
        (0.08, 0.40, "direct_conversion_at_or_below_8pct_after_100"),
        (0.20, 0.27, "normalized_payoff_at_or_below_0.27_after_100"),
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
    assert gate["promotion_ready"] is True
    assert gate["guardrails"]["manual_review_required"] is True


def test_persuasion_promotes_with_standardized_lift_and_zero_sale_safety():
    gate = evaluate_gate(_pers_experiment())

    assert gate is not None
    assert gate["decision"] == "promote"
    assert gate["promotion_ready"] is True
    revenue = gate["statistics"]["standardized_revenue"]
    assert revenue["difference"] == pytest.approx(0.10)
    assert revenue["lower_95_one_sided"] > 0
    zero = gate["statistics"]["zero_sale_noninferiority"]
    assert zero["risk_difference"] == pytest.approx(-0.05)
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
    assert gate["decision"] == "promote"
    support = gate["statistics"]["standardized_revenue"]["support"]
    assert support["common_groups"] == 2
    assert support["treatment_common_fraction"] == 1
    assert support["control_common_fraction"] == 1
    temporal = gate["statistics"]["temporal_support_diagnostic"]
    assert temporal["common_groups"] == 0


def test_persuasion_waits_for_each_arm_in_every_observed_p_stratum():
    experiment = _pers_experiment()
    experiment["metrics"]["treatment"]["p_strata"]["0.25"]["resolved"] = 149

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert gate["decision"] == "continue"
    assert not gate["p_strata_checks"]["0.25"]["treatment_resolved"]["passed"]


def test_persuasion_zero_sale_upper_bound_can_block_promotion():
    experiment = _pers_experiment()
    experiment["metrics"]["treatment"]["zero_sales"] = 200
    experiment["metrics"]["control"]["zero_sales"] = 100

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert gate["decision"] == "continue"
    assert not gate["statistics"]["zero_sale_noninferiority"]["check"]["passed"]


def _barg_cell(role, horizon, phase, maximum, resolved, mean):
    return {
        "cell": {
            "role": role,
            "horizon": horizon,
            "phase": phase,
            "max_rounds": maximum,
        },
        "resolved": resolved,
        "mean_normalized_payoff": mean,
    }


def test_bargaining_tiny_favorable_common_support_cannot_promote():
    experiment = _barg_experiment()
    experiment["metrics"]["treatment"]["cells"] = {
        "common": _barg_cell("player_1", "finite", "offer", "6", 30, 0.9),
        "t-only": _barg_cell(
            "player_2", "unlimited", "offer", "unlimited", 270, 0.9
        ),
    }
    experiment["metrics"]["control"]["cells"] = {
        "common": _barg_cell("player_1", "finite", "offer", "6", 90, 0.1),
        "c-only": _barg_cell(
            "player_1", "unlimited", "decision", "unlimited", 810, 0.1
        ),
    }

    gate = evaluate_gate(experiment)

    assert gate is not None
    standardized = gate["statistics"]["standardized_payoff"]
    assert standardized["difference"] == pytest.approx(0.8)
    assert standardized["lower_95_one_sided"] > 0
    assert gate["decision"] == "continue"
    assert not gate["statistics"]["support_checks"][
        "treatment_common_coverage"
    ]["passed"]
    assert not gate["statistics"]["support_checks"][
        "control_common_coverage"
    ]["passed"]


def test_persuasion_tiny_favorable_common_support_cannot_promote():
    experiment = _pers_experiment()
    base = {
        "p": 0.25,
        "price": 100.0,
        "total_rounds": 10,
        "opponent_type": "hidden",
        "start_block_15m": 0,
    }
    experiment["metrics"]["treatment"]["cells"] = {
        "common": {
            "cell": {**base, "message_type": "common"},
            "resolved": 100,
            "mean_revenue_share": 0.9,
        },
        "t-only": {
            "cell": {**base, "message_type": "treatment-only"},
            "resolved": 900,
            "mean_revenue_share": 0.9,
        },
    }
    experiment["metrics"]["control"]["cells"] = {
        "common": {
            "cell": {**base, "message_type": "common"},
            "resolved": 100,
            "mean_revenue_share": 0.1,
        },
        "c-only": {
            "cell": {**base, "message_type": "control-only"},
            "resolved": 900,
            "mean_revenue_share": 0.1,
        },
    }

    gate = evaluate_gate(experiment)

    assert gate is not None
    standardized = gate["statistics"]["standardized_revenue"]
    assert standardized["difference"] == pytest.approx(0.8)
    assert standardized["lower_95_one_sided"] > 0
    assert gate["decision"] == "continue"
    assert not all(
        check["passed"]
        for check in gate["statistics"]["support_checks"].values()
    )


def test_arm_censor_and_error_excess_guardrails_are_explicit():
    censor = _pers_experiment()
    censor["metrics"]["treatment"]["censored"] = 100
    gate = evaluate_gate(censor)
    assert gate is not None
    censor_gate = gate["arm_rate_guardrails"]["censoring"]
    assert censor_gate["treatment"]["rate"] == pytest.approx(100 / 1100)
    assert not censor_gate["check"]["passed"]
    assert gate["decision"] == "continue"

    error = _pers_experiment()
    error["agents"]["test_a"] = _agent(errors=3)
    gate = evaluate_gate(error)
    assert gate is not None
    error_gate = gate["arm_rate_guardrails"]["errors"]
    assert error_gate["treatment"]["rate_upper_bound"] == pytest.approx(3 / 200)
    assert error_gate["control"]["rate_upper_bound"] == 0
    assert not error_gate["check"]["passed"]


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
