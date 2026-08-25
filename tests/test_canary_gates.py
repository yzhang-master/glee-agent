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
                "games": finite_n,
                "resolved": finite_n,
                "censored": 0,
                "direct_resolved": finite_n,
                "direct_converted": converted_finite,
                "direct_conversion_rate": converted_finite / finite_n,
            },
            "unlimited": {
                "games": unlimited_n,
                "resolved": unlimited_n,
                "censored": 0,
                "direct_resolved": unlimited_n,
                "direct_converted": converted_unlimited,
                "direct_conversion_rate": converted_unlimited / unlimited_n,
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
        "terminal_conflicts": 0,
        "normalized_outcome_sum": total,
        "normalized_outcome_sum_squares": resolved * mean * mean,
        "mean_normalized_outcome": total / n,
        "sample_variance_normalized_outcome": None,
        "zero_sales": censored,
        "zero_sales_rate": censored / n,
    }


def _allocate_count(total, sizes):
    denominator = sum(sizes)
    allocations = [total * size // denominator for size in sizes]
    for index in range(total - sum(allocations)):
        allocations[index % len(allocations)] += 1
    return allocations


def _aggregate_itt_entries(entries):
    entries = list(entries)
    count_fields = (
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
    output = {key: sum(entry[key] for entry in entries) for key in count_fields}
    output["normalized_outcome_sum"] = sum(
        entry["normalized_outcome_sum"] for entry in entries
    )
    output["normalized_outcome_sum_squares"] = sum(
        entry["normalized_outcome_sum_squares"] for entry in entries
    )
    matured = output["matured"]
    output["mean_normalized_outcome"] = (
        output["normalized_outcome_sum"] / matured if matured else None
    )
    output["sample_variance_normalized_outcome"] = None
    output["zero_sales_rate"] = output["zero_sales"] / matured if matured else None
    return output


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
        "population": "all strictly enrolled explicit blind-seller games",
        "maturity_lag_s": 1800,
        "cells": {},
        "p_strata": {},
    }
    weighted_cells = list(PERS_CELL_WEIGHTS.items())
    cell_sizes = [round(weight * 1762) * scale for _, weight in weighted_cells]
    censored_by_cell = _allocate_count(censored, cell_sizes)
    total_zero_sales = zero_sales if not censor_fraction else censored
    zero_by_cell = _allocate_count(total_zero_sales, cell_sizes)
    for index, (((p, message, price, rounds), _), count, cell_censored, cell_zero) in enumerate(
        zip(
            weighted_cells,
            cell_sizes,
            censored_by_cell,
            zero_by_cell,
            strict=True,
        )
    ):
        entry = _itt_leaf(count, revenue, censored=cell_censored)
        entry["zero_sales"] = cell_zero
        entry["zero_sales_rate"] = cell_zero / count
        metric["cells"][str(index)] = {
            "cell": {
                "p": 1 / 3 if p == "0.333333" else float(p),
                "message_type": message,
                "price": price,
                "total_rounds": rounds,
            },
            **entry,
        }
    for p in PERS_P_WEIGHTS:
        p_entries = [
            entry
            for entry in metric["cells"].values()
            if ("0.333333" if entry["cell"]["p"] == 1 / 3 else str(entry["cell"]["p"]))
            == p
        ]
        metric["p_strata"][p] = _aggregate_itt_entries(p_entries)
    metric.update(_aggregate_itt_entries(metric["cells"].values()))
    assert metric["games"] == n
    return metric


def _rebuild_pers_itt_from_cells(metric):
    metric["p_strata"] = {}
    for p in PERS_P_WEIGHTS:
        p_entries = [
            entry
            for entry in metric["cells"].values()
            if ("0.333333" if entry["cell"]["p"] == 1 / 3 else str(entry["cell"]["p"]))
            == p
        ]
        metric["p_strata"][p] = _aggregate_itt_entries(p_entries)
    metric.update(_aggregate_itt_entries(metric["cells"].values()))


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
    assert gate["statistics"]["payoff_checks"]["itt_raw_lower_bound"]["passed"]
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
    treatment["direct_resolved"] = 100
    treatment["direct_converted"] = round(100 * rate)
    converted_left = treatment["direct_converted"]
    for index, horizon in enumerate(("finite", "unlimited")):
        entry = treatment["horizon_strata"][horizon]
        trials = 50
        converted = converted_left if index else converted_left // 2
        if index == 0:
            converted_left -= converted
        entry["direct_resolved"] = trials
        entry["direct_converted"] = converted
        entry["direct_conversion_rate"] = converted / trials
    # The scheduled rollback must use the sufficient statistics, not this
    # mutable precomputed convenience field.
    treatment["direct_conversion_rate"] = 0.99 if rate <= 0.08 else rate
    treatment["mean_normalized_payoff"] = payoff
    if payoff <= 0.27:
        itt_treatment = experiment["itt"]["treatment"]
        matured = itt_treatment["matured"]
        itt_treatment["mean_normalized_outcome"] = payoff
        itt_treatment["normalized_outcome_sum"] = matured * payoff
        itt_treatment["normalized_outcome_sum_squares"] = matured * payoff * payoff
        for cell in itt_treatment["cells"].values():
            cell_n = cell["matured"]
            cell["mean_normalized_outcome"] = payoff
            cell["normalized_outcome_sum"] = cell_n * payoff
            cell["normalized_outcome_sum_squares"] = cell_n * payoff * payoff

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert trigger in gate["rollback_triggers"]
    if rate <= 0.08:
        # The count-derived harm signal is retained, but the contradictory
        # reported convenience rate makes it nonbinding.
        assert gate["binding_rollback"] is False
        assert gate["decision"] == "continue"
    else:
        assert gate["binding_rollback"] is True
        assert gate["decision"] == "rollback"


def test_bargaining_spoofed_rates_and_means_cannot_trigger_rollback():
    experiment = _barg_experiment()
    treatment = experiment["metrics"]["treatment"]
    treatment["direct_conversion_rate"] = 0.0
    experiment["itt"]["treatment"]["mean_normalized_outcome"] = 0.0

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert "scheduled affected direct conversion <= 0.08" not in gate["rollback_triggers"]
    assert "scheduled treatment deadline payoff <= 0.27" not in gate["rollback_triggers"]
    assert not gate["data_integrity"]["passed"]
    assert "direct conversion rate inconsistent with counts" in gate["data_integrity"][
        "failures"
    ]["treatment_affected"]


@pytest.mark.parametrize(
    "mutation",
    ["top_count_bound", "horizon_bound", "horizon_additivity"],
)
def test_bargaining_direct_conversion_integrity_is_fail_closed(mutation):
    experiment = _barg_experiment()
    treatment = experiment["metrics"]["treatment"]
    if mutation == "top_count_bound":
        treatment["direct_converted"] = treatment["direct_resolved"] + 1
    elif mutation == "horizon_bound":
        finite = treatment["horizon_strata"]["finite"]
        finite["direct_converted"] = finite["direct_resolved"] + 1
        finite["direct_conversion_rate"] = finite["direct_converted"] / finite[
            "direct_resolved"
        ]
    else:
        treatment["horizon_strata"]["finite"]["direct_resolved"] -= 1

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert gate["decision"] != "screen_pass"
    assert not gate["data_integrity"]["passed"]


def test_bargaining_raw_itt_lower_bound_is_binding():
    experiment = _barg_experiment()
    for arm, mean in (("treatment", 0.501), ("control", 0.5)):
        metric = experiment["itt"][arm]
        for entry in metric["cells"].values():
            n = entry["matured"]
            entry["normalized_outcome_sum"] = n * mean
            entry["normalized_outcome_sum_squares"] = n * mean
            entry["mean_normalized_outcome"] = mean
        metric["normalized_outcome_sum"] = sum(
            entry["normalized_outcome_sum"] for entry in metric["cells"].values()
        )
        metric["normalized_outcome_sum_squares"] = sum(
            entry["normalized_outcome_sum_squares"]
            for entry in metric["cells"].values()
        )
        metric["mean_normalized_outcome"] = mean

    gate = evaluate_gate(experiment)

    assert gate is not None
    checks = gate["statistics"]["payoff_checks"]
    assert checks["itt_raw_point_nonnegative"]["passed"]
    assert not checks["itt_raw_lower_bound"]["passed"]
    assert gate["decision"] == "continue"


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


@pytest.mark.parametrize("factory", [_barg_experiment, _pers_experiment])
def test_legacy_injected_randomization_booleans_can_never_promote(factory):
    experiment = factory()
    experiment["randomization_evidence"] = {
        "manifest_backed": True,
        "approved_manifest_identity": True,
        "within_game_randomization": True,
        "mirrored_simultaneous_periods": True,
        "strictly_post_v2_checkpoint": True,
        "agent_blocks": {
            label: {
                "treatment_blocks": 100,
                "control_blocks": 100,
                "treatment_n": 10_000,
                "control_n": 10_000,
            }
            for label in ("main", "test_a", "test_b", "test_c")
        },
    }

    gate = evaluate_gate(experiment)

    assert gate is not None
    assert gate["screen_ready"] is True
    assert gate["promotion_ready"] is False
    assert gate["decision"] == "screen_pass"
    causal = gate["causal_confirmation"]
    assert causal["pass"] is False
    assert causal["legacy_randomization_evidence_ignored"] is True
    assert not causal["design_checks"]["structured_reporter_contract"]
    assert not causal["design_checks"]["immutable_prefix_identity"]


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
    assert zero["raw_check"]["passed"]
    assert zero["fixed_weight_check"]["passed"]
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


@pytest.mark.parametrize(
    ("field", "delta"),
    [
        ("games", 1),
        ("matured", 1),
        ("resolved", 1),
        ("censored", 1),
        ("invalid_terminals", 1),
        ("zero_sales", 1),
        ("normalized_outcome_sum", 0.1),
        ("normalized_outcome_sum_squares", 0.1),
    ],
)
def test_persuasion_p_strata_must_exactly_link_to_cells(field, delta):
    experiment = _pers_experiment()
    experiment["itt"]["treatment"]["p_strata"]["0.333333"][field] += delta

    gate = evaluate_gate(experiment)

    assert gate is not None
    failures = gate["data_integrity"]["failures"]["treatment_itt"]
    assert any(f"{field} does not match cells" in failure for failure in failures)
    assert gate["decision"] != "screen_pass"


def test_persuasion_zero_sales_must_add_from_cells_to_top():
    experiment = _pers_experiment()
    experiment["itt"]["treatment"]["cells"]["0"]["zero_sales"] += 1

    gate = evaluate_gate(experiment)

    assert gate is not None
    failures = gate["data_integrity"]["failures"]["treatment_itt"]
    assert "ITT cell zero_sales statistics are not additive" in failures
    assert gate["decision"] != "screen_pass"


def test_fixed_weight_zero_sale_guard_blocks_pooled_dilution_counterexample():
    experiment = _pers_experiment()
    for arm, mean in (("treatment", 0.60), ("control", 0.50)):
        metric = experiment["itt"][arm]
        for index, entry in metric["cells"].items():
            cell = entry["cell"]
            n = 10 if index == "0" else 200
            entry.clear()
            entry.update({"cell": cell, **_itt_leaf(n, mean)})
        metric["cells"]["0"]["zero_sales"] = 5 if arm == "treatment" else 0
        metric["cells"]["0"]["zero_sales_rate"] = (
            metric["cells"]["0"]["zero_sales"] / 10
        )
        _rebuild_pers_itt_from_cells(metric)

    gate = evaluate_gate(experiment)

    assert gate is not None
    zero = gate["statistics"]["zero_sale_noninferiority"]
    assert gate["data_integrity"]["passed"]
    assert zero["raw_check"]["passed"]
    assert not zero["fixed_weight_check"]["passed"]
    assert not zero["check"]["passed"]
    assert gate["decision"] == "continue"


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


def test_reporter_arm_health_keeps_crossed_label_errors_in_treatment_arm():
    experiment = _pers_experiment()
    # Static labels would put main's errors in control.  The reporter contract
    # attributes raw occurrences by immutable game assignment instead.
    experiment["agents"]["main"] = _agent(errors=5)
    experiment["agents"]["test_a"] = _agent(errors=0)
    experiment["analysis_as_of_ts"] = 1787691601
    empty = {
        "turn_events": 0,
        "result_events": 0,
        "turn_errors": 0,
        "result_errors": 0,
        "invalid_results": 0,
    }
    experiment["arm_health"] = {
        "schema_version": 1,
        "attribution": "immutable first-game assignment plus raw event occurrences",
        "prospective_activation": 1787691600.0,
        "cohorts": {
            "legacy": {"treatment": dict(empty), "control": dict(empty)},
            "prospective": {
                "treatment": {
                    "turn_events": 1000,
                    "result_events": 1000,
                    "turn_errors": 5,
                    "result_errors": 0,
                    "invalid_results": 0,
                },
                "control": {
                    "turn_events": 1000,
                    "result_events": 1000,
                    "turn_errors": 0,
                    "result_errors": 0,
                    "invalid_results": 0,
                },
            },
        },
        "integrity": {
            "unknown_turn_events": 0,
            "unknown_result_events": 0,
            "unassigned_or_missing_after_activation": 0,
        },
        "integrity_pass": True,
    }

    gate = evaluate_gate(experiment)

    assert gate is not None
    errors = gate["arm_rate_guardrails"]["errors"]
    assert errors["attribution"] == "reporter_game_arm:prospective"
    assert errors["treatment"]["failures_lower_bound"] == 5
    assert errors["treatment"]["failures_upper_bound"] == 5
    assert errors["treatment"]["rate_upper_bound"] == pytest.approx(5 / 2000)
    assert errors["control"]["failures_lower_bound"] == 0
    assert errors["control"]["rate_lower_bound"] == 0


def test_malformed_reporter_arm_health_cannot_downgrade_to_static_labels():
    experiment = _pers_experiment()
    experiment["analysis_as_of_ts"] = 1787691601
    experiment["arm_health"] = {
        "schema_version": True,
        "cohorts": {},
        "integrity_pass": True,
    }

    gate = evaluate_gate(experiment)

    assert gate is not None
    errors = gate["arm_rate_guardrails"]["errors"]
    assert errors["attribution"] == "invalid_reporter_arm_health"
    assert not errors["treatment"]["valid"]
    assert not errors["control"]["valid"]
    assert not errors["check"]["passed"]


@pytest.mark.parametrize(
    ("name", "family"),
    (("barg_dis_anchor", "bargaining"), ("pers_blind_lie", "persuasion")),
)
def test_missing_or_malformed_evidence_never_masquerades_as_rollback(name, family):
    gate = evaluate_gate({"name": name, "family": family})

    assert gate is not None
    assert gate["data_integrity"]["passed"] is False
    assert gate["guardrails"]["evidence_complete"] is False
    assert gate["binding_rollback"] is False
    assert gate["decision"] == "continue"


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

    cutoff = first["barg_dis_anchor"]["amendment_provenance"][
        "prospective_confirmatory_cutoff"
    ]
    assert "bargaining" not in cutoff
    assert "persuasion" not in cutoff
    cutoff["prior_rows_status"] = "mutated caller copy"
    third = evaluate_report_gates(report)
    assert (
        third["barg_dis_anchor"]["amendment_provenance"][
            "prospective_confirmatory_cutoff"
        ]["prior_rows_status"]
        == "pilot_or_exploratory_screen_only"
    )
