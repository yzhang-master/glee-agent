"""Focused regression tests for the raw-JSONL canary report."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pytest

import scripts.canary_gates as canary_gates
import scripts.canary_report as canary_report
from scripts.canary_report import (
    EXPERIMENTS,
    LogSlice,
    NEG_TERMINAL_GATE_DESIGN,
    Experiment,
    _neg_gate_unsupported_reason,
    _neg_terminal_gate_from_rows,
    build_report,
    discover_log_slices,
    iter_log_records,
    seek_timestamp,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _turn(
    agent,
    gid,
    ts,
    family="bargaining",
    round_=1,
    phase="offer",
    history=None,
    action=None,
    your_player="player_1",
    opponent_type="hidden",
    **state,
):
    game_state = {
        "game_family": family,
        "round": round_,
        "phase": phase,
        "history": [] if history is None else history,
        **state,
    }
    game = {
        "game_id": gid,
        "game_family": family,
        "your_player": your_player,
        "phase": phase,
        "opponent": {"type": opponent_type, "name": None},
        "game_state": game_state,
        "valid_actions": {"type": phase, "fields": {}},
    }
    return {
        "type": "turn",
        "ts": ts,
        "_agent": agent,
        "game": game,
        "action": action or {},
        "corrections": [],
        "error": None,
    }


def _result(
    agent,
    gid,
    ts,
    result=None,
    *,
    valid=True,
    game_over=True,
    error=None,
    result_source=None,
    reaped=None,
):
    record = {
        "type": "result",
        "ts": ts,
        "_agent": agent,
        "agent": agent,
        "game_id": gid,
        "valid": valid,
        "game_over": game_over,
        "error": error,
        "result": result,
    }
    if result_source is not None:
        record["result_source"] = result_source
    if reaped is not None:
        record["reaped"] = reaped
    return record


def _experiment(name="barg_dis_anchor", family="bargaining", cutoff=100):
    return Experiment(
        name,
        family,
        cutoff,
        ("test_b",),
        ("main",),
        {
            "bargaining": "barg_dis_anchor",
            "negotiation": "neg_terminal_close",
            "persuasion": "pers_blind_lie",
        }[family],
        {"bargaining": 0.50, "negotiation": True, "persuasion": 0.40}[family],
        {"bargaining": 0.58, "negotiation": False, "persuasion": 1.0}[family],
    )


def _barg_replay(game, knobs):
    mine = round(100 * knobs.barg_dis_anchor)
    return {"alice_gain": mine, "bob_gain": 100 - mine}


def _gate_row(
    agent,
    variant,
    value,
    *,
    direct,
    normalized_payoff=0.01,
    payoff_percentile=0.70,
    role="buyer",
    phase="decision",
    max_rounds="10",
    opponent_type="hidden",
    complete_information=False,
    direction_violation=False,
    assignment_epoch_id=None,
    first_observed_ts=100.0,
):
    sequence = next(_GATE_ROW_SEQUENCE)
    cell = {
        "role": role,
        "own_value_grid": str(value),
        "phase": phase,
        "horizon": "finite",
        "max_rounds": max_rounds,
        "opponent_type": opponent_type,
        "complete_information": complete_information,
    }
    cell_id = json.dumps(cell, sort_keys=True, separators=(",", ":"))
    supported = (
        role == "buyer"
        and str(value) in {"80", "100", "120", "150"}
        and phase == "decision"
        and max_rounds == "10"
        and opponent_type in {"agent", "hidden"}
        and complete_information is False
    )
    row = {
        "agent": agent,
        "variant": variant,
        "game_id": f"{agent}-{value}-{sequence}",
        "cell": cell,
        "cell_id": cell_id,
        "supported": supported,
        "unsupported_reason": (
            None if supported else _neg_gate_unsupported_reason(cell)
        ),
        "maturity_status": (
            "resolved" if direct is not None else "deadline_censored"
        ),
        "maturity_lag_s": 600,
        "first_observed_ts": first_observed_ts,
        "deadline_ts": first_observed_ts + 600,
        "analysis_ts": first_observed_ts + 700,
        "matured": True,
        "resolved": direct is not None,
        "pending_maturation": False,
        "censored": direct is None,
        "invalid_timely_terminals": 0,
        "late_terminals": 0,
        "malformed_terminal_events": 0,
        "conflicting_timely_terminals": False,
        "terminal_reaped": False,
        "direct": direct,
        "effective_offer_round": 10,
        "normalized_payoff": normalized_payoff if direct is not None else None,
        "payoff_percentile": payoff_percentile if direct is not None else None,
        "compatibility_rate": None,
        "direction_violation": direction_violation,
        "assigned_match": True,
        "assignment_evidence": {
            "prospective": False,
            "valid": True,
            "approved": False,
            "reason": "synthetic_legacy_fixture",
        },
    }
    if assignment_epoch_id is not None:
        row["assignment_epoch_id"] = assignment_epoch_id
        row["assignment_source"] = "synthetic_timestamped_assignment"
    return row


_GATE_ROW_SEQUENCE = itertools.count()


def _promotable_gate_rows(*, switchback=True):
    rows = []
    cells = [
        (opponent_type, value)
        for opponent_type in ("agent", "hidden")
        for value in (80, 100, 120, 150)
    ]
    for cell_index, (opponent_type, value) in enumerate(cells):
        treatment_total = 43 if cell_index < 4 else 42
        first = (treatment_total + 1) // 2
        for agent, n, epoch_id in (
            ("test_a", first, "static:test_a:treatment"),
            ("test_b", treatment_total - first, "runtime:test_b:on"),
        ):
            rows.extend(
                _gate_row(
                    agent,
                    "treatment",
                    value,
                    direct=index < 5,
                    opponent_type=opponent_type,
                    assignment_epoch_id=epoch_id,
                )
                for index in range(n)
            )
        control_n = 128 if cell_index < 4 else 127
        if switchback:
            for agent, epoch_id in (
                ("test_a", "runtime:test_a:off"),
                ("test_b", "static:test_b:control"),
            ):
                rows.extend(
                    _gate_row(
                        agent,
                        "control",
                        value,
                        direct=index < 2,
                        opponent_type=opponent_type,
                        assignment_epoch_id=epoch_id,
                    )
                    for index in range(30)
                )
            remaining = control_n - 60
            rows.extend(
                _gate_row(
                    "main",
                    "control",
                    value,
                    direct=index < 6,
                    opponent_type=opponent_type,
                )
                for index in range(remaining)
            )
        else:
            rows.extend(
                _gate_row(
                    "main",
                    "control",
                    value,
                    direct=index < 10,
                    opponent_type=opponent_type,
                )
                for index in range(control_n)
            )
    return rows


def test_strict_enrollment_dedup_latest_terminal_and_health():
    experiment = _experiment()
    records = [
        # A turn before the cut excludes the game even when round 1 is retried.
        _turn("main", "precut", 99, action={"alice_gain": 58, "bob_gain": 42}, money_to_divide=100),
        _turn(
            "main",
            "precut",
            101,
            action={"alice_gain": 58, "bob_gain": 42},
            money_to_divide=100,
        ),
        # A first sighting with embedded history is partial and excluded.
        _turn(
            "main",
            "partial",
            101,
            round_=2,
            history=[{"round": 1}],
            action={"alice_gain": 58, "bob_gain": 42},
            money_to_divide=100,
        ),
        # Latest duplicate wins for routing and metrics.
        _turn("main", "done", 101, action={"alice_gain": 50, "bob_gain": 50}, money_to_divide=100),
        _turn("main", "done", 102, action={"alice_gain": 58, "bob_gain": 42}, money_to_divide=100),
        _result("main", "done", 103, game_over=False, valid=False, error="HTTP 503"),
        _result(
            "main",
            "done",
            104,
            {"outcome": "agreement", "agreed_round": 1, "player_1_payoff": 20},
        ),
    # A differing later reaper terminal is conflicting causal evidence.
        _result(
            "main",
            "done",
            105,
            {"outcome": "agreement", "agreed_round": 1, "player_1_payoff": 30},
            valid=None,
        ),
        _turn(
            "test_b",
            "open",
            101,
            action={"alice_gain": 50, "bob_gain": 50},
            money_to_divide=100,
        ),
        # This looks clean in the post-cut slice but prefix scanning found it.
        _turn(
            "main",
            "prefix-only",
            106,
            action={"alice_gain": 58, "bob_gain": 42},
            money_to_divide=100,
        ),
    ]
    records[2]["corrections"] = ["first repair", "second repair"]

    report = build_report(
        records,
        preexisting={("main", "prefix-only")},
        experiments=(experiment,),
        replay=_barg_replay,
    )["experiments"][0]

    main = report["agents"]["main"]
    assert main["enrollment"] == {
        "enrolled": 1,
        "resolved": 1,
        "censored": 0,
        "terminal_reaped": 1,
        "excluded_pre_cut": 2,
        "excluded_partial": 1,
    }
    assert report["agents"]["test_b"]["enrollment"]["censored"] == 1
    assert main["health"]["duplicate_turns"] == 2
    assert main["health"]["invalid_results"] == 1
    assert main["health"]["result_errors"] == 1
    assert main["health"]["http_503"] == 1
    assert main["health"]["corrections"] == 2
    assert main["health"]["turns_with_corrections"] == 1
    assert main["routing"]["assigned_matches"] == 1
    assert main["routing"]["affected_wrong_variant"] == 0

    control = report["metrics"]["control"]
    assert control["affected_games"] == 1
    assert control["direct_converted"] == 0
    assert control["direct_resolved"] == 0
    assert control["terminal_conflicts"] == 1
    assert control["mean_normalized_payoff"] is None
    assert control["normalized_payoff_sum"] == 0
    assert control["normalized_payoff_sum_squares"] == 0
    assert control["sample_variance_normalized_payoff"] is None
    assert report["metrics"]["treatment"]["resolved"] == 0


def test_negotiation_counter_uses_next_effective_offer_round():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("test_a",),
        ("main",),
        "neg_terminal_close",
        True,
        False,
    )

    def replay(game, knobs):
        state = game["game_state"]
        if state["round"] == 1:
            return {"product_price": 140}
        price = 102 if knobs.neg_terminal_close else 110
        return {"decision": "RejectOffer", "product_price": price}

    records = []
    for agent, counter in (("test_a", 102), ("main", 110)):
        records.extend(
            [
                _turn(
                    agent,
                    f"g-{agent}",
                    101,
                    family="negotiation",
                    action={"product_price": 140},
                    max_rounds=10,
                    horizon_known=True,
                    player_1_role="seller",
                    player_1_value=100,
                ),
                _turn(
                    agent,
                    f"g-{agent}",
                    102,
                    family="negotiation",
                    round_=9,
                    phase="decision",
                    history=[{"round": 1}],
                    action={"decision": "RejectOffer", "product_price": counter},
                    max_rounds=10,
                    horizon_known=True,
                    player_1_role="seller",
                    player_1_value=100,
                ),
                _result(
                    agent,
                    f"g-{agent}",
                    103,
                    {
                        "outcome": "agreement",
                        "agreed_round": "10.0",
                        "player_1_payoff": 2 if agent == "test_a" else 10,
                    },
                ),
            ]
        )

    report = build_report(records, experiments=(experiment,), replay=replay)["experiments"][0]
    assert len(report["affected_turns"]) == 2
    assert {item["effective_offer_round"] for item in report["affected_turns"]} == {10}
    assert not any(item["direction_violation"] for item in report["affected_turns"])
    assert report["metrics"]["treatment"]["direct_conversion_rate"] == 1
    assert report["metrics"]["control"]["direct_conversion_rate"] == 1
    assert report["metrics"]["treatment"]["max_rounds_strata"]["10"]["resolved"] == 1
    assert next(iter(report["metrics"]["treatment"]["cells"].values()))["cell"] == {
        "complete_information": False,
        "horizon": "finite",
        "max_rounds": "10",
        "opponent_type": "hidden",
        "phase": "decision",
        "role": "seller",
    }
    assert report["gate"]["design"]["pilot_checkpoint"] == {
        "treatment": {"direct_converted": 0, "direct_resolved": 2},
        "control": {"direct_converted": 1, "direct_resolved": 6},
        "used_to_tune_thresholds": False,
        "note": (
            "Pre-gate pilot was T 0/2 versus C 1/6; later outcomes were not "
            "used to set gates."
        ),
            "analysis_window": (
                "The report retains prior rows as pilot/screen evidence; only a "
                "future pinned manifest-backed assignment can be confirmatory."
            ),
    }
    assert report["gate"]["counts"]["unsupported"]["reasons"] == {
        "role=seller": 2
    }
    assert report["gate"]["standardized"]["direct"][
        "reference_weight_coverage"
    ] == 0


def test_negotiation_gate_extracts_scaled_buyer_joint_cell_without_renormalizing():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("test_a",),
        ("main",),
        "neg_terminal_close",
        True,
        False,
    )

    def replay(game, knobs):
        if game["game_state"]["round"] == 1:
            return {"product_price": 6000}
        return {
            "decision": "RejectOffer",
            "product_price": 9000 if knobs.neg_terminal_close else 8500,
        }

    records = []
    for agent, counter in (("test_a", 9000), ("main", 8500)):
        gid = f"buyer-{agent}"
        common = {
            "family": "negotiation",
            "your_player": "player_2",
            "max_rounds": 10,
            "horizon_known": True,
            "complete_information": False,
            "player_1_role": "seller",
            "player_2_role": "buyer",
            "player_2_value": 10000,
        }
        records.extend(
            [
                _turn(agent, gid, 101, action={"product_price": 6000}, **common),
                _turn(
                    agent,
                    gid,
                    102,
                    round_=9,
                    phase="decision",
                    history=[{"round": 1}],
                    action={"decision": "RejectOffer", "product_price": counter},
                    **common,
                ),
                _result(
                    agent,
                    gid,
                    103,
                    {
                        "outcome": "agreement",
                        "agreed_round": 10,
                        "player_2_payoff": 1000,
                    },
                ),
            ]
        )

    report = build_report(records, experiments=(experiment,), replay=replay)[
        "experiments"
    ][0]
    gate = report["gate"]
    value_100 = next(
        cell
        for cell in gate["counts"]["cells"].values()
        if cell["cell"]["own_value_grid"] == "100"
        and cell["cell"]["opponent_type"] == "hidden"
    )
    assert value_100["treatment"]["direct_trials"] == 1
    assert value_100["control"]["direct_trials"] == 1
    assert value_100["weight"] == pytest.approx(207 / 1382)
    assert gate["standardized"]["direct"][
        "reference_weight_coverage"
    ] == pytest.approx(207 / 1382)
    assert gate["standardized"]["direct"]["uplift"] is None
    assert gate["promotion"]["passes"]["complete_fixed_support"] is False


def test_legacy_negotiation_evidence_is_capped_at_screen_pass():
    rows = _promotable_gate_rows()

    gate = _neg_terminal_gate_from_rows(rows)

    assert sum(
        cell["weight"] for cell in gate["design"]["reference_cells"]
    ) == pytest.approx(1)
    assert gate["design"]["estimand"]["role_weight"] == {"buyer": 1.0}
    assert gate["counts"]["variants"]["treatment"]["primary"][
        "direct_trials"
    ] == 340
    assert gate["counts"]["variants"]["control"]["primary"][
        "direct_trials"
    ] == 1020
    assert all(
        cell["treatment"]["direct_trials"] >= 42
        and cell["control"]["direct_trials"] >= 127
        for cell in gate["counts"]["cells"].values()
    )
    assert gate["standardized"]["direct"]["uplift"] > 0.10
    assert gate["standardized"]["direct"]["one_sided_95_lower"] > 0
    assert gate["agent_confirmation"]["confirmed"] == 2
    assert gate["promotion"]["status"] == "screen_pass"
    assert gate["promotion"]["passes"]["approved_prospective_manifest_assignment"] is False
    assert gate["prospective_confirmation"]["pass"] is False


def test_fixed_label_evidence_is_capped_at_screen_pass_without_switchback():
    gate = _neg_terminal_gate_from_rows(
        _promotable_gate_rows(switchback=False)
    )

    assert gate["agent_confirmation"]["pass"] is True
    assert gate["switchback_confirmation"]["pass"] is False
    assert gate["promotion"]["passes"]["balanced_manifest_switchback"] is False
    assert gate["promotion"]["status"] == "screen_pass"
    assert gate["promotion"]["failed_checks"] == [
        "balanced_manifest_switchback",
        "approved_prospective_manifest_assignment",
    ]


def test_second_treatment_epoch_cannot_confirm_on_four_games():
    rows = []
    cells = [
        (opponent_type, value)
        for opponent_type in ("agent", "hidden")
        for value in (80, 100, 120, 150)
    ]
    for cell_index, (opponent_type, value) in enumerate(cells):
        rows.extend(
            _gate_row(
                "test_a",
                "treatment",
                value,
                direct=index < 10,
                opponent_type=opponent_type,
                assignment_epoch_id="static:test_a",
            )
            for index in range(42)
        )
        if cell_index < 4:
            rows.append(
                _gate_row(
                    "test_b",
                    "treatment",
                    value,
                    direct=True,
                    opponent_type=opponent_type,
                    assignment_epoch_id="runtime:test_b:200",
                )
            )
        control_n = 128 if cell_index < 4 else 127
        rows.extend(
            _gate_row(
                "main",
                "control",
                value,
                direct=index < 10,
                opponent_type=opponent_type,
            )
            for index in range(control_n)
        )

    gate = _neg_terminal_gate_from_rows(rows)
    blocks = {
        block["assignment_epoch_id"]: block
        for block in gate["agent_confirmation"]["blocks"]
    }

    assert gate["counts"]["variants"]["treatment"]["primary"][
        "direct_trials"
    ] == 340
    assert blocks["static:test_a"]["sample_pass"] is True
    assert blocks["runtime:test_b:200"]["direct_trials"] == 4
    assert blocks["runtime:test_b:200"]["sample_pass"] is False
    assert gate["agent_confirmation"]["confirmed"] == 1
    assert gate["agent_confirmation"]["pass"] is False
    assert gate["promotion"]["status"] == "screen_pass"
    # The epoch check still fails, but the pinned amendment retired it to a
    # diagnostic, so it no longer appears in the promotion conjunction.
    assert gate["agent_confirmation"]["formal_promotion_gate"] is False
    assert "two_supported_nonnegative_treatment_epochs" not in gate["promotion"]["passes"]
    assert (
        "two_supported_nonnegative_treatment_epochs"
        not in gate["promotion"]["failed_checks"]
    )


def test_payoff_target_artifact_drift_blocks_promotion():
    gate = _neg_terminal_gate_from_rows(
        _promotable_gate_rows(),
        target_artifact_identity={
            "path": "data/targets.json",
            "sha256": "0" * 64,
            "bytes": 642520,
            "available": True,
        },
    )

    assert gate["payoff_target_integrity"]["pass"] is False
    assert gate["promotion"]["passes"][
        "payoff_target_artifact_matches_cutoff"
    ] is False
    assert gate["promotion"]["status"] == "continue"


def test_unsupported_policy_slice_harm_blocks_otherwise_promotable_gate():
    rows = _promotable_gate_rows()
    rows.extend(
        _gate_row("test_a", "treatment", 100, direct=False, role="seller")
        for _ in range(1000)
    )
    rows.extend(
        _gate_row("main", "control", 100, direct=True, role="seller")
        for _ in range(1000)
    )

    gate = _neg_terminal_gate_from_rows(rows)

    assert gate["counts"]["unsupported"]["total"] == 2000
    assert gate["unsupported_safety"]["present"] is True
    assert gate["unsupported_safety"]["harm_fail"] is True
    assert gate["promotion"]["passes"][
        "unsupported_policy_slices_noninferior"
    ] is False
    assert gate["promotion"]["status"] == "rollback"


def test_runtime_assignment_does_not_reclassify_earlier_control_game():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("test_a",),
        ("main", "test_b"),
        "neg_terminal_close",
        True,
        False,
    )

    def replay(game, knobs):
        if game["game_state"]["round"] == 1:
            return {"product_price": 6000}
        return {
            "decision": "RejectOffer",
            "product_price": 9000 if knobs.neg_terminal_close else 8500,
        }

    common = {
        "family": "negotiation",
        "your_player": "player_2",
        "max_rounds": 10,
        "horizon_known": True,
        "complete_information": False,
        "player_1_role": "seller",
        "player_2_role": "buyer",
        "player_2_value": 10000,
    }
    records = [
        _turn("test_b", "before", 110, action={"product_price": 6000}, **common),
        _turn(
            "test_b",
            "before",
            120,
            round_=9,
            phase="decision",
            history=[{"round": 1}],
            action={"decision": "RejectOffer", "product_price": 8500},
            **common,
        ),
        _result(
            "test_b",
            "before",
            125,
            {"outcome": "no_deal", "agreed_round": None, "player_2_payoff": 0},
        ),
        {
            "type": "runtime",
            "ts": 150,
            "_agent": "test_b",
            "agent": "test_b",
            "pid": 77,
            "knobs": {"neg_terminal_close": True},
            "git_head": "a" * 40,
            "content_hashes": {
                "strategy_python": {"aggregate_sha256": "b" * 64}
            },
        },
        _turn("test_b", "after", 160, action={"product_price": 6000}, **common),
        _turn(
            "test_b",
            "after",
            170,
            round_=9,
            phase="decision",
            history=[{"round": 1}],
            action={"decision": "RejectOffer", "product_price": 9000},
            **common,
        ),
        _result(
            "test_b",
            "after",
            175,
            {"outcome": "no_deal", "agreed_round": None, "player_2_payoff": 0},
        ),
    ]

    report = build_report(records, experiments=(experiment,), replay=replay)[
        "experiments"
    ][0]

    assert [item["variant"] for item in report["affected_turns"]] == [
        "control",
        "treatment",
    ]
    assert report["metrics"]["control"]["direct_resolved"] == 1
    assert report["metrics"]["treatment"]["direct_resolved"] == 1
    blocks = report["gate"]["agent_confirmation"]["blocks"]
    assert len(blocks) == 1
    assert blocks[0]["assignment_epoch_id"].startswith("runtime:test_b:150.000000")
    assert blocks[0]["sample_pass"] is False


def test_frozen_negotiation_gate_interim_zero_conversion_and_health_rollback():
    rows = []
    treatment_sizes = (13, 13, 12, 12)
    control_sizes = (38, 38, 37, 37)
    for value, treatment_n, control_n in zip(
        (80, 100, 120, 150), treatment_sizes, control_sizes, strict=True
    ):
        rows.extend(
            _gate_row("test_a", "treatment", value, direct=False)
            for _ in range(treatment_n)
        )
        rows.extend(
            _gate_row("main", "control", value, direct=False)
            for _ in range(control_n)
        )
    rows[0]["direction_violation"] = True

    gate = _neg_terminal_gate_from_rows(rows)

    assert gate["interim"]["stage"] == "interim_1"
    assert gate["interim"]["conditional_power_binding"] is False
    assert gate["interim"]["rollback"] is True
    assert "T>=50/C>=150 with zero treatment conversions" in gate["interim"][
        "reasons"
    ]
    assert gate["health"]["hard_fail"] is True
    assert gate["promotion"]["status"] == "rollback"


def test_amended_gate_discloses_inspected_live_outcomes_and_pre_cut_maturation():
    assert NEG_TERMINAL_GATE_DESIGN["frozen_before_subsequent_outcomes"] is False
    provenance = NEG_TERMINAL_GATE_DESIGN["amendment_provenance"]
    assert provenance["live_outcomes_inspected_before_amendment"] is True
    assert provenance["pre_amendment_observed"] == {
        "treatment": {"unresolved": 1, "affected": 11},
        "control": {"unresolved": 2, "affected": 33},
        "raw_treatment_minus_control_unresolved_rate": 0.0303,
    }
    assert NEG_TERMINAL_GATE_DESIGN["pilot_checkpoint"]["treatment"] == {
        "direct_converted": 0,
        "direct_resolved": 2,
    }
    assert NEG_TERMINAL_GATE_DESIGN["pilot_checkpoint"]["control"] == {
        "direct_converted": 1,
        "direct_resolved": 6,
    }
    assert (
        NEG_TERMINAL_GATE_DESIGN["pilot_checkpoint"]["used_to_tune_thresholds"]
        is False
    )
    assert sum(
        cell["historical_resolved"]
        for cell in NEG_TERMINAL_GATE_DESIGN["reference_cells"]
    ) == 1382
    assert sum(
        cell["historical_direct_converted"]
        for cell in NEG_TERMINAL_GATE_DESIGN["reference_cells"]
    ) == 103
    assert {
        cell["opponent_type"]
        for cell in NEG_TERMINAL_GATE_DESIGN["reference_cells"]
    } == {"agent", "hidden"}
    assert NEG_TERMINAL_GATE_DESIGN["payoff_target_artifact"]["sha256"] == (
        "1d24a579ca2b611e3b30af4ddf7af5b84ad13e7198fa55b93a2f5e6617e65e25"
    )
    maturation = NEG_TERMINAL_GATE_DESIGN["historical_reference"][
        "maturation_derivation"
    ]
    assert maturation["eligible"] == 1418
    assert maturation["terminals"] == 1382
    assert maturation["no_terminal"] == 36
    assert maturation["frozen_lag_seconds"] == 600


def test_bargaining_accept_instead_of_reject_is_in_direction():
    experiment = _experiment()

    def replay(_game, knobs):
        return {
            "decision": "accept" if knobs.barg_dis_anchor == 0.50 else "reject"
        }

    records = [
        _turn(
            "test_b",
            "accept-fast",
            101,
            phase="decision",
            action={"decision": "accept"},
            money_to_divide=100,
        )
    ]
    report = build_report(records, experiments=(experiment,), replay=replay)["experiments"][0]
    assert len(report["affected_turns"]) == 1
    assert report["affected_turns"][0]["direction_violation"] is False


def test_blind_persuasion_revenue_p_strata_and_deterministic_routing():
    experiment = Experiment(
        "pers_blind_lie",
        "persuasion",
        100,
        ("test_b",),
        ("main",),
        "pers_blind_lie",
        0.40,
        1.0,
    )

    def replay(_game, knobs):
        return {"decision": "yes" if knobs.pers_blind_lie == 1.0 else "no"}

    records = []
    for agent, action, payoff in (("test_b", "no", 100), ("main", "yes", 0)):
        gid = f"pers-{agent}"
        records.extend(
            [
                _turn(
                    agent,
                    gid,
                    101,
                    family="persuasion",
                    phase="seller_recommendation",
                    action={"decision": action},
                    total_rounds=2,
                    product_price=100,
                    p=0.5,
                    is_seller_know_cv=False,
                    current_quality="low",
                    player_1_role="seller",
                ),
                _result(
                    agent,
                    gid,
                    102,
                    {"outcome": "completed", "player_1_payoff": payoff},
                ),
            ]
        )

    # Missing v is not enough to call a current-format payload blind when its
    # authoritative flag explicitly says the seller knows the value.
    records.extend(
        [
            _turn(
                "main",
                "visible-contract",
                101,
                family="persuasion",
                phase="seller_recommendation",
                action={"decision": "yes"},
                total_rounds=2,
                product_price=100,
                p=0.5,
                is_seller_know_cv=True,
                current_quality="low",
                player_1_role="seller",
            ),
            _result(
                "main",
                "visible-contract",
                102,
                {"outcome": "completed", "player_1_payoff": 200},
            ),
        ]
    )

    report = build_report(records, experiments=(experiment,), replay=replay)["experiments"][0]
    treatment = report["metrics"]["treatment"]
    control = report["metrics"]["control"]
    assert treatment["mean_revenue_share"] == pytest.approx(0.5)
    assert treatment["revenue_share_sum"] == pytest.approx(0.5)
    assert treatment["revenue_share_sum_squares"] == pytest.approx(0.25)
    assert treatment["zero_sales_rate"] == 0
    assert control["mean_revenue_share"] == 0
    assert control["zero_sales_rate"] == 1
    assert treatment["p_strata"]["0.5"]["resolved"] == 1
    assert next(iter(treatment["cells"].values()))["cell"] == {
        "message_type": "unknown",
        "opponent_type": "hidden",
        "p": 0.5,
        "price": 100.0,
        "start_block_15m": 0,
        "total_rounds": 2,
    }
    assert treatment["deterministic_route_matches"] == 1
    assert control["deterministic_route_matches"] == 1
    assert report["agents"]["main"]["routing"]["affected_assigned_matches"] == 2
    assert not any(item["direction_violation"] for item in report["affected_turns"])


def test_bargaining_itt_includes_unaffected_games_and_separates_invalid_terminals():
    experiment = _experiment()

    def replay(game, knobs):
        state = game["game_state"]
        if state.get("same_policy"):
            return {"alice_gain": 50, "bob_gain": 50}
        return _barg_replay(game, knobs)

    records = [
        _turn(
            "test_b",
            "unaffected-valid",
            101,
            action={"alice_gain": 50, "bob_gain": 50},
            money_to_divide=100,
            max_rounds=6,
            horizon_known=True,
            same_policy=True,
        ),
            _result(
                "test_b",
                "unaffected-valid",
                102,
                {
                    "outcome": "agreement",
                    "agreed_round": 1,
                    "player_1_payoff": 40,
                },
        ),
        _turn(
            "test_b",
            "invalid-terminal",
            103,
            action={"alice_gain": 50, "bob_gain": 50},
            money_to_divide=100,
            max_rounds=6,
            horizon_known=True,
            same_policy=True,
        ),
        _result(
            "test_b",
            "invalid-terminal",
            104,
            {"outcome": "agreement", "player_1_payoff": 90},
            valid="false",
        ),
        # The deterministic report clock advances beyond both 1200-second
        # maturity deadlines without touching either game or live state.
        {
            "type": "runtime",
            "ts": 1400,
            "_agent": "observer",
            "agent": "observer",
            "pid": 1,
            "knobs": {},
            "git_head": "a" * 40,
            "content_hashes": {
                "strategy_python": {"aggregate_sha256": "b" * 64}
            },
        },
    ]

    full_report = build_report(
        records,
        experiments=(experiment,),
        replay=replay,
    )
    report = full_report["experiments"][0]
    treatment = report["itt"]["treatment"]

    assert report["metrics"]["treatment"]["affected_games"] == 0
    assert treatment["games"] == 2
    assert treatment["matured"] == 2
    assert treatment["resolved"] == 1
    assert treatment["invalid_terminals"] == 1
    assert treatment["censored"] == 0
    assert treatment["deadline_zeroes"] == 1
    assert treatment["normalized_outcome_sum"] == pytest.approx(0.4)
    assert treatment["normalized_outcome_sum_squares"] == pytest.approx(0.16)
    assert treatment["mean_normalized_outcome"] == pytest.approx(0.2)
    assert report["analysis_as_of_ts"] == 1400
    assert full_report["gates"][experiment.name]["data_integrity"]["passed"] is False


def test_duplicate_turn_cannot_move_itt_deadline_or_erase_raw_health_error():
    experiment = _experiment()
    first = _turn(
        "test_b",
        "duplicate-deadline",
        101,
        action={"alice_gain": 50, "bob_gain": 50},
        money_to_divide=100,
    )
    first["error"] = "first occurrence failed"
    duplicate = _turn(
        "test_b",
        "duplicate-deadline",
        1001,
        action={"alice_gain": 50, "bob_gain": 50},
        money_to_divide=100,
    )
    records = [
        first,
        duplicate,
        _result(
            "test_b",
            "duplicate-deadline",
            1302,
            {"outcome": "agreement", "player_1_payoff": 90},
        ),
        {
            "type": "runtime",
            "ts": 2300,
            "_agent": "observer",
            "agent": "observer",
            "pid": 1,
            "knobs": {},
            "git_head": "a" * 40,
            "content_hashes": {
                "strategy_python": {"aggregate_sha256": "b" * 64}
            },
        },
    ]

    report = build_report(
        records,
        experiments=(experiment,),
        replay=lambda _game, _knobs: {"alice_gain": 50, "bob_gain": 50},
    )["experiments"][0]

    assert report["itt"]["treatment"]["matured"] == 1
    assert report["itt"]["treatment"]["resolved"] == 0
    assert report["itt"]["treatment"]["deadline_censored"] == 1
    assert report["itt"]["treatment"]["normalized_outcome_sum"] == 0
    assert report["agents"]["test_b"]["health"]["turns"] == 2
    assert report["agents"]["test_b"]["health"]["turn_errors"] == 1
    assert report["agents"]["test_b"]["health"]["duplicate_turns"] == 1


def test_malformed_terminal_fields_cannot_become_favorable_itt():
    experiment = _experiment()
    malformed = _result(
        "test_b",
        "malformed-terminal",
        102,
        {"outcome": "garbage", "player_1_payoff": "90"},
    )
    malformed["game_over"] = "false"
    records = [
        _turn(
            "test_b",
            "malformed-terminal",
            101,
            action={"alice_gain": 50, "bob_gain": 50},
            money_to_divide=100,
        ),
        malformed,
        {
            "type": "runtime",
            "ts": 1400,
            "_agent": "observer",
            "agent": "observer",
            "pid": 1,
            "knobs": {},
            "git_head": "a" * 40,
            "content_hashes": {
                "strategy_python": {"aggregate_sha256": "b" * 64}
            },
        },
    ]

    report = build_report(
        records,
        experiments=(experiment,),
        replay=lambda _game, _knobs: {"alice_gain": 50, "bob_gain": 50},
    )["experiments"][0]

    assert report["itt"]["treatment"]["resolved"] == 0
    assert report["itt"]["treatment"]["deadline_censored"] == 1
    assert report["itt"]["treatment"]["normalized_outcome_sum"] == 0


def test_conflicting_timely_itt_terminals_are_not_silently_selected():
    experiment = _experiment()
    records = [
        _turn(
            "test_b",
            "conflicting-itt-terminal",
            101,
            action={"alice_gain": 50, "bob_gain": 50},
            money_to_divide=100,
        ),
            _result(
                "test_b",
                "conflicting-itt-terminal",
                102,
                {
                    "outcome": "agreement",
                    "agreed_round": 1,
                    "player_1_payoff": 40,
                },
        ),
            _result(
                "test_b",
                "conflicting-itt-terminal",
                103,
                {
                    "outcome": "agreement",
                    "agreed_round": 1,
                    "player_1_payoff": 90,
                },
        ),
        {
            "type": "runtime",
            "ts": 1400,
            "_agent": "observer",
            "agent": "observer",
            "pid": 1,
            "knobs": {},
            "git_head": "a" * 40,
            "content_hashes": {
                "strategy_python": {"aggregate_sha256": "b" * 64}
            },
        },
    ]

    full_report = build_report(
        records,
        experiments=(experiment,),
        replay=lambda _game, _knobs: {"alice_gain": 50, "bob_gain": 50},
    )
    treatment = full_report["experiments"][0]["itt"]["treatment"]

    assert treatment["terminal_conflicts"] == 1
    assert treatment["resolved"] == 0
    assert treatment["invalid_terminals"] == 1
    assert treatment["normalized_outcome_sum"] == 0
    assert full_report["gates"][experiment.name]["data_integrity"]["passed"] is False


def test_root_gates_are_additive_and_do_not_replace_negotiation_gate():
    negotiation = _experiment(
        name="neg_terminal_close", family="negotiation", cutoff=100
    )
    bargaining = _experiment()

    report = build_report(
        [],
        experiments=(negotiation, bargaining),
        replay=lambda _game, _knobs: {},
    )
    by_name = {entry["name"]: entry for entry in report["experiments"]}

    assert "gate" in by_name["neg_terminal_close"]
    assert "gate" not in by_name["barg_dis_anchor"]
    assert set(report["gates"]) == {"barg_dis_anchor"}
    assert report["gates"]["barg_dis_anchor"]["rule_version"].endswith(
        "amended-v2"
    )


def test_malformed_negotiation_runtime_assignment_fails_closed_without_crashing():
    experiment = _experiment(
        name="neg_terminal_close", family="negotiation", cutoff=100
    )
    records = [
        {
            "type": "runtime",
            "ts": 101,
            "_agent": "test_b",
            "agent": "test_b",
            "pid": 77,
            "knobs": {"neg_terminal_close": "false"},
            "git_head": "a" * 40,
            "content_hashes": {
                "strategy_python": {"aggregate_sha256": "b" * 64}
            },
        },
        _turn(
            "test_b",
            "unknown-arm",
            102,
            family="negotiation",
            action={"product_price": 100},
            max_rounds=10,
            horizon_known=True,
            player_1_role="seller",
            player_1_value=100,
        ),
    ]

    report = build_report(
        records,
        experiments=(experiment,),
        replay=lambda _game, _knobs: {"product_price": 100},
    )["experiments"][0]

    assert report["agents"]["test_b"]["routing"]["assignment_integrity_errors"] == 1
    assert report["affected_turns"] == []
    assert report["gate"]["health"]["hard_fail"] is False
    assert report["gate"]["promotion"]["status"] == "continue"


def _neg_deadline_records(agent, gid, terminal_records):
    common = {
        "family": "negotiation",
        "your_player": "player_2",
        "max_rounds": 10,
        "horizon_known": True,
        "complete_information": False,
        "player_1_role": "seller",
        "player_2_role": "buyer",
        "player_2_value": 100,
    }
    return [
        _turn(agent, gid, 101, action={"product_price": 60}, **common),
        _turn(
            agent,
            gid,
            102,
            round_=9,
            phase="decision",
            history=[{"round": 1}],
            action={"decision": "RejectOffer", "product_price": 90},
            **common,
        ),
        *terminal_records,
    ]


_PLAN_SHA = "b002b688d02df3233b7dd4f21a5595cf149b4cc8dd501a0bfc2ee5bccd11d745"
_STRATEGY_SHA = "631ef69862d572644ba855174a411f80a220b11ed5c20e30b43ffc31f1303388"
_PLAN_ID = "confirmation-v2-20260825-2100z"
_NEG_RULE_ID = "neg-terminal-confirm-v2"
_PLAN_SALT = "730f45c9167e0c39136c20b30dcbdda3"
_PLAN_ACTIVATION = 1787691600
_PLAN_EXPIRY = 1787950800


def _prospective_neg_records(agent="main", gid="prospective-neg"):
    rules = [
        {
            "family": "bargaining",
            "rule_id": "barg-anchor-confirm-v2",
            "knob": "barg_dis_anchor",
            "control": 0.58,
            "treatment": 0.5,
            "treatment_probability": 0.25,
        },
        {
            "family": "negotiation",
            "rule_id": _NEG_RULE_ID,
            "knob": "neg_terminal_close",
            "control": False,
            "treatment": True,
            "treatment_probability": 0.5,
        },
        {
            "family": "persuasion",
            "rule_id": "pers-blind-confirm-v2",
            "knob": "pers_blind_lie",
            "control": 1.0,
            "treatment": 0.4,
            "treatment_probability": 0.5,
        },
    ]
    contract = {
        "schema_version": 1,
        "plan_id": _PLAN_ID,
        "assignment_salt": _PLAN_SALT,
        "assignment_salt_visibility": "public_replay_seed",
        "activated_at": _PLAN_ACTIVATION,
        "expires_at": _PLAN_EXPIRY,
        "agents": ["main", "test_a", "test_b", "test_c"],
        "assignment_algorithm": "sha256-u64-v1",
        "enrollment": {
            "first_seen_round": 1,
            "requires_empty_history": True,
            "assigned_games_remain_stable_after_expiry": True,
        },
        "rules": rules,
    }
    runtime = {
        "type": "runtime",
        "ts": _PLAN_ACTIVATION,
        "_agent": agent,
        "agent": agent,
        "pid": 77,
        "knobs": {"neg_terminal_close": False},
        "git_head": "a" * 40,
        "canary_assignment": {
            "loader_status": "valid",
            "error_code": None,
            "artifact": {
                "path": "data/canary_assignment.json",
                "available": True,
                "sha256": _PLAN_SHA,
                "bytes": 837,
            },
            "contract": contract,
            "receipt_store": {
                "format": "append-only-jsonl-v1",
                "path": f"logs/canary-assignments/{agent}/{_PLAN_SHA}.jsonl",
                "write_ahead_fsync": True,
                "max_bytes": 64 * 1024 * 1024,
            },
        },
        "content_hashes": {
            "strategy_python": {"aggregate_sha256": _STRATEGY_SHA},
            "targets": {
                "data/targets.json": {
                    "sha256": "1d24a579ca2b611e3b30af4ddf7af5b84ad13e7198fa55b93a2f5e6617e65e25"
                },
                "data/live_targets.json": {
                    "sha256": "3dcaff69f17175648e4b46499859bf183bba03b1321364de329d01bed0e618a3"
                },
            },
        },
    }
    digest = hashlib.sha256(
        b"\0".join(
            value.encode()
            for value in (
                "glee-canary-assignment-v1",
                _PLAN_SALT,
                _PLAN_ID,
                _NEG_RULE_ID,
                agent,
                "negotiation",
                gid,
            )
        )
    ).digest()
    bucket = int.from_bytes(digest[:8], "big")
    threshold = 1 << 63
    arm = "treatment" if bucket < threshold else "control"
    metadata = {
        "status": "assigned",
        "reason": "eligible",
        "artifact_sha256": _PLAN_SHA,
        "plan_id": _PLAN_ID,
        "rule_id": _NEG_RULE_ID,
        "family": "negotiation",
        "knob": "neg_terminal_close",
        "arm": arm,
        "value": arm == "treatment",
        "treatment_probability": 0.5,
        "assignment_algorithm": "sha256-u64-v1",
        "assignment_sha256": digest.hex(),
        "assignment_u64": bucket,
        "treatment_threshold_u64": threshold,
        "enrollment": "new",
        "enrolled_at": _PLAN_ACTIVATION + 5.0,
    }
    common = {
        "family": "negotiation",
        "your_player": "player_2",
        "max_rounds": 10,
        "horizon_known": True,
        "complete_information": False,
        "player_1_role": "seller",
        "player_2_role": "buyer",
        "player_2_value": 100,
    }
    first = _turn(
        agent,
        gid,
        _PLAN_ACTIVATION + 10,
        action={"product_price": 60},
        **common,
    )
    first["canary_assignment"] = dict(metadata)
    counter = 90 if arm == "treatment" else 85
    affected = _turn(
        agent,
        gid,
        _PLAN_ACTIVATION + 20,
        round_=9,
        phase="decision",
        history=[{"round": 1}],
        action={"decision": "RejectOffer", "product_price": counter},
        **common,
    )
    affected["canary_assignment"] = dict(metadata)
    result = _result(
        agent,
        gid,
        _PLAN_ACTIVATION + 100,
        {"outcome": "agreement", "agreed_round": 10, "player_2_payoff": 10},
        result_source="move",
        reaped=False,
    )
    return [runtime, first, affected, result], arm


def _prospective_nonneg_records(family, agent="main", gid="prospective-game"):
    base, _ = _prospective_neg_records(agent=agent, gid=f"seed-{gid}")
    runtime = base[0]
    specifications = {
        "bargaining": {
            "rule_id": "barg-anchor-confirm-v2",
            "knob": "barg_dis_anchor",
            "control": 0.58,
            "treatment": 0.5,
            "probability": 0.25,
        },
        "persuasion": {
            "rule_id": "pers-blind-confirm-v2",
            "knob": "pers_blind_lie",
            "control": 1.0,
            "treatment": 0.4,
            "probability": 0.5,
        },
    }
    spec = specifications[family]
    runtime["knobs"] = {spec["knob"]: spec["control"]}
    digest = hashlib.sha256(
        b"\0".join(
            value.encode()
            for value in (
                "glee-canary-assignment-v1",
                _PLAN_SALT,
                _PLAN_ID,
                spec["rule_id"],
                agent,
                family,
                gid,
            )
        )
    ).digest()
    bucket = int.from_bytes(digest[:8], "big")
    threshold = int(spec["probability"] * (1 << 64))
    arm = "treatment" if bucket < threshold else "control"
    metadata = {
        "status": "assigned",
        "reason": "eligible",
        "artifact_sha256": _PLAN_SHA,
        "plan_id": _PLAN_ID,
        "rule_id": spec["rule_id"],
        "family": family,
        "knob": spec["knob"],
        "arm": arm,
        "value": spec[arm],
        "treatment_probability": spec["probability"],
        "assignment_algorithm": "sha256-u64-v1",
        "assignment_sha256": digest.hex(),
        "assignment_u64": bucket,
        "treatment_threshold_u64": threshold,
        "enrollment": "new",
        "enrolled_at": _PLAN_ACTIVATION + 5.0,
    }
    if family == "bargaining":
        mine = 50 if arm == "treatment" else 58
        first = _turn(
            agent,
            gid,
            _PLAN_ACTIVATION + 10,
            action={"alice_gain": mine, "bob_gain": 100 - mine},
            money_to_divide=100,
            max_rounds=6,
            horizon_known=True,
        )
        result = _result(
            agent,
            gid,
            _PLAN_ACTIVATION + 100,
            {
                "outcome": "agreement",
                "agreed_round": 1,
                "player_1_payoff": mine,
            },
            result_source="move",
            reaped=False,
        )
    else:
        decision = "no" if arm == "treatment" else "yes"
        first = _turn(
            agent,
            gid,
            _PLAN_ACTIVATION + 10,
            family="persuasion",
            phase="seller_recommendation",
            action={"decision": decision},
            total_rounds=2,
            product_price=100,
            p=0.5,
            is_seller_know_cv=False,
            current_quality="low",
            player_1_role="seller",
        )
        result = _result(
            agent,
            gid,
            _PLAN_ACTIVATION + 100,
            {"outcome": "completed", "player_1_payoff": 100},
            result_source="move",
            reaped=False,
        )
    first["canary_assignment"] = dict(metadata)
    return [runtime, first, result], arm


def test_prospective_negotiation_receipt_is_cryptographically_linked_and_separate():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "neg_terminal_close",
        True,
        False,
    )
    records, arm = _prospective_neg_records()

    def replay(game, knobs):
        if game["game_state"]["round"] == 1:
            return {"product_price": 60}
        return {
            "decision": "RejectOffer",
            "product_price": 90 if knobs.neg_terminal_close else 85,
        }

    report = build_report(records, experiments=(experiment,), replay=replay)[
        "experiments"
    ][0]
    gate = report["gate"]

    assert gate["cohort_basis"] == "approved_prospective_manifest_receipts"
    assert gate["data_integrity"]["pass"] is True
    assert gate["counts"]["variants"][arm]["primary"]["resolved"] == 1
    assert gate["prospective_confirmation"]["prospective_rows"] == 1
    assert gate["prospective_confirmation"]["checks"][
        "exact_artifact_plan_rule_cutoff"
    ] is True
    assert gate["prospective_confirmation"]["pass"] is False
    assert report["arm_health"]["cohorts"]["prospective"][arm][
        "result_events"
    ] == 1
    assert sum(
        report["arm_health"]["cohorts"]["legacy"][candidate]["turn_events"]
        for candidate in ("treatment", "control")
    ) == 0


def test_prospective_negotiation_receipt_arm_tampering_blocks_without_rollback():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "neg_terminal_close",
        True,
        False,
    )
    records, _arm = _prospective_neg_records(gid="tampered")
    records[1]["canary_assignment"]["arm"] = (
        "control"
        if records[1]["canary_assignment"]["arm"] == "treatment"
        else "treatment"
    )

    report = build_report(
        records,
        experiments=(experiment,),
        replay=lambda _game, _knobs: {"product_price": 60},
    )["experiments"][0]

    assert report["arm_health"]["integrity_pass"] is False
    assert report["gate"]["promotion"]["status"] != "promote"


def test_latest_runtime_without_manifest_cannot_reuse_stale_valid_manifest():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "neg_terminal_close",
        True,
        False,
    )
    records, _arm = _prospective_neg_records(gid="stale-runtime-manifest")
    restarted = json.loads(json.dumps(records[0]))
    restarted["ts"] = _PLAN_ACTIVATION + 7
    restarted["pid"] = 88
    restarted.pop("canary_assignment")
    records.insert(1, restarted)

    report = build_report(
        records,
        experiments=(experiment,),
        replay=lambda game, _knobs: (
            {"product_price": 60}
            if game["game_state"]["round"] == 1
            else records[3]["action"]
        ),
    )["experiments"][0]

    assert report["arm_health"]["integrity_pass"] is False
    assert report["gate"]["prospective_confirmation"]["approved_rows"] == 0
    assert report["gate"]["promotion"]["status"] == "continue"


def test_prospective_assignment_allows_new_to_recovered_only_across_restart():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "neg_terminal_close",
        True,
        False,
    )
    records, _arm = _prospective_neg_records(gid="restart-stable")
    restarted_runtime = json.loads(json.dumps(records[0]))
    restarted_runtime["ts"] = _PLAN_ACTIVATION + 15
    restarted_runtime["pid"] = 88
    records[2]["canary_assignment"]["enrollment"] = "recovered"
    records.insert(2, restarted_runtime)

    def replay(game, knobs):
        if game["game_state"]["round"] == 1:
            return {"product_price": 60}
        return {
            "decision": "RejectOffer",
            "product_price": 90 if knobs.neg_terminal_close else 85,
        }

    gate = build_report(records, experiments=(experiment,), replay=replay)[
        "experiments"
    ][0]["gate"]

    assert gate["data_integrity"]["pass"] is True
    assert gate["prospective_confirmation"]["prospective_rows"] == 1


def test_nondivergent_game_still_validates_full_receipt_sequence():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "neg_terminal_close",
        True,
        False,
    )
    records, _arm = _prospective_neg_records(gid="bad-nondivergent-sequence")
    records[2]["canary_assignment"]["enrollment"] = "recovered"

    report = build_report(
        records,
        experiments=(experiment,),
        replay=lambda game, _knobs: (
            {"product_price": 60}
            if game["game_state"]["round"] == 1
            else records[2]["action"]
        ),
    )["experiments"][0]

    assert report["affected_turns"] == []
    assert report["arm_health"]["integrity_pass"] is False
    assert report["gate"]["promotion"]["status"] == "continue"


def test_prospective_cohort_selection_does_not_require_an_affected_divergence():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "neg_terminal_close",
        True,
        False,
    )
    prospective, _arm = _prospective_neg_records(gid="nondivergent-prospective")
    legacy = _neg_deadline_records("test_b", "legacy-pilot-row", [])
    legacy[1]["action"]["product_price"] = 85

    def replay(game, knobs):
        if game["game_state"]["round"] == 1:
            return {"product_price": 60}
        if game["game_id"] == "nondivergent-prospective":
            return prospective[2]["action"]
        return {
            "decision": "RejectOffer",
            "product_price": 90 if knobs.neg_terminal_close else 85,
        }

    report = build_report(
        [*legacy, *prospective], experiments=(experiment,), replay=replay
    )["experiments"][0]

    assert any(item["game_id"] == "legacy-pilot-row" for item in report["affected_turns"])
    assert not any(
        item["game_id"] == "nondivergent-prospective"
        for item in report["affected_turns"]
    )
    assert report["gate"]["cohort_basis"] == "approved_prospective_manifest_receipts"
    assert report["gate"]["data_integrity"]["evaluated_rows"] == 0
    assert report["gate"]["promotion"]["status"] == "continue"


def test_post_activation_unassigned_game_is_integrity_only_not_legacy_evidence():
    experiment = _experiment(
        name="neg_terminal_close", family="negotiation", cutoff=100
    )
    turn = _turn(
        "test_b",
        "post-activation-unassigned",
        _PLAN_ACTIVATION + 1,
        family="negotiation",
        action={"product_price": 60},
    )
    turn["canary_assignment"] = {
        "status": "unassigned",
        "reason": "missing_artifact",
    }

    report = build_report(
        [turn],
        experiments=(experiment,),
        replay=lambda _game, _knobs: {"product_price": 60},
    )["experiments"][0]

    assert report["arm_health"]["integrity"]["unknown_turn_events"] == 1
    assert (
        report["arm_health"]["integrity"][
            "unassigned_or_missing_after_activation"
        ]
        == 1
    )
    assert report["arm_health"]["cohorts"]["legacy"]["control"][
        "turn_events"
    ] == 0
    assert report["gate"]["promotion"]["status"] == "continue"


def test_pre_activation_legacy_game_keeps_arm_across_activation_boundary():
    experiment = _experiment(
        name="neg_terminal_close", family="negotiation", cutoff=100
    )
    first = _turn(
        "test_b",
        "legacy-crosses-activation",
        101,
        family="negotiation",
        action={"product_price": 60},
    )
    later = _turn(
        "test_b",
        "legacy-crosses-activation",
        _PLAN_ACTIVATION + 1,
        family="negotiation",
        round_=2,
        history=[{"round": 1}],
        action={"product_price": 60},
    )

    report = build_report(
        [first, later],
        experiments=(experiment,),
        replay=lambda _game, _knobs: {"product_price": 60},
    )["experiments"][0]

    assert report["arm_health"]["integrity_pass"] is True
    assert report["arm_health"]["cohorts"]["legacy"]["control"][
        "turn_events"
    ] == 2
    assert report["arm_health"]["prospective_events"] == 0


def test_legacy_game_assignment_is_fixed_at_first_turn_across_runtime_switch():
    experiment = _experiment(
        name="neg_terminal_close", family="negotiation", cutoff=100
    )
    common = {
        "family": "negotiation",
        "your_player": "player_2",
        "max_rounds": 10,
        "horizon_known": True,
        "complete_information": False,
        "player_1_role": "seller",
        "player_2_role": "buyer",
        "player_2_value": 100,
    }

    def runtime(ts, value):
        return {
            "type": "runtime",
            "ts": ts,
            "_agent": "test_b",
            "agent": "test_b",
            "pid": int(ts),
            "knobs": {"neg_terminal_close": value},
            "git_head": "a" * 40,
            "content_hashes": {
                "strategy_python": {"aggregate_sha256": "b" * 64}
            },
        }

    records = [
        runtime(100, False),
        _turn("test_b", "runtime-switch-game", 101, action={"product_price": 60}, **common),
        runtime(102, True),
        _turn(
            "test_b",
            "runtime-switch-game",
            103,
            round_=9,
            phase="decision",
            history=[{"round": 1}],
            action={"decision": "RejectOffer", "product_price": 85},
            **common,
        ),
    ]

    def replay(game, knobs):
        if game["game_state"]["round"] == 1:
            return {"product_price": 60}
        return {
            "decision": "RejectOffer",
            "product_price": 90 if knobs.neg_terminal_close else 85,
        }

    report = build_report(records, experiments=(experiment,), replay=replay)[
        "experiments"
    ][0]

    assert report["affected_turns"][0]["variant"] == "control"
    assert report["gate"]["counts"]["variants"]["control"]["primary"][
        "affected"
    ] == 1
    assert report["gate"]["counts"]["variants"]["treatment"]["primary"][
        "affected"
    ] == 0


def test_first_seen_after_plan_expiry_is_outside_confirmation_not_integrity_harm():
    experiment = _experiment(
        name="neg_terminal_close", family="negotiation", cutoff=100
    )
    turn = _turn(
        "test_b",
        "after-plan-expiry",
        _PLAN_EXPIRY + 1,
        family="negotiation",
        action={"product_price": 60},
    )
    turn["canary_assignment"] = {
        "status": "unassigned",
        "reason": "plan_expired",
    }

    report = build_report(
        [turn],
        experiments=(experiment,),
        replay=lambda _game, _knobs: {"product_price": 60},
    )["experiments"][0]

    assert report["arm_health"]["integrity_pass"] is True
    assert report["arm_health"]["integrity"][
        "outside_confirmation_turn_events"
    ] == 1
    assert report["arm_health"]["prospective_events"] == 0
    assert report["gate"]["promotion"]["status"] == "continue"


def test_conflicting_timely_terminals_integrity_block_without_empirical_rollback():
    experiment = _experiment(
        name="neg_terminal_close", family="negotiation", cutoff=100
    )

    def replay(game, knobs):
        if game["game_state"]["round"] == 1:
            return {"product_price": 60}
        return {
            "decision": "RejectOffer",
            "product_price": 90 if knobs.neg_terminal_close else 85,
        }

    records = _neg_deadline_records(
        "test_b",
        "conflicting-terminals",
        [
            _result(
                "test_b",
                "conflicting-terminals",
                600,
                {
                    "outcome": "agreement",
                    "agreed_round": 10,
                    "player_2_payoff": 10,
                },
            ),
            _result(
                "test_b",
                "conflicting-terminals",
                650,
                {
                    "outcome": "agreement",
                    "agreed_round": 10,
                    "player_2_payoff": 20,
                },
            ),
        ],
    )
    # ``test_b`` is a frozen legacy control label; keep routing correct so this
    # regression isolates malformed terminal evidence from observed harm.
    records[1]["action"]["product_price"] = 85
    gate = build_report(records, experiments=(experiment,), replay=replay)[
        "experiments"
    ][0]["gate"]

    assert gate["data_integrity"]["pass"] is False
    assert any(
        "conflicting timely valid terminals" in failure
        for failures in gate["data_integrity"]["row_failures"].values()
        for failure in failures
    )
    assert gate["promotion"]["status"] == "continue"


def test_conflicting_duplicate_turn_cannot_rewrite_first_causal_divergence():
    experiment = _experiment(
        name="neg_terminal_close", family="negotiation", cutoff=100
    )
    common = {
        "family": "negotiation",
        "your_player": "player_2",
        "max_rounds": 10,
        "horizon_known": True,
        "complete_information": False,
        "player_1_role": "seller",
        "player_2_role": "buyer",
        "player_2_value": 100,
    }
    records = [
        _turn("test_b", "duplicate-divergence", 101, action={"product_price": 60}, **common),
        _turn(
            "test_b",
            "duplicate-divergence",
            102,
            round_=9,
            phase="decision",
            history=[{"round": 1}],
            action={"decision": "RejectOffer", "product_price": 85},
            **common,
        ),
        _turn(
            "test_b",
            "duplicate-divergence",
            103,
            round_=9,
            phase="decision",
            history=[{"round": 1}],
            action={"decision": "RejectOffer", "product_price": 85},
            same_policy=True,
            **common,
        ),
    ]

    def replay(game, knobs):
        state = game["game_state"]
        if state["round"] == 1:
            return {"product_price": 60}
        price = 85 if state.get("same_policy") else (90 if knobs.neg_terminal_close else 85)
        return {"decision": "RejectOffer", "product_price": price}

    report = build_report(records, experiments=(experiment,), replay=replay)[
        "experiments"
    ][0]

    assert report["affected_turns"] == []
    assert report["agents"]["test_b"]["routing"][
        "duplicate_causal_turn_conflicts"
    ] == 1
    assert report["gate"]["health"]["integrity_blocked"] is True
    assert report["gate"]["promotion"]["status"] == "continue"


def test_malformed_prospective_terminal_schema_is_integrity_not_censor_harm():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "neg_terminal_close",
        True,
        False,
    )
    records, _arm = _prospective_neg_records(gid="malformed-terminal-schema")
    records[3]["game_over"] = "true"
    records.append(
        {
            "type": "runtime",
            "ts": _PLAN_ACTIVATION + 700,
            "_agent": "observer",
            "agent": "observer",
            "pid": 1,
            "knobs": {},
            "git_head": "a" * 40,
            "content_hashes": {},
        }
    )

    def replay(game, knobs):
        if game["game_state"]["round"] == 1:
            return {"product_price": 60}
        return {
            "decision": "RejectOffer",
            "product_price": 90 if knobs.neg_terminal_close else 85,
        }

    report = build_report(records, experiments=(experiment,), replay=replay)[
        "experiments"
    ][0]

    assert report["gate"]["data_integrity"]["pass"] is False
    assert any(
        "malformed prospective terminal schema" in failure
        for failures in report["gate"]["data_integrity"]["row_failures"].values()
        for failure in failures
    )
    assert report["gate"]["censor_safety"]["rollback_look_scheduled"] is False
    assert report["gate"]["promotion"]["binding_rollback"] is False
    assert report["gate"]["promotion"]["status"] == "continue"


def test_negotiation_deadline_is_immutable_and_keeps_earliest_timely_terminal():
    experiment = _experiment(
        name="neg_terminal_close", family="negotiation", cutoff=100
    )

    def replay(game, knobs):
        if game["game_state"]["round"] == 1:
            return {"product_price": 60}
        return {
            "decision": "RejectOffer",
            "product_price": 90 if knobs.neg_terminal_close else 85,
        }

    timely = _result(
        "test_b",
        "timely-then-late",
        701,
        {"outcome": "agreement", "agreed_round": 10, "player_2_payoff": 10},
    )
    later_duplicate = _result(
        "test_b",
        "timely-then-late",
        702,
        {"outcome": "agreement", "agreed_round": 10, "player_2_payoff": 20},
    )
    late_only = _result(
        "test_b",
        "late-only",
        702,
        {"outcome": "agreement", "agreed_round": 10, "player_2_payoff": 20},
    )
    invalid_before = _result(
        "test_b",
        "post-deadline-repair",
        650,
        {"outcome": "agreement", "agreed_round": 10, "player_2_payoff": 20},
        valid=False,
    )
    repaired_late = _result(
        "test_b",
        "post-deadline-repair",
        702,
        {"outcome": "agreement", "agreed_round": 10, "player_2_payoff": 20},
    )
    records = [
        *_neg_deadline_records(
            "test_b", "timely-then-late", [timely, later_duplicate]
        ),
        *_neg_deadline_records("test_b", "late-only", [late_only]),
        *_neg_deadline_records(
            "test_b", "post-deadline-repair", [invalid_before, repaired_late]
        ),
        {
            "type": "runtime",
            "ts": 900,
            "_agent": "observer",
            "agent": "observer",
            "pid": 1,
            "knobs": {},
            "git_head": "a" * 40,
            "content_hashes": {
                "strategy_python": {"aggregate_sha256": "b" * 64}
            },
        },
    ]

    gate = build_report(
        records, experiments=(experiment,), replay=replay
    )["experiments"][0]["gate"]
    treatment = gate["counts"]["variants"]["control"]["primary"]

    assert treatment["affected"] == 2
    assert treatment["matured"] == 2
    assert treatment["resolved"] == 1
    assert treatment["censored"] == 1
    assert treatment["pending_maturation"] == 0
    assert treatment["direct_trials"] == 1
    assert treatment["late_terminals"] == 2
    assert gate["data_integrity"]["pass"] is False
    assert any(
        "invalid timely terminal evidence" in failure
        for failures in gate["data_integrity"]["row_failures"].values()
        for failure in failures
    )


def test_young_unresolved_negotiation_game_is_pending_not_censored():
    experiment = _experiment(
        name="neg_terminal_close", family="negotiation", cutoff=100
    )

    def replay(game, knobs):
        if game["game_state"]["round"] == 1:
            return {"product_price": 60}
        return {
            "decision": "RejectOffer",
            "product_price": 90 if knobs.neg_terminal_close else 85,
        }

    records = _neg_deadline_records("test_b", "young", [])
    gate = build_report(
        records, experiments=(experiment,), replay=replay
    )["experiments"][0]["gate"]
    treatment = gate["counts"]["variants"]["control"]["primary"]

    assert treatment["affected"] == 1
    assert treatment["matured"] == 0
    assert treatment["pending_maturation"] == 1
    assert treatment["censored"] == 0
    assert gate["censor_safety"]["rollback_look_scheduled"] is False


def _neg_censor_rows(treatment_n, control_n, treatment_censored, control_censored):
    cells = [
        (identity, value)
        for identity in ("agent", "hidden")
        for value in (80, 100, 120, 150)
    ]
    rows = []
    for arm, total, censored, agent in (
        ("treatment", treatment_n, treatment_censored, "test_a"),
        ("control", control_n, control_censored, "main"),
    ):
        for index in range(total):
            identity, value = cells[index % len(cells)]
            rows.append(
                _gate_row(
                    agent,
                    arm,
                    value,
                    direct=None if index < censored else False,
                    opponent_type=identity,
                )
            )
    return rows


def test_negotiation_censor_harm_does_not_rollback_before_scheduled_mature_look():
    gate = _neg_terminal_gate_from_rows(
        _neg_censor_rows(49, 149, 45, 135)
    )

    assert gate["censor_safety"]["rollback_look_scheduled"] is False
    assert gate["censor_safety"]["rollback_reasons"] == []
    assert gate["promotion"]["status"] != "rollback"


def test_bilateral_heavy_negotiation_censoring_rolls_back_at_scheduled_look():
    gate = _neg_terminal_gate_from_rows(
        _neg_censor_rows(50, 150, 45, 135)
    )

    safety = gate["censor_safety"]
    assert safety["rollback_look_scheduled"] is True
    assert any("Wilson lower exceeds" in reason for reason in safety["rollback_reasons"])
    assert gate["promotion"]["status"] == "rollback"


def test_malformed_row_makes_scheduled_censor_signal_nonbinding():
    rows = _neg_censor_rows(50, 150, 45, 135)
    malformed = dict(rows[0])
    malformed["game_id"] += "-malformed"
    malformed["malformed_terminal_events"] = 1
    rows.append(malformed)

    gate = _neg_terminal_gate_from_rows(rows)

    assert gate["health"]["integrity_blocked"] is True
    assert gate["promotion"]["empirical_rollback_signal"] is True
    assert gate["promotion"]["binding_rollback"] is False
    assert gate["promotion"]["status"] == "continue"


def test_pre_amendment_small_censor_snapshot_is_nonbinding():
    gate = _neg_terminal_gate_from_rows(_neg_censor_rows(11, 33, 1, 2))

    contrast = gate["censor_safety"]["treatment_minus_control"]
    assert contrast["difference"] == pytest.approx(1 / 11 - 2 / 33)
    assert gate["censor_safety"]["rollback_look_scheduled"] is False
    assert gate["promotion"]["status"] != "rollback"


def test_malformed_negotiation_maturity_evidence_blocks_without_binding_rollback():
    row = _gate_row("test_a", "treatment", 100, direct=False)
    row["maturity_lag_s"] = "600"

    gate = _neg_terminal_gate_from_rows([row])

    assert gate["data_integrity"]["pass"] is False
    assert gate["health"]["integrity_blocked"] is True
    assert gate["promotion"]["status"] == "continue"


def test_offer_outcomes_use_only_first_exact_divergence_per_game():
    experiment = _experiment()

    def replay(_game, knobs):
        mine = round(100 * knobs.barg_dis_anchor)
        return {"alice_gain": mine, "bob_gain": 100 - mine}

    records = [
        _turn(
            "main",
            "multi",
            101,
            action={"alice_gain": 58, "bob_gain": 42},
            money_to_divide=100,
            max_rounds=6,
            horizon_known=True,
        ),
        _turn(
            "main",
            "multi",
            102,
            round_=3,
            history=[{"round": 1}],
            action={"alice_gain": 58, "bob_gain": 42},
            money_to_divide=100,
            max_rounds=6,
            horizon_known=True,
        ),
        _result(
            "main",
            "multi",
            103,
            {"outcome": "agreement", "agreed_round": "1", "player_1_payoff": 58},
        ),
    ]
    report = build_report(records, experiments=(experiment,), replay=replay)["experiments"][0]

    assert report["agents"]["main"]["routing"]["affected"] == 2
    assert [item["first_for_game"] for item in report["affected_turns"]] == [True, False]
    control = report["metrics"]["control"]
    assert control["affected_turns"] == 1
    assert control["direct_offers"] == 1
    assert control["direct_converted"] == 1
    assert control["effective_offer_rounds"] == {
        "1": {"offers": 1, "resolved": 1, "converted": 1, "conversion_rate": 1}
    }


def test_bargaining_confirmation_scope_includes_test_a():
    experiment = next(
        item for item in EXPERIMENTS if item.name == "barg_dis_anchor"
    )
    assert set(experiment.agents) == {"main", "test_a", "test_b", "test_c"}
    assert "test_a" in experiment.control_agents


def test_analysis_cohorts_separate_legacy_prospective_and_postexpiry_games():
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("test_b",),
        ("main",),
        "barg_dis_anchor",
        0.50,
        0.58,
    )
    prospective, arm = _prospective_nonneg_records(
        "bargaining", gid="cohort-prospective"
    )
    legacy = [
        _turn(
            "test_b",
            "cohort-legacy",
            101,
            action={"alice_gain": 50, "bob_gain": 50},
            money_to_divide=100,
        ),
        _result(
            "test_b",
            "cohort-legacy",
            102,
            {
                "outcome": "agreement",
                "agreed_round": 1,
                "player_1_payoff": 50,
            },
        ),
    ]
    outside = _turn(
        "main",
        "cohort-outside",
        _PLAN_EXPIRY + 1,
        action={"alice_gain": 58, "bob_gain": 42},
        money_to_divide=100,
    )
    outside["canary_assignment"] = {
        "status": "unassigned",
        "reason": "plan_expired",
    }

    report = build_report(
        [*legacy, *prospective, outside],
        experiments=(experiment,),
        replay=_barg_replay,
    )["experiments"][0]
    cohorts = report["analysis_cohorts"]

    assert set(cohorts) == {
        "legacy",
        "prospective",
        "outside_confirmation",
    }
    assert cohorts["legacy"]["enrolled_games"] == 1
    assert cohorts["prospective"]["enrolled_games"] == 1
    assert cohorts["prospective"]["binding_eligible"] is True
    assert cohorts["outside_confirmation"]["enrolled_games"] == 0
    assert cohorts["outside_confirmation"]["excluded_games"] == 1
    assert sum(report["itt"][candidate]["games"] for candidate in ("treatment", "control")) == 2
    assert sum(
        cohorts["prospective"]["itt"][candidate]["games"]
        for candidate in ("treatment", "control")
    ) == 1
    assert sum(
        cohorts["outside_confirmation"]["itt"][candidate]["games"]
        for candidate in ("treatment", "control")
    ) == 0
    confirmation = report["prospective_confirmation"]
    assert confirmation["linked_itt_rows"][arm] == 1
    assert confirmation["labels"]["main"][f"{arm}_rows"] == 1
    assert confirmation["labels"]["main"][f"{arm}_affected_rows"] == 1
    assert confirmation["contract"]["artifact_sha256"] == _PLAN_SHA
    assert confirmation["scheduled_look"]["declaration_artifact"] == {
        "schema_version": 1,
        "path": "data/canary_analysis_plan.json",
        "sha256": "39943a3877adafae71f6bdacfab13a02f0065dc1b955ef6184fbb14dfe20e260",
        "bytes": 5950,
        "look_id": "final-confirmatory-expiry-plus-persuasion-maturity-v1",
        "declared_at_ts": 1787689408,
        "analysis_as_of_ts": 1787952600,
        "enrollment_cutoff_exclusive": _PLAN_EXPIRY,
        "prefix_procedure_version": "stable-filtered-jsonl-snapshot-v2",
        "prefix_capture_not_before": 1787952900,
        "prefix_output_path_template": (
            "data/canary-confirmation-prefix/{declaration_sha256}.json"
        ),
    }
    assert confirmation["reporter_verification"] == {
        "schema_version": 1,
        "producer": "scripts.canary_report:confirmation-verifier-v1",
        "prefix_recomputed_from_sources": False,
        "declaration_recomputed_from_artifact": False,
        "prefix_sha256": None,
        "declaration_sha256": "39943a3877adafae71f6bdacfab13a02f0065dc1b955ef6184fbb14dfe20e260",
    }


def test_persuasion_uses_validated_game_arm_not_static_agent_label():
    records, arm = _prospective_nonneg_records(
        "persuasion", gid="pers-dynamic-arm"
    )
    opposite = "control" if arm == "treatment" else "treatment"
    experiment = Experiment(
        "pers_blind_lie",
        "persuasion",
        100,
        ("main",) if opposite == "treatment" else (),
        ("main",) if opposite == "control" else (),
        "pers_blind_lie",
        0.40,
        1.0,
    )
    duplicate = json.loads(json.dumps(records[1]))
    duplicate["ts"] = _PLAN_ACTIVATION + 20
    records.insert(2, duplicate)

    def replay(_game, knobs):
        return {"decision": "yes" if knobs.pers_blind_lie == 1.0 else "no"}

    report = build_report(records, experiments=(experiment,), replay=replay)[
        "experiments"
    ][0]

    assert report["variants"][arm]["games"] == 1
    assert report["variants"][opposite]["games"] == 0
    assert report["metrics"][arm]["blind_seller_games"] == 1
    assert report["metrics"][opposite]["blind_seller_games"] == 0
    assert report["itt"][arm]["by_agent"]["main"]["games"] == 1
    assert report["analysis_cohorts"]["prospective"]["integrity"]["pass"] is True


def test_new_enrollment_after_runtime_restart_fails_closed():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "neg_terminal_close",
        True,
        False,
    )
    records, _arm = _prospective_neg_records(gid="new-new-restart")
    restarted = json.loads(json.dumps(records[0]))
    restarted["ts"] = _PLAN_ACTIVATION + 15
    restarted["pid"] = 88
    records.insert(2, restarted)

    report = build_report(
        records,
        experiments=(experiment,),
        replay=lambda _game, _knobs: {"product_price": 60},
    )["experiments"][0]

    prospective = report["analysis_cohorts"]["prospective"]
    assert prospective["enrolled_games"] == 0
    assert prospective["excluded_games"] == 1
    assert prospective["routing"]["integrity"][
        "unapproved_prospective_games"
    ] == 1
    assert prospective["binding_eligible"] is False


def test_wrong_runtime_hash_on_recovered_turn_invalidates_whole_game():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "neg_terminal_close",
        True,
        False,
    )
    records, _arm = _prospective_neg_records(gid="wrong-recovered-runtime")
    restarted = json.loads(json.dumps(records[0]))
    restarted["ts"] = _PLAN_ACTIVATION + 15
    restarted["pid"] = 88
    restarted["content_hashes"]["strategy_python"]["aggregate_sha256"] = "f" * 64
    records[2]["canary_assignment"]["enrollment"] = "recovered"
    records.insert(2, restarted)

    report = build_report(
        records,
        experiments=(experiment,),
        replay=lambda _game, _knobs: {"product_price": 60},
    )["experiments"][0]

    assert report["analysis_cohorts"]["prospective"]["enrolled_games"] == 0
    assert report["arm_health"]["integrity_pass"] is False


@pytest.mark.parametrize("identity", ["strategy", "targets"])
def test_terminal_must_link_to_latest_exact_runtime_identity(identity):
    records, arm = _prospective_nonneg_records(
        "bargaining", gid=f"terminal-runtime-{identity}"
    )
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("main",),
        (),
        "barg_dis_anchor",
        0.50,
        0.58,
    )
    restarted = json.loads(json.dumps(records[0]))
    restarted["ts"] = _PLAN_ACTIVATION + 50
    restarted["pid"] = 88
    if identity == "strategy":
        restarted["content_hashes"]["strategy_python"][
            "aggregate_sha256"
        ] = "f" * 64
    else:
        restarted["content_hashes"]["targets"]["data/targets.json"][
            "sha256"
        ] = "f" * 64
    records.insert(2, restarted)

    report = build_report(
        records, experiments=(experiment,), replay=_barg_replay
    )["experiments"][0]
    prospective = report["analysis_cohorts"]["prospective"]

    assert prospective["itt"][arm]["invalid_terminals"] == 1
    assert prospective["health"][arm]["invalid_results"] == 1
    assert prospective["integrity"]["pass"] is False
    assert prospective["binding_eligible"] is False


@pytest.mark.parametrize("agreed_round", ["1", True, 0, 7])
def test_prospective_agreement_round_must_be_literal_int_and_in_bounds(
    agreed_round,
):
    records, arm = _prospective_nonneg_records(
        "bargaining", gid=f"bad-agreed-round-{agreed_round!r}"
    )
    records[-1]["result"]["agreed_round"] = agreed_round
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("main",),
        (),
        "barg_dis_anchor",
        0.50,
        0.58,
    )

    prospective = build_report(
        records, experiments=(experiment,), replay=_barg_replay
    )["experiments"][0]["analysis_cohorts"]["prospective"]

    assert prospective["itt"][arm]["invalid_terminals"] == 1
    assert prospective["metrics"][arm]["invalid_terminals"] == 1
    assert prospective["binding_eligible"] is False


def test_terminal_conflict_signature_includes_agreement_round():
    records, arm = _prospective_nonneg_records(
        "bargaining", gid="round-conflict"
    )
    conflict = json.loads(json.dumps(records[-1]))
    conflict["ts"] += 1
    conflict["result"]["agreed_round"] = 2
    records.append(conflict)
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("main",),
        (),
        "barg_dis_anchor",
        0.50,
        0.58,
    )

    prospective = build_report(
        records, experiments=(experiment,), replay=_barg_replay
    )["experiments"][0]["analysis_cohorts"]["prospective"]

    assert prospective["itt"][arm]["terminal_conflicts"] == 1
    assert prospective["metrics"][arm]["terminal_conflicts"] == 1
    assert prospective["integrity"]["pass"] is False


def test_hostile_unhashable_payoff_is_integrity_failure_not_crash():
    records, arm = _prospective_nonneg_records(
        "bargaining", gid="hostile-payoff"
    )
    records[-1]["result"]["player_1_payoff"] = [50]
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("main",),
        (),
        "barg_dis_anchor",
        0.50,
        0.58,
    )

    prospective = build_report(
        records, experiments=(experiment,), replay=_barg_replay
    )["experiments"][0]["analysis_cohorts"]["prospective"]

    assert prospective["itt"][arm]["invalid_terminals"] == 1
    assert prospective["integrity"]["pass"] is False


def test_hostile_unhashable_outcome_is_integrity_failure_not_crash():
    records, arm = _prospective_nonneg_records(
        "bargaining", gid="hostile-outcome"
    )
    records[-1]["result"]["outcome"] = ["agreement"]
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("main",),
        (),
        "barg_dis_anchor",
        0.50,
        0.58,
    )

    prospective = build_report(
        records, experiments=(experiment,), replay=_barg_replay
    )["experiments"][0]["analysis_cohorts"]["prospective"]

    assert prospective["itt"][arm]["invalid_terminals"] == 1
    assert prospective["health"][arm]["invalid_results"] == 1
    assert prospective["integrity"]["pass"] is False
    assert prospective["binding_eligible"] is False


@pytest.mark.parametrize("field", ["player_1_payoff", "agreed_round"])
def test_gigantic_terminal_integer_fails_closed_without_overflow(field):
    records, arm = _prospective_nonneg_records(
        "bargaining", gid=f"gigantic-terminal-{field}"
    )
    records[-1]["result"][field] = 10**10000
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("main",),
        (),
        "barg_dis_anchor",
        0.50,
        0.58,
    )

    prospective = build_report(
        records, experiments=(experiment,), replay=_barg_replay
    )["experiments"][0]["analysis_cohorts"]["prospective"]

    assert prospective["itt"][arm]["invalid_terminals"] == 1
    assert prospective["health"][arm]["invalid_results"] == 1
    assert prospective["integrity"]["pass"] is False
    assert prospective["binding_eligible"] is False


def test_prospective_persuasion_missing_visibility_flag_is_out_of_population():
    records, arm = _prospective_nonneg_records(
        "persuasion", gid="missing-visibility-flag"
    )
    state = records[1]["game"]["game_state"]
    state.pop("is_seller_know_cv")
    state["v"] = None
    experiment = Experiment(
        "pers_blind_lie",
        "persuasion",
        100,
        ("main",),
        (),
        "pers_blind_lie",
        0.40,
        1.0,
    )

    def replay(_game, knobs):
        return {"decision": "yes" if knobs.pers_blind_lie == 1.0 else "no"}

    report = build_report(
        records, experiments=(experiment,), replay=replay
    )["experiments"][0]
    prospective = report["analysis_cohorts"]["prospective"]

    assert prospective["enrolled_games"] == 1
    assert sum(
        prospective["metrics"][candidate]["blind_seller_games"]
        for candidate in ("treatment", "control")
    ) == 0
    assert sum(
        prospective["itt"][candidate]["games"]
        for candidate in ("treatment", "control")
    ) == 0
    assert report["metrics"][arm]["blind_seller_games"] == 0
    assert prospective["integrity"]["pass"] is True
    assert prospective["binding_eligible"] is False
    assert report["prospective_confirmation"]["prospective_rows"] == 0


def test_extreme_neg_probability_receipt_fails_closed_without_overflow():
    records, _arm = _prospective_neg_records(gid="extreme-probability")
    for record in records:
        receipt = record.get("canary_assignment")
        if isinstance(receipt, dict):
            receipt["treatment_probability"] = 1e308
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "neg_terminal_close",
        True,
        False,
    )

    def replay(game, knobs):
        if game["game_state"]["round"] == 1:
            return {"product_price": 60}
        return {
            "decision": "RejectOffer",
            "product_price": 90 if knobs.neg_terminal_close else 85,
        }

    prospective = build_report(
        records, experiments=(experiment,), replay=replay
    )["experiments"][0]["analysis_cohorts"]["prospective"]

    assert prospective["enrolled_games"] == 0
    assert prospective["excluded_games"] == 1
    assert prospective["integrity"]["pass"] is False
    assert prospective["binding_eligible"] is False


@pytest.mark.parametrize("family", ["bargaining", "negotiation", "persuasion"])
def test_gigantic_your_player_fails_closed_for_every_family(family):
    if family == "negotiation":
        records, arm = _prospective_neg_records(
            gid="gigantic-player-negotiation"
        )
        experiment = Experiment(
            "neg_terminal_close",
            "negotiation",
            100,
            ("main", "test_a", "test_b", "test_c"),
            (),
            "neg_terminal_close",
            True,
            False,
        )

        def replay(game, knobs):
            if game["game_state"]["round"] == 1:
                return {"product_price": 60}
            return {
                "decision": "RejectOffer",
                "product_price": 90 if knobs.neg_terminal_close else 85,
            }

    else:
        records, arm = _prospective_nonneg_records(
            family, gid=f"gigantic-player-{family}"
        )
        if family == "bargaining":
            experiment = Experiment(
                "barg_dis_anchor",
                "bargaining",
                100,
                ("main",),
                (),
                "barg_dis_anchor",
                0.50,
                0.58,
            )
            replay = _barg_replay
        else:
            experiment = Experiment(
                "pers_blind_lie",
                "persuasion",
                100,
                ("main",),
                (),
                "pers_blind_lie",
                0.40,
                1.0,
            )

            def replay(_game, knobs):
                return {
                    "decision": (
                        "yes" if knobs.pers_blind_lie == 1.0 else "no"
                    )
                }

    for record in records:
        if record.get("type") == "turn":
            record["game"]["your_player"] = 10**10000

    prospective = build_report(
        records, experiments=(experiment,), replay=replay
    )["experiments"][0]["analysis_cohorts"]["prospective"]

    assert prospective["enrolled_games"] == 1
    assert prospective["health"][arm]["invalid_results"] == 1
    assert prospective["integrity"]["pass"] is False
    assert prospective["binding_eligible"] is False


def test_gigantic_persuasion_message_type_is_normalized_and_blocks_binding():
    records, arm = _prospective_nonneg_records(
        "persuasion", gid="gigantic-message-type"
    )
    records[1]["game"]["game_state"]["seller_message_type"] = 10**10000
    experiment = Experiment(
        "pers_blind_lie",
        "persuasion",
        100,
        ("main",),
        (),
        "pers_blind_lie",
        0.40,
        1.0,
    )

    def replay(_game, knobs):
        return {"decision": "yes" if knobs.pers_blind_lie == 1.0 else "no"}

    prospective = build_report(
        records, experiments=(experiment,), replay=replay
    )["experiments"][0]["analysis_cohorts"]["prospective"]

    assert prospective["metrics"][arm]["blind_seller_games"] == 1
    assert {
        entry["cell"]["message_type"]
        for entry in prospective["metrics"][arm]["cells"].values()
    } == {"unknown"}
    assert prospective["integrity"]["pass"] is False
    assert prospective["binding_eligible"] is False


def test_legacy_preprovenance_reaper_is_accepted_only_before_activation():
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        (),
        ("main",),
        "barg_dis_anchor",
        0.50,
        0.58,
    )
    records = []
    for gid, terminal_ts in (
        ("legacy-reaper-before", _PLAN_ACTIVATION - 10),
        ("legacy-reaper-after", _PLAN_ACTIVATION + 10),
    ):
        records.extend(
            [
                _turn(
                    "main",
                    gid,
                    _PLAN_ACTIVATION - 20,
                    action={"alice_gain": 58, "bob_gain": 42},
                    money_to_divide=100,
                ),
                _result(
                    "main",
                    gid,
                    terminal_ts,
                    {
                        "outcome": "agreement",
                        "agreed_round": 1,
                        "player_1_payoff": 58,
                    },
                    valid=None,
                ),
            ]
        )
    records.append(
        {
            "type": "runtime",
            "ts": _PLAN_ACTIVATION + 2000,
            "_agent": "observer",
            "agent": "observer",
            "pid": 1,
            "knobs": {},
            "content_hashes": {},
        }
    )

    legacy = build_report(
        records, experiments=(experiment,), replay=_barg_replay
    )["experiments"][0]["analysis_cohorts"]["legacy"]

    assert legacy["itt"]["control"]["games"] == 2
    assert legacy["itt"]["control"]["resolved"] == 1
    assert legacy["itt"]["control"]["invalid_terminals"] == 1


def test_move_transport_error_exact_null_envelope_is_health_only():
    records, arm = _prospective_nonneg_records(
        "bargaining", gid="transport-null"
    )
    records.insert(
        2,
        _result(
            "main",
            "transport-null",
            _PLAN_ACTIVATION + 50,
            None,
            valid=None,
            game_over=None,
            error="HTTP 500",
            result_source="move_transport_error",
            reaped=False,
        ),
    )
    records.append(
        {
            "type": "runtime",
            "ts": _PLAN_ACTIVATION + 1300,
            "_agent": "observer",
            "agent": "observer",
            "pid": 1,
            "knobs": {},
            "content_hashes": {},
        }
    )
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("main",),
        (),
        "barg_dis_anchor",
        0.50,
        0.58,
    )

    prospective = build_report(
        records, experiments=(experiment,), replay=_barg_replay
    )["experiments"][0]["analysis_cohorts"]["prospective"]

    assert prospective["health"][arm]["result_errors"] == 1
    assert prospective["health"][arm]["invalid_results"] == 0
    assert prospective["itt"][arm]["resolved"] == 1
    assert prospective["integrity"]["pass"] is True


@pytest.mark.parametrize("error", [False, 0, [], {}, ""])
def test_transport_error_requires_literal_nonempty_error_and_stays_visible(error):
    records, arm = _prospective_nonneg_records(
        "bargaining", gid=f"transport-falsey-{type(error).__name__}"
    )
    records.insert(
        2,
        _result(
            "main",
            records[1]["game"]["game_id"],
            _PLAN_ACTIVATION + 50,
            None,
            valid=None,
            game_over=None,
            error=error,
            result_source="move_transport_error",
            reaped=False,
        ),
    )
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("main",),
        (),
        "barg_dis_anchor",
        0.50,
        0.58,
    )

    prospective = build_report(
        records, experiments=(experiment,), replay=_barg_replay
    )["experiments"][0]["analysis_cohorts"]["prospective"]

    assert prospective["health"][arm]["result_errors"] == 1
    assert prospective["health"][arm]["invalid_results"] == 1
    assert prospective["integrity"]["pass"] is False
    assert prospective["binding_eligible"] is False


def test_transport_error_with_outcome_payload_poison_is_rejected():
    records, arm = _prospective_nonneg_records(
        "bargaining", gid="transport-payload"
    )
    records.insert(
        2,
        _result(
            "main",
            "transport-payload",
            _PLAN_ACTIVATION + 50,
            {
                "outcome": "agreement",
                "agreed_round": 1,
                "player_1_payoff": 100,
            },
            valid=None,
            game_over=None,
            error="HTTP 500",
            result_source="move_transport_error",
            reaped=False,
        ),
    )
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("main",),
        (),
        "barg_dis_anchor",
        0.50,
        0.58,
    )

    prospective = build_report(
        records, experiments=(experiment,), replay=_barg_replay
    )["experiments"][0]["analysis_cohorts"]["prospective"]

    assert prospective["health"][arm]["invalid_results"] == 1
    assert prospective["itt"][arm]["invalid_terminals"] == 1
    assert prospective["integrity"]["pass"] is False


def test_future_skew_and_out_of_range_timestamp_share_fail_closed_clock():
    records, _arm = _prospective_nonneg_records(
        "bargaining", gid="future-clock"
    )
    records.extend(
        [
            {
                "type": "runtime",
                "ts": _PLAN_ACTIVATION + 1_000_000,
                "_agent": "observer",
                "agent": "observer",
                "pid": 1,
                "knobs": {},
                "content_hashes": {},
            },
            {
                "type": "runtime",
                "ts": 1e20,
                "_agent": "observer",
                "agent": "observer",
                "pid": 2,
                "knobs": {},
                "content_hashes": {},
            },
        ]
    )
    bargaining = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("main",),
        (),
        "barg_dis_anchor",
        0.50,
        0.58,
    )
    persuasion = Experiment(
        "pers_blind_lie",
        "persuasion",
        100,
        (),
        ("main",),
        "pers_blind_lie",
        0.40,
        1.0,
    )

    report = build_report(
        records,
        experiments=(bargaining, persuasion),
        replay=_barg_replay,
        wall_clock_ts=_PLAN_ACTIVATION + 200,
    )

    assert report["analysis_clock"]["valid"] is False
    assert report["analysis_clock"]["future_skew_events"] == 2
    assert report["analysis_clock"]["out_of_range_timestamp_events"] == 1
    assert {
        item["analysis_as_of_ts"] for item in report["experiments"]
    } == {_PLAN_ACTIVATION + 100}
    assert report["experiments"][0]["analysis_cohorts"]["prospective"][
        "binding_eligible"
    ] is False


def test_nondivergent_negotiation_replay_error_is_attributed_to_assigned_arm():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "neg_terminal_close",
        True,
        False,
    )
    records, arm = _prospective_neg_records(gid="nondivergent-replay-error")

    def broken_replay(_game, _knobs):
        raise RuntimeError("synthetic replay failure")

    report = build_report(
        records, experiments=(experiment,), replay=broken_replay
    )["experiments"][0]

    assert report["affected_turns"] == []
    assert report["analysis_cohorts"]["prospective"]["routing"][arm][
        "replay_errors"
    ] == 2
    assert report["gate"]["health"]["by_variant"][arm]["replay_errors"] == 2


def test_unprintable_replay_error_is_attributed_without_crashing_report():
    experiment = Experiment(
        "neg_terminal_close",
        "negotiation",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "neg_terminal_close",
        True,
        False,
    )
    records, arm = _prospective_neg_records(gid="unprintable-replay-error")

    def broken_replay(_game, _knobs):
        raise ValueError(10**10_000)

    report = build_report(
        records, experiments=(experiment,), replay=broken_replay
    )["experiments"][0]

    assert report["affected_turns"] == []
    assert report["analysis_cohorts"]["prospective"]["routing"][arm][
        "replay_errors"
    ] == 2
    assert report["gate"]["health"]["by_variant"][arm]["replay_errors"] == 2


def test_lone_surrogate_receipt_identity_fails_closed_without_crashing_report():
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "barg_dis_anchor",
        0.50,
        0.58,
    )
    records, _arm = _prospective_nonneg_records("bargaining")
    surrogate_gid = "\ud800"
    records[1]["game"]["game_id"] = surrogate_gid
    records[2]["game_id"] = surrogate_gid

    report = build_report(records, experiments=(experiment,), replay=_barg_replay)[
        "experiments"
    ][0]

    prospective = report["analysis_cohorts"]["prospective"]
    assert prospective["binding_eligible"] is False
    assert prospective["enrolled_games"] == 0
    assert canary_report.render_text(
        {"experiments": [report], "gates": {}}
    ).encode("utf-8")


def test_lone_surrogate_inside_receipt_fails_assignment_without_crashing():
    experiment = Experiment(
        "barg_dis_anchor",
        "bargaining",
        100,
        ("main", "test_a", "test_b", "test_c"),
        (),
        "barg_dis_anchor",
        0.50,
        0.58,
    )
    records, _arm = _prospective_nonneg_records("bargaining")
    records[0]["canary_assignment"]["contract"]["assignment_salt"] = "\ud800"

    report = build_report(records, experiments=(experiment,), replay=_barg_replay)[
        "experiments"
    ][0]

    prospective = report["analysis_cohorts"]["prospective"]
    assert prospective["binding_eligible"] is False
    assert prospective["integrity"]["assignment_failures"] > 0


def test_neg_pure_gate_rows_reject_bounds_invalid_duplicates_and_clock_drift():
    cases = []
    out_of_bounds = _gate_row("test_a", "treatment", 100, direct=False)
    out_of_bounds["normalized_payoff"] = -10
    cases.append(([out_of_bounds], "normalized_payoff must be within [0,1]"))

    invalid_terminal = _gate_row("test_a", "treatment", 100, direct=False)
    invalid_terminal["invalid_timely_terminals"] = 1
    cases.append(([invalid_terminal], "invalid timely terminal evidence"))

    unhashable_status = _gate_row(
        "test_a", "treatment", 100, direct=False
    )
    unhashable_status["maturity_status"] = ["deadline_censored"]
    cases.append(
        ([unhashable_status], "maturity status/count flags inconsistent")
    )

    unhashable_cell = _gate_row("test_a", "treatment", 100, direct=False)
    unhashable_cell["cell"]["own_value_grid"] = ["100"]
    unhashable_cell["cell_id"] = json.dumps(
        unhashable_cell["cell"], sort_keys=True, separators=(",", ":")
    )
    cases.append(
        ([unhashable_cell], "cell.own_value_grid must be a literal string")
    )

    duplicate = _gate_row("test_a", "treatment", 100, direct=False)
    cases.append(
        ([duplicate, dict(duplicate)], "duplicate (agent,game_id) observation")
    )

    first = _gate_row("test_a", "treatment", 100, direct=False)
    second = _gate_row("main", "control", 100, direct=False)
    second["analysis_ts"] += 1
    cases.append(([first, second], "analysis_ts differs from common report prefix"))

    for rows, expected in cases:
        gate = _neg_terminal_gate_from_rows(rows)
        assert gate["data_integrity"]["pass"] is False
        assert any(
            expected in failure
            for failures in gate["data_integrity"]["row_failures"].values()
            for failure in failures
        )
        assert gate["promotion"]["binding_rollback"] is False


def test_neg_gate_rejects_forged_supported_flag_before_cell_counting():
    row = _gate_row("test_a", "treatment", 100, direct=False)
    row["cell"]["own_value_grid"] = "999"
    row["cell_id"] = json.dumps(
        row["cell"], sort_keys=True, separators=(",", ":")
    )
    row["supported"] = True
    row["unsupported_reason"] = None

    gate = _neg_terminal_gate_from_rows([row])

    assert gate["data_integrity"]["pass"] is False
    assert any(
        "supported flag does not match frozen cell membership" in failure
        for failures in gate["data_integrity"]["row_failures"].values()
        for failure in failures
    )
    assert gate["counts"]["variants"]["treatment"]["all"]["affected"] == 0
    assert gate["promotion"]["binding_rollback"] is False


def test_seek_and_prefix_scan_only_decode_post_cut(tmp_path):
    experiment = next(exp for exp in EXPERIMENTS if exp.name == "barg_dis_anchor")
    path = tmp_path / "main-20260825.jsonl"
    before = _turn(
        "main",
        "before",
        experiment.cutoff - 1,
        action={"alice_gain": 58, "bob_gain": 42},
        money_to_divide=100,
    )
    after = _turn(
        "main",
        "after",
        experiment.cutoff + 1,
        action={"alice_gain": 58, "bob_gain": 42},
        money_to_divide=100,
    )
    path.write_text(json.dumps(before) + "\n" + json.dumps(after) + "\n")

    offset = seek_timestamp(path, experiment.cutoff)
    assert offset == len((json.dumps(before) + "\n").encode())
    slices, preexisting = discover_log_slices(tmp_path, (experiment,))
    assert ("main", "before") in preexisting
    records = list(iter_log_records(slices))
    assert [record["game"]["game_id"] for record in records] == ["after"]


def test_jsonl_iterator_skips_overlong_and_invalid_numeric_lines(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(canary_report, "_MAX_JSONL_LINE_BYTES", 8192)
    path = tmp_path / "main-hostile.jsonl"
    overlong = b'{"padding":"' + b"x" * 9000 + b'"}'
    deeply_nested = (
        b'{"type":"runtime","ts":1,"nested":'
        + b"[" * 1500
        + b"0"
        + b"]" * 1500
        + b"}"
    )
    huge_integer = (
        b'{"type":"runtime","ts":1,"value":'
        + b"9" * 5000
        + b"}"
    )
    exponent_overflow = b'{"type":"runtime","ts":1,"value":1e10000}'
    nonstandard_infinity = b'{"type":"runtime","ts":1,"value":Infinity}'
    valid = json.dumps({"type": "runtime", "ts": 2}).encode()
    path.write_bytes(
        b"\n".join(
            (
                overlong,
                deeply_nested,
                huge_integer,
                exponent_overflow,
                nonstandard_infinity,
                valid,
            )
        )
        + b"\n"
    )

    records = list(iter_log_records((LogSlice("main", path, 0),)))

    # The stdlib decoder safely accepts this depth in the current runtime;
    # irrelevant nested extension data need not incur a second Python byte
    # scan.  Decoder recursion failures at greater depth are caught above.
    assert [record["ts"] for record in records] == [1, 2]
    assert records[0]["_agent"] == "main"
    assert records[1] == {"type": "runtime", "ts": 2, "_agent": "main"}


def test_amendment_pin_matches_the_on_disk_artifact():
    """The negotiation deviation is pinned, so the artifact cannot drift."""
    pin = canary_report._NEG_CONFIRMATION_AMENDMENT
    raw = (REPOSITORY_ROOT / pin["path"]).read_bytes()

    assert len(raw) == pin["bytes"]
    assert hashlib.sha256(raw).hexdigest() == pin["sha256"]

    body = json.loads(raw)
    amendment = body["amendments"][0]
    assert amendment["amendment_id"] == pin["amendment_id"]
    assert amendment["amended_at"] == pin["amended_at_ts"]
    assert amendment["rule_id"] == "neg-terminal-confirm-v2"
    assert amendment["family"] == "negotiation"
    assert amendment["pre_registered"] is False
    assert amendment["declared_before_activation"] is False
    # The deviation was made after enrollment opened; the record must say so
    # rather than presenting itself as pre-registered evidence.
    assert amendment["amended_at"] > amendment["activated_at"]
    assert amendment["unblinding_disclosure"]["live_outcomes_inspected"] is True
    assert (
        amendment["retired_promotion_check"] == pin["retired_promotion_check"]
    )

    # Every further deviation must be pinned too, so none can be made silently.
    recorded = [item["amendment_id"] for item in body["amendments"]]
    assert recorded[0] == pin["amendment_id"]
    assert tuple(recorded[1:]) == pin["additional_amendment_ids"]

    measurement = body["amendments"][1]
    assert measurement["kind"] == "post_firing_measurement_correction"
    assert measurement["made_after_the_trigger_fired"] is True
    assert measurement["pre_registered"] is False
    assert measurement["affected_check"] == "treatment_invalid_and_corrections_clean"
    # Zero tolerance on treatment-caused invalidity must still be claimed and
    # must still be true in the code the gate actually runs.
    assert "zero tolerance retained" in measurement["change"]["hard_fail_scope_after"]
    assert measurement["change"]["new_check"] == "reporter_fault_excess_within_0.01"


def test_amendment_leaves_the_frozen_declaration_and_other_families_untouched():
    """Bargaining and persuasion keep their pre-registered declaration."""
    declaration_pin = canary_report._CONFIRMATION_DECLARATION_PIN
    raw = (REPOSITORY_ROOT / declaration_pin["path"]).read_bytes()

    assert len(raw) == declaration_pin["bytes"]
    assert hashlib.sha256(raw).hexdigest() == declaration_pin["sha256"]
    assert canary_gates._CONFIRMATION_LOOK_DECLARATION["sha256"] == (
        declaration_pin["sha256"]
    )

    body = json.loads(raw)
    assert body["declaration_id"] == "confirmation-v2-final-look-20260828-2130z-r2"
    # Declared strictly before activation -- the property the loader enforces
    # and the reason the deviation could not be folded in here.
    assert body["declared_at"] < body["assignment"]["activated_at"]

    amendment = json.loads(
        (REPOSITORY_ROOT / canary_report._NEG_CONFIRMATION_AMENDMENT["path"]).read_bytes()
    )["amendments"][0]
    assert amendment["declaration_sha256"] == declaration_pin["sha256"]
    assert amendment["claim_strength"]["bargaining_and_persuasion_unaffected"] is True


def test_retired_epoch_check_is_diagnostic_and_verdicts_carry_the_label():
    gate = _neg_terminal_gate_from_rows(_promotable_gate_rows())
    pin = canary_report._NEG_CONFIRMATION_AMENDMENT

    assert gate["agent_confirmation"]["formal_promotion_gate"] is False
    assert gate["agent_confirmation"]["retired_by_amendment"] == pin["amendment_id"]
    assert "two_supported_nonnegative_treatment_epochs" not in gate["promotion"]["passes"]
    assert gate["amendment"] == pin

    # The verdict itself must never be readable as pre-registered evidence.
    assert gate["promotion"]["pre_registered"] is False
    assert gate["promotion"]["analysis_label"] == (
        "amended_analysis_not_fully_pre_registered"
    )
    assert gate["promotion"]["amendment_id"] == pin["amendment_id"]

    # The retained per-agent switchback unit still gates promotion.
    assert "balanced_manifest_switchback" in gate["promotion"]["passes"]
    assert "approved_prospective_manifest_assignment" in gate["promotion"]["passes"]


def test_retiring_the_epoch_check_cannot_promote_a_failing_switchback():
    """Dropping the unreachable check must not weaken the retained ones."""
    gate = _neg_terminal_gate_from_rows(_promotable_gate_rows(switchback=False))

    assert gate["switchback_confirmation"]["pass"] is False
    assert gate["promotion"]["status"] == "screen_pass"
    assert "balanced_manifest_switchback" in gate["promotion"]["failed_checks"]


def _health_from_epoch(*, treatment, control):
    """Drive the negotiation health check straight from per-arm counters."""

    def leaf(counts):
        base = {
            "traffic_events": 20000,
            "errors": 0,
            "invalid_results": 0,
            "invalid_moves": 0,
            "invalid_terminals": 0,
            "provenance_faults": 0,
            "corrections": 0,
        }
        base.update(counts)
        base["invalid_results"] = (
            base["invalid_moves"]
            + base["invalid_terminals"]
            + base["provenance_faults"]
        )
        return base

    return canary_report._neg_gate_health(
        [],
        None,
        None,
        epoch_health={"treatment": leaf(treatment), "control": leaf(control)},
    )


def test_shared_baseline_reporter_faults_do_not_hard_fail():
    """The live 7-vs-4 split is reporter-side evidence quality, not harm."""
    health = _health_from_epoch(
        treatment={"provenance_faults": 7},
        control={"provenance_faults": 4},
    )

    assert health["treatment_validity_faults"] == 0
    assert health["treatment_reported_validity_faults"] == 7
    assert health["checks"]["treatment_invalid_and_corrections_clean"] is True
    assert health["checks"]["reporter_fault_excess_within_0.01"] is True
    assert health["hard_fail"] is False
    assert health["pass"] is True


def test_treatment_caused_invalidity_still_hard_fails_at_zero_tolerance():
    for counts in ({"invalid_moves": 1}, {"corrections": 1}):
        health = _health_from_epoch(treatment=counts, control={})

        assert health["treatment_validity_faults"] == 1
        assert health["checks"]["treatment_invalid_and_corrections_clean"] is False
        assert health["hard_fail"] is True
        assert health["pass"] is False


def test_treatment_specific_reporter_fault_explosion_hard_fails():
    """Relaxing zero tolerance must not blind the gate to a real divergence."""
    health = _health_from_epoch(
        treatment={"invalid_terminals": 400},
        control={"invalid_terminals": 4},
    )

    assert health["reporter_fault_excess"] > 0.010
    assert health["checks"]["reporter_fault_excess_within_0.01"] is False
    assert health["hard_fail"] is True


def test_invalid_result_components_reconcile_with_the_retained_total():
    health = _health_from_epoch(
        treatment={"invalid_moves": 2, "invalid_terminals": 3, "provenance_faults": 5},
        control={},
    )
    treatment = health["by_variant"]["treatment"]

    assert treatment["invalid_results"] == 10
    assert health["treatment_validity_faults"] == 2
    assert health["reporter_faults"]["treatment"] == 8
