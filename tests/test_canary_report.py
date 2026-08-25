"""Focused regression tests for the raw-JSONL canary report."""

from __future__ import annotations

import hashlib
import json

import pytest

from scripts.canary_report import (
    EXPERIMENTS,
    NEG_TERMINAL_GATE_DESIGN,
    Experiment,
    _neg_terminal_gate_from_rows,
    build_report,
    discover_log_slices,
    iter_log_records,
    seek_timestamp,
)


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
        "game_id": f"{agent}-{value}",
        "cell": cell,
        "cell_id": cell_id,
        "supported": supported,
        "unsupported_reason": None if supported else f"role={role}",
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
        # A later reaper terminal supersedes the direct terminal.
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
    assert control["direct_converted"] == 1
    assert control["direct_resolved"] == 1
    assert control["mean_normalized_payoff"] == pytest.approx(0.30)
    assert control["normalized_payoff_sum"] == pytest.approx(0.30)
    assert control["normalized_payoff_sum_squares"] == pytest.approx(0.09)
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
    assert gate["promotion"]["status"] == "screen_pass"
    assert gate["promotion"]["passes"][
        "two_supported_nonnegative_treatment_epochs"
    ] is False


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
            {"outcome": "agreement", "player_1_payoff": 40},
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
            {"outcome": "agreement", "player_1_payoff": 40},
        ),
        _result(
            "test_b",
            "conflicting-itt-terminal",
            103,
            {"outcome": "agreement", "player_1_payoff": 90},
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

    assert treatment["affected"] == 3
    assert treatment["matured"] == 3
    assert treatment["resolved"] == 1
    assert treatment["censored"] == 2
    assert treatment["pending_maturation"] == 0
    assert treatment["direct_trials"] == 1
    assert treatment["late_terminals"] == 3


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
