"""Focused regression tests for the raw-JSONL canary report."""

from __future__ import annotations

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


def _result(agent, gid, ts, result=None, *, valid=True, game_over=True, error=None):
    return {
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
        "resolved": direct is not None,
        "censored": direct is None,
        "terminal_reaped": False,
        "direct": direct,
        "effective_offer_round": 10,
        "normalized_payoff": normalized_payoff if direct is not None else None,
        "payoff_percentile": payoff_percentile if direct is not None else None,
        "compatibility_rate": None,
        "direction_violation": direction_violation,
        "assigned_match": True,
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
            "The report retains all strictly enrolled games from the experiment "
            "cutoff, including the disclosed pilot."
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


def test_frozen_negotiation_gate_promotes_only_with_joint_support_and_two_agents():
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
    assert gate["promotion"]["status"] == "promote"
    assert all(gate["promotion"]["passes"].values())


def test_fixed_label_evidence_is_capped_at_screen_pass_without_switchback():
    gate = _neg_terminal_gate_from_rows(
        _promotable_gate_rows(switchback=False)
    )

    assert gate["agent_confirmation"]["pass"] is True
    assert gate["switchback_confirmation"]["pass"] is False
    assert gate["promotion"]["passes"]["balanced_manifest_switchback"] is False
    assert gate["promotion"]["status"] == "screen_pass"
    assert gate["promotion"]["failed_checks"] == [
        "balanced_manifest_switchback"
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
    assert gate["promotion"]["status"] == "continue"
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


def test_frozen_gate_design_was_not_tuned_on_post_pilot_outcomes():
    assert NEG_TERMINAL_GATE_DESIGN["frozen_before_subsequent_outcomes"] is True
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
