"""Focused regression tests for the raw-JSONL canary report."""

from __future__ import annotations

import json

import pytest

from scripts.canary_report import (
    EXPERIMENTS,
    Experiment,
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
        "your_player": "player_1",
        "phase": phase,
        "opponent": {"type": "hidden", "name": None},
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
