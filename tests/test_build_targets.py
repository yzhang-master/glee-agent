"""Regression tests for dataset role-marginal payoff pools."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from glee_agent.theory import targets as T

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_targets", ROOT / "scripts" / "build_targets.py"
)
build_targets = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(build_targets)


def test_incomplete_bargaining_emits_one_pool_per_visible_role():
    agg = build_targets.Agg()
    game_args = {
        "money_to_divide": 100,
        "delta_1": 0.9,
        "delta_2": 0.8,
        "max_rounds": 12,
        "messages_allowed": False,
        "complete_information": False,
    }
    rows = [
        {
            "alice_gain": "60",
            "bob_gain": "40",
            "decision": "",
            "round": "1",
            "player": "Alice",
        },
        {
            "alice_gain": "",
            "bob_gain": "",
            "decision": "accept",
            "round": "1",
            "player": "Bob",
        },
    ]

    build_targets.do_bargaining(
        {}, game_args, rows, agg, False, False, None, None
    )

    canonical = {
        "money_to_divide": 100.0,
        "delta_1": 0.9,
        "delta_2": 0.8,
        "max_rounds": 12,
        "horizon_known": True,
        "messages_allowed": False,
        "complete_information": False,
    }
    exact = agg.cfg["bargaining"][T.config_key_bargaining(canonical)]
    p1 = agg.cfg["bargaining"][
        T.config_key_bargaining_marginal(canonical, "player_1")
    ]
    p2 = agg.cfg["bargaining"][
        T.config_key_bargaining_marginal(canonical, "player_2")
    ]
    assert exact["p1"] == [60.0]
    assert exact["p2"] == [40.0]
    assert p1["p1"] == [60.0]
    assert p1["p2"] == []
    assert p2["p1"] == []
    assert p2["p2"] == [40.0]


def test_blind_persuasion_emits_seller_only_marginal_pool():
    agg = build_targets.Agg()
    game_args = {
        "product_price": 100,
        "p": 0.5,
        "v": 1.2,
        "c": 0,
        "total_rounds": 20,
        "seller_message_type": "binary",
        "is_seller_know_cv": False,
    }
    config = {"player_2_args": {"public_name": "Bob"}}
    rows = [
        {"player": "Nature", "product_worth": "120", "decision": ""},
        {"player": "Bob", "product_worth": "", "decision": "yes"},
    ]

    build_targets.do_persuasion(
        config, game_args, rows, agg, False, False, None, None
    )

    exact_key = T.config_key_persuasion({
        "product_price": 100,
        "p": 0.5,
        "v": 120,
        "u": 0,
        "total_rounds": 20,
        "seller_message_type": "binary",
    })
    marginal_key = T.config_key_persuasion_marginal(game_args, "player_1")
    exact = agg.cfg["persuasion"][exact_key]
    marginal = agg.cfg["persuasion"][marginal_key]
    assert exact["p1"] == [100.0]
    assert exact["p2"] == [20.0]
    assert marginal["p1"] == [100.0]
    assert marginal["p2"] == []
