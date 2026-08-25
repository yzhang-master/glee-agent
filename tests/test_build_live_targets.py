"""Offline regression tests for live acceptance-curve round labeling."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

from glee_agent.theory import targets as T

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_live_targets", ROOT / "scripts" / "build_live_targets.py"
)
build_live_targets = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(build_live_targets)


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE games (
            game_id TEXT, family TEXT, outcome TEXT, agreed_round INTEGER,
            config_json TEXT, opp_type TEXT, your_player TEXT,
            my_payoff REAL, opp_payoff REAL
        );
        CREATE TABLE turns (
            game_id TEXT, family TEXT, your_player TEXT, round INTEGER,
            action_json TEXT, action_type TEXT, opp_type TEXT, state_json TEXT
        );
        """
    )
    return con


def _add_game(
    con: sqlite3.Connection,
    game_id: str,
    family: str,
    agreed_round: int,
    *,
    max_rounds: int = 4,
) -> None:
    con.execute(
        "INSERT INTO games "
        "(game_id, family, outcome, agreed_round, config_json, opp_type) "
        "VALUES (?, ?, 'agreement', ?, ?, 'agent')",
        (
            game_id,
            family,
            agreed_round,
            json.dumps({"max_rounds": max_rounds, "money_to_divide": 100}),
        ),
    )


def _add_turn(
    con: sqlite3.Connection,
    game_id: str,
    family: str,
    round_: int,
    action_type: str,
    action: dict,
) -> None:
    con.execute(
        "INSERT INTO turns "
        "(game_id, family, your_player, round, action_json, action_type, opp_type) "
        "VALUES (?, ?, 'player_1', ?, ?, ?, 'agent')",
        (game_id, family, round_, json.dumps(action), action_type),
    )


def _barg_key(share: float, rounds_left: str) -> str:
    return T._dumps(
        {
            "share_bucket": share,
            "rounds_left_bucket": rounds_left,
            "human": False,
        }
    )


def _neg_key(rel: float, rounds_left: str) -> str:
    return T._dumps(
        {
            "rel_bucket": rel,
            "role": "buyer",
            "rounds_left_bucket": rounds_left,
            "human": False,
        }
    )


def test_bargaining_labels_same_round_accept_and_prior_offer_rejection(monkeypatch):
    con = _db()
    monkeypatch.setattr(build_live_targets, "MIN_BUCKET_N", 1)
    _add_game(con, "barg", "bargaining", agreed_round=3, max_rounds=5)

    # The round-2 offer was rejected; agreement on round 3 accepted only the
    # round-3 offer. Their horizon buckets use inclusive rounds remaining.
    _add_turn(
        con, "barg", "bargaining", 2, "offer",
        {"player_1_gain": 80, "player_2_gain": 20},
    )
    _add_turn(
        con, "barg", "bargaining", 3, "offer",
        {"player_1_gain": 60, "player_2_gain": 40},
    )
    # An offer-shaped payload on a decision must not become an observation.
    _add_turn(
        con, "barg", "bargaining", 3, "decision",
        {"player_1_gain": 50, "player_2_gain": 50},
    )

    barg, neg = build_live_targets.build_accept_curves(con, {})

    assert neg == {}
    assert barg == {
        _barg_key(0.20, "4+"): [1, 0],
        _barg_key(0.40, "2-3"): [1, 1],
    }


def test_negotiation_labels_initial_offer_and_decision_counter(monkeypatch):
    con = _db()
    monkeypatch.setattr(build_live_targets, "MIN_BUCKET_N", 1)
    _add_game(con, "initial", "negotiation", agreed_round=1)
    _add_game(con, "counter", "negotiation", agreed_round=2)
    _add_game(con, "decision", "negotiation", agreed_round=1)

    # A standalone round-1 offer can be accepted in round 1.
    _add_turn(
        con, "initial", "negotiation", 1, "offer", {"product_price": 80}
    )
    # A RejectOffer counter made during decision round 1 is the round-2 offer.
    _add_turn(
        con,
        "counter",
        "negotiation",
        1,
        "decision",
        {"decision": "RejectOffer", "product_price": 85},
    )
    # Other decisions remain outcomes, even if they carry an offer-like price.
    _add_turn(
        con,
        "decision",
        "negotiation",
        1,
        "decision",
        {"decision": "AcceptOffer", "product_price": 90},
    )
    neg_state = {
        game_id: {
            "player_1_role": "seller",
            "player_1_value": 50,
            "player_2_role": "buyer",
            "player_2_value": 100,
        }
        for game_id in ("initial", "counter", "decision")
    }

    barg, neg = build_live_targets.build_accept_curves(con, neg_state)

    assert barg == {}
    assert neg == {
        _neg_key(0.80, "4+"): [1, 1],
        _neg_key(0.85, "2-3"): [1, 1],
    }


def _add_payoff_game(
    con: sqlite3.Connection,
    game_id: str,
    family: str,
    role: str,
    config: dict,
    my_payoff: float,
    opp_payoff: float,
) -> None:
    con.execute(
        "INSERT INTO games "
        "(game_id, family, outcome, config_json, your_player, my_payoff, opp_payoff) "
        "VALUES (?, ?, 'agreement', ?, ?, ?, ?)",
        (game_id, family, json.dumps(config), role, my_payoff, opp_payoff),
    )


def test_role_marginal_payoff_pools_never_include_opponent_payoff():
    con = _db()
    hidden_barg = {
        "money_to_divide": 100,
        "delta_1": 0.9,
        "delta_2": None,
        "max_rounds": 12,
        "horizon_known": True,
        "messages_allowed": False,
        "complete_information": False,
    }
    blind_seller = {
        "product_price": 100,
        "p": 0.5,
        "v": None,
        "u": None,
        "total_rounds": 20,
        "seller_message_type": "binary",
    }
    _add_payoff_game(
        con, "hidden-barg", "bargaining", "player_1", hidden_barg, 61, 39
    )
    _add_payoff_game(
        con, "blind-seller", "persuasion", "player_1", blind_seller, 700, -200
    )

    pools, n_games = build_live_targets.build_payoff_pools(con, {})

    barg_key = T.config_key_bargaining_marginal(hidden_barg, "player_1")
    pers_key = T.config_key_persuasion_marginal(blind_seller, "player_1")
    assert pools["bargaining"][barg_key] == {"player_1": [61.0]}
    assert pools["persuasion"][pers_key] == {"player_1": [700.0]}
    assert n_games == {"bargaining": 1, "persuasion": 1}


def test_exact_payoff_pool_keeps_opponent_field_sample():
    con = _db()
    config = {
        "money_to_divide": 100,
        "delta_1": 0.9,
        "delta_2": 0.8,
        "max_rounds": 12,
        "horizon_known": True,
        "messages_allowed": True,
        "complete_information": True,
    }
    _add_payoff_game(con, "exact", "bargaining", "player_2", config, 55, 45)

    pools, _ = build_live_targets.build_payoff_pools(con, {})

    exact_key = T.config_key_bargaining(config)
    assert pools["bargaining"][exact_key] == {
        "player_2": [55.0],
        "player_1": [45.0],
    }
