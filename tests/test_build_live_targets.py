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
            config_json TEXT, opp_type TEXT
        );
        CREATE TABLE turns (
            game_id TEXT, family TEXT, your_player TEXT, round INTEGER,
            action_json TEXT, action_type TEXT, opp_type TEXT
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
        "INSERT INTO games VALUES (?, ?, 'agreement', ?, ?, 'agent')",
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
        "INSERT INTO turns VALUES (?, ?, 'player_1', ?, ?, ?, 'agent')",
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
