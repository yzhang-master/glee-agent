"""The guard is a total function: never raises, output always valid.
Includes a deterministic fuzz over mutated/garbage game dicts."""

import copy
import json
import math
import random
from types import SimpleNamespace

import pytest

from fixtures import (
    bargaining_decision,
    bargaining_game,
    negotiation_decision,
    negotiation_game,
    persuasion_game,
)

import glee_agent.dispatcher as dispatcher
from glee_agent.config import Knobs
from glee_agent.guard import MAX_MESSAGE, fallback_action, guard
from glee_agent.schema import parse_game, parse_negotiation


def check_valid(action: dict, game: dict) -> None:
    """Assert `action` matches the documented server contract for the turn."""
    va = game.get("valid_actions") or {}
    atype = va.get("type")
    family = game.get("game_family")
    assert isinstance(action, dict) and action, f"empty action for {atype}"
    json.dumps(action)  # must be JSON-serializable

    msg = action.get("message")
    if msg is not None:
        assert isinstance(msg, str) and len(msg) <= 2000

    if atype == "offer" and family == "bargaining":
        state = game.get("game_state") or {}
        pot = state.get("money_to_divide")
        a, b = action.get("alice_gain"), action.get("bob_gain")
        assert isinstance(a, (int, float)) and isinstance(b, (int, float))
        assert a >= 0 and b >= 0
        if isinstance(pot, (int, float)) and math.isfinite(pot) and pot > 0:
            assert abs((a + b) - pot) < 1e-6, f"gains {a}+{b} != pot {pot}"
    elif atype == "offer" and family == "negotiation":
        assert isinstance(action.get("product_price"), (int, float))
        assert math.isfinite(action["product_price"])
        assert action["product_price"] >= 0
        state = parse_negotiation(parse_game(game))
        if state.my_value is not None:
            reservation = max(state.my_value, 0.0)
            if state.my_role == "seller":
                assert action["product_price"] >= reservation
            else:
                assert action["product_price"] <= reservation
    elif atype == "decision" and family == "bargaining":
        assert action.get("decision") in ("accept", "reject", "walkaway")
    elif atype == "decision" and family == "negotiation":
        assert action.get("decision") in ("AcceptOffer", "RejectOffer", "WalkAway")
        if "product_price" in action:
            assert isinstance(action["product_price"], (int, float))
            assert math.isfinite(action["product_price"])
            assert action["product_price"] >= 0
            state = parse_negotiation(parse_game(game))
            if state.my_value is not None:
                reservation = max(state.my_value, 0.0)
                if state.my_role == "seller":
                    assert action["product_price"] >= reservation
                else:
                    assert action["product_price"] <= reservation
    elif atype == "seller_message":
        assert isinstance(action.get("message"), str) and action["message"]
    elif atype in ("seller_recommendation", "buyer_decision"):
        assert action.get("decision") in ("yes", "no")


ALL_GAMES = [
    bargaining_game(),
    bargaining_decision(),
    negotiation_game(role="seller"),
    negotiation_game(role="buyer"),
    negotiation_decision(role="buyer"),
    negotiation_decision(role="seller"),
    persuasion_game(actor="seller"),
    persuasion_game(actor="buyer"),
]


class TestGuardBasics:
    def test_valid_actions_pass_through(self):
        game = bargaining_game()
        action, notes = guard({"alice_gain": 600, "bob_gain": 400}, parse_game(game))
        check_valid(action, game)
        assert action["alice_gain"] == 600
        assert not notes

    def test_fixes_sum_mismatch(self):
        game = bargaining_game()
        action, _ = guard({"alice_gain": 600, "bob_gain": 500}, parse_game(game))
        check_valid(action, game)

    def test_clamps_negative_gain(self):
        game = bargaining_game()
        action, notes = guard({"alice_gain": -50, "bob_gain": 1050}, parse_game(game))
        check_valid(action, game)
        assert action["alice_gain"] >= 0

    def test_string_numbers_coerced(self):
        game = bargaining_game()
        action, _ = guard({"alice_gain": "$600", "bob_gain": "400"}, parse_game(game))
        check_valid(action, game)

    def test_blocks_losing_negotiation_accept(self):
        # Buyer value 100, offered 150: guard must block AcceptOffer.
        game = negotiation_decision(role="buyer", offer_price=150.0)
        action, notes = guard({"decision": "AcceptOffer"}, parse_game(game))
        check_valid(action, game)
        assert action["decision"] != "AcceptOffer"
        assert any("negative-payoff" in n for n in notes)

    def test_message_truncated(self):
        game = bargaining_game()
        action, _ = guard(
            {"alice_gain": 600, "bob_gain": 400, "message": "x" * 5000}, parse_game(game)
        )
        check_valid(action, game)
        assert len(action["message"]) <= MAX_MESSAGE

    def test_bad_decision_string_replaced(self):
        game = bargaining_decision()
        action, _ = guard({"decision": "ACCEPT!!"}, parse_game(game))
        check_valid(action, game)

    def test_persuasion_yes_no_normalized(self):
        game = persuasion_game(actor="buyer")
        action, _ = guard({"decision": "YES"}, parse_game(game))
        check_valid(action, game)
        assert action["decision"] == "yes"

    def test_none_action(self):
        for game in ALL_GAMES:
            action, _ = guard(None, parse_game(game))
            check_valid(action, game)

    def test_fallbacks_are_valid(self):
        for game in ALL_GAMES:
            action = fallback_action(parse_game(game))
            action, _ = guard(action, parse_game(game))
            check_valid(action, game)

    def test_negative_negotiation_reservation_fallbacks_are_clamped(self):
        offer = negotiation_game(
            role="seller", game_state={"player_1_value": -1e308}
        )
        decision = negotiation_decision(
            role="seller",
            game_state={"player_1_value": -1e308, "last_offer": None},
        )

        assert fallback_action(parse_game(offer))["product_price"] == 0.0
        assert fallback_action(parse_game(decision)) == {
            "decision": "RejectOffer",
            "product_price": 0.0,
        }

    def test_negative_negotiation_reservation_guard_counter_is_clamped(self):
        game = negotiation_decision(
            role="seller", game_state={"player_1_value": -1e308}
        )

        action, notes = guard(
            {"decision": "RejectOffer", "product_price": float("nan")},
            parse_game(game),
        )

        check_valid(action, game)
        assert action["product_price"] == 0.0
        assert any("added missing counteroffer" in note for note in notes)

    @pytest.mark.parametrize(
        ("role", "proposed", "expected", "note"),
        [
            ("seller", 50.0, 100.0, "raised seller offer"),
            ("buyer", 150.0, 100.0, "lowered buyer offer"),
        ],
    )
    def test_negotiation_offer_guard_enforces_own_reservation(
        self, role, proposed, expected, note
    ):
        game = negotiation_game(role=role)

        action, notes = guard({"product_price": proposed}, parse_game(game))

        check_valid(action, game)
        assert action["product_price"] == expected
        assert any(note in correction for correction in notes)

    @pytest.mark.parametrize(
        ("role", "proposed", "expected", "note"),
        [
            ("seller", 50.0, 100.0, "raised seller counteroffer"),
            ("buyer", 150.0, 100.0, "lowered buyer counteroffer"),
        ],
    )
    def test_negotiation_counter_guard_enforces_own_reservation(
        self, role, proposed, expected, note
    ):
        game = negotiation_decision(role=role, offer_price=150.0)

        action, notes = guard(
            {"decision": "RejectOffer", "product_price": proposed},
            parse_game(game),
        )

        check_valid(action, game)
        assert action["product_price"] == expected
        assert any(note in correction for correction in notes)

    @pytest.mark.parametrize("action_type", ["offer", "counter"])
    @pytest.mark.parametrize(
        ("role", "proposed", "expected", "note"),
        [
            ("seller", 1000.006, 1000.005, "buyer reservation"),
            ("buyer", 1000.0, 1000.001, "seller reservation"),
        ],
    )
    def test_negotiation_guard_enforces_public_opponent_reservation(
        self, action_type, role, proposed, expected, note
    ):
        state = {
            "complete_information": True,
            "round": 1,
            "max_rounds": 2,
            "messages_allowed": False,
            "player_1_value": 1000.001,
            "player_2_value": 1000.005,
        }
        if action_type == "offer":
            game = negotiation_game(role=role, game_state=state)
            proposed_action = {"product_price": proposed}
        else:
            incoming = 1000.0 if role == "seller" else 1000.006
            game = negotiation_decision(
                role=role, offer_price=incoming, game_state=state
            )
            proposed_action = {
                "decision": "RejectOffer",
                "product_price": proposed,
            }

        action, notes = guard(proposed_action, parse_game(game))

        assert action["product_price"] == expected
        assert any(note in correction for correction in notes)

    @pytest.mark.parametrize("action_type", ["offer", "counter"])
    @pytest.mark.parametrize("role", ["seller", "buyer"])
    def test_dispatcher_preserves_subcent_feasible_price(
        self, monkeypatch, action_type, role
    ):
        seller_value, buyer_value = 1000.001, 1000.005
        state = {
            "complete_information": True,
            "round": 2 if action_type == "offer" else 1,
            "max_rounds": 2,
            "messages_allowed": False,
            "player_1_value": seller_value,
            "player_2_value": buyer_value,
        }
        if action_type == "offer":
            game = negotiation_game(role=role, game_state=state)
        else:
            incoming = (
                seller_value - 0.001
                if role == "seller"
                else buyer_value + 0.001
            )
            game = negotiation_decision(
                role=role, offer_price=incoming, game_state=state
            )

        logged = []
        monkeypatch.setattr(
            dispatcher,
            "log_turn",
            lambda *args, **kwargs: logged.append((args, kwargs)),
        )
        strategy = dispatcher.build_strategy(
            SimpleNamespace(
                knobs=Knobs(llm_enabled=False),
                agent_label="subcent-test",
            )
        )

        action = strategy(game)

        if action_type == "counter":
            assert action["decision"] == "RejectOffer"
        price = action["product_price"]
        assert seller_value < price < buyer_value
        assert price == pytest.approx(1000.003)
        assert price != round(price, 2)
        assert logged[0][0][3] == []


class TestGuardFuzz:
    def test_dispatcher_handles_nonfinite_game_numbers(self, monkeypatch):
        monkeypatch.setattr(dispatcher, "log_turn", lambda *args, **kwargs: None)
        strategy = dispatcher.build_strategy(
            SimpleNamespace(knobs=Knobs(llm_enabled=False), agent_label="test")
        )
        cases = [
            (
                bargaining_game,
                ["round", "max_rounds", "money_to_divide", "delta_1", "delta_2"],
            ),
            (
                negotiation_game,
                ["round", "max_rounds", "player_1_value", "player_2_value"],
            ),
            (
                lambda: persuasion_game(actor="buyer"),
                [
                    "round",
                    "max_rounds",
                    "product_price",
                    "p",
                    "v",
                    "u",
                    "total_rounds",
                    "buyer_total_payoff",
                ],
            ),
        ]

        for raw in (float("nan"), float("inf"), float("-inf"), "nan", "inf", "-inf"):
            for factory, fields in cases:
                game = factory()
                game["game_state"].update({field: raw for field in fields})

                action = strategy(game)

                check_valid(action, game)
                json.dumps(action, allow_nan=False)
                for value in action.values():
                    if isinstance(value, float):
                        assert math.isfinite(value)

    def test_dispatcher_exception_uses_clamped_negotiation_fallback(self, monkeypatch):
        def fail_strategy(*args, **kwargs):
            raise OverflowError("synthetic strategy failure")

        monkeypatch.setitem(dispatcher.FAMILIES, "negotiation", fail_strategy)
        monkeypatch.setattr(dispatcher, "log_turn", lambda *args, **kwargs: None)
        strategy = dispatcher.build_strategy(
            SimpleNamespace(knobs=Knobs(llm_enabled=False), agent_label="test")
        )
        game = negotiation_decision(
            role="seller",
            game_state={"player_1_value": -1e308, "last_offer": None},
        )

        action = strategy(game)

        check_valid(action, game)
        assert action == {"decision": "RejectOffer", "product_price": 0.0}

    def test_garbage_actions_never_raise(self):
        garbage = [
            None, 42, "accept", [], {}, {"decision": None}, {"decision": 7},
            {"alice_gain": float("nan")}, {"alice_gain": float("inf"), "bob_gain": 1},
            {"product_price": "lots"}, {"message": 123}, {"unknown_field": True},
            {"decision": "walkaway", "product_price": -5},
        ]
        for game in ALL_GAMES:
            for g in garbage:
                action, _ = guard(copy.deepcopy(g), parse_game(game))
                check_valid(action, game)

    def test_mutated_games_never_raise(self):
        rng = random.Random(42)
        mutations = 300
        for _ in range(mutations):
            game = copy.deepcopy(rng.choice(ALL_GAMES))
            # Randomly delete or corrupt fields at both levels.
            for target in (game, game.get("game_state") or {}):
                keys = list(target.keys())
                if not keys:
                    continue
                key = rng.choice(keys)
                op = rng.random()
                if op < 0.4:
                    del target[key]
                elif op < 0.7:
                    target[key] = rng.choice([None, "garbage", -1, [], {}])
                else:
                    target[key] = rng.choice([0, "", float("nan")])
            view = parse_game(game)
            action, _ = guard({"decision": "accept"}, view)
            assert isinstance(action, dict) and action
            json.dumps(action, default=str)

    def test_game_not_a_dict(self):
        for bad in (None, "hi", 7, []):
            view = parse_game(bad)
            action, _ = guard({"decision": "accept"}, view)
            assert isinstance(action, dict) and action
