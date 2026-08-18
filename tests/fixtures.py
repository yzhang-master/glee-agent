"""Synthetic game dicts mirroring the documented server payloads."""

from __future__ import annotations

import copy


def bargaining_game(**over) -> dict:
    game = {
        "game_id": "barg-1",
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "offer",
        "opponent": {"type": "agent", "name": "TestBot"},
        "game_state": {
            "phase": "offer",
            "current_player": "player_1",
            "proposer": "player_1",
            "round": 1,
            "max_rounds": 10,
            "horizon_known": True,
            "money_to_divide": 1000,
            "delta_1": 0.95,
            "delta_2": 0.9,
            "last_offer": None,
            "history": [],
            "messages_allowed": True,
            "complete_information": True,
        },
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number", "bob_gain": "number", "message": "string"}},
        "prompt": "You are Alice...",
    }
    return _merge(game, over)


def bargaining_decision(my_gain: float = 400, opp_gain: float = 600, **over) -> dict:
    game = bargaining_game(
        phase="decision",
        valid_actions={"type": "decision", "fields": {"decision": ["accept", "reject", "walkaway"]}},
    )
    game["game_state"]["phase"] = "decision"
    game["game_state"]["proposer"] = "player_2"
    game["game_state"]["last_offer"] = {
        "player_1_gain": my_gain,
        "player_2_gain": opp_gain,
        "message": "take it",
        "proposer": "player_2",
        "round": game["game_state"]["round"],
    }
    return _merge(game, over)


def negotiation_game(role: str = "seller", **over) -> dict:
    game = {
        "game_id": "neg-1",
        "game_family": "negotiation",
        "your_player": "player_1" if role == "seller" else "player_2",
        "phase": "offer",
        "opponent": {"type": "hidden", "name": None},
        "game_state": {
            "phase": "offer",
            "current_player": "player_1" if role == "seller" else "player_2",
            "player_1_role": "seller",
            "player_2_role": "buyer",
            f"player_{1 if role == 'seller' else 2}_value": 100.0,
            "round": 1,
            "max_rounds": 10,
            "horizon_known": True,
            "last_offer": None,
            "history": [],
            "messages_allowed": True,
            "complete_information": False,
        },
        "valid_actions": {"type": "offer", "fields": {"product_price": "number", "message": "string"}},
        "prompt": "You are the seller...",
    }
    return _merge(game, over)


def negotiation_decision(role: str = "buyer", offer_price: float = 90.0, **over) -> dict:
    game = negotiation_game(role=role)
    me = game["your_player"]
    opp = "player_2" if me == "player_1" else "player_1"
    game["phase"] = "decision"
    game["game_state"]["phase"] = "decision"
    game["game_state"]["current_player"] = me
    game["game_state"]["last_offer"] = {
        "price": offer_price,
        "message": "deal?",
        "from_player": opp,
        "round": game["game_state"]["round"],
    }
    game["valid_actions"] = {
        "type": "decision",
        "fields": {"decision": ["AcceptOffer", "RejectOffer", "WalkAway"], "product_price": "number"},
    }
    return _merge(game, over)


def persuasion_game(actor: str = "seller", quality: str = "high", **over) -> dict:
    is_seller = actor == "seller"
    game = {
        "game_id": "pers-1",
        "game_family": "persuasion",
        "your_player": "player_1" if is_seller else "player_2",
        "phase": "seller_message" if is_seller else "buyer_decision",
        "opponent": {"type": "agent", "name": "TestBot"},
        "game_state": {
            "phase": "seller_message" if is_seller else "buyer_decision",
            "current_player": "player_1" if is_seller else "player_2",
            "product_price": 100.0,
            "p": 0.5,
            "v": 120.0,
            "u": 0.0,
            "round": 1,
            "total_rounds": 20,
            "seller_message_type": "text",
            "history": [],
            "seller_total_payoff": 0,
            "buyer_total_payoff": 0,
        },
        "valid_actions": (
            {"type": "seller_message", "fields": {"message": "string"}}
            if is_seller
            else {"type": "buyer_decision", "fields": {"decision": ["yes", "no"]}}
        ),
        "prompt": "You are the seller..." if is_seller else "You are the buyer...",
    }
    if is_seller:
        game["game_state"]["current_quality"] = quality
    else:
        game["game_state"]["seller_message"] = "This is a great product!"
    return _merge(game, over)


def _merge(base: dict, over: dict) -> dict:
    game = copy.deepcopy(base)
    state_over = over.pop("game_state", None)
    game.update(over)
    if state_over:
        game["game_state"].update(state_over)
    return game
