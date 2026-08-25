"""Regression coverage for defensive numeric parsing."""

from __future__ import annotations

import pytest

from fixtures import bargaining_game, negotiation_game, persuasion_game

from glee_agent.schema import (
    parse_bargaining,
    parse_game,
    parse_negotiation,
    parse_persuasion,
)


NONFINITE_VALUES = [
    pytest.param(float("nan"), id="nan-number"),
    pytest.param(float("inf"), id="positive-infinity-number"),
    pytest.param(float("-inf"), id="negative-infinity-number"),
    pytest.param("nan", id="nan-string"),
    pytest.param("+Infinity", id="positive-infinity-string"),
    pytest.param("-inf", id="negative-infinity-string"),
]


@pytest.mark.parametrize("raw", NONFINITE_VALUES)
def test_parse_game_nonfinite_rounds_use_defaults(raw):
    view = parse_game(bargaining_game(game_state={"round": raw, "max_rounds": raw}))

    assert view.round == 1
    assert view.max_rounds is None


@pytest.mark.parametrize("raw", NONFINITE_VALUES)
def test_parse_bargaining_nonfinite_numbers_use_defaults(raw):
    view = parse_game(
        bargaining_game(
            game_state={
                "money_to_divide": raw,
                "delta_1": raw,
                "delta_2": raw,
                "last_offer": {
                    "player_1_gain": raw,
                    "player_2_gain": raw,
                },
            }
        )
    )

    state = parse_bargaining(view)
    assert state.money == 0.0
    assert state.my_delta == 1.0
    assert state.opp_delta is None
    assert state.last_offer_my_gain is None
    assert state.last_offer_opp_gain is None


@pytest.mark.parametrize("raw", NONFINITE_VALUES)
def test_parse_negotiation_nonfinite_numbers_use_defaults(raw):
    view = parse_game(
        negotiation_game(
            game_state={
                "player_1_value": raw,
                "player_2_value": raw,
                "last_offer": {"price": raw},
            }
        )
    )

    state = parse_negotiation(view)
    assert state.my_value is None
    assert state.opp_value is None
    assert state.last_offer_price is None


@pytest.mark.parametrize("raw", NONFINITE_VALUES)
def test_parse_persuasion_nonfinite_numbers_use_defaults(raw):
    view = parse_game(
        persuasion_game(
            game_state={
                "product_price": raw,
                "p": raw,
                "v": raw,
                "u": raw,
                "total_rounds": raw,
                "seller_total_payoff": raw,
            }
        )
    )

    state = parse_persuasion(view)
    assert state.price == 0.0
    assert state.p == 0.5
    assert state.v is None
    assert state.u == 0.0
    assert state.total_rounds is None
    assert state.my_total_payoff == 0.0


def test_finite_numeric_strings_still_parse():
    bargaining_view = parse_game(
        bargaining_game(
            game_state={
                "round": "3",
                "max_rounds": "12",
                "money_to_divide": "$1,000.50",
                "delta_1": "0.95",
                "delta_2": "0.90",
                "last_offer": {
                    "player_1_gain": "600.25",
                    "player_2_gain": "400.25",
                },
            }
        )
    )
    bargaining_state = parse_bargaining(bargaining_view)
    assert bargaining_view.round == 3
    assert bargaining_view.max_rounds == 12
    assert bargaining_state.money == 1000.5
    assert bargaining_state.my_delta == 0.95
    assert bargaining_state.opp_delta == 0.9
    assert bargaining_state.last_offer_my_gain == 600.25
    assert bargaining_state.last_offer_opp_gain == 400.25

    negotiation_state = parse_negotiation(
        parse_game(
            negotiation_game(
                game_state={
                    "player_1_value": "100",
                    "player_2_value": "150",
                    "last_offer": {"price": "$125.50"},
                }
            )
        )
    )
    assert negotiation_state.my_value == 100.0
    assert negotiation_state.opp_value == 150.0
    assert negotiation_state.last_offer_price == 125.5

    persuasion_state = parse_persuasion(
        parse_game(
            persuasion_game(
                game_state={
                    "product_price": "$100",
                    "p": "0.25",
                    "v": "200",
                    "u": "-10",
                    "total_rounds": "20",
                    "seller_total_payoff": "15.5",
                }
            )
        )
    )
    assert persuasion_state.price == 100.0
    assert persuasion_state.p == 0.25
    assert persuasion_state.v == 200.0
    assert persuasion_state.u == -10.0
    assert persuasion_state.total_rounds == 20
    assert persuasion_state.my_total_payoff == 15.5
