"""Incomplete-information opening price exploration.

The production policy emits one deterministic price per value cell, which
censors the acceptance curve: every price comparison in the logs collapses
into a comparison between different game cells.  These tests pin the
exploration arm that breaks that determinism, and pin that it is completely
inert until switched on.
"""

from dataclasses import replace

from fixtures import negotiation_game

from glee_agent.config import Knobs
from glee_agent.families import negotiation
from glee_agent.schema import parse_game

OFF = Knobs(llm_enabled=False)
ON = replace(OFF, neg_ii_explore_frac=1.0)


def price(game, knobs):
    action = negotiation.decide(parse_game(game), knobs)
    return action.get("product_price")


def seller_game(gid, value=100.0, **over):
    game = negotiation_game(role="seller", **over)
    game["game_id"] = gid
    game["game_state"]["player_1_value"] = value
    return game


def test_exploration_is_off_by_default():
    assert Knobs().neg_ii_explore_frac == 0.0
    for gid in ("a", "b", "c", "d", "e"):
        assert price(seller_game(gid), OFF) == price(seller_game(gid), Knobs(
            llm_enabled=False, neg_ii_explore_frac=0.0))


def test_exploration_changes_the_opening_when_enabled():
    changed = sum(
        1 for i in range(200)
        if price(seller_game(f"g{i}"), ON) != price(seller_game(f"g{i}"), OFF)
    )
    # The default anchor sits on a ladder rung, so a few draws coincide.
    assert changed > 150, changed


def test_exploration_is_deterministic_per_game_id():
    for gid in ("alpha", "beta", "gamma"):
        first = price(seller_game(gid), ON)
        for _ in range(5):
            assert price(seller_game(gid), ON) == first


def test_exploration_visits_every_rung():
    seen = set()
    for i in range(600):
        p = price(seller_game(f"rung{i}"), ON)
        if p is not None:
            seen.add(round(p / 100.0 - 1.0, 4))
    for rung in negotiation._II_EXPLORE_LADDER:
        assert rung in seen, (rung, sorted(seen))


def test_exploration_share_tracks_the_configured_fraction():
    knobs = replace(OFF, neg_ii_explore_frac=0.25)
    n = 800
    explored = sum(
        1 for i in range(n)
        if negotiation._ii_explore_markup(
            f"share{i}", _norm(seller_game(f"share{i}")), knobs) is not None
    )
    assert 0.20 * n <= explored <= 0.30 * n, explored


def _norm(game):
    from glee_agent.schema import parse_negotiation
    view = parse_game(game)
    return parse_negotiation(view)


def test_explored_seller_opening_never_prices_below_reservation():
    for i in range(300):
        game = seller_game(f"floor{i}", value=120.0)
        p = price(game, ON)
        assert p is not None and p >= 120.0, (i, p)


def test_explored_buyer_opening_never_prices_above_its_value():
    for i in range(300):
        game = negotiation_game(role="buyer")
        game["game_id"] = f"buyer{i}"
        game["game_state"]["player_2_value"] = 120.0
        p = price(game, ON)
        assert p is not None and p <= 120.0, (i, p)


def test_complete_information_games_are_untouched():
    for i in range(120):
        game = seller_game(f"ci{i}")
        game["game_state"]["complete_information"] = True
        game["game_state"]["player_2_value"] = 150.0
        assert price(game, ON) == price(game, OFF)


def test_missing_game_id_falls_back_to_the_anchor():
    game = seller_game("")
    assert price(game, ON) == price(game, OFF)
