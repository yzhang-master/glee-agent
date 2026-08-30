"""Bargaining late-game acceptance floor.

Rejecting a near-even offer is correct early and increasingly wrong late.
Measured per game (per-rejection aggregates double-count long games and
invert the sign): -0.013 pot from round 1, +0.034 from round 5, +0.085 from
round 15, +0.217 from round 30.
"""

from dataclasses import replace

from fixtures import bargaining_decision

from glee_agent.config import Knobs
from glee_agent.families import bargaining
from glee_agent.schema import parse_game

OFF = Knobs(llm_enabled=False)
LATE = replace(OFF, barg_late_accept_round=25, barg_late_accept_share=0.45)


def decide(game, knobs):
    return bargaining.decide(parse_game(game), knobs)


def near_even(round_no, my_share=0.49, pot=10000, delta=0.95):
    """A near-even offer on the table at a given round."""
    game = bargaining_decision()
    st = game["game_state"]
    st["round"] = round_no
    st["money_to_divide"] = pot
    st["horizon_known"] = False
    st["max_rounds"] = None
    st["delta_1"] = delta
    st["delta_2"] = 1.0
    st["complete_information"] = True
    st["proposer"] = "player_2"
    st["last_offer"] = {
        "player_1_gain": pot * my_share,
        "player_2_gain": pot * (1 - my_share),
        "proposer": "player_2",
        "round": round_no,
    }
    return game


def test_late_accept_is_off_by_default():
    assert Knobs().barg_late_accept_round == 0
    for rnd in (1, 10, 30, 60):
        assert decide(near_even(rnd), OFF) == decide(
            near_even(rnd), replace(OFF, barg_late_accept_round=0))


def test_late_accept_fires_only_at_depth():
    early = decide(near_even(5), LATE)
    late = decide(near_even(30), LATE)
    assert late["decision"] == "accept"
    # Early rounds keep whatever the unmodified policy did.
    assert early == decide(near_even(5), OFF)


def test_late_accept_respects_its_share_floor():
    # Below the floor the late rule must not fire.
    low = decide(near_even(40, my_share=0.30), LATE)
    assert low == decide(near_even(40, my_share=0.30), OFF)
    assert decide(near_even(40, my_share=0.46), LATE)["decision"] == "accept"


def test_late_accept_boundary_round_is_inclusive():
    assert decide(near_even(24), LATE) == decide(near_even(24), OFF)
    assert decide(near_even(25), LATE)["decision"] == "accept"


def test_late_accept_never_overrides_a_better_existing_accept():
    """It only ever converts a reject into an accept, never the reverse."""
    for rnd in (1, 5, 15, 25, 40, 80):
        for share in (0.20, 0.35, 0.45, 0.49, 0.60, 0.75):
            base = decide(near_even(rnd, my_share=share), OFF)
            new = decide(near_even(rnd, my_share=share), LATE)
            if base.get("decision") == "accept":
                assert new.get("decision") == "accept", (rnd, share)


def test_late_accept_threshold_is_configurable():
    strict = replace(OFF, barg_late_accept_round=25, barg_late_accept_share=0.55)
    assert decide(near_even(30, my_share=0.49), strict) == decide(
        near_even(30, my_share=0.49), OFF)
    assert decide(near_even(30, my_share=0.56), strict)["decision"] == "accept"
