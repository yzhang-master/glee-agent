"""Bargaining strategy: Rubinstein/backward-induction bound + Boulware
concession, with a human-fairness branch.

Share convention: fraction of the CURRENT pot that goes to me.
"""

from __future__ import annotations

from ..config import Knobs
from ..schema import GameView, parse_bargaining
from ..theory.concession import boulware
from ..theory.rubinstein import finite_spe_share, infinite_spe_share

# Discount grid used when the opponent's delta is hidden (GLEE-style values);
# refined from the dataset in a later version.
HIDDEN_DELTA_PESSIMISTIC = 0.95


def _gain_names(view: GameView) -> tuple[str, str]:
    names = {1: "alice_gain", 2: "bob_gain"}
    return names[view.my_index], names[3 - view.my_index]


def _rounds_left(view: GameView) -> int | None:
    if view.max_rounds is None:
        return None
    return max(view.max_rounds - view.round + 1, 1)


def _spe_bound(view: GameView, my_delta: float, opp_delta: float) -> float:
    """My SPE share as the current proposer."""
    left = _rounds_left(view)
    if left is None:
        return infinite_spe_share(my_delta, opp_delta)
    # Cap recursion depth; beyond ~40 rounds the finite game ≈ infinite one.
    if left > 40:
        return infinite_spe_share(my_delta, opp_delta)
    return finite_spe_share(left, my_delta, opp_delta)


def _my_rejected_offers(view: GameView) -> int:
    count = 0
    for entry in view.history:
        if not isinstance(entry, dict):
            continue
        if entry.get("proposer") == view.your_player and entry.get("decision") == "reject":
            count += 1
    return count


def _target_share(view: GameView, b, knobs: Knobs) -> float:
    """Boulware-decaying target share for my offers this round."""
    human = view.opponent_type == "human"
    opp_delta = b.opp_delta if b.opp_delta is not None else HIDDEN_DELTA_PESSIMISTIC
    spe = _spe_bound(view, b.my_delta, opp_delta)

    if human:
        start, floor = knobs.barg_anchor_human, knobs.barg_floor_human
    else:
        # SPE can sit very high near an endgame that favors me; cap the greed —
        # real opponents reject "rational" lowballs and deadlock costs the pot.
        start, floor = knobs.barg_anchor_agent, max(knobs.barg_floor_agent, min(spe, 0.70))

    # Effective schedule length: horizon if known, else a virtual deadline set
    # by impatience — with heavy discounting the pot decays fast, settle early.
    horizon = _rounds_left(view)
    joint_delta = min(b.my_delta, opp_delta)
    virtual = 3 if joint_delta < 0.9 else (6 if joint_delta < 1.0 else 10)
    T = min(horizon, virtual) if horizon is not None else virtual

    # Round index within MY schedule: each of my proposals advances it.
    my_round = (view.round + 1) // 2
    target = boulware(my_round, T, start, floor, knobs.barg_beta)

    # Deadlock pressure: every rejected offer of mine says my price is too
    # high — concede beyond the schedule rather than ride into the endgame.
    rejected = _my_rejected_offers(view)
    if rejected >= 2:
        target = max(target - 0.05 * (rejected - 1), 0.50)

    # Endgame parity: when the horizon is near, SPE is what a rational
    # opponent will actually hold out for. If their continuation is worth
    # more than my aspiration leaves them, they reject and I face their
    # ultimatum — so near the end the SPE becomes an UPPER cap on my share.
    left = _rounds_left(view)
    if left is not None and left <= 4:
        cap = max(spe, 0.45 if human else 0.25)
        target = min(target, cap)
    return target


def _continuation_value(view: GameView, b, knobs: Knobs) -> float:
    """My expected share (of the CURRENT pot) from rejecting: next round I
    propose, discounted by my delta; 0 if this is the last round."""
    left = _rounds_left(view)
    if left is not None and left <= 1:
        return 0.0
    opp_delta = b.opp_delta if b.opp_delta is not None else HIDDEN_DELTA_PESSIMISTIC
    # Next round I'm the proposer; assume I secure roughly my next target,
    # bounded by what an SPE proposer could get. SPE overstates what real
    # opponents accept (they reject "rational" lowballs), so haircut the
    # whole thing by the odds my aggressive counter actually lands.
    next_view_round = view.round + 1
    spe_next = _spe_bound_at(view, next_view_round, b.my_delta, opp_delta)
    target_next = min(_target_share_at(view, b, knobs, next_view_round), max(spe_next, 0.5))
    return b.my_delta * target_next * knobs.barg_cont_realism


def _spe_bound_at(view: GameView, round_: int, my_delta: float, opp_delta: float) -> float:
    if view.max_rounds is None:
        return infinite_spe_share(my_delta, opp_delta)
    left = max(view.max_rounds - round_ + 1, 1)
    if left > 40:
        return infinite_spe_share(my_delta, opp_delta)
    return finite_spe_share(left, my_delta, opp_delta)


def _target_share_at(view: GameView, b, knobs: Knobs, round_: int) -> float:
    saved = view.round
    try:
        view.round = round_
        return _target_share(view, b, knobs)
    finally:
        view.round = saved


def decide(view: GameView, knobs: Knobs) -> dict:
    b = parse_bargaining(view)
    mine_key, opp_key = _gain_names(view)

    if view.action_type == "offer":
        pot = b.money
        share = _target_share(view, b, knobs)

        # Final-round proposer: offer the empirical minimum the field accepts,
        # not epsilon — spite rejection is real.
        left = _rounds_left(view)
        if left is not None and left <= 1:
            share = 1.0 - knobs.barg_final_round_give

        my_gain = pot * share
        # Humans accept clean numbers more readily.
        if view.opponent_type == "human" and pot >= 100:
            my_gain = round(my_gain / 10) * 10
        my_gain = min(max(my_gain, 0.0), pot)
        return {mine_key: my_gain, opp_key: pot - my_gain}

    # Decision phase.
    offered = b.last_offer_my_gain
    if offered is None:
        return {"decision": "accept"}  # can't evaluate: accepting beats risking timeout

    pot = b.money if b.money > 0 else (offered + (b.last_offer_opp_gain or 0.0))
    offered_share = offered / pot if pot > 0 else 0.5

    left = _rounds_left(view)
    if left is not None and left <= 1:
        # Last decision of the game: anything >= 0 beats the $0 no-deal.
        return {"decision": "accept"}

    # An offer this good is already top-of-field under percentile scoring;
    # take the sure thing over a risky squeeze.
    if offered_share >= knobs.barg_accept_great:
        return {"decision": "accept"}

    # Endgame parity: in the decision phase the current proposer is the
    # opponent, so they also propose the final round iff (max_rounds - round)
    # is even. Facing their ultimatum power, a modest offer now beats the
    # near-zero share their last proposal will carry.
    if left is not None and left <= 4:
        opp_proposes_last = (view.max_rounds - view.round) % 2 == 0
        if opp_proposes_last and offered_share >= 0.35:
            return {"decision": "accept"}

    cont = _continuation_value(view, b, knobs)
    # Accept when the offer beats waiting (small tolerance avoids knife-edge
    # rejections over rounding).
    if offered_share >= cont - 0.02:
        return {"decision": "accept"}
    return {"decision": "reject"}
