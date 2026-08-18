"""Negotiation strategy: anchored Boulware concession toward reservation,
never accept a losing deal, final-round accept-anything-positive.

Prices are absolute; my payoff = price - value (seller) or value - price (buyer).
"""

from __future__ import annotations

from ..config import Knobs
from ..schema import GameView, parse_negotiation
from ..theory.concession import boulware


def _my_payoff(role: str, value: float, price: float) -> float:
    return price - value if role == "seller" else value - price


def _is_final_round(view: GameView) -> bool:
    return view.max_rounds is not None and view.round >= view.max_rounds


def _schedule_length(view: GameView, knobs: Knobs) -> int:
    if view.max_rounds is not None:
        return max(min(view.max_rounds, knobs.neg_max_planned_rounds), 1)
    return knobs.neg_max_planned_rounds


def _anchor_and_floor(n, knobs: Knobs) -> tuple[float, float]:
    """(opening price, worst price I'll concede to) for my role."""
    v = n.my_value if n.my_value is not None else 100.0
    margin = max(v * knobs.neg_min_margin_frac, 0.01)
    if n.my_role == "seller":
        return v * (1.0 + knobs.neg_anchor_markup), v + margin
    return v * (1.0 - knobs.neg_anchor_markup), max(v - margin, 0.0)


def _target_price(view: GameView, n, knobs: Knobs) -> float:
    anchor, floor = _anchor_and_floor(n, knobs)
    T = _schedule_length(view, knobs)
    my_round = min(view.round, T)
    if n.my_role == "seller":
        # Concede downward from anchor to floor.
        return boulware(my_round, T, anchor, floor, knobs.neg_beta)
    # Buyer concedes upward: mirror the curve.
    return boulware(my_round, T, anchor, floor, knobs.neg_beta)


def _opponent_best_price(view: GameView, n) -> float | None:
    """The most favorable price the opponent has ever put on the table for me."""
    best: float | None = None
    for entry in view.history:
        if not isinstance(entry, dict):
            continue
        for key in ("offer", "counteroffer"):
            item = entry.get(key)
            if not isinstance(item, dict):
                continue
            if item.get("from_player") == view.your_player:
                continue
            price = item.get("price", item.get("product_price"))
            try:
                price = float(price)
            except (TypeError, ValueError):
                continue
            if best is None:
                best = price
            elif n.my_role == "seller":
                best = max(best, price)
            else:
                best = min(best, price)
    lo = n.last_offer_price
    if lo is not None and not n.last_offer_mine:
        if best is None:
            best = lo
        else:
            best = max(best, lo) if n.my_role == "seller" else min(best, lo)
    return best


def _should_walk_away(view: GameView, n, knobs: Knobs) -> bool:
    """Walk only when the horizon is nearly exhausted and the opponent's
    trajectory cannot cross our reservation. With no round cap we keep the
    free option open (countering costs nothing)."""
    if view.max_rounds is None:
        return False
    if n.my_value is None:
        return False
    rounds_left = view.max_rounds - view.round
    if rounds_left > 1:
        return False
    best = _opponent_best_price(view, n)
    if best is None:
        return False
    return _my_payoff(n.my_role, n.my_value, best) < 0


def decide(view: GameView, knobs: Knobs) -> dict:
    n = parse_negotiation(view)
    value = n.my_value if n.my_value is not None else 100.0

    if view.action_type == "offer":
        price = _target_price(view, n, knobs)
        # Single-round ultimatum: no counteroffers exist, price to close.
        if view.max_rounds == 1:
            anchor, floor = _anchor_and_floor(n, knobs)
            # Without the dataset CDF yet, split the difference between a
            # moderate markup and reservation — closing matters most.
            price = (anchor + floor) / 2
        return {"product_price": round(max(price, 0.0), 2)}

    # Decision phase.
    offer = n.last_offer_price
    if offer is None:
        return {"decision": "RejectOffer", "product_price": round(value, 2)}

    payoff = _my_payoff(n.my_role, value, offer)
    final = _is_final_round(view)

    if final:
        if payoff > 0:
            return {"decision": "AcceptOffer"}
        # Losing deal on the last round: plain rejection ends at $0, same as
        # walkaway; reject is the safer enum.
        return {"decision": "RejectOffer"}

    # Unlimited-horizon stalemate exit: once both concession schedules are
    # exhausted, an endless reject/counter loop just blocks a concurrency
    # slot (observed live: games running past round 79). Take any profit on
    # the table; if the opponent has NEVER crossed our reservation by deep
    # into the marathon, walk and free the slot.
    if view.max_rounds is None:
        stall = knobs.neg_max_planned_rounds + 4
        if view.round >= stall and payoff > 0:
            return {"decision": "AcceptOffer"}
        if view.round >= stall + 8:
            best = _opponent_best_price(view, n)
            best_payoff = (
                _my_payoff(n.my_role, value, best) if best is not None else None
            )
            if best_payoff is None or best_payoff <= 0:
                return {"decision": "WalkAway"}

    # Accept when the offer already beats the price we planned to counter at
    # (they met or beat our own trajectory).
    my_next = _target_price(view, n, knobs)
    counter_payoff = _my_payoff(n.my_role, value, my_next)
    if payoff > 0 and payoff >= counter_payoff * 0.9:
        return {"decision": "AcceptOffer"}

    if _should_walk_away(view, n, knobs):
        return {"decision": "WalkAway"}

    return {"decision": "RejectOffer", "product_price": round(max(my_next, 0.0), 2)}
