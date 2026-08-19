"""Negotiation strategy: anchored Boulware concession toward reservation,
never accept a losing deal, final-round accept-anything-positive.

Prices are absolute; my payoff = price - value (seller) or value - price (buyer).
"""

from __future__ import annotations

from ..config import Knobs
from ..llm import client as llm_client
from ..llm import messages as llm_messages
from ..schema import GameView, parse_negotiation
from ..theory.concession import boulware
from ..theory.targets import config_key_negotiation, get_targets


def _my_payoff(role: str, value: float, price: float) -> float:
    return price - value if role == "seller" else value - price


def _rounds_left(view: GameView) -> int | None:
    if view.max_rounds is None:
        return None
    return max(view.max_rounds - view.round + 1, 1)


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
        anchor, floor = v * (1.0 + knobs.neg_anchor_markup), v + margin
    else:
        markup = (
            knobs.neg_anchor_markup_buyer
            if knobs.neg_anchor_markup_buyer is not None
            else knobs.neg_anchor_markup
        )
        anchor, floor = v * (1.0 - markup), max(v - margin, 0.0)

    # Complete information: an ask above the buyer's value (bid below the
    # seller's) can never be profitably accepted — those offers burn every
    # leverage round. Clamp the anchor feasible, and floor the concession at
    # a real share of the public surplus: with the floor at own reservation,
    # patient opponents just wait the Boulware curve out and collect ~95% of
    # the surplus at round T.
    frac = knobs.neg_ci_floor_frac
    if frac > 0 and n.opp_value is not None and n.my_value is not None:
        surplus = (
            n.opp_value - v if n.my_role == "seller" else v - n.opp_value
        )
        if surplus > 0:
            if n.my_role == "seller":
                anchor = min(anchor, v + 0.95 * surplus)
                floor = max(floor, v + frac * surplus)
            else:
                anchor = max(anchor, v - 0.95 * surplus)
                floor = min(floor, v - frac * surplus)
    return anchor, floor


def _target_price(view: GameView, n, knobs: Knobs) -> float:
    anchor, floor = _anchor_and_floor(n, knobs)
    # Finite endgame: the schedule only reaches the floor at round T, a round
    # where we may never place an offer (measured live: every T=10 no-deal
    # died with our best ask still >= 1.1x value). With <= 1 round after this
    # one, price at the floor — a closed floor deal beats certain no-deal.
    if view.max_rounds is not None and view.round >= view.max_rounds - 1:
        return floor
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
            if isinstance(item, dict):
                if item.get("from_player") == view.your_player:
                    continue
                price = item.get("price", item.get("product_price"))
            elif key == "counteroffer" and isinstance(item, (int, float)):
                # Live history stores counteroffers as bare floats: the
                # counter answers this entry's offer, so it comes from the
                # opposite player — ours iff the offer was theirs.
                off = entry.get("offer")
                if not isinstance(off, dict) or off.get("from_player") != view.your_player:
                    continue
                price = item
            else:
                continue
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
    """Never walk in finite games: walking pays 0, exactly what horizon
    expiry pays, but forecloses our remaining offers — measured live, 757
    round-9 walks each forfeited a free final floor offer the opponent
    could still have accepted. (Unlimited-horizon walks are handled by the
    stalemate exit in decide(), which frees the concurrency slot.)"""
    return False


def _payoff_percentile(view: GameView, n, value: float, payoff: float) -> float | None:
    """Percentile of `payoff` vs the live scoring pool for my config+role.
    Tries the full config key (complete info), then the role-marginal key."""
    tg = get_targets()
    full_key, role_key = config_key_negotiation(view.state, n.my_role, value)
    pct = None
    if full_key is not None:
        pct = tg.payoff_percentile("negotiation", full_key, view.your_player, payoff)
    if pct is None and role_key is not None:
        pct = tg.payoff_percentile("negotiation", role_key, view.your_player, payoff)
    return pct


def _pool_quantile(view: GameView, n, value: float, q: float) -> float | None:
    """Pool payoff at quantile q, same key fallback as _payoff_percentile."""
    tg = get_targets()
    full_key, role_key = config_key_negotiation(view.state, n.my_role, value)
    out = None
    if full_key is not None:
        out = tg.payoff_quantile("negotiation", full_key, view.your_player, q)
    if out is None and role_key is not None:
        out = tg.payoff_quantile("negotiation", role_key, view.your_player, q)
    return out


def _optimized_price(view: GameView, n, knobs: Knobs, target_price: float) -> float | None:
    """Pick my offer/counter price by maximizing EV against the empirical
    accept curve keyed on price relative to the RESPONDER's value.

    Only runs when the opponent's value is visible (their rel bucket is
    unknowable otherwise — the pooled marginal is constant in price, so it
    cannot rank candidates and we keep the Boulware schedule instead).
    Returns None when the curve is too thin or agrees with the schedule."""
    if n.opp_value is None or n.opp_value <= 0:
        return None
    tg = get_targets()
    value = n.my_value if n.my_value is not None else 100.0
    their_role = "buyer" if n.my_role == "seller" else "seller"
    human = view.opponent_type == "human"
    left = _rounds_left(view)
    ultimatum = view.max_rounds == 1
    # Continuation if they reject: roughly my scheduled counter, haircut;
    # nothing left to continue to on the last round.
    if ultimatum or (left is not None and left <= 1):
        cont = 0.0
    else:
        cont = max(_my_payoff(n.my_role, value, target_price), 0.0) * 0.8
    anchor, floor = _anchor_and_floor(n, knobs)
    lo, hi = (floor, anchor) if floor <= anchor else (anchor, floor)
    best_price = None
    best_ev = -1.0
    n_with_data = 0
    for i in range(21):  # candidate prices between reservation and anchor
        price = lo + (hi - lo) * i / 20.0
        if knobs.neg_ci_floor_frac > 0:
            # The dataset accept curve contains irrational losing-price
            # accepts; live opponents never take a deal that loses money.
            responder_payoff = _my_payoff(their_role, n.opp_value, price)
            if responder_payoff < 0:
                continue
        p_accept = tg.neg_accept_prob(price / n.opp_value, their_role, left, human)
        if p_accept is None:
            continue
        n_with_data += 1
        ev = max(_my_payoff(n.my_role, value, price), 0.0) * p_accept + (1.0 - p_accept) * cont
        if ev > best_ev:
            best_ev, best_price = ev, price
    if best_price is None or n_with_data < 5:
        return None
    if not ultimatum and abs(best_price - target_price) <= 0.02 * max(value, 1.0):
        return None  # empirics agree with the schedule; keep it
    return best_price


def decide(view: GameView, knobs: Knobs) -> dict:
    n = parse_negotiation(view)
    value = n.my_value if n.my_value is not None else 100.0

    if view.action_type == "offer":
        price = _target_price(view, n, knobs)
        # Single-round ultimatum: no counteroffers exist, price to close.
        if view.max_rounds == 1:
            anchor, floor = _anchor_and_floor(n, knobs)
            # Without the dataset CDF, split the difference between a
            # moderate markup and reservation — closing matters most.
            price = (anchor + floor) / 2
        # Empirical accept-curve optimizer (ultimatum seller always prefers
        # it when curve data exists; otherwise it overrides Boulware only
        # when it disagrees materially).
        optimized = _optimized_price(view, n, knobs, price)
        if optimized is not None:
            price = optimized
        out = {"product_price": round(max(price, 0.0), 2)}
        if view.messages_allowed and llm_client.llm_available(knobs):
            msg = llm_messages.negotiation_offer_message(
                n.my_role, out["product_price"], view.round
            )
            if msg:
                out["message"] = msg
        return out

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

    # Empirical percentile accept: this profit already beats
    # knobs.neg_accept_pct of every payoff scored on my config+role pool.
    # Two guards: never score a DEFAULTED value (my_value missing), and —
    # because many pools are mostly no-deal zeros, where any epsilon profit
    # "ranks" at the 98th percentile — demand the offer also carries a real
    # fraction of what top games extract (q95), so we never trade a 200k
    # surplus for a 1k crumb just because the pool is full of zeros.
    if payoff > 0 and n.my_value is not None:
        pct = _payoff_percentile(view, n, value, payoff)
        q95 = _pool_quantile(view, n, value, 0.95)
        if (
            pct is not None
            and pct >= knobs.neg_accept_pct
            and q95 is not None
            and payoff >= 0.25 * q95
        ):
            return {"decision": "AcceptOffer"}

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
    # (they met or beat our own trajectory). Optional percentile gate: once
    # the Boulware schedule hits its floor (round >= T), counter_payoff
    # collapses to the margin and this rule accepts any crumb — measured
    # live, 616 such accepts averaged pct 0.25. The gate refuses offers the
    # scoring pool ranks poorly while still closing in no-deal-heavy pools
    # (where a small profit genuinely ranks high).
    my_next = _target_price(view, n, knobs)
    counter_payoff = _my_payoff(n.my_role, value, my_next)
    if payoff > 0 and payoff >= counter_payoff * knobs.neg_accept_factor:
        gate = knobs.neg_traj_pct_gate
        if gate <= 0 or n.my_value is None:
            return {"decision": "AcceptOffer"}
        pct = _payoff_percentile(view, n, value, payoff)
        if pct is None or pct >= gate:
            return {"decision": "AcceptOffer"}

    if _should_walk_away(view, n, knobs):
        return {"decision": "WalkAway"}

    # Counter at the optimizer's price when the accept curve supports one
    # (the accept test above still used the Boulware trajectory).
    counter = _optimized_price(view, n, knobs, my_next)
    if counter is None:
        counter = my_next
    out = {"decision": "RejectOffer", "product_price": round(max(counter, 0.0), 2)}
    if view.messages_allowed and llm_client.llm_available(knobs):
        msg = llm_messages.negotiation_offer_message(
            n.my_role, out["product_price"], view.round
        )
        if msg:
            out["message"] = msg
    return out
