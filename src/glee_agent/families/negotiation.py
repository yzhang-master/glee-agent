"""Negotiation strategy: anchored Boulware concession toward reservation,
never accept a losing deal, final-round accept-anything-positive.

Prices are absolute; my payoff = price - value (seller) or value - price (buyer).
"""

from __future__ import annotations

import hashlib
import math

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


def _is_ultimatum(view: GameView) -> bool:
    """No later round exists to keep the opponent engaged in."""
    return view.max_rounds is not None and view.round >= view.max_rounds


def _terminal_close(view: GameView, knobs: Knobs) -> bool:
    """Whether this is the last finite-horizon offer we can improve."""
    return (
        knobs.neg_terminal_close
        and view.max_rounds is not None
        and view.max_rounds > 1
        and view.round >= view.max_rounds - 1
    )


def _schedule_length(view: GameView, knobs: Knobs) -> int:
    if view.max_rounds is not None:
        return max(min(view.max_rounds, knobs.neg_max_planned_rounds), 1)
    return knobs.neg_max_planned_rounds


def _anchor_and_floor(n, knobs: Knobs, ultimatum: bool = False) -> tuple[float, float]:
    """(opening price, worst price I'll concede to) for my role.

    `ultimatum` marks a take-it-or-leave-it round (T=1, or the final round of
    a capped game): there is no later round to engage the opponent in, so the
    anchor goes to the feasibility wall rather than leaving them room.
    """
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

    # In incomplete information the platform draws values from a small,
    # stable grid. A generic 50% own-value anchor sits outside that grid for
    # many buyer states and wastes the high-leverage opening round. The
    # prior anchor targets the EV-optimal opponent-type boundary while leaving
    # the existing Boulware floor and reciprocal concession intact. Setting
    # its capture fraction to zero restores the generic anchor.
    prior_price = _ii_prior_price(n, knobs)
    if prior_price is not None:
        anchor = prior_price

    # Complete information: an ask above the buyer's value (bid below the
    # seller's) can never be profitably accepted — those offers burn every
    # leverage round. Clamp the anchor feasible, and floor the concession at
    # a real share of the public surplus: with the floor at own reservation,
    # patient opponents just wait the Boulware curve out and collect ~95% of
    # the surplus at round T.
    frac = knobs.neg_ci_floor_frac
    if n.opp_value is not None and n.my_value is not None:
        surplus = (
            n.opp_value - v if n.my_role == "seller" else v - n.opp_value
        )
        if surplus > 0:
            # The anchor cap is NOT knob-gated: measured live, ~58% of our
            # complete-info offers left the opponent negative surplus and were
            # accepted 0-0.6% of the time, while an offer leaving them ~10-20%
            # of the surplus is accepted 22%. An offer they cannot profitably
            # accept is not an aggressive offer, it is a wasted round.
            # The anchor must stay ABOVE the floor, or the concession range
            # collapses to a point and the floor knob silently stops doing
            # anything: floors of 0.85 and 0.95 both clamped to the 0.80 cap,
            # which turned a ladder into an accidental A/A test.
            cap = 0.98 if ultimatum else max(knobs.neg_ci_anchor_frac, frac + 0.10)
            cap = min(cap, 0.98)
            if n.my_role == "seller":
                anchor = min(anchor, v + cap * surplus)
                floor = max(floor, v + frac * surplus)
            else:
                anchor = max(anchor, v - cap * surplus)
                floor = min(floor, v - frac * surplus)
            if n.my_role == "seller":
                floor = min(floor, anchor)
            else:
                floor = max(floor, anchor)
    return anchor, floor


def _feasible_price(price: float, n, eps_frac: float = 0.01) -> float:
    """Clamp a price so a known-value opponent can still profit by accepting.

    A hard invariant, independent of any knob and applied at every return
    site: the acceptance model has a strict gate at the responder's own
    value, so an infeasible offer has literally zero chance and merely burns
    a round (and, late in a game, invites a walk-away).
    """
    if n.opp_value is None or n.opp_value <= 0:
        return price
    eps = max(abs(n.opp_value) * eps_frac, 1e-9)
    if n.my_role == "seller":
        return min(price, n.opp_value - eps)   # buyer must gain by buying
    return max(price, n.opp_value + eps)       # seller must gain by selling


def _ci_ultimatum_price(n, knobs: Knobs) -> float | None:
    """Direct surplus price for a one-round complete-info offer.

    In this branch there is no continuation value and both reservation values
    are public. The historical control captures a median 95% of the visible
    surplus with 100% agreement, whereas the dataset-CDF optimizer gives away
    roughly 27% with the same agreement rate. A zero knob remains available
    as a rollback to the previous heuristic.
    """
    frac = knobs.neg_ci_ultimatum_frac
    if frac <= 0 or n.my_value is None or n.opp_value is None:
        return None
    frac = min(max(frac, 0.0), 0.99)
    if n.my_role == "seller":
        surplus = n.opp_value - n.my_value
        if surplus <= 0:
            return None
        return n.my_value + frac * surplus
    surplus = n.my_value - n.opp_value
    if surplus <= 0:
        return None
    return n.my_value - frac * surplus


def _ii_ultimatum_price(n, knobs: Knobs) -> float | None:
    """Opt-in direct price for a one-round hidden-value offer.

    This is deliberately independent of the complete-information surplus
    rule: only my reservation value is visible here, so the knob is a markup
    (or buyer discount) rather than a share of surplus. Zero preserves the
    established midpoint heuristic for clean live comparison.
    """
    markup = knobs.neg_ii_ultimatum_markup
    if markup <= 0 or n.my_value is None or n.opp_value is not None:
        return None
    markup = min(max(markup, 0.0), 0.99)
    if n.my_role == "seller":
        return n.my_value * (1.0 + markup)
    return n.my_value * (1.0 - markup)


def _ii_prior_price(n, knobs: Knobs) -> float | None:
    """Bayesian anchor from the platform's discrete hidden-value prior.

    Complete-information live games expose the same value grid used for
    hidden-information games. Conditional EV is maximized at these opponent
    type boundaries (scaled by order of magnitude):

    * seller 8 -> target buyer 12; seller 10/12 -> target buyer 15
    * buyer 10 -> target seller 8; buyer 12/15 -> target seller 10

    ``frac`` captures that share of surplus at the selected boundary. Values
    outside the known grid fall back to the established generic anchor.
    """
    frac = knobs.neg_ii_prior_capture_frac
    value = n.my_value
    if frac <= 0 or value is None or value <= 0 or n.opp_value is not None:
        return None
    frac = min(max(frac, 0.0), 0.99)
    order = 10.0 ** math.floor(math.log10(value))
    mantissa = value / order

    def near(expected: float) -> bool:
        return abs(mantissa - expected) <= 1e-6

    if n.my_role == "seller":
        if near(8.0):
            boundary = 12.0 * order
        elif near(1.0) or near(1.2):
            boundary = 1.5 * order
        else:
            return None
        surplus = boundary - value
        return value + frac * surplus if surplus > 0 else None

    if near(1.0):
        boundary = 0.8 * order
    elif near(1.2) or near(1.5):
        boundary = 1.0 * order
    else:
        return None
    surplus = value - boundary
    return value - frac * surplus if surplus > 0 else None


def _target_price(view: GameView, n, knobs: Knobs) -> float:
    anchor, floor = _anchor_and_floor(n, knobs, ultimatum=_is_ultimatum(view))
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


def _my_offer_prices(view: GameView) -> list[float]:
    """Chronological prices I have proposed, oldest first."""
    out = []
    for entry in view.history:
        if not isinstance(entry, dict):
            continue
        off = entry.get("offer")
        if isinstance(off, dict) and off.get("from_player") == view.your_player:
            try:
                out.append(float(off.get("price", off.get("product_price"))))
            except (TypeError, ValueError):
                continue
        co = entry.get("counteroffer")
        if isinstance(co, (int, float)) and isinstance(off, dict) \
                and off.get("from_player") != view.your_player:
            out.append(float(co))
    return out


def _opp_offer_prices(view: GameView) -> list[float]:
    """Chronological prices the opponent has proposed, oldest first."""
    out = []
    for entry in view.history:
        if not isinstance(entry, dict):
            continue
        off = entry.get("offer")
        if isinstance(off, dict) and off.get("from_player") != view.your_player:
            try:
                out.append(float(off.get("price", off.get("product_price"))))
            except (TypeError, ValueError):
                continue
        co = entry.get("counteroffer")
        if isinstance(co, (int, float)) and isinstance(off, dict) \
                and off.get("from_player") == view.your_player:
            out.append(float(co))
    return out


def _opp_last_concession(view: GameView, n) -> float:
    """How much their latest offer improved FOR ME over their previous one."""
    prices = _opp_offer_prices(view)
    if len(prices) < 2:
        return 0.0
    prev, last = prices[-2], prices[-1]
    gain = (last - prev) if n.my_role == "seller" else (prev - last)
    return max(gain, 0.0)


def _reciprocal_cap(view: GameView, n, knobs: Knobs, target: float,
                    anchor: float, floor: float) -> float:
    """Never concede faster than the opponent does (MiCRO).

    The validated counterpart model makes BOTH their acceptance probability
    and their own concession size strictly decreasing in our concession
    speed, so a purely time-based schedule pays an opponent to sit still —
    which is what 30-45%% of our opponents do. The schedule therefore becomes
    an upper bound on generosity: between two of my own offers I move at most
    what they just moved, plus a token drip.
    """
    if not knobs.neg_reciprocal:
        return target
    mine = _my_offer_prices(view)
    if not mine:
        return target
    my_last = mine[-1]
    drip = abs(anchor - floor) * knobs.neg_drip
    step = max(_opp_last_concession(view, n), drip)
    if n.my_role == "seller":
        # I concede by lowering the price: never below my last minus step.
        return min(max(target, my_last - step), my_last)
    # Buyer concedes by raising: never above my last plus step.
    return max(min(target, my_last + step), my_last)


def _terminal_generosity_guard(n, terminal: float, reciprocal: float) -> float:
    """A terminal close may improve an offer, never retract one.

    Reciprocity combines two jobs: limiting the speed of a concession and
    preventing walkback from our own prior offer.  The canary bypasses only
    the speed limit.  Comparing against the reciprocal candidate retains the
    no-walkback invariant even if an optimizer or unusual history would make
    the raw terminal candidate less generous to the opponent.
    """
    if n.my_role == "seller":
        return min(terminal, reciprocal)
    return max(terminal, reciprocal)


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
    anchor, floor = _anchor_and_floor(n, knobs, ultimatum=ultimatum)
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


_NEG_OPEN = (
    "This price reflects what the item is worth to me; I have room to talk but not much.",
    "That's my opening number. I'd rather settle quickly than grind through rounds.",
)
_NEG_HOLD = (
    "I've moved and you haven't. My number stands until yours does.",
    "You're asking me to bid against myself. Show me movement and I'll respond in kind.",
)
_NEG_CONCEDE = (
    "I've come down again — that's another concession from me. We're close; let's finish it.",
    "That's a real move toward you. I've got very little left to give.",
)
_NEG_FINAL = (
    "This is my final price. It beats the nothing we both get if we don't close.",
    "Last offer — this is my limit, and I'd rather we both walked away with something.",
)


def _neg_message(view: GameView, price: float, conceded: bool) -> str:
    """Deterministic, zero-cost offer text in the validated register.

    Write-only: we never parse the opponent's prose, because reading it is
    measured to LOWER surplus for every model tested.
    """
    left = _rounds_left(view)
    if left is not None and left <= 1:
        pool = _NEG_FINAL
    elif view.round <= 1:
        pool = _NEG_OPEN
    elif conceded:
        pool = _NEG_CONCEDE
    else:
        pool = _NEG_HOLD
    seed = f"nmsg:{view.game_id}:{view.round}".encode()
    return pool[int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") % len(pool)]


def decide(view: GameView, knobs: Knobs) -> dict:
    n = parse_negotiation(view)
    value = n.my_value if n.my_value is not None else 100.0
    terminal_close = _terminal_close(view, knobs)

    if view.action_type == "offer":
        price = _target_price(view, n, knobs)
        direct_ultimatum = None
        # Single-round ultimatum: no counteroffers exist, price to close.
        if view.max_rounds == 1:
            direct_ultimatum = _ci_ultimatum_price(n, knobs)
            if direct_ultimatum is None:
                direct_ultimatum = _ii_prior_price(n, knobs)
            if direct_ultimatum is None:
                direct_ultimatum = _ii_ultimatum_price(n, knobs)
            if direct_ultimatum is not None:
                price = direct_ultimatum
            else:
                anchor, floor = _anchor_and_floor(n, knobs, ultimatum=True)
                # Without the dataset CDF, split the difference between a
                # moderate markup and reservation — closing matters most.
                price = (anchor + floor) / 2
        # Empirical accept-curve optimizer (ultimatum seller always prefers
        # it when curve data exists; otherwise it overrides Boulware only
        # when it disagrees materially).
        if direct_ultimatum is None and not terminal_close:
            optimized = _optimized_price(view, n, knobs, price)
            if optimized is not None:
                price = optimized
        if view.max_rounds != 1:
            anchor, floor = _anchor_and_floor(n, knobs, ultimatum=_is_ultimatum(view))
            reciprocal = _reciprocal_cap(view, n, knobs, price, anchor, floor)
            price = (
                _terminal_generosity_guard(n, price, reciprocal)
                if terminal_close
                else reciprocal
            )
        # The direct CI price already leaves the opponent a positive share of
        # the actual surplus. Do not replace that with the generic 1%-of-value
        # epsilon, which can consume most of a thin surplus.
        price = _feasible_price(price, n, eps_frac=0.0 if direct_ultimatum is not None else 0.01)
        out = {"product_price": round(max(price, 0.0), 2)}
        if view.messages_allowed:
            mine = _my_offer_prices(view)
            conceded = bool(mine) and (
                out["product_price"] < mine[-1] - 1e-9 if n.my_role == "seller"
                else out["product_price"] > mine[-1] + 1e-9
            )
            msg = None
            if llm_client.llm_available(knobs):
                msg = llm_messages.negotiation_offer_message(
                    n.my_role, out["product_price"], view.round
                )
            out["message"] = msg or _neg_message(view, out["product_price"], conceded)
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
        stall = knobs.neg_stall_accept or (knobs.neg_max_planned_rounds + 4)
        if view.round >= stall and payoff > 0:
            return {"decision": "AcceptOffer"}
        if not knobs.neg_never_walk and view.round >= stall + 8:
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
    counter = None if terminal_close else _optimized_price(view, n, knobs, my_next)
    if counter is None:
        counter = my_next
    anchor, floor = _anchor_and_floor(n, knobs, ultimatum=_is_ultimatum(view))
    reciprocal = _reciprocal_cap(view, n, knobs, counter, anchor, floor)
    counter = (
        _terminal_generosity_guard(n, counter, reciprocal)
        if terminal_close
        else reciprocal
    )
    # Walkback resistance: never counter below the best they have already
    # offered us — that price is already banked, and bidding under it hands
    # back surplus they had conceded.
    best = _opponent_best_price(view, n)
    if best is not None:
        counter = max(counter, best) if n.my_role == "seller" else min(counter, best)
    counter = _feasible_price(counter, n)
    out = {"decision": "RejectOffer", "product_price": round(max(counter, 0.0), 2)}
    if view.messages_allowed:
        mine = _my_offer_prices(view)
        conceded = bool(mine) and (
            out["product_price"] < mine[-1] - 1e-9 if n.my_role == "seller"
            else out["product_price"] > mine[-1] + 1e-9
        )
        msg = None
        if llm_client.llm_available(knobs):
            msg = llm_messages.negotiation_offer_message(
                n.my_role, out["product_price"], view.round
            )
        out["message"] = msg or _neg_message(view, out["product_price"], conceded)
    return out
