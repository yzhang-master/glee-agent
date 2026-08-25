"""The validity guard: a TOTAL function from (proposed action, game view) to a
schema-valid action dict. It must never raise and never emit an action the
server could reject — an invalid move burns one of 5 attempts and five of them
score the game at the 5th percentile.

Every correction it makes is appended to `corrections` (the caller logs them
at WARNING: a correction in production is a bug elsewhere).
"""

from __future__ import annotations

import math
from typing import Any

from .schema import GameView, parse_bargaining, parse_negotiation

MAX_MESSAGE = 1990  # server cap is 2000; keep margin

BARGAINING_DECISIONS = ("accept", "reject", "walkaway")
NEGOTIATION_DECISIONS = ("AcceptOffer", "RejectOffer", "WalkAway")
YESNO = ("yes", "no")


def _finite(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _granularity(pot: float) -> float:
    """Round to integers for integer pots, else cents."""
    return 1.0 if float(pot).is_integer() else 0.01


def _snap(x: float, unit: float) -> float:
    snapped = round(x / unit) * unit
    return round(snapped, 2)


def _clean_message(action: dict, view: GameView, corrections: list[str]) -> None:
    msg = action.get("message")
    if msg is None:
        return
    # messages_allowed only gates OPTIONAL messages riding on offers/decisions;
    # a persuasion seller_message action IS the message and must survive.
    if not view.messages_allowed and view.action_type != "seller_message":
        action.pop("message", None)
        corrections.append("dropped message (messages not allowed)")
        return
    if not isinstance(msg, str):
        msg = str(msg)
        corrections.append("coerced non-string message")
    # Strip control characters that could corrupt the JSON body.
    msg = "".join(ch for ch in msg if ch >= " " or ch in "\n\t")
    if len(msg) > MAX_MESSAGE:
        msg = msg[:MAX_MESSAGE]
        corrections.append("truncated message to length cap")
    action["message"] = msg


def _bargaining_fallback(view: GameView) -> dict:
    if view.action_type == "decision":
        return {"decision": "accept"}
    b = parse_bargaining(view)
    pot = b.money if b.money > 0 else 100.0
    unit = _granularity(pot)
    half = _snap(pot / 2, unit)
    names = {1: "alice_gain", 2: "bob_gain"}
    return {names[view.my_index]: half, names[3 - view.my_index]: round(pot - half, 2)}


def _negotiation_fallback(view: GameView) -> dict:
    n = parse_negotiation(view)
    if view.action_type == "decision":
        # Accept only when profitable; otherwise reject with a counter at our value.
        price = n.last_offer_price
        if price is not None and n.my_value is not None:
            profitable = price >= n.my_value if n.my_role == "seller" else price <= n.my_value
            if profitable:
                return {"decision": "AcceptOffer"}
        if n.my_value is not None and not _is_final_round(view):
            return {
                "decision": "RejectOffer",
                "product_price": round(max(n.my_value, 0.0), 2),
            }
        return {"decision": "RejectOffer"}
    value = max(n.my_value, 0.0) if n.my_value is not None else 100.0
    return {"product_price": round(value, 2)}


def _persuasion_fallback(view: GameView) -> dict:
    if view.action_type == "seller_message":
        return {"message": "I recommend this product."}
    return {"decision": "yes"}


def _is_final_round(view: GameView) -> bool:
    return view.max_rounds is not None and view.round >= view.max_rounds


def fallback_action(view: GameView) -> dict:
    """A safe, always-valid move for the current turn."""
    if view.family == "bargaining":
        return _bargaining_fallback(view)
    if view.family == "negotiation":
        return _negotiation_fallback(view)
    if view.family == "persuasion":
        return _persuasion_fallback(view)
    # Unknown family: guess from action_type.
    if view.action_type == "offer":
        return _bargaining_fallback(view)
    if view.action_type == "seller_message":
        return {"message": "Hello."}
    return {"decision": "accept"}


def guard(proposed: Any, view: GameView) -> tuple[dict, list[str]]:
    """Validate/repair `proposed` for this turn. Returns (action, corrections).

    Never raises. On unrepairable input returns the family fallback.
    """
    corrections: list[str] = []
    try:
        return _guard_inner(proposed, view, corrections)
    except Exception as e:  # noqa: BLE001 — total function by contract
        corrections.append(f"guard exception: {type(e).__name__}: {e}")
        try:
            action = fallback_action(view)
            _clean_message(action, view, corrections)
            return action, corrections
        except Exception:  # noqa: BLE001
            return {"decision": "accept"}, corrections


def _guard_inner(proposed: Any, view: GameView, corrections: list[str]) -> tuple[dict, list[str]]:
    if not isinstance(proposed, dict):
        corrections.append(f"proposed action not a dict: {type(proposed).__name__}")
        proposed = fallback_action(view)
    action = dict(proposed)

    family = view.family
    atype = view.action_type

    if family == "bargaining":
        action = _guard_bargaining(action, view, corrections)
    elif family == "negotiation":
        action = _guard_negotiation(action, view, corrections)
    elif family == "persuasion":
        action = _guard_persuasion(action, view, corrections)
    else:
        corrections.append(f"unknown family {family!r}; using fallback")
        action = fallback_action(view)

    _clean_message(action, view, corrections)

    # Belt-and-suspenders: decisions must be plain strings.
    if "decision" in action and not isinstance(action["decision"], str):
        corrections.append("decision was not a string; using fallback")
        action = fallback_action(view)
        _clean_message(action, view, corrections)
    return action, corrections


def _guard_bargaining(action: dict, view: GameView, corrections: list[str]) -> dict:
    if view.action_type == "decision":
        d = action.get("decision")
        if d not in BARGAINING_DECISIONS:
            corrections.append(f"bad bargaining decision {d!r}")
            return {"decision": "accept"}
        return {"decision": d, **({"message": action["message"]} if "message" in action else {})}

    # Offer: exact-sum by construction from my_gain.
    b = parse_bargaining(view)
    pot = b.money
    if pot <= 0:
        corrections.append("money_to_divide missing/invalid; fallback offer")
        return _bargaining_fallback(view)
    unit = _granularity(pot)
    names = {1: "alice_gain", 2: "bob_gain"}
    mine_key, opp_key = names[view.my_index], names[3 - view.my_index]

    my_gain = _finite(action.get(mine_key))
    if my_gain is None:
        # Maybe the strategy filled only the opponent's side.
        opp_gain = _finite(action.get(opp_key))
        my_gain = pot - opp_gain if opp_gain is not None else pot / 2
        corrections.append("my gain missing; derived")
    my_gain = _snap(min(max(my_gain, 0.0), pot), unit)
    opp_gain = round(pot - my_gain, 2)

    out = {mine_key: my_gain, opp_key: opp_gain}
    if "message" in action:
        out["message"] = action["message"]
    return out


def _guard_negotiation(action: dict, view: GameView, corrections: list[str]) -> dict:
    n = parse_negotiation(view)

    if view.action_type == "decision":
        d = action.get("decision")
        if d not in NEGOTIATION_DECISIONS:
            corrections.append(f"bad negotiation decision {d!r}")
            return _negotiation_fallback(view)

        # Never accept a losing deal — invariant enforced here, not only in strategy.
        if d == "AcceptOffer" and n.last_offer_price is not None and n.my_value is not None:
            price = n.last_offer_price
            losing = price < n.my_value if n.my_role == "seller" else price > n.my_value
            if losing:
                corrections.append("blocked acceptance of negative-payoff deal")
                return _negotiation_fallback(view)

        out: dict = {"decision": d}
        if d == "RejectOffer":
            counter = _finite(action.get("product_price"))
            if counter is None and not _is_final_round(view):
                # Mid-game rejection should carry a counter; use our value.
                if n.my_value is not None:
                    counter = n.my_value
                    corrections.append("added missing counteroffer at own value")
            if counter is not None:
                out["product_price"] = _guard_negotiation_price(
                    counter, n, corrections, "counteroffer"
                )
        if "message" in action:
            out["message"] = action["message"]
        return out

    # Offer phase.
    price = _finite(action.get("product_price"))
    if price is None:
        corrections.append("offer price missing/invalid; using own value")
        price = n.my_value if n.my_value is not None else 100.0
    price = _guard_negotiation_price(price, n, corrections, "offer")
    out = {"product_price": price}
    if "message" in action:
        out["message"] = action["message"]
    return out


def _guard_negotiation_price(
    price: float, n, corrections: list[str], label: str
) -> float:
    """Enforce nonnegative and own-reservation bounds on an emitted price."""
    cleaned = round(max(price, 0.0), 2)
    if price < 0:
        corrections.append(f"clamped negotiation {label} to nonnegative")

    if n.my_value is None:
        return cleaned
    reservation = max(n.my_value, 0.0)
    if n.my_role == "seller" and cleaned < reservation:
        corrections.append(f"raised seller {label} to own reservation")
        return reservation
    if n.my_role == "buyer" and cleaned > reservation:
        corrections.append(f"lowered buyer {label} to own reservation")
        return reservation
    return cleaned


def _guard_persuasion(action: dict, view: GameView, corrections: list[str]) -> dict:
    atype = view.action_type
    if atype == "seller_message":
        msg = action.get("message")
        if not isinstance(msg, str) or not msg.strip():
            corrections.append("empty seller message; using template")
            msg = "I recommend this product."
        return {"message": msg}

    # seller_recommendation and buyer_decision are both yes/no.
    d = action.get("decision")
    if isinstance(d, str):
        d_norm = d.strip().lower()
        if d_norm in YESNO:
            out: dict = {"decision": d_norm}
            if "message" in action:
                out["message"] = action["message"]
            return out
    corrections.append(f"bad persuasion decision {d!r}")
    return {"decision": "yes"}
