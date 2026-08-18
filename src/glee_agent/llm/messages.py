"""Prompt builders + post-processing for the three game families.

Each generator returns a cleaned message string or None, and NEVER raises.
Prompts must never leak our private state: no reservation values, no
discount factors, and — when we recommend a product we do not know to be
good — no hint of that in the seller prompt (bluffing is legal; leaking is
just bad play).
"""

from __future__ import annotations

import logging

from . import client

logger = logging.getLogger("glee_agent.llm")

MAX_LEN = 600

_REFUSAL_MARKERS = (
    "i cannot",
    "i can't",
    "i am unable",
    "i'm unable",
    "as an ai",
    "as a language model",
    "i'm sorry",
    "i am sorry",
)

_SYSTEM = (
    "You ghostwrite short in-game chat messages for a player in an economic "
    "game. Respond with ONLY the message itself: a single paragraph of plain "
    "text, no markdown, no quotation marks, no preamble, no explanations, "
    "under 500 characters."
)


def _finish(text: str | None) -> str | None:
    """Strip quotes, flatten newlines, cap at MAX_LEN; None if unusable."""
    if not text:
        return None
    text = " ".join(text.replace("\r", " ").replace("\n", " ").replace("\t", " ").split())
    text = text.strip().strip('"').strip("'").strip()
    if not text:
        return None
    lowered = text.lower()
    if any(marker in lowered for marker in _REFUSAL_MARKERS):
        return None
    if len(text) > MAX_LEN:
        text = text[:MAX_LEN].rstrip()
    return text


def persuasion_seller_message(
    recommend: bool, quality_known_high: bool, round_: int, history_summary: str
) -> str | None:
    """A varied, natural 1-3 sentence sales message.

    When recommending without knowing the product is good, the prompt is a
    plain enthusiastic-pitch request: it must never mention product grade or
    any doubt about it (see module docstring).
    """
    try:
        history = history_summary.strip() or "first round"
        if recommend and quality_known_high:
            prompt = (
                f"Round {round_} of a repeated sales game; recent outcomes: {history}. "
                "Today's product is genuinely top quality and you are recommending it. "
                "Write a confident, concrete 1-3 sentence pitch to the buyer that "
                "stands behind the product and invites them to buy."
                "IMPORTANT: never use the words don't, no, not, pass, skip, or avoid — pure positive phrasing only."
                " Vary your phrasing "
                f"from earlier rounds (this is round {round_})."
            )
        elif recommend:
            prompt = (
                f"Round {round_} of a repeated sales game; recent outcomes: {history}. "
                "You are recommending today's product. Write an enthusiastic, upbeat "
                "1-2 sentence generic pitch that highlights your track record and the "
                "deal's value, and invites the buyer to purchase."
                "IMPORTANT: never use the words don't, no, not, pass, skip, or avoid — pure positive phrasing only."
                " Keep it general — "
                "no specific product claims. Vary your phrasing from earlier rounds "
                f"(this is round {round_})."
            )
        else:
            prompt = (
                f"Round {round_} of a repeated sales game; recent outcomes: {history}. "
                "You are honestly steering the buyer AWAY from today's product to "
                "build long-term trust. Write a candid 1-2 sentence message advising "
                "them to pass this round, signalling that your recommendations mean "
                "something. Vary your phrasing from earlier rounds."
            )
        return _finish(client.complete(prompt, system=_SYSTEM))
    except Exception as e:  # noqa: BLE001 — message layer must never raise
        logger.warning("persuasion_seller_message failed: %s", e)
        return None


def bargaining_offer_message(my_share_pct: int, round_: int, human_opponent: bool) -> str | None:
    """A short fairness-framed pitch for a proposed split. Never mentions our
    discount factor or strategy."""
    try:
        tone = (
            "You are talking to a person, so sound warm and reasonable."
            if human_opponent
            else "Keep it brief and businesslike."
        )
        prompt = (
            f"You are splitting a pot of money with a partner. In round {round_} you "
            f"propose keeping {int(my_share_pct)}% and giving them {100 - int(my_share_pct)}%. "
            "Write a 1-2 sentence message framing this split as fair and worth taking "
            "now, noting that the pot shrinks as rounds pass so agreeing early leaves "
            f"more for both of you. {tone} Do not mention percentages other than the "
            "offer itself, and never discuss your own tactics."
        )
        return _finish(client.complete(prompt, system=_SYSTEM))
    except Exception as e:  # noqa: BLE001
        logger.warning("bargaining_offer_message failed: %s", e)
        return None


def negotiation_offer_message(role: str, price: float, round_: int) -> str | None:
    """A one-liner justifying a price. Never mentions our reservation value."""
    try:
        stance = "asking" if role == "seller" else "offering to pay"
        prompt = (
            f"You are the {role} in a price negotiation over a product, round {round_}. "
            f"You are {stance} {price:.2f}. Write ONE short sentence to the other side "
            "justifying this price as fair and serious (appeal to the product's merit "
            "or market conditions). Do not mention any limits, walk-away points, or "
            "private numbers of your own — only the price itself."
        )
        return _finish(client.complete(prompt, system=_SYSTEM))
    except Exception as e:  # noqa: BLE001
        logger.warning("negotiation_offer_message failed: %s", e)
        return None
