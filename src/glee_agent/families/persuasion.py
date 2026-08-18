"""Persuasion strategy.

Seller: honest when honesty is free (p*v >= price); otherwise Kamenica-
Gentzkow mixing with a safety haircut and trust repair.
Buyer: Beta-tracked P(high | recommendation) anchored at the KG-honest prior,
UCB exploration early, expected-value rule throughout.
"""

from __future__ import annotations

import hashlib

from ..config import Knobs
from ..llm import client as llm_client
from ..llm import messages as llm_messages
from ..schema import GameView, parse_persuasion
from ..theory.bayes import BetaTracker, kg_lie_rate, posterior_high_given_rec

POSITIVE_WORDS = ("recommend", "great", "excellent", "worth", "quality", "buy", "good", "yes")
NEGATIVE_WORDS = ("don't", "do not", "not recommend", "skip", "pass", "avoid", "no ", "poor", "low")


def _det_coin(view: GameView, threshold: float) -> bool:
    """Deterministic pseudo-coin (game_id + round) so replays are stable:
    True with probability ~= threshold."""
    seed = f"{view.game_id}:{view.round}".encode()
    h = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")
    return (h / 2**64) < threshold


def _msg_polarity(text: str | None) -> str:
    """'pos' | 'neg' | 'neutral' from a seller message (regex-free heuristic;
    the LLM extractor can refine this in a later version)."""
    if not text:
        return "neutral"
    t = text.lower()
    if t.strip() in ("yes", "no"):
        return "pos" if t.strip() == "yes" else "neg"
    neg = any(w in t for w in NEGATIVE_WORDS)
    pos = any(w in t for w in POSITIVE_WORDS)
    if neg and not pos:
        return "neg"
    if pos and not neg:
        return "pos"
    if neg and pos:
        return "neg"  # mixed message: hedged sellers are hiding something
    return "neutral"


# ------------------------------------------------------------------- seller

def _sold_lies_and_trust(view: GameView) -> tuple[int, int, int]:
    """(low-quality sales, consecutive recent non-purchases, rounds played)."""
    lies = 0
    consecutive_pass = 0
    played = 0
    for entry in view.history:
        if not isinstance(entry, dict):
            continue
        played += 1
        bought = bool(entry.get("bought"))
        if bought and entry.get("quality") == "low":
            lies += 1
        consecutive_pass = 0 if bought else consecutive_pass + 1
    return lies, consecutive_pass, played


def _seller_wants_to_recommend(view: GameView, ps, knobs: Knobs) -> bool:
    quality = ps.current_quality
    if quality == "high":
        return True

    # Low quality (or unknown — treat unknown as low for safety).
    price, p = ps.price, ps.p
    v = ps.v
    if v is not None and p * v >= price and price > 0:
        # Buying is ex-ante profitable for the buyer: recommendations stay
        # credible even when uninformative; recommending always maximizes sales.
        return True
    if v is None:
        # Blind seller (buyer's valuation hidden): the KG rate is
        # uncomputable, and the dataset field simply pools (recommends
        # everything) — its sellers earn the pool median doing so. Guessing
        # a rate under-sells against that pool.
        return True

    haircut = (
        knobs.pers_kg_haircut_human
        if view.opponent_type == "human"
        else knobs.pers_kg_haircut
    )
    x_star = kg_lie_rate(p, v if v is not None else price * 1.5, price)
    x = x_star * haircut

    _, consecutive_pass, _ = _sold_lies_and_trust(view)
    if consecutive_pass >= 2:
        # Trust repair: strict honesty until purchases resume.
        return False

    # Was the previous low-quality product already oversold recently? Track my
    # own empirical lie rate and stay under x.
    my_recs_on_low = 0
    my_lows = 0
    for entry in view.history:
        if not isinstance(entry, dict) or entry.get("quality") != "low":
            continue
        my_lows += 1
        if _msg_polarity(str(entry.get("seller_message", ""))) == "pos":
            my_recs_on_low += 1
    if my_lows > 0 and my_recs_on_low / my_lows >= x:
        return False

    return _det_coin(view, x)


def _history_summary(view: GameView) -> str:
    """Compact last-3-rounds digest for the LLM prompt, e.g.
    "r5:rec/buy-hi". Quality is encoded hi/lo so the prompt never carries
    the word "low"."""
    entries = [e for e in view.history if isinstance(e, dict)]
    start = len(entries) - len(entries[-3:]) + 1
    parts = []
    for i, entry in enumerate(entries[-3:], start=start):
        pol = _msg_polarity(str(entry.get("seller_message", "")))
        rec = {"pos": "rec", "neg": "norec"}.get(pol, "neut")
        buy = "buy" if entry.get("bought") else "pass"
        q = {"high": "hi", "low": "lo"}.get(entry.get("quality"), "?")
        parts.append(f"r{i}:{rec}/{buy}-{q}")
    return " ".join(parts)


def _seller_message(recommend: bool, quality: str | None) -> str:
    if recommend:
        if quality == "high":
            return (
                "I can honestly say this is one of the strongest products I've "
                "offered you — I stand behind its quality and recommend it."
            )
        return "This product meets the standard you've seen from me. I recommend it."
    return (
        "I'll be straight with you: I don't think this one is worth it for you. "
        "I'd rather skip a sale than lose your trust — pass on this round."
    )


# -------------------------------------------------------------------- buyer

def _buyer_tracker(view: GameView, ps) -> tuple[BetaTracker, BetaTracker]:
    """Rebuild (rec_tracker, neg_tracker) posteriors from history every turn —
    stateless across calls, so concurrency and restarts are free."""
    x_star = kg_lie_rate(ps.p, ps.v if ps.v is not None else 0.0, ps.price)
    prior_rec = posterior_high_given_rec(ps.p, x_star) if ps.v else ps.p
    rec = BetaTracker(prior_mean=max(prior_rec, ps.p), prior_strength=2.0)
    neg = BetaTracker(prior_mean=min(ps.p * 0.3 + 1e-6, 1.0), prior_strength=2.0)

    for entry in view.history:
        if not isinstance(entry, dict) or not entry.get("bought"):
            continue
        quality = entry.get("quality")
        if quality not in ("high", "low"):
            continue
        polarity = _msg_polarity(str(entry.get("seller_message", "")))
        if polarity == "neg":
            neg.update(quality == "high")
        else:
            rec.update(quality == "high")
    return rec, neg


def _buyer_decide(view: GameView, ps, knobs: Knobs) -> dict:
    price, p, v = ps.price, ps.p, ps.v
    if v is None or price <= 0:
        # Degenerate config; buy iff prior EV is non-negative with v unknown -> pass.
        return {"decision": "no"}

    polarity = _msg_polarity(ps.seller_message)
    rec, neg = _buyer_tracker(view, ps)

    if polarity == "neg":
        # A seller advising against a sale forfeits revenue — highly credible.
        p_high = neg.mean
    elif polarity == "pos":
        p_high = rec.mean
    else:
        p_high = p  # uninformative message: prior

    # Exploratory (UCB) buying pays for information with expected losses, so
    # it must be affordable: skip it on thin margins (v/price close to 1
    # needs near-perfect honesty to profit), stop after the game has already
    # lost money, and never sink more than ~2 speculative buys per game.
    total = ps.total_rounds or 10
    thin_margin = (v - price) / price < 0.35 if price > 0 else True
    explore_buys = sum(
        1
        for e in view.history
        if isinstance(e, dict) and e.get("bought") and e.get("quality") == "low"
    )
    exploring = (
        view.round <= max(1, int(total * knobs.pers_explore_frac))
        and not thin_margin
        and ps.my_total_payoff >= 0
        and explore_buys < 2
    )
    if exploring and polarity == "pos":
        p_high = rec.ucb(c=1.0)

    # Side-dependent margin. THIN configs: the KG-honest prior sits exactly
    # at indifference, so demand a real edge — any extra seller lying turns
    # ">=" into guaranteed losses. WIDE configs: the field earns the pool
    # median by buying at the knife-edge; sitting out scores ~0, below most
    # of the pool, so tolerate slight prior slack and let evidence cut us
    # off if the seller over-lies.
    margin = knobs.pers_buy_margin if thin_margin else knobs.pers_buy_margin_wide
    return {"decision": "yes" if p_high * v > price * (1.0 + margin) else "no"}


# ----------------------------------------------------------------- dispatch

def decide(view: GameView, knobs: Knobs) -> dict:
    ps = parse_persuasion(view)

    if view.action_type == "seller_message":
        recommend = _seller_wants_to_recommend(view, ps, knobs)
        if llm_client.llm_available(knobs):
            text = llm_messages.persuasion_seller_message(
                recommend, ps.current_quality == "high", view.round, _history_summary(view)
            )
            # The generated pitch must READ as the intended signal under the
            # same keyword heuristics opponents (and our own lie accounting)
            # use — sales English is full of "don't miss out", which
            # classifies as an anti-recommendation. Mismatch -> template.
            if text and _msg_polarity(text) == ("pos" if recommend else "neg"):
                return {"message": text}
        return {"message": _seller_message(recommend, ps.current_quality)}

    if view.action_type == "seller_recommendation":
        recommend = _seller_wants_to_recommend(view, ps, knobs)
        return {"decision": "yes" if recommend else "no"}

    return _buyer_decide(view, ps, knobs)
