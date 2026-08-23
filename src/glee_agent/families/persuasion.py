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
from ..theory.bayes import LieRatePosterior, kg_lie_rate, lie_rate_for_surplus

POSITIVE_WORDS = ("recommend", "great", "excellent", "worth", "quality", "buy", "good", "yes")
# Soft negatives matter: honest sellers phrase anti-recommendations as
# "ordinary unit" or "not the premium one", which must not read as neutral.
NEGATIVE_WORDS = (
    "don't", "do not", "not recommend", "skip", "pass", "avoid", "no ", "poor", "low",
    "ordinary", "standard", "basic", "average", "mediocre", "nothing special",
    "not the", "regular", "typical", "unremarkable", "isn't",
)


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
    if v is not None and price > 0 and p * v >= price * (1.0 + knobs.pers_buyer_surplus):
        # Buying is ex-ante profitable for the buyer BY A REAL MARGIN, so
        # recommendations stay credible even when uninformative and pooling
        # maximises sales. The margin is essential: this test used to be
        # `p*v >= price`, which fires at exact indifference, where a pooling
        # seller leaves the buyer precisely zero and they stop buying. In
        # those configs we ran a lie rate of 1.000 and earned 0.39-0.56 of
        # max while the field ran 0.28-0.43 and earned 0.70-0.77.
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
    # Cap the rate so the buyer keeps a real surplus. KG's x* is their
    # indifference point; sitting on it earns them nothing and kills sales.
    x_star = kg_lie_rate(p, v if v is not None else price * 1.5, price)
    x = min(
        x_star * haircut,
        lie_rate_for_surplus(p, v if v is not None else price * 1.5, price,
                             knobs.pers_buyer_surplus),
    )

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


def _honest_rounds(view: GameView) -> str:
    """Round numbers where we advised the buyer to PASS, as "3, 7".

    Anti-recommendations are the only observable that certifies the signal:
    they cost us the sale, so citing them is evidence a recommendation means
    something. Empty when we have never passed.
    """
    rounds = []
    for i, entry in enumerate(view.history, start=1):
        if not isinstance(entry, dict):
            continue
        if _msg_polarity(str(entry.get("seller_message", ""))) == "neg":
            rounds.append(str(entry.get("round", i)))
    return ", ".join(rounds[-3:])


# Recommendation templates. CRITICAL: these are drawn from ONE pool regardless
# of the product's quality. Previously a high-quality recommendation read "one
# of the strongest products I've offered ... I stand behind its quality" while
# a low-quality one read "in line with what you've seen from me" — two
# perfectly separating messages, so every lie was labelled as a lie and the
# mixing scheme carried no information at all. A signalling scheme requires a
# single message distribution; only the DECISION to recommend may depend on
# quality.
_RECOMMEND_TEMPLATES = (
    "I recommend this one — it's worth your money at this price.",
    "This is one worth taking. I recommend it.",
    "I'd buy this one in your position. Recommended.",
    "Worth it this round — I recommend going ahead.",
    "I recommend this product; it earns the price.",
)

_PASS_TEMPLATES = (
    "I'll be straight with you: I don't think this one is worth it for you. "
    "I'd rather skip a sale than lose your trust — pass on this round.",
    "Not this one. I'm giving up the sale to tell you that — pass this round.",
    "I'd pass on this one. Telling you so costs me the sale, which is exactly "
    "why my recommendations are worth something.",
)


def _seller_message(recommend: bool, view: GameView) -> str:
    """Pick a template from the signal-appropriate pool.

    Selection is deterministic in (game_id, round) so replays are stable, and
    is INDEPENDENT of the product's quality by construction — the pool is
    chosen by `recommend` alone.
    """
    pool = _RECOMMEND_TEMPLATES if recommend else _PASS_TEMPLATES
    seed = f"msg:{view.game_id}:{view.round}".encode()
    idx = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") % len(pool)
    return pool[idx]


# -------------------------------------------------------------------- buyer

def _buyer_posterior(view: GameView, ps, knobs: Knobs) -> LieRatePosterior:
    """Rebuild the seller's lie-rate posterior from history every turn.

    Stateless across calls, so concurrency and restarts are free. Both
    channels are fed: the recommendation FREQUENCY on every past round
    (available whether or not we bought), and the realized quality on the
    rounds we did buy. Older rounds are down-weighted so a seller who switches
    behaviour mid-game (trust repair, endgame pooling) is tracked.
    """
    # Prior on the seller's lie rate. NOT the KG-rational x*: measured over
    # ~900k live rounds the field lies at ~0.19-0.29, far below x* in most
    # configs, so an x*-anchored prior refuses recommendations that are
    # actually profitable. The frequency channel corrects this within a few
    # rounds anyway; the prior only has to be sane at round 1.
    post = LieRatePosterior(ps.p, knobs.pers_prior_lie, prior_strength=2.0)

    entries = [e for e in view.history if isinstance(e, dict)]
    n = len(entries)
    for age, entry in enumerate(reversed(entries)):
        w = knobs.pers_forget ** age
        polarity = _msg_polarity(str(entry.get("seller_message", "")))
        recommended = polarity == "pos"
        if polarity != "neutral" or entry.get("seller_message") is not None:
            post.observe_message(recommended, weight=w)
        if entry.get("bought") and entry.get("quality") in ("high", "low"):
            post.observe_outcome(recommended, entry.get("quality") == "high", weight=w)
    return post


def _buyer_value(p_high: float, ps) -> float:
    """Expected surplus from buying: p*v + (1-p)*u - price.

    `u` is 0 in every live config, but carrying it keeps the arithmetic right
    if a config ever prices a low-quality unit above zero.
    """
    u = ps.u if getattr(ps, "u", None) is not None else 0.0
    v = ps.v if ps.v is not None else 0.0
    return p_high * v + (1.0 - p_high) * u - ps.price


def _probe_is_worth_it(view: GameView, ps, knobs: Knobs, post, p_high: float) -> bool:
    """Knowledge-gradient: is a losing buy worth what it teaches us?

    Buying is the only way to observe a round's quality, so a purchase that is
    myopically negative can still pay if the resulting posterior would let us
    buy profitably for the remaining rounds. Replaces the old fixed
    exploration window + hard caps, which explored on a schedule rather than
    on the value of the information, and which shut off exactly in the thin
    configs where the question is live.
    """
    total = ps.total_rounds or 10
    remaining = max(total - view.round, 0)
    if remaining <= 0:
        return False

    myopic = _buyer_value(p_high, ps)
    q_hi = post.hypothetical_p_high(True, True)
    q_lo = post.hypothetical_p_high(True, False)
    future = p_high * max(0.0, _buyer_value(q_hi, ps)) + (1.0 - p_high) * max(
        0.0, _buyer_value(q_lo, ps)
    )
    if myopic + (remaining - 1) * future < 0:
        return False

    # Spend cap: never sink more than a fixed share of the game's maximum
    # attainable surplus into information.
    v = ps.v if ps.v is not None else 0.0
    max_attainable = max(total * (v - ps.price), 1e-9)
    spent = sum(
        ps.price - (v if e.get("quality") == "high" else 0.0)
        for e in view.history
        if isinstance(e, dict) and e.get("bought") and e.get("quality") == "low"
    )
    return spent < knobs.pers_probe_budget * max_attainable


def _buyer_decide(view: GameView, ps, knobs: Knobs) -> dict:
    price, p, v = ps.price, ps.p, ps.v
    if v is None or price <= 0:
        # Degenerate config; buy iff prior EV is non-negative with v unknown -> pass.
        return {"decision": "no"}

    polarity = _msg_polarity(ps.seller_message)
    total = ps.total_rounds or 10

    # Endgame: in the last round the seller faces no future punishment, so a
    # payoff-maximising one recommends everything and the message carries no
    # information. Fall back to the prior and ignore what they said.
    if view.round >= total:
        return {"decision": "yes" if _buyer_value(p, ps) > 0 else "no"}

    post = _buyer_posterior(view, ps, knobs)
    p_high = post.p_high_given(polarity == "pos")

    # Side-dependent margin, as before: thin configs need a real edge because
    # the KG-honest prior sits exactly at indifference; wide configs tolerate
    # slight slack because the field profits at the knife-edge and sitting out
    # scores below the pool median.
    thin_margin = (v - price) / price < 0.35
    margin = knobs.pers_buy_margin if thin_margin else knobs.pers_buy_margin_wide
    if p_high * v + (1.0 - p_high) * (ps.u or 0.0) > price * (1.0 + margin):
        return {"decision": "yes"}

    if polarity == "pos" and _probe_is_worth_it(view, ps, knobs, post, p_high):
        return {"decision": "yes"}
    return {"decision": "no"}


# ----------------------------------------------------------------- dispatch

def decide(view: GameView, knobs: Knobs) -> dict:
    ps = parse_persuasion(view)

    if view.action_type == "seller_message":
        recommend = _seller_wants_to_recommend(view, ps, knobs)
        if llm_client.llm_available(knobs):
            # NOTE: quality is deliberately NOT passed — see the docstring of
            # persuasion_seller_message. The signal is the only input.
            text = llm_messages.persuasion_seller_message(
                recommend, view.round, _history_summary(view), _honest_rounds(view)
            )
            # The generated pitch must READ as the intended signal under the
            # same keyword heuristics opponents (and our own lie accounting)
            # use — sales English is full of "don't miss out", which
            # classifies as an anti-recommendation. Mismatch -> template.
            if text and _msg_polarity(text) == ("pos" if recommend else "neg"):
                return {"message": text}
        return {"message": _seller_message(recommend, view)}

    if view.action_type == "seller_recommendation":
        recommend = _seller_wants_to_recommend(view, ps, knobs)
        return {"decision": "yes" if recommend else "no"}

    return _buyer_decide(view, ps, knobs)
