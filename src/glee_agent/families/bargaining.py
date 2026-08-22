"""Bargaining strategy: Rubinstein/backward-induction bound + patience-regime
play with behavior-dependent (MiCRO-style) concession and a human-fairness
branch.

Share convention: fraction of the CURRENT pot that goes to me.

Regimes (patience edge e = my_delta - opp_delta_est):
- ADVANTAGE (e >= edge, or I'm ~undiscounted vs a discounted opponent): my
  delay is cheap and theirs is not — hold near max(SPE, 0.65) capped at 0.90
  and NEVER concede on the clock; concessions must be bought by the
  opponent's own concessions (drip = 0).
- NEUTRAL (|e| < edge): the classic Boulware schedule, but the schedule is
  only an upper bound on generosity — unreciprocated concessions are capped
  at a small drip per offer.
- DISADVANTAGE (e <= -edge): my pot melts faster than theirs — anchor near
  0.58 and accept anything >= ~0.48 from round 1; a quick near-half beats a
  discounted haggle.
"""

from __future__ import annotations

import hashlib

from ..config import Knobs
from ..llm import client as llm_client
from ..llm import messages as llm_messages
from ..schema import GameView, parse_bargaining
from ..theory import opponents
from ..theory.concession import boulware
from ..theory.rubinstein import finite_spe_share, infinite_spe_share
from ..theory.targets import config_key_bargaining, get_targets

# Discount grid used when the opponent's delta is hidden (GLEE-style values);
# refined from the dataset in a later version.
HIDDEN_DELTA_PESSIMISTIC = 0.95

ADVANTAGE = "advantage"
NEUTRAL = "neutral"
DISADVANTAGE = "disadvantage"


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


# ------------------------------------------------------------ opponent model

def _bounded_adj(x: float) -> float:
    """Profile-driven adjustments are bounded to [0.02, 0.10] in magnitude."""
    return min(max(abs(x), 0.02), 0.10)


def _profile_adjust(view: GameView) -> dict:
    """Strategy adjustments from the opponent's behavioral profile.

    floor/anchor/give are additive share adjustments; `patient` marks an
    opponent who behaves as if undiscounted (stingy toward lowballs).
    """
    adj = {"floor": 0.0, "anchor": 0.0, "give": 0.0, "patient": False}
    prof = opponents.profile(view.opponent_name)
    if not prof:
        return adj
    rate = prof.get("barg_accept_rate_when_offered_lt40pct")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)):
        return adj
    n = prof.get("barg_n")
    if not isinstance(n, (int, float)) or n < 200:
        return adj  # a thin profile must not steer real money
    if rate >= 0.2:
        # Accepts lowballs: hold a higher floor, give less in the ultimatum.
        adj["floor"] = _bounded_adj(0.05)
        adj["give"] = -_bounded_adj(0.05)
    elif rate < 0.05:
        # Never takes a small share: anchoring high just burns rounds.
        adj["anchor"] = -_bounded_adj(0.05)
        adj["patient"] = True
    return adj


def _opp_delta_est(b, adj: dict) -> float:
    est = b.opp_delta if b.opp_delta is not None else HIDDEN_DELTA_PESSIMISTIC
    if adj.get("patient"):
        est = max(est, 0.99)
    return est


def _regime(view: GameView, b, knobs: Knobs, adj: dict) -> str:
    # Regimes fire only on a KNOWN opponent delta: classifying by the 0.95
    # guess put half of all games (hidden delta) into hold-forever or
    # cave-from-round-1 postures on pure speculation — mutual stonewalls
    # against mirror agents end at $0 for both.
    if b.opp_delta is None:
        return NEUTRAL
    e = b.my_delta - b.opp_delta
    if e >= knobs.barg_patience_edge or (b.my_delta >= 0.999 and b.opp_delta <= 0.97):
        return ADVANTAGE
    if e <= -knobs.barg_patience_edge:
        return DISADVANTAGE
    return NEUTRAL


def _advantage_hold(spe: float, human: bool) -> float:
    """The share an advantage-regime proposer holds at (never below)."""
    hold = min(max(spe, 0.65), 0.90)
    if human:
        # Humans spite-reject lopsided splits; shade the hold, keep a floor.
        hold = max(hold - 0.05, 0.58)
    return hold


# ------------------------------------------------------ behavior tracking

def _entry_share_to_me(entry: dict, view: GameView, pot: float) -> float | None:
    """My share of the pot in a history offer entry.

    The platform nests the gains under entry["offer"]
    ({"round", "proposer", "offer": {"player_1_gain": ...}, "decision"});
    flat entries are kept as a fallback for older fixtures."""
    if pot <= 0:
        return None
    names = {1: "alice_gain", 2: "bob_gain"}
    offer = entry.get("offer")
    sources = [offer, entry] if isinstance(offer, dict) else [entry]
    for src in sources:
        for key in (f"player_{view.my_index}_gain", names[view.my_index]):
            raw = src.get(key)
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            return min(max(float(raw) / pot, 0.0), 1.0)
    return None


def _offer_shares_by(view: GameView, b, proposer: str) -> list[float]:
    """Chronological shares offered TO ME by `proposer` in the history."""
    shares: list[float] = []
    for entry in view.history:
        if not isinstance(entry, dict) or entry.get("proposer") != proposer:
            continue
        s = _entry_share_to_me(entry, view, b.money)
        if s is not None:
            shares.append(s)
    return shares


def _my_last_offer_share(view: GameView, b) -> float | None:
    shares = _offer_shares_by(view, b, view.your_player)
    return shares[-1] if shares else None


def _opp_last_concession(view: GameView, b) -> float:
    """How much MORE the opponent's latest offer gives me vs their previous
    one (their concession toward me); 0 with fewer than two offers."""
    shares = _offer_shares_by(view, b, view.opp_player)
    if len(shares) < 2:
        return 0.0
    return max(shares[-1] - shares[-2], 0.0)


# ------------------------------------------------------------ offer targets

def _schedule(view: GameView, b, knobs: Knobs) -> tuple[str, float, float]:
    """(regime, target share for my offer this round, optimizer floor)."""
    adj = _profile_adjust(view)
    regime = _regime(view, b, knobs, adj)
    human = view.opponent_type == "human"
    opp_delta = b.opp_delta if b.opp_delta is not None else HIDDEN_DELTA_PESSIMISTIC
    spe = _spe_bound(view, b.my_delta, opp_delta)

    if human:
        start, floor = knobs.barg_anchor_human, knobs.barg_floor_human
    else:
        # SPE can sit very high near an endgame that favors me; cap the greed —
        # real opponents reject "rational" lowballs and deadlock costs the pot.
        start, floor = knobs.barg_anchor_agent, max(knobs.barg_floor_agent, min(spe, 0.70))
    start = max(start + adj["anchor"], 0.50)
    floor = min(max(floor + adj["floor"], 0.0), start)

    # Effective schedule length: horizon if known, else a virtual deadline set
    # by impatience — with heavy discounting the pot decays fast, settle early.
    horizon = _rounds_left(view)
    joint_delta = min(b.my_delta, opp_delta)
    virtual = 3 if joint_delta < 0.9 else (6 if joint_delta < 1.0 else 10)
    T = min(horizon, virtual) if horizon is not None else virtual

    # Round index within MY schedule: each of my proposals advances it.
    my_round = (view.round + 1) // 2
    target = boulware(my_round, T, start, floor, knobs.barg_beta)

    if regime == ADVANTAGE:
        # My delay is cheap, theirs is not: the schedule may open above the
        # hold but never decays below it — time alone buys them nothing.
        # Marathon escape: against an equally stubborn opponent an eternal
        # hold ends at $0 for both, so the hold steps down late.
        hold = _advantage_hold(spe, human)
        if view.round > 30:
            hold = min(hold, 0.65)
        elif view.round > 16:
            hold = min(hold, 0.75)
        target = max(target, hold)
        opt_floor = hold
    elif regime == DISADVANTAGE:
        # My pot melts faster than theirs: anchor low and close fast.
        anchor = min(max(0.58 + adj["floor"] + adj["anchor"], 0.50), 0.65)
        target = min(target, anchor)
        opt_floor = knobs.barg_dis_accept
    else:
        opt_floor = floor

    # Deadlock pressure: every rejected offer of mine says my price is too
    # high — concede beyond the schedule rather than ride into the endgame.
    # Never in the advantage regime: there, deadlock costs THEM more.
    if regime != ADVANTAGE:
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
    return regime, target, opt_floor


def _target_share(view: GameView, b, knobs: Knobs) -> float:
    """Regime-aware target share for my offers this round."""
    return _schedule(view, b, knobs)[1]


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


def _endgame_cap(view: GameView, b, knobs: Knobs) -> float | None:
    """Near the horizon SPE is what a rational opponent holds out for; it
    caps my share (same rule as inside _schedule)."""
    left = _rounds_left(view)
    if left is None or left > 4:
        return None
    human = view.opponent_type == "human"
    opp_delta = b.opp_delta if b.opp_delta is not None else HIDDEN_DELTA_PESSIMISTIC
    spe = _spe_bound(view, b.my_delta, opp_delta)
    return max(spe, 0.45 if human else 0.25)


def _optimized_offer_share(view: GameView, b, knobs: Knobs, boulware_share: float) -> float | None:
    """Pick my share by maximizing EV against the empirical field accept
    curve. Returns None when the curve is too thin or agrees with Boulware."""
    tg = get_targets()
    left = _rounds_left(view)
    human = view.opponent_type == "human"
    # Continuation if they reject: my next-round Boulware aspiration,
    # discounted and haircut by the odds it actually lands — CAPPED at the
    # fleet's realized post-reject share (0.418 measured, mild headroom):
    # pricing continuation off aspiration instead of reality is what drove
    # the optimizer into the keep-0.75 corner (3.7% round-1 conversion).
    cont = b.my_delta * min(boulware_share * knobs.barg_cont_realism, 0.46)
    best_share = None
    best_ev = -1.0
    n_with_data = 0
    for i in range(21):  # candidates 0.30 .. 0.80 step 0.025
        share = 0.30 + 0.025 * i
        p = tg.barg_accept_prob(1.0 - share, left, human)
        if p is None:
            continue
        n_with_data += 1
        ev = share * p + (1.0 - p) * cont
        if ev > best_ev:
            best_ev, best_share = ev, share
    if best_share is None or n_with_data < 5:
        return None
    if abs(best_share - boulware_share) <= 0.05:
        return None  # empirics agree with theory; keep the schedule
    return best_share


def _final_round_share(view: GameView, knobs: Knobs, adj: dict) -> float:
    """Final-round proposer: argmax over give of (1-give)*P(accept | give,
    rl=1) on the empirical accept curve (continuation is 0 — the responder
    gets nothing by rejecting, but spite rejection is real). Falls back to
    the flat knob when the rl=1 curve has no usable buckets."""
    tg = get_targets()
    human = view.opponent_type == "human"
    best_give = None
    best_ev = -1.0
    n_with_data = 0
    for i in range(9):  # give in 0.05 .. 0.45 step 0.05
        give = round(0.05 + 0.05 * i, 2)
        p = tg.barg_accept_prob(give, 1, human)
        if p is None:
            continue
        n_with_data += 1
        ev = (1.0 - give) * p
        if ev > best_ev:
            best_ev, best_give = ev, give
    if best_give is None or n_with_data < 3:
        best_give = knobs.barg_final_round_give  # curve too thin to trust
    give = min(max(best_give + adj["give"], 0.05), 0.45)
    if human:
        # The human accept curve silently backs off to the agent-dominated
        # pooled counts when thin; a near-epsilon ultimatum against a human
        # is a spite-rejection machine. Hard floor regardless of the curve.
        give = max(give, 0.25)
    return 1.0 - give


# Offer messages. Deterministic, zero-cost, and written in the register the
# literature actually validates: explicit concession accounting ("I've moved,
# you haven't"), a stated reason, and firm commitment language late. Hedged or
# analytic phrasing measurably does not move an LLM opponent; declarative
# commitment does. We never READ the opponent's text — reading it lowers
# surplus for every model tested — so this is a write-only channel.
_BARG_OPEN = (
    "This split reflects what each round of delay costs us — I'd rather close now than shrink the pot for both of us.",
    "Here's my opening: the sooner we agree, the more there is to divide. Delay only burns value.",
)
_BARG_HOLD = (
    "I've already moved toward you and you've held still. This is where I am until I see movement back.",
    "My position hasn't changed because yours hasn't. Meet me and I'll meet you.",
)
_BARG_CONCEDE = (
    "I've come down again — that's another move on my part. We're close now; let's not burn the pot arguing over the rest.",
    "That's me moving toward you a second time. Take it and we both stop losing value to delay.",
)
_BARG_FINAL = (
    "This is my final offer. Rejecting it leaves us both with nothing, and I'd rather we both walked away with something.",
    "Last round — this is my limit. Accepting beats the zero we both get otherwise.",
)


def _barg_message(view: GameView, share: float, conceded: bool, left: int | None) -> str:
    if left is not None and left <= 1:
        pool = _BARG_FINAL
    elif view.round <= 1:
        pool = _BARG_OPEN
    elif conceded:
        pool = _BARG_CONCEDE
    else:
        pool = _BARG_HOLD
    seed = f"bmsg:{view.game_id}:{view.round}".encode()
    return pool[int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") % len(pool)]


def decide(view: GameView, knobs: Knobs) -> dict:
    b = parse_bargaining(view)
    mine_key, opp_key = _gain_names(view)

    if view.action_type == "offer":
        pot = b.money
        regime, share, opt_floor = _schedule(view, b, knobs)

        left = _rounds_left(view)
        if left is not None and left <= 1:
            # Final-round proposer: EV-optimal give from the field accept
            # curve, never epsilon — spite rejection is real.
            share = _final_round_share(view, knobs, _profile_adjust(view))
        else:
            # The optimizer runs only in the NEUTRAL regime, and may only
            # ACCELERATE concession, never ask above the schedule: the
            # dataset accept curve is 2-4x optimistic vs the live field
            # (give=0.25 priced 14.8%, live 3.7%), so its greedy direction
            # is exactly where it is least trustworthy. Advantage holds by
            # design; disadvantage keeps its fast-closing anchor.
            if regime == NEUTRAL:
                optimized = _optimized_offer_share(view, b, knobs, share)
                if optimized is not None:
                    share = min(max(optimized, opt_floor), share)
            # Behavior-dependent concession (MiCRO): between my consecutive
            # offers I concede at most what the opponent just conceded to me
            # (plus a drip; no drip at all in the advantage regime). The time
            # schedule above is an upper bound on generosity, not a promise.
            my_last = _my_last_offer_share(view, b)
            if my_last is not None:
                drip = 0.0 if regime == ADVANTAGE else knobs.barg_drip
                step = max(_opp_last_concession(view, b), drip)
                share = min(max(share, my_last - step), my_last)
            # Endgame SPE cap always wins, even over the behavior cap.
            cap = _endgame_cap(view, b, knobs)
            if cap is not None:
                share = min(share, cap)

        my_gain = pot * share
        # Humans accept clean numbers more readily.
        if view.opponent_type == "human" and pot >= 100:
            my_gain = round(my_gain / 10) * 10
        my_gain = min(max(my_gain, 0.0), pot)
        out = {mine_key: my_gain, opp_key: pot - my_gain}
        if view.messages_allowed:
            my_last = _my_last_offer_share(view, b)
            conceded = my_last is not None and share < my_last - 1e-9
            msg = None
            if llm_client.llm_available(knobs):
                share_pct = int(round(100.0 * my_gain / pot)) if pot > 0 else 50
                msg = llm_messages.bargaining_offer_message(
                    share_pct, view.round, view.opponent_type == "human"
                )
            out["message"] = msg or _barg_message(view, share, conceded, left)
        return out

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

    adj = _profile_adjust(view)
    regime = _regime(view, b, knobs, adj)

    # Empirical percentile accept: my realized payoff if I accept NOW
    # (discount included — the pool holds realized payoffs) already beats
    # knobs.barg_accept_pct of every payoff ever scored on this exact
    # config+role. A top-of-pool payoff in hand beats a gamble. This runs
    # before the regime gates: the pool already encodes what patience is
    # worth on this config.
    accept_payoff = offered * (b.my_delta ** max(view.round - 1, 0))
    if accept_payoff > 0:
        tg = get_targets()
        key = config_key_bargaining(view.state)
        pct = tg.payoff_percentile("bargaining", key, view.your_player, accept_payoff)
        # q95 gate mirrors negotiation: on a zero-heavy pool a crumb can
        # "rank" high; require a real fraction of what top games get.
        q95 = tg.payoff_quantile("bargaining", key, view.your_player, 0.95)
        if (
            pct is not None
            and pct >= knobs.barg_accept_pct
            and q95 is not None
            and accept_payoff >= 0.25 * q95
        ):
            return {"decision": "accept"}

    if regime == ADVANTAGE:
        # Waiting is nearly free for me and expensive for them: hold out for
        # a hold-level share until the endgame parity rules take over.
        # Humans spite-punish holds; the gate shades down for them. Marathon
        # escape mirrors the offer path so mutual stonewalls resolve.
        late = left is not None and left <= 4
        opp_delta = b.opp_delta if b.opp_delta is not None else HIDDEN_DELTA_PESSIMISTIC
        spe = _spe_bound(view, b.my_delta, opp_delta)
        gate = min(spe * 0.85, knobs.barg_adv_hold)
        if view.opponent_type == "human":
            gate = min(gate, 0.62)
        if view.round > 30:
            gate = min(gate, 0.52)
        elif view.round > 16:
            gate = min(gate, 0.60)
        if not late and offered_share < gate:
            return {"decision": "reject"}
    elif regime == DISADVANTAGE and offered_share >= knobs.barg_dis_accept:
        # My pot melts faster than theirs: a near-half NOW beats a
        # discounted haggle, from round 1.
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
