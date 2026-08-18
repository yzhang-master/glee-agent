"""Alternating-offers bargaining solutions.

Conventions: shares are the PROPOSER's fraction of the current pot.
delta_p = proposer's per-round discount, delta_r = responder's.
"""

from __future__ import annotations


def infinite_spe_share(delta_p: float, delta_r: float) -> float:
    """Proposer's SPE share in the infinite-horizon Rubinstein game.

    share = (1 - delta_r) / (1 - delta_p * delta_r); both players patient
    (deltas -> 1) gives 1/2 in the limit, fully impatient responder gives 1.
    """
    delta_p = min(max(delta_p, 0.0), 1.0)
    delta_r = min(max(delta_r, 0.0), 1.0)
    denom = 1.0 - delta_p * delta_r
    if denom <= 1e-12:  # both deltas 1: equal split is the standard limit
        return 0.5
    return (1.0 - delta_r) / denom


def finite_spe_share(rounds_left: int, delta_p: float, delta_r: float) -> float:
    """Proposer's SPE share with `rounds_left` rounds remaining (this one
    included), by backward induction. In the last round the proposer takes
    everything (responder accepts anything over 0).

    Roles alternate each round; discounts swap accordingly.
    """
    if rounds_left <= 1:
        return 1.0
    # Next round the current responder proposes; their SPE share then is s'.
    s_next = finite_spe_share(rounds_left - 1, delta_r, delta_p)
    # Responder accepts x iff x >= delta_r * (their payoff proposing next round).
    responder_keep = delta_r * s_next
    return max(0.0, min(1.0, 1.0 - responder_keep))
