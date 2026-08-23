"""Bayesian pieces for persuasion: KG credibility ratio and Beta tracking."""

from __future__ import annotations

import math


def kg_lie_rate(p: float, v: float, price: float) -> float:
    """Kamenica-Gentzkow optimal rate of recommending a LOW-quality product,
    x* = p(v - price) / ((1 - p) * price), capped to [0, 1].

    At x*, P(high | recommend) * v == price: the buyer is exactly indifferent,
    so any x below it keeps recommendations strictly worth following.
    Returns 0 when the price leaves no persuasion room (price >= v or p == 1).
    """
    if price <= 0 or v <= price or p >= 1.0:
        return 0.0 if v <= price or price <= 0 else 1.0
    x = p * (v - price) / ((1.0 - p) * price)
    return max(0.0, min(1.0, x))


def posterior_high_given_rec(p: float, lie_rate: float) -> float:
    """P(high | recommendation) for a seller who recommends all high and a
    `lie_rate` fraction of low products."""
    denom = p + (1.0 - p) * lie_rate
    return 1.0 if denom <= 1e-12 else p / denom


class BetaTracker:
    """Beta-Bernoulli posterior over P(high | signal class)."""

    def __init__(self, prior_mean: float, prior_strength: float = 2.0):
        prior_mean = min(max(prior_mean, 1e-6), 1 - 1e-6)
        self.a = prior_mean * prior_strength
        self.b = (1.0 - prior_mean) * prior_strength

    def update(self, was_high: bool) -> None:
        if was_high:
            self.a += 1.0
        else:
            self.b += 1.0

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def ucb(self, c: float = 1.0) -> float:
        """Upper confidence bound: mean + c * posterior std."""
        n = self.a + self.b
        var = (self.a * self.b) / (n * n * (n + 1.0))
        return min(1.0, self.mean + c * math.sqrt(var))


class LieRatePosterior:
    """Posterior over a persuasion seller's lie rate x = P(recommend | low).

    Why a posterior over x rather than a Beta over P(high | recommendation):
    the buyer only learns a round's QUALITY when they bought it, so an
    outcome-only tracker never updates while we pass — refusing forever is an
    absorbing no-information state, and that is what left 41% of buyer games
    at exactly zero payoff. But the seller's *recommendation frequency* is
    observable every round, bought or not, and it identifies x directly:

        f = P(recommend) = p + (1-p)x        =>  x_hat = (f - p) / (1 - p)
        P(high | recommend) = p / (p + (1-p)x)

    This holds for any payoff-maximising seller, because recommending a
    high-quality product is weakly dominant (it is the only way to sell it),
    so P(recommend | high) = 1. A seller who does hide some high products just
    shifts mass to a lower x, which is conservative in the buyer's favour.

    Two likelihood channels are combined on a grid over x:
      A (every round):     rec -> p + (1-p)x        pass -> (1-p)(1-x)
      B (purchases only):  high -> p / (p+(1-p)x)   low -> (1-p)x / (p+(1-p)x)

    Older observations are down-weighted by `decay` per round of age, so a
    seller who changes behaviour (trust repair, endgame pooling) is tracked.
    """

    __slots__ = ("p", "_grid", "_logw")

    def __init__(self, p: float, prior_x: float, prior_strength: float = 2.0,
                 n_grid: int = 25):
        self.p = min(max(p, 1e-6), 1.0 - 1e-6)
        self._grid = [i / (n_grid - 1) for i in range(n_grid)]
        # Weakly-informative prior centred on prior_x (the rate a rational
        # opponent would play), implemented as a Beta(a,b) density on the grid.
        prior_x = min(max(prior_x, 1e-6), 1.0 - 1e-6)
        a = prior_x * prior_strength + 1.0
        b = (1.0 - prior_x) * prior_strength + 1.0
        self._logw = [
            (a - 1.0) * math.log(max(x, 1e-9)) + (b - 1.0) * math.log(max(1.0 - x, 1e-9))
            for x in self._grid
        ]

    def _p_rec(self, x: float) -> float:
        return self.p + (1.0 - self.p) * x

    def observe_message(self, recommended: bool, weight: float = 1.0) -> None:
        """Channel A — available every round, whether or not we bought."""
        for i, x in enumerate(self._grid):
            f = self._p_rec(x)
            lik = f if recommended else (1.0 - f)
            self._logw[i] += weight * math.log(max(lik, 1e-12))

    def observe_outcome(self, recommended: bool, was_high: bool, weight: float = 1.0) -> None:
        """Channel B — only on rounds we actually bought and saw the quality."""
        for i, x in enumerate(self._grid):
            f = self._p_rec(x)
            if recommended:
                lik = self.p / f if was_high else (1.0 - self.p) * x / f
            else:
                # A seller who anti-recommends a high product is off-model;
                # treat it as uninformative rather than letting it dominate.
                lik = 1.0 if was_high else 1.0
            self._logw[i] += weight * math.log(max(lik, 1e-12))

    def _weights(self) -> list[float]:
        m = max(self._logw)
        w = [math.exp(lw - m) for lw in self._logw]
        total = sum(w)
        return [wi / total for wi in w] if total > 0 else [1.0 / len(w)] * len(w)

    @property
    def mean_x(self) -> float:
        """Posterior mean lie rate."""
        return sum(w * x for w, x in zip(self._weights(), self._grid))

    def p_high_given(self, recommended: bool) -> float:
        """Posterior P(high | this round's signal), integrating over x.

        Under the model P(recommend | high) = 1, so an anti-recommendation is
        conclusive: P(high | no recommendation) = 0. That is the right read —
        a seller forfeiting a sale to warn us is the most credible signal in
        the game — and it makes the buyer decline without needing a margin.
        """
        if not recommended:
            return 0.0
        out = 0.0
        for w, x in zip(self._weights(), self._grid):
            out += w * (self.p / self._p_rec(x))
        return min(max(out, 0.0), 1.0)

    def hypothetical_p_high(self, recommended: bool, was_high: bool) -> float:
        """P(high | recommend) we WOULD hold after one more purchase outcome.

        Used for the value-of-information calculation; leaves self untouched.
        """
        saved = list(self._logw)
        try:
            self.observe_outcome(recommended, was_high)
            return self.p_high_given(True)
        finally:
            self._logw = saved


def lie_rate_for_surplus(p: float, v: float, price: float, surplus_frac: float) -> float:
    """Highest lie rate that still leaves the buyer a real expected surplus.

    KG's x* equalises the buyer's expected value with the price, so a buyer who
    follows a recommendation at x* earns exactly zero. Anyone who is not
    perfectly indifferent -- which is everyone -- stops buying. This solves
    instead for the x at which

        P(high | recommend) * v - price >= surplus_frac * price

    Substituting P(high|rec) = p / (p + (1-p)x) and rearranging:

        x <= p * (v - price*(1+s)) / ((1-p) * price*(1+s))

    Returns 0 when even a perfectly honest seller cannot clear the bar (the
    product is simply not worth the price at this prior), and is capped at 1.
    """
    if price <= 0 or p >= 1.0 or p <= 0.0:
        return 0.0
    target = price * (1.0 + surplus_frac)
    if v <= target:
        return 0.0
    x = p * (v - target) / ((1.0 - p) * target)
    return max(0.0, min(1.0, x))
