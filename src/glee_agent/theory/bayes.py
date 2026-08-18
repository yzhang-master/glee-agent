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
