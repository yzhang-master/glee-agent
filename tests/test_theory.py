import math

from glee_agent.theory.bayes import BetaTracker, kg_lie_rate, posterior_high_given_rec
from glee_agent.theory.concession import boulware
from glee_agent.theory.rubinstein import finite_spe_share, infinite_spe_share


class TestRubinstein:
    def test_infinite_symmetric(self):
        # delta = 0.9 both: share = (1-0.9)/(1-0.81) = 0.5263...
        assert math.isclose(infinite_spe_share(0.9, 0.9), 0.1 / 0.19, rel_tol=1e-9)

    def test_infinite_impatient_responder(self):
        # Responder delta 0 -> proposer takes all.
        assert infinite_spe_share(0.9, 0.0) == 1.0

    def test_infinite_patient_limit(self):
        assert infinite_spe_share(1.0, 1.0) == 0.5

    def test_finite_last_round(self):
        assert finite_spe_share(1, 0.9, 0.9) == 1.0

    def test_finite_two_rounds(self):
        # Responder can get everything next round, discounted: keep = 0.9.
        assert math.isclose(finite_spe_share(2, 0.9, 0.9), 0.1, rel_tol=1e-9)

    def test_finite_converges_to_infinite(self):
        fin = finite_spe_share(39, 0.9, 0.8)
        inf = infinite_spe_share(0.9, 0.8)
        assert abs(fin - inf) < 0.01

    def test_proposer_advantage(self):
        # Proposer share always >= 1/2 when deltas equal.
        for d in (0.5, 0.8, 0.95, 0.99):
            assert infinite_spe_share(d, d) >= 0.5


class TestBoulware:
    def test_endpoints(self):
        assert boulware(1, 10, 0.8, 0.55, 2.5) == 0.8
        assert boulware(10, 10, 0.8, 0.55, 2.5) == 0.55

    def test_monotone_decreasing(self):
        values = [boulware(t, 10, 0.8, 0.55, 2.5) for t in range(1, 11)]
        assert all(a >= b for a, b in zip(values, values[1:]))

    def test_concedes_late(self):
        # With beta > 1, at the midpoint we're still closer to the start.
        mid = boulware(5, 9, 1.0, 0.0, 3.0)
        assert mid > 0.7

    def test_degenerate_schedule(self):
        assert boulware(1, 1, 0.8, 0.55, 2.5) == 0.55
        assert boulware(99, 10, 0.8, 0.55, 2.5) == 0.55


class TestKG:
    def test_lie_rate_formula(self):
        # p=0.5, v=120, price=100: x* = 0.5*20/(0.5*100) = 0.2
        assert math.isclose(kg_lie_rate(0.5, 120, 100), 0.2, rel_tol=1e-9)

    def test_no_room_when_price_exceeds_value(self):
        assert kg_lie_rate(0.5, 90, 100) == 0.0

    def test_indifference_at_x_star(self):
        p, v, price = 0.4, 200, 100
        x = kg_lie_rate(p, v, price)
        # At x*, posterior * v == price.
        assert math.isclose(posterior_high_given_rec(p, x) * v, price, rel_tol=1e-9)

    def test_capped_at_one(self):
        # Very cheap product relative to value: lying freely still credible.
        assert kg_lie_rate(0.9, 1000, 10) == 1.0


class TestBetaTracker:
    def test_prior_mean(self):
        t = BetaTracker(prior_mean=0.7, prior_strength=2.0)
        assert math.isclose(t.mean, 0.7, rel_tol=1e-9)

    def test_updates_move_mean(self):
        t = BetaTracker(prior_mean=0.5)
        for _ in range(10):
            t.update(True)
        assert t.mean > 0.8

    def test_ucb_above_mean(self):
        t = BetaTracker(prior_mean=0.5)
        assert t.ucb() > t.mean

    def test_ucb_shrinks_with_data(self):
        t = BetaTracker(prior_mean=0.5)
        width_before = t.ucb() - t.mean
        for i in range(20):
            t.update(i % 2 == 0)
        assert (t.ucb() - t.mean) < width_before
