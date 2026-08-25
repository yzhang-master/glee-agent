"""Targets loader, config-key canonicalization, and strategy integration."""

from __future__ import annotations

import json

import pytest

from fixtures import (
    bargaining_decision,
    bargaining_game,
    negotiation_decision,
    negotiation_game,
    persuasion_game,
)
from glee_agent.config import Knobs
from glee_agent.families import bargaining, negotiation
from glee_agent.schema import parse_game
from glee_agent.theory import targets as T

KNOBS = Knobs()

GRID = [round(0.05 * i, 2) for i in range(1, 20)]
LINEAR = [i * 50.0 for i in range(19)]                    # 0 .. 900
ZEROS_TAIL = [0.0] * 13 + [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]

# Hand-written canonical keys — byte-exact contract with the builder.
BARG_KEY = (
    '{"complete_information":true,"delta_1":0.95,"delta_2":0.9,'
    '"horizon_known":true,"max_rounds":10,"messages_allowed":true,'
    '"money_to_divide":1000.0}'
)
NEG_ROLE_KEY = (
    '{"complete_information":false,"horizon_known":true,"max_rounds":10,'
    '"messages_allowed":true,"my_value":100.0,"role":"buyer"}'
)
NEG_FULL_KEY = (
    '{"buyer_value":150.0,"complete_information":true,"horizon_known":true,'
    '"max_rounds":10,"messages_allowed":true,"seller_value":100.0}'
)
PERS_KEY = (
    '{"p":0.5,"product_price":100.0,"seller_message_type":"text",'
    '"total_rounds":20,"u":0.0,"v":120.0}'
)


def _accept_key(share: float, rl: str, human: bool) -> str:
    return json.dumps(
        {"share_bucket": share, "rounds_left_bucket": rl, "human": human},
        sort_keys=True, separators=(",", ":"),
    )


def _neg_key(rel: float, role: str, rl: str, human: bool) -> str:
    return json.dumps(
        {"rel_bucket": rel, "role": role, "rounds_left_bucket": rl, "human": human},
        sort_keys=True, separators=(",", ":"),
    )


def _fixture_data() -> dict:
    barg_accept = {
        # Agent responder, deep game: sharp curve for the offer optimizer.
        _accept_key(0.45, "4+", False): [40, 38],
        _accept_key(0.40, "4+", False): [40, 36],
        _accept_key(0.35, "4+", False): [40, 8],
        _accept_key(0.30, "4+", False): [40, 6],
        _accept_key(0.25, "4+", False): [40, 0],
        _accept_key(0.20, "4+", False): [50, 0],
        # Thin human split -> backs off to pooled with the agent bucket.
        _accept_key(0.40, "4+", True): [10, 9],
        # Thin with no pooled backup -> None.
        _accept_key(0.30, "1", True): [10, 5],
    }
    neg_accept = {}
    for b in (0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        neg_accept[_neg_key(b, "buyer", "1", False)] = [40, 38]
    for b in (0.95, 1.00):
        neg_accept[_neg_key(b, "buyer", "1", False)] = [40, 20]
    for b in (1.05, 1.10, 1.15, 1.20, 1.25):
        neg_accept[_neg_key(b, "buyer", "1", False)] = [40, 2]
    # Seller responder buckets for the pooled-rel marginal.
    neg_accept[_neg_key(0.80, "seller", "2-3", False)] = [40, 10]
    neg_accept[_neg_key(0.85, "seller", "2-3", False)] = [20, 16]
    # Thin, no pooled backup.
    neg_accept[_neg_key(1.50, "buyer", "inf", True)] = [5, 1]

    return {
        "built_at": "2026-08-18T00:00:00",
        "n_games": {"bargaining": 3, "negotiation": 2, "persuasion": 1},
        "quantile_grid": GRID,
        "bargaining": {
            BARG_KEY: {
                "n": 800,
                "deal_rate": 0.35,
                "payoff_q": {"player_1": ZEROS_TAIL, "player_2": ZEROS_TAIL},
            },
            "cfgA": {
                "n": 100,
                "deal_rate": 0.9,
                "payoff_q": {"player_1": LINEAR, "player_2": LINEAR},
            },
        },
        "negotiation": {
            NEG_FULL_KEY: {
                "n": 50,
                "deal_rate": 0.5,
                "payoff_q": {"player_1": LINEAR, "player_2": LINEAR},
            },
        },
        "neg_by_role": {
            NEG_ROLE_KEY: {
                "n": 500,
                "deal_rate": 0.4,
                "payoff_q": [0.0] * 13 + [1.0, 2.0, 3.0, 4.0, 5.0, 20.0],
            },
        },
        "persuasion": {
            PERS_KEY: {
                "n": 60,
                "seller_sell_rate": 0.7,
                "payoff_q": {"player_1": LINEAR, "player_2": LINEAR},
            },
        },
        "barg_accept": barg_accept,
        "neg_accept": neg_accept,
        "models": {"gpt-4o": {"barg_n": 100}},
    }


@pytest.fixture
def tfix(tmp_path):
    path = tmp_path / "targets.json"
    path.write_text(json.dumps(_fixture_data()))
    t = T.load_targets(path, live=False)
    T.set_targets(t)
    return t


# ------------------------------------------------------------ loading

class TestLoading:
    def test_missing_file_gives_null(self, tmp_path):
        t = T.load_targets(tmp_path / "nope.json", live=False)
        assert t.is_null
        assert t.payoff_quantile("bargaining", "cfgA", "player_1", 0.5) is None
        assert t.payoff_percentile("bargaining", "cfgA", "player_1", 100) is None
        assert t.barg_accept_prob(0.4, 5, False) is None
        assert t.neg_accept_prob(0.9, "buyer", 1, False) is None
        assert t.deal_rate("negotiation", NEG_ROLE_KEY) is None

    def test_corrupt_file_gives_null(self, tmp_path):
        path = tmp_path / "targets.json"
        path.write_text("{not json!!")
        t = T.load_targets(path, live=False)
        assert t.is_null

    def test_get_targets_never_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(T, "DEFAULT_PATH", tmp_path / "absent.json")
        T.set_targets(None)
        t = T.get_targets()
        assert t.payoff_quantile("bargaining", "x", "player_1", 0.5) is None
        assert T.get_targets() is t  # cached singleton

    def test_deal_rate(self, tfix):
        assert tfix.deal_rate("bargaining", BARG_KEY) == pytest.approx(0.35)
        assert tfix.deal_rate("negotiation", NEG_ROLE_KEY) == pytest.approx(0.4)
        assert tfix.deal_rate("bargaining", "unknown") is None


# ------------------------------------------------------------ quantiles

class TestQuantiles:
    def test_quantile_at_grid_point(self, tfix):
        assert tfix.payoff_quantile("bargaining", "cfgA", "player_1", 0.5) == pytest.approx(450.0)
        assert tfix.payoff_quantile("bargaining", "cfgA", "player_1", 0.05) == pytest.approx(0.0)
        assert tfix.payoff_quantile("bargaining", "cfgA", "player_1", 0.95) == pytest.approx(900.0)

    def test_quantile_interpolates(self, tfix):
        assert tfix.payoff_quantile("bargaining", "cfgA", "player_1", 0.525) == pytest.approx(475.0)

    def test_quantile_clamps_q(self, tfix):
        assert tfix.payoff_quantile("bargaining", "cfgA", "player_1", 0.0) == pytest.approx(0.0)
        assert tfix.payoff_quantile("bargaining", "cfgA", "player_1", 1.0) == pytest.approx(900.0)

    def test_quantile_unknown_config(self, tfix):
        assert tfix.payoff_quantile("bargaining", "missing", "player_1", 0.5) is None
        assert tfix.payoff_quantile("bargaining", None, "player_1", 0.5) is None

    def test_persuasion_and_neg_full_rows(self, tfix):
        assert tfix.payoff_quantile("persuasion", PERS_KEY, "player_2", 0.5) == pytest.approx(450.0)
        assert tfix.payoff_percentile("negotiation", NEG_FULL_KEY, "player_1", 450.0) == pytest.approx(0.5)

    def test_percentile_inverse_and_monotone(self, tfix):
        pcts = [
            tfix.payoff_percentile("bargaining", "cfgA", "player_1", x)
            for x in (-5, 0, 100, 450, 475, 899, 2000)
        ]
        assert pcts == sorted(pcts)
        assert pcts[0] == pytest.approx(0.02)   # below the pool -> clamp
        assert pcts[3] == pytest.approx(0.5)
        assert pcts[4] == pytest.approx(0.525)
        assert pcts[-1] == pytest.approx(0.98)  # above the pool -> clamp

    def test_percentile_mass_at_zero(self, tfix):
        # 13 grid points sit at 0: a zero payoff sits at the top of that mass.
        assert tfix.payoff_percentile("bargaining", BARG_KEY, "player_1", 0.0) == pytest.approx(0.65)
        assert tfix.payoff_percentile("bargaining", BARG_KEY, "player_1", 500.0) == pytest.approx(0.90)

    def test_neg_by_role_fallback(self, tfix):
        # Key absent from the full negotiation dict resolves via neg_by_role.
        assert tfix.payoff_percentile("negotiation", NEG_ROLE_KEY, "player_2", 10.0) == pytest.approx(
            0.9 + 0.05 * (5.0 / 15.0)
        )


# ------------------------------------------------------------ config keys

class TestConfigKeys:
    def test_bargaining_key_byte_exact(self):
        state = parse_game(bargaining_game()).state
        assert T.config_key_bargaining(state) == BARG_KEY

    def test_bargaining_key_unbuildable(self):
        assert T.config_key_bargaining({}) is None
        state = parse_game(bargaining_game()).state.copy()
        del state["delta_2"]  # hidden opponent delta -> no key
        assert T.config_key_bargaining(state) is None

    def test_negotiation_keys(self):
        state = parse_game(negotiation_decision(role="buyer")).state
        full, role = T.config_key_negotiation(state, "buyer", 100.0)
        assert full is None  # seller value hidden
        assert role == NEG_ROLE_KEY

    def test_negotiation_full_key_byte_exact(self):
        game = negotiation_game(
            role="seller",
            game_state={"player_2_value": 150.0, "complete_information": True},
        )
        state = parse_game(game).state
        full, role = T.config_key_negotiation(state, "seller", 100.0)
        assert full == NEG_FULL_KEY
        assert role is not None and '"role":"seller"' in role

    def test_persuasion_key_byte_exact(self):
        state = parse_game(persuasion_game()).state
        assert T.config_key_persuasion(state) == PERS_KEY
        assert T.config_key_persuasion({}) is None


# ------------------------------------------------------------ accept curves

class TestAcceptCurves:
    def test_barg_bucketing(self, tfix):
        # 0.42 floors into the 0.40 bucket.
        assert tfix.barg_accept_prob(0.42, 10, False) == pytest.approx(36 / 40)
        assert tfix.barg_accept_prob(0.35, 10, False) == pytest.approx(8 / 40)

    def test_barg_human_backoff_to_pooled(self, tfix):
        # Human split has n=10 < 20 -> pooled with the agent bucket: 45/50.
        assert tfix.barg_accept_prob(0.40, 10, True) == pytest.approx(0.9)

    def test_barg_thin_bucket_is_none(self, tfix):
        # n=10 total for this bucket even after pooling.
        assert tfix.barg_accept_prob(0.30, 1, True) is None
        assert tfix.barg_accept_prob(0.90, 10, False) is None  # no data at all

    def test_neg_bucketing(self, tfix):
        assert tfix.neg_accept_prob(0.83, "seller", 3, False) == pytest.approx(10 / 40)
        assert tfix.neg_accept_prob(0.94, "buyer", 1, False) == pytest.approx(38 / 40)
        assert tfix.neg_accept_prob(1.5, "buyer", None, True) is None  # n=5

    def test_neg_pooled_rel_marginal(self, tfix):
        # rel=None pools across rel buckets for (seller, "2-3", agent).
        assert tfix.neg_accept_prob(None, "seller", 3, False) == pytest.approx(26 / 60)


# ------------------------------------------------------------ decide() e2e

class TestDecideIntegration:
    def test_barg_percentile_accept_fires(self, tfix):
        # 500/1000 in round 1 = q90 of a pool that mostly ended in no-deal
        # zeros. Baseline theory rejects (share 0.50 < continuation), the
        # percentile path accepts.
        game = bargaining_decision(my_gain=500, opp_gain=500)
        T.set_targets(T.Targets.null())
        assert bargaining.decide(parse_game(game), KNOBS)["decision"] == "reject"
        T.set_targets(tfix)
        assert bargaining.decide(parse_game(game), KNOBS)["decision"] == "accept"

    def test_barg_lowball_still_rejected(self, tfix):
        game = bargaining_decision(my_gain=50, opp_gain=950)
        assert bargaining.decide(parse_game(game), KNOBS)["decision"] == "reject"

    def test_barg_offer_optimizer_uses_accept_curve(self, tfix):
        # NEUTRAL regime only (equal deltas — asymmetric deltas now route to
        # the patience regimes where the optimizer is deliberately off):
        # curve says shares beyond 0.70 almost never get accepted; the
        # optimizer accelerates the round-1 anchor from 0.80 down to 0.70
        # (it may only ever move DOWN from the schedule).
        game = bargaining_game(game_state={"delta_2": 0.95})
        view = parse_game(game)
        action = bargaining.decide(view, KNOBS)
        # With continuation capped at the realized 0.46 (not the aspirational
        # schedule), the EV argmax concedes deeper: 0.60, down from 0.80.
        assert action["alice_gain"] == pytest.approx(600.0, abs=0.01)
        assert action["alice_gain"] < 800.0  # only ever DOWN from the schedule
        assert action["alice_gain"] + action["bob_gain"] == pytest.approx(1000.0)
        # Without curve data: the plain Boulware anchor.
        T.set_targets(T.Targets.null())
        assert bargaining.decide(view, KNOBS)["alice_gain"] == pytest.approx(800.0, abs=0.01)

    def test_neg_percentile_accept_fires(self, tfix):
        # Buyer offered price 90 (payoff 10) vs a pool where q95 = 20:
        # baseline counters, percentile path accepts.
        game = negotiation_decision(role="buyer", offer_price=90.0)
        T.set_targets(T.Targets.null())
        assert negotiation.decide(parse_game(game), KNOBS)["decision"] == "RejectOffer"
        T.set_targets(tfix)
        assert negotiation.decide(parse_game(game), KNOBS)["decision"] == "AcceptOffer"

    def test_neg_losing_deal_never_accepted(self, tfix):
        game = negotiation_decision(role="buyer", offer_price=130.0)
        decision = negotiation.decide(parse_game(game), KNOBS)["decision"]
        assert decision != "AcceptOffer"

    def test_neg_ultimatum_rollback_uses_optimizer(self, tfix):
        # Complete info, one round: EV-optimal price 141.95 (rel 0.946, 95%
        # accept) beats the (anchor+floor)/2 heuristic of 125.5. The optimum
        # sits just under the curve's cliff at rel 0.95, where acceptance
        # halves — pricing nearer the buyer's value would raise the payoff
        # but more than halve the odds of collecting it.
        game = negotiation_game(
            role="seller",
            game_state={
                "max_rounds": 1,
                "player_2_value": 150.0,
                "complete_information": True,
            },
        )
        view = parse_game(game)
        rollback = Knobs(neg_ci_ultimatum_frac=0.0)
        assert negotiation.decide(view, rollback)["product_price"] == pytest.approx(141.95, abs=0.01)
        T.set_targets(T.Targets.null())
        assert negotiation.decide(view, rollback)["product_price"] == pytest.approx(125.5, abs=0.01)


class TestLiveOverlay:
    """The live overlay corrects a dataset key that is 2-24x optimistic about
    this field. It must never widen a lookup's blast radius: thin live cells
    are ignored, and a missing/corrupt live file leaves the prior untouched."""

    def _base(self, tmp_path):
        p = tmp_path / "targets.json"
        p.write_text(json.dumps(_fixture_data()))
        return p

    def test_missing_live_file_leaves_dataset_intact(self, tmp_path, monkeypatch):
        monkeypatch.setattr(T, "LIVE_PATH", tmp_path / "absent.json")
        t = T.load_targets(self._base(tmp_path), live=True)
        assert t.barg_accept_prob(0.4, 5, False) is not None

    def test_corrupt_live_file_leaves_dataset_intact(self, tmp_path, monkeypatch):
        bad = tmp_path / "live.json"
        bad.write_text("{not json")
        monkeypatch.setattr(T, "LIVE_PATH", bad)
        t = T.load_targets(self._base(tmp_path), live=True)
        assert t.barg_accept_prob(0.4, 5, False) is not None

    def test_well_sampled_live_bucket_overrides(self, tmp_path, monkeypatch):
        key = T._dumps({"share_bucket": 0.4, "rounds_left_bucket": "4+", "human": False})
        live = tmp_path / "live.json"
        live.write_text(json.dumps({"barg_accept": {key: [5000, 50]}}))
        monkeypatch.setattr(T, "LIVE_PATH", live)
        t = T.load_targets(self._base(tmp_path), live=True)
        assert t.barg_accept_prob(0.4, 5, False) == pytest.approx(50 / 5000)

    def test_thin_live_bucket_is_ignored(self, tmp_path, monkeypatch):
        key = T._dumps({"share_bucket": 0.4, "rounds_left_bucket": "4+", "human": False})
        live = tmp_path / "live.json"
        live.write_text(json.dumps({"barg_accept": {key: [5, 0]}}))
        monkeypatch.setattr(T, "LIVE_PATH", live)
        t = T.load_targets(self._base(tmp_path), live=True)
        assert t.barg_accept_prob(0.4, 5, False) != pytest.approx(0.0)
