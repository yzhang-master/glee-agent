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
from glee_agent.families import bargaining, negotiation, persuasion
from glee_agent.schema import parse_game
from glee_agent.theory import targets as T

KNOBS = Knobs()


def run(module, game):
    return module.decide(parse_game(game), KNOBS)


class TestBargaining:
    def test_offer_sums_to_pot(self):
        action = run(bargaining, bargaining_game())
        assert action["alice_gain"] + action["bob_gain"] == 1000

    def test_opening_offer_favors_me(self):
        action = run(bargaining, bargaining_game())
        assert action["alice_gain"] > 500

    def test_accept_generous_offer(self):
        action = run(bargaining, bargaining_decision(my_gain=700, opp_gain=300))
        assert action["decision"] == "accept"

    def test_reject_lowball_early(self):
        action = run(bargaining, bargaining_decision(my_gain=50, opp_gain=950))
        assert action["decision"] == "reject"

    def test_final_round_accept_anything(self):
        game = bargaining_decision(my_gain=1, opp_gain=999)
        game["game_state"]["round"] = 10
        action = run(bargaining, game)
        assert action["decision"] == "accept"

    def test_final_round_proposer_not_epsilon(self):
        game = bargaining_game()
        game["game_state"]["round"] = 10
        action = run(bargaining, game)
        # Gives the responder a real share, not epsilon.
        assert action["bob_gain"] >= 200

    def test_unknown_horizon(self):
        game = bargaining_game()
        del game["game_state"]["max_rounds"]
        game["game_state"]["horizon_known"] = False
        action = run(bargaining, game)
        assert action["alice_gain"] + action["bob_gain"] == 1000

    def test_never_walkaway(self):
        # Walkaway is dominated; strategy must never emit it.
        for gain in (0, 10, 400, 900):
            action = run(bargaining, bargaining_decision(my_gain=gain, opp_gain=1000 - gain))
            assert action["decision"] != "walkaway"

    def test_human_offer_is_fairer(self):
        agent_offer = run(bargaining, bargaining_game())
        human_game = bargaining_game(opponent={"type": "human", "name": "Sam"})
        human_offer = run(bargaining, human_game)
        assert human_offer["bob_gain"] > agent_offer["bob_gain"]

    def test_player2_perspective(self):
        game = bargaining_game(your_player="player_2")
        game["game_state"]["current_player"] = "player_2"
        game["game_state"]["proposer"] = "player_2"
        action = run(bargaining, game)
        assert action["bob_gain"] > 500  # I'm Bob now


class TestBargainingRegimes:
    """Patience regimes, behavior-dependent concession, final-round EV."""

    @pytest.fixture(autouse=True)
    def clean_targets(self):
        T.set_targets(T.Targets.null())
        yield
        T.set_targets(None)

    @staticmethod
    def _adv_game(**state):
        # My delta 1.0 vs their 0.8, unlimited horizon: strong patience edge.
        game = bargaining_game()
        del game["game_state"]["max_rounds"]
        game["game_state"].update(
            {"horizon_known": False, "delta_1": 1.0, "delta_2": 0.8}
        )
        game["game_state"].update(state)
        return game

    @staticmethod
    def _stonewall_history(my_gain, opp_gives_me):
        # PLATFORM SHAPE: gains nested under "offer" — exactly what live
        # game_state.history carries (the flat shape was a fixture-only
        # fiction that let a dead parser pass its tests).
        pot = 1000

        def entry(proposer, p1, rnd):
            return {
                "round": rnd, "proposer": proposer, "decision": "reject",
                "offer": {"player_1_gain": p1, "player_2_gain": pot - p1,
                          "message": None},
            }

        return [
            entry("player_1", my_gain, 1),
            entry("player_2", opp_gives_me, 2),
            entry("player_1", my_gain, 3),
            entry("player_2", opp_gives_me, 4),
        ]

    def test_advantage_holds_high_and_off_schedule(self):
        # delta 1.0 vs 0.8: hold at 0.90; round 8 offer == round 2 offer —
        # time alone must buy the opponent nothing.
        early = self._adv_game(round=2)
        late = self._adv_game(round=8)
        a_early = run(bargaining, early)
        a_late = run(bargaining, late)
        assert a_early["alice_gain"] / 1000 >= 0.65
        assert a_early["alice_gain"] == pytest.approx(a_late["alice_gain"])

    def test_advantage_no_drip_vs_stonewaller(self):
        # Opponent stonewalls (their offers to me never improve): my offer
        # must not move at all — advantage drip is zero, and the rejected>=2
        # deadlock pressure is off in the advantage regime.
        game = self._adv_game(round=5, history=self._stonewall_history(900, 100))
        action = run(bargaining, game)
        assert action["alice_gain"] == pytest.approx(900.0)

    def test_neutral_stonewaller_gets_drip_only(self):
        # Equal deltas (neutral): opponent stonewalls at 0.2-to-me; despite
        # the schedule AND deadlock pressure wanting ~0.725, my offer moves
        # at most the 0.01 drip per own offer.
        game = bargaining_game(game_state={
            "round": 5, "delta_1": 0.95, "delta_2": 0.95,
            "history": self._stonewall_history(800, 200),
        })
        action = run(bargaining, game)
        assert action["alice_gain"] == pytest.approx(1000 * (0.80 - KNOBS.barg_drip))

    def test_neutral_concession_follows_opponents(self):
        # Same spot but the opponent conceded 0.05 to me between their two
        # offers: my step may now match it (schedule wants 0.725 -> allowed).
        history = self._stonewall_history(800, 150)
        history[3]["offer"]["player_1_gain"] = 200
        history[3]["offer"]["player_2_gain"] = 800
        game = bargaining_game(game_state={
            "round": 5, "delta_1": 0.95, "delta_2": 0.95, "history": history,
        })
        action = run(bargaining, game)
        assert action["alice_gain"] == pytest.approx(750.0)  # 0.80 - 0.05 step

    def test_disadvantage_anchors_low(self):
        # delta 0.8 vs 1.0: my pot melts faster — anchor down to ~0.58.
        game = bargaining_game(game_state={"delta_1": 0.8, "delta_2": 1.0})
        action = run(bargaining, game)
        assert action["alice_gain"] == pytest.approx(580.0)

    def test_disadvantage_accepts_near_half_round_1(self):
        game = bargaining_decision(
            my_gain=480, opp_gain=520,
            game_state={"delta_1": 0.8, "delta_2": 1.0},
        )
        assert run(bargaining, game)["decision"] == "accept"

    def test_advantage_rejects_below_hold(self):
        # 0.70 clears barg_accept_great, but under a strong patience edge the
        # advantage threshold min(SPE*0.85, 0.75) = 0.75 rejects it early.
        game = bargaining_decision(
            my_gain=700, opp_gain=300,
            game_state={"delta_1": 1.0, "delta_2": 0.8, "horizon_known": False},
        )
        del game["game_state"]["max_rounds"]
        assert run(bargaining, game)["decision"] == "reject"

    def test_final_round_ev_optimizer_uses_curve(self):
        # Fixture curve (>=3 buckets — thinner curves fall back to the flat
        # knob): argmax of (1-give)*P is give=0.15.
        def k(share):
            return json.dumps(
                {"human": False, "rounds_left_bucket": "1", "share_bucket": share},
                sort_keys=True, separators=(",", ":"),
            )
        T.set_targets(T.Targets({"barg_accept": {
            k(0.10): [40, 20],   # p=0.5 -> ev 0.45
            k(0.15): [40, 32],   # p=0.8 -> ev 0.68  <- argmax
            k(0.30): [40, 34],   # p=0.85 -> ev 0.595
        }}))
        game = bargaining_game(game_state={"round": 10})
        action = run(bargaining, game)
        assert action["bob_gain"] == pytest.approx(150.0)
        assert action["bob_gain"] < 300.0

    def test_final_round_falls_back_to_flat_knob(self):
        game = bargaining_game(game_state={"round": 10})
        action = run(bargaining, game)
        assert action["bob_gain"] == pytest.approx(1000 * KNOBS.barg_final_round_give)

    def test_profile_shaves_final_round_give(self):
        # Opponent known (dataset) to accept lowballs >= 20% of the time:
        # final-round give drops by 0.05.
        T.set_targets(T.Targets({"models": {
            "gpt-4o": {"barg_n": 5000, "barg_accept_rate_when_offered_lt40pct": 0.25},
        }}))
        game = bargaining_game(
            opponent={"type": "agent", "name": "gpt-4o"},
            game_state={"round": 10},
        )
        action = run(bargaining, game)
        assert action["bob_gain"] == pytest.approx(
            1000 * (KNOBS.barg_final_round_give - 0.05)
        )

    def test_endgame_parity_caps_advantage_hold(self):
        # Known horizon, 2 rounds left, responder proposes last: SPE caps my
        # share far below the hold — the parity rule survives the regimes.
        game = bargaining_game(game_state={
            "round": 9, "delta_1": 1.0, "delta_2": 0.8,
        })
        action = run(bargaining, game)
        assert action["alice_gain"] <= 300.0

    def test_new_knobs_env_overrides(self, monkeypatch):
        from glee_agent.config import _knobs_from_env
        monkeypatch.setenv("GLEE_KNOB_BARG_DRIP", "0.03")
        monkeypatch.setenv("GLEE_KNOB_BARG_ADV_HOLD", "0.8")
        monkeypatch.setenv("GLEE_KNOB_BARG_DIS_ACCEPT", "0.5")
        monkeypatch.setenv("GLEE_KNOB_BARG_PATIENCE_EDGE", "0.06")
        knobs = _knobs_from_env()
        assert knobs.barg_drip == pytest.approx(0.03)
        assert knobs.barg_adv_hold == pytest.approx(0.8)
        assert knobs.barg_dis_accept == pytest.approx(0.5)
        assert knobs.barg_patience_edge == pytest.approx(0.06)


class TestNegotiation:
    def test_seller_opens_above_value(self):
        action = run(negotiation, negotiation_game(role="seller"))
        assert action["product_price"] > 100

    def test_buyer_opens_below_value(self):
        action = run(negotiation, negotiation_game(role="buyer"))
        assert action["product_price"] < 100

    def test_buyer_accepts_profitable_final_round(self):
        game = negotiation_decision(role="buyer", offer_price=99.0)
        game["game_state"]["round"] = 10
        action = run(negotiation, game)
        assert action["decision"] == "AcceptOffer"

    def test_never_accept_losing_deal(self):
        # Buyer with value 100 offered price 150 — must not accept, any round.
        for rnd in (1, 5, 10):
            game = negotiation_decision(role="buyer", offer_price=150.0)
            game["game_state"]["round"] = rnd
            action = run(negotiation, game)
            assert action["decision"] != "AcceptOffer"

    def test_seller_never_accept_below_value(self):
        game = negotiation_decision(role="seller", offer_price=50.0)
        game["your_player"] = "player_1"
        action = run(negotiation, game)
        assert action["decision"] != "AcceptOffer"

    def test_counteroffer_present_midgame(self):
        game = negotiation_decision(role="buyer", offer_price=150.0)
        action = run(negotiation, game)
        if action["decision"] == "RejectOffer":
            assert "product_price" in action

    def test_ultimatum_seller_prices_to_close(self):
        game = negotiation_game(role="seller")
        game["game_state"]["max_rounds"] = 1
        action = run(negotiation, game)
        # Between reservation and anchor, not at the extreme.
        assert 100 < action["product_price"] < 100 * (1 + KNOBS.neg_anchor_markup)

    def test_no_horizon_no_walkaway_early(self):
        game = negotiation_decision(role="buyer", offer_price=150.0)
        del game["game_state"]["max_rounds"]
        game["game_state"]["horizon_known"] = False
        action = run(negotiation, game)
        assert action["decision"] != "WalkAway"


class TestPersuasion:
    def test_seller_recommends_high(self):
        action = run(persuasion, persuasion_game(actor="seller", quality="high"))
        assert "recommend" in action["message"].lower()

    def test_seller_binary_mode(self):
        game = persuasion_game(actor="seller", quality="high")
        game["valid_actions"] = {"type": "seller_recommendation", "fields": {"decision": ["yes", "no"]}}
        action = run(persuasion, game)
        assert action["decision"] == "yes"

    def test_buyer_buys_when_ev_positive(self):
        # p=0.5, v=250, price=100 -> prior EV 125 > 100.
        game = persuasion_game(actor="buyer", game_state={"v": 250.0})
        action = run(persuasion, game)
        assert action["decision"] == "yes"

    def test_buyer_passes_when_ev_negative_and_no_trust(self):
        # p=0.1, v=120, price=100 -> EV 12 << 100, no history to build trust.
        game = persuasion_game(actor="buyer", game_state={"p": 0.1})
        game["game_state"]["round"] = 15  # past exploration window
        action = run(persuasion, game)
        assert action["decision"] == "no"

    def test_buyer_respects_anti_recommendation(self):
        game = persuasion_game(actor="buyer", game_state={"v": 250.0})
        game["game_state"]["seller_message"] = "I don't recommend this one, skip it."
        action = run(persuasion, game)
        assert action["decision"] == "no"

    def test_buyer_learns_from_history(self):
        # Seller recommended and delivered high 6 times: trust recommendation.
        history = [
            {"round": i, "seller_message": "Great product!", "buyer_decision": "yes",
             "bought": True, "quality": "high", "seller_payoff": 100, "buyer_payoff": 20}
            for i in range(1, 7)
        ]
        game = persuasion_game(actor="buyer", game_state={"p": 0.3, "history": history, "round": 7})
        action = run(persuasion, game)
        assert action["decision"] == "yes"

    def test_seller_trust_repair_after_passes(self):
        # Buyer stopped buying: seller must go honest on a low product.
        history = [
            {"round": 1, "seller_message": "Great!", "buyer_decision": "yes",
             "bought": True, "quality": "low", "seller_payoff": 100, "buyer_payoff": -100},
            {"round": 2, "seller_message": "Great!", "buyer_decision": "no", "bought": False},
            {"round": 3, "seller_message": "Great!", "buyer_decision": "no", "bought": False},
        ]
        game = persuasion_game(actor="seller", quality="low",
                               game_state={"history": history, "round": 4, "p": 0.3})
        action = run(persuasion, game)
        msg = action["message"].lower()
        assert "pass" in msg or "don't" in msg or "not" in msg


class TestPersuasionStopLoss:
    def _thin_game(self, **state):
        # v=12000, price=10000: needs >83% honesty to profit — thin margin.
        from fixtures import persuasion_game
        base = {"p": 0.5, "v": 12000.0, "product_price": 10000.0,
                "total_rounds": 20, "round": 3}
        base.update(state)
        return persuasion_game(actor="buyer", game_state=base)

    def test_no_explore_buy_on_thin_margin(self):
        # Prior EV = 0.5*12000 = 6000 < 10000; thin margin forbids UCB buys.
        game = self._thin_game()
        action = run(persuasion, game)
        assert action["decision"] == "no"

    def test_no_explore_after_losses(self):
        # Wide margin but game already under water: exploration off.
        game = self._thin_game(v=30000.0, p=0.2, buyer_total_payoff=-10000.0)
        history = [{"round": 1, "seller_message": "Great!", "buyer_decision": "yes",
                    "bought": True, "quality": "low", "buyer_payoff": -10000}]
        game["game_state"]["history"] = history
        action = run(persuasion, game)
        # p=0.2 prior EV=6000 < 10000, and stop-loss blocks the UCB boost.
        assert action["decision"] == "no"

    def test_profitable_prior_still_buys(self):
        game = self._thin_game(v=25000.0, p=0.5)  # EV 12500 > 10000
        action = run(persuasion, game)
        assert action["decision"] == "yes"


class TestPersuasionNeutralMessages:
    """Regression: soft anti-recs classified 'neutral' were bought at the raw
    prior forever (verified -15M game: 15 'Ordinary unit' lows bought)."""

    def _wide_game(self, message, history=None, rnd=1):
        from fixtures import persuasion_game
        # p=1/3, v=4x price: prior EV positive, wide margin.
        state = {"p": 1 / 3, "v": 4000000.0, "product_price": 1000000.0,
                 "total_rounds": 20, "round": rnd, "seller_message": message}
        if history is not None:
            state["history"] = history
        return persuasion_game(actor="buyer", game_state=state)

    def test_soft_anti_rec_words_classify_negative(self):
        from glee_agent.families.persuasion import _msg_polarity
        assert _msg_polarity("Ordinary unit this round.") == "neg"
        assert _msg_polarity("This is not the premium one.") == "neg"
        assert _msg_polarity("A standard offering today.") == "neg"

    def test_neutral_buys_stop_after_lows(self):
        # Message with no keyword hits at all -> neutral; after two neutral
        # lows the neutral tracker must cut off further buying.
        msg = "Item #7 ships today."
        assert __import__("glee_agent.families.persuasion", fromlist=["_msg_polarity"])._msg_polarity(msg) == "neutral"
        history = [
            {"round": i, "seller_message": msg, "buyer_decision": "yes",
             "bought": True, "quality": "low", "buyer_payoff": -1000000}
            for i in (1, 2)
        ]
        game = self._wide_game(msg, history=history, rnd=3)
        action = run(persuasion, game)
        assert action["decision"] == "no"

    def test_neutral_lows_do_not_poison_rec_tracker(self):
        # Neutral lows in history must not stop us trusting a genuinely
        # positive recommendation afterwards.
        neutral = "Item #7 ships today."
        history = [
            {"round": i, "seller_message": neutral, "buyer_decision": "yes",
             "bought": True, "quality": "low", "buyer_payoff": -1000000}
            for i in (1, 2)
        ]
        game = self._wide_game("Good unit — worth taking.", history=history, rnd=3)
        action = run(persuasion, game)
        assert action["decision"] == "yes"

    def test_thin_explore_knob_enables_rec_following(self):
        from fixtures import persuasion_game
        from glee_agent.config import Knobs
        from glee_agent.families import persuasion as pers_mod
        from glee_agent.schema import parse_game
        # p=0.8, v=1.25x: thin margin — default passes, knob explores the rec.
        state = {"p": 0.8, "v": 12500.0, "product_price": 10000.0,
                 "total_rounds": 20, "round": 2,
                 "seller_message": "yes"}
        game = persuasion_game(actor="buyer", game_state=state)
        view = parse_game(game)
        off = pers_mod.decide(view, Knobs(llm_enabled=False))
        on = pers_mod.decide(view, Knobs(llm_enabled=False, pers_thin_explore=True))
        assert off["decision"] == "no"
        assert on["decision"] == "yes"


class TestNegotiationStalemate:
    def _marathon(self, offer_price, rnd):
        from fixtures import negotiation_decision
        game = negotiation_decision(role="buyer", offer_price=offer_price)
        del game["game_state"]["max_rounds"]
        game["game_state"]["horizon_known"] = False
        game["game_state"]["round"] = rnd
        return game

    def test_accepts_profit_in_marathon(self):
        # Buyer value 100, offered 95 at round 20: take the profit, end it.
        action = run(negotiation, self._marathon(95.0, 20))
        assert action["decision"] == "AcceptOffer"

    def test_walks_away_from_hopeless_marathon(self):
        # Offered 150 (> value 100) at round 30, no better offer ever: walk.
        action = run(negotiation, self._marathon(150.0, 30))
        assert action["decision"] == "WalkAway"

    def test_keeps_countering_before_stall(self):
        # Round 5: too early for stalemate logic, keep negotiating.
        action = run(negotiation, self._marathon(150.0, 5))
        assert action["decision"] == "RejectOffer"
