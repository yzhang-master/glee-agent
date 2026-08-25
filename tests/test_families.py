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
from glee_agent.schema import parse_game, parse_negotiation
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

    def test_thin_config_refuses_even_after_a_lucky_run(self):
        """v = 1.2 x price at p = 0.3 needs a >91%-honest seller (x* = 0.086),
        and the field averages ~75% honest — so passing is correct here.

        The six high-quality purchases are NOT evidence of honesty: a liar
        also recommends every high product, so P(recommend, high | x) = p is
        independent of the lie rate. Only a low-quality sale, or the seller
        declining to recommend, moves the posterior.
        """
        history = [
            {"round": i, "seller_message": "Great product!", "buyer_decision": "yes",
             "bought": True, "quality": "high", "seller_payoff": 100, "buyer_payoff": 20}
            for i in range(1, 7)
        ]
        game = persuasion_game(actor="buyer", game_state={"p": 0.3, "history": history, "round": 7})
        assert run(persuasion, game)["decision"] == "no"

    def test_wide_config_buys_on_the_same_history(self):
        """Same history, v = 2.5 x price: now the prior alone clears the bar."""
        history = [
            {"round": i, "seller_message": "Great product!", "buyer_decision": "yes",
             "bought": True, "quality": "high", "seller_payoff": 100, "buyer_payoff": 150}
            for i in range(1, 7)
        ]
        game = persuasion_game(actor="buyer", game_state={
            "p": 0.3, "v": 250.0, "history": history, "round": 7})
        assert run(persuasion, game)["decision"] == "yes"

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
                "total_rounds": 20, "round": 3, "history": []}
        base.update(state)
        return persuasion_game(actor="buyer", game_state=base)

    def test_no_explore_buy_on_thin_margin(self):
        # Prior EV = 0.5*12000 = 6000 < 10000; thin margin forbids UCB buys.
        game = self._thin_game()
        action = run(persuasion, game)
        assert action["decision"] == "no"

    def test_buying_shuts_off_against_a_persistent_liar(self):
        """Evidence, not a fixed budget, is what stops us.

        p=0.2, v=3x price: x* = 0.5, so a recommendation is genuinely
        ambiguous at the prior and a few probe buys are the price of the
        information. What matters is that the posterior closes the tap and
        keeps it closed once the seller has proved dishonest.
        """
        decisions, history = [], []
        for r in range(1, 9):
            game = self._thin_game(v=30000.0, p=0.2, round=r,
                                   history=list(history))
            game["game_state"]["seller_message"] = "Great product!"
            d = run(persuasion, game)["decision"]
            decisions.append(d)
            history.append({"round": r, "seller_message": "Great product!",
                            "buyer_decision": d, "bought": d == "yes",
                            "quality": "low",
                            "buyer_payoff": -10000 if d == "yes" else 0})
        assert decisions[-4:] == ["no"] * 4, decisions
        assert decisions.count("yes") <= 4, decisions

    def test_keeps_buying_from_an_honest_seller(self):
        """The mirror: an honest seller must not be starved out."""
        history = []
        for r in range(1, 7):
            rec = r % 2 == 1
            msg = "I recommend it" if rec else "pass on this one"
            history.append({"round": r, "seller_message": msg,
                            "buyer_decision": "yes" if rec else "no",
                            "bought": rec, "quality": "high" if rec else "low"})
        game = persuasion_game(actor="buyer", game_state={
            "p": 0.5, "v": 250.0, "product_price": 100.0, "total_rounds": 20,
            "round": 7, "history": history, "seller_message": "I recommend it"})
        assert run(persuasion, game)["decision"] == "yes"

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

    def test_buyer_learns_lie_rate_without_ever_buying(self):
        """The core fix: a buyer who never buys still learns from the seller's
        recommendation FREQUENCY. Previously this state was absorbing — the
        tracker only updated on purchases, so 41% of buyer games ended at 0."""
        from fixtures import persuasion_game
        from glee_agent.config import Knobs
        from glee_agent.families import persuasion as pers_mod
        from glee_agent.schema import parse_game

        def posterior_after(messages):
            hist = [{"round": i + 1, "seller_message": m, "buyer_decision": "no",
                     "bought": False} for i, m in enumerate(messages)]
            game = persuasion_game(actor="buyer", game_state={
                "p": 0.5, "v": 250.0, "product_price": 100.0, "total_rounds": 20,
                "round": len(messages) + 1, "history": hist,
                "seller_message": "I recommend this one"})
            view = parse_game(game)
            ps = pers_mod.parse_persuasion(view)
            return pers_mod._buyer_posterior(view, ps, Knobs(llm_enabled=False))

        honest = posterior_after(["I recommend it", "pass on this one"] * 5)
        liar = posterior_after(["I recommend it"] * 10)
        # Zero purchases in both, yet the frequency channel separates them.
        assert liar.mean_x > honest.mean_x + 0.25
        assert honest.p_high_given(True) > liar.p_high_given(True) + 0.15

    def _final_round_game(self, message, p=0.5, **state):
        game_state = {
            "p": p, "v": 250.0, "product_price": 100.0,
            "total_rounds": 20, "round": 20, "seller_message": message,
        }
        game_state.update(state)
        return persuasion_game(actor="buyer", game_state=game_state)

    def test_buyer_respects_binary_no_in_final_round(self):
        game = self._final_round_game(
            {"decision": "no"}, seller_message_type="binary"
        )
        assert run(persuasion, game)["decision"] == "no"

    def test_buyer_respects_text_anti_recommendation_in_final_round(self):
        game = self._final_round_game("honestly, skip this one")
        assert run(persuasion, game)["decision"] == "no"

    @pytest.mark.parametrize(
        "message", ["I recommend this one", "Item #7 ships today."]
    )
    @pytest.mark.parametrize(("p", "expected"), [(0.5, "yes"), (0.2, "no")])
    def test_positive_and_neutral_final_messages_fall_back_to_prior(
        self, message, p, expected
    ):
        game = self._final_round_game(message, p=p)
        assert run(persuasion, game)["decision"] == expected


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


class TestNegotiationCompleteInfo:
    """CI feasibility clamp + surplus floor (neg_ci_floor_frac knob)."""

    def _ci_offer(self, rnd=5, my_value=100.0, opp_value=200.0):
        from fixtures import negotiation_game
        return negotiation_game(role="seller", game_state={
            "complete_information": True, "round": rnd,
            "player_1_value": my_value, "player_2_value": opp_value,
        })

    def _decide(self, game, **knob_over):
        from glee_agent.config import Knobs
        from glee_agent.families import negotiation as neg_mod
        from glee_agent.schema import parse_game
        return neg_mod.decide(parse_game(game), Knobs(llm_enabled=False, **knob_over))

    def test_ci_knob_keeps_offers_inside_feasible_band(self):
        # Seller 100 vs buyer 200, S=100: ask must sit in [140, 195].
        act = self._decide(self._ci_offer(), neg_ci_floor_frac=0.4)
        assert 140.0 - 1e-6 <= act["product_price"] <= 195.0 + 1e-6

    def test_ci_knob_off_keeps_old_anchor(self):
        # Default markup 0.9: round-1 ask is 190 (possibly optimizer-shifted,
        # but never floored up to 140 by the knob).
        act = self._decide(self._ci_offer(rnd=1))
        assert act["product_price"] <= 190.0 + 1e-6

    def test_ci_floor_never_lowers_capture(self):
        # Late round with knob: counter never drops below value + 0.4*S.
        from fixtures import negotiation_decision
        game = negotiation_decision(role="seller", offer_price=105.0, game_state={
            "complete_information": True, "round": 8,
            "player_1_value": 100.0, "player_2_value": 200.0,
        })
        act = self._decide(game, neg_ci_floor_frac=0.4)
        assert act["decision"] == "RejectOffer"
        assert act["product_price"] >= 140.0 - 1e-6

    def test_ci_ultimatum_default_captures_promoted_surplus(self):
        game = self._ci_offer(rnd=1, my_value=100.0, opp_value=200.0)
        game["game_state"]["max_rounds"] = 1
        act = self._decide(game)
        assert act["product_price"] == pytest.approx(195.0)

    def test_ci_ultimatum_default_is_symmetric_for_buyer(self):
        from fixtures import negotiation_game
        game = negotiation_game(role="buyer", game_state={
            "complete_information": True,
            "round": 1,
            "max_rounds": 1,
            "player_1_value": 100.0,
            "player_2_value": 200.0,
        })
        act = self._decide(game)
        assert act["product_price"] == pytest.approx(105.0)

    def test_ci_ultimatum_default_does_not_affect_incomplete_information(self):
        from fixtures import negotiation_game
        game = negotiation_game(role="seller", game_state={"max_rounds": 1})
        promoted = self._decide(game)
        rollback = self._decide(game, neg_ci_ultimatum_frac=0.0)
        assert promoted == rollback

    def test_ii_ultimatum_canary_uses_configured_seller_markup(self):
        from fixtures import negotiation_game
        game = negotiation_game(role="seller", game_state={"max_rounds": 1})
        act = self._decide(
            game, neg_ii_ultimatum_markup=0.475, neg_ii_prior_capture_frac=0.0
        )
        assert act["product_price"] == pytest.approx(147.5)

    def test_ii_ultimatum_canary_is_symmetric_for_buyer(self):
        from fixtures import negotiation_game
        game = negotiation_game(role="buyer", game_state={"max_rounds": 1})
        act = self._decide(
            game, neg_ii_ultimatum_markup=0.475, neg_ii_prior_capture_frac=0.0
        )
        assert act["product_price"] == pytest.approx(52.5)

    def test_ii_ultimatum_canary_does_not_override_complete_information(self):
        game = self._ci_offer(rnd=1, my_value=100.0, opp_value=200.0)
        game["game_state"]["max_rounds"] = 1
        act = self._decide(game, neg_ii_ultimatum_markup=0.475)
        assert act["product_price"] == pytest.approx(195.0)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(80.0, 118.0), (100.0, 147.5), (120.0, 148.5), (8000.0, 11800.0)],
    )
    def test_ii_prior_uses_value_aware_seller_anchor(self, value, expected):
        from fixtures import negotiation_game
        game = negotiation_game(role="seller", game_state={"player_1_value": value})
        act = self._decide(game, neg_ii_prior_capture_frac=0.95)
        assert act["product_price"] == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(100.0, 81.0), (120.0, 101.0), (150.0, 102.5), (1500000.0, 1025000.0)],
    )
    def test_ii_prior_uses_value_aware_buyer_anchor(self, value, expected):
        from fixtures import negotiation_game
        game = negotiation_game(role="buyer", game_state={"player_2_value": value})
        act = self._decide(game, neg_ii_prior_capture_frac=0.95)
        assert act["product_price"] == pytest.approx(expected)

    def test_ii_prior_falls_back_outside_known_value_grid(self):
        from fixtures import negotiation_game
        game = negotiation_game(role="seller", game_state={"player_1_value": 90.0})
        baseline = self._decide(game)
        prior = self._decide(game, neg_ii_prior_capture_frac=0.95)
        assert prior == baseline

    def test_ii_prior_prices_one_round_game_directly(self):
        from fixtures import negotiation_game
        game = negotiation_game(
            role="seller", game_state={"max_rounds": 1, "player_1_value": 120.0}
        )
        act = self._decide(game, neg_ii_prior_capture_frac=0.95)
        assert act["product_price"] == pytest.approx(148.5)

    def test_ii_prior_is_the_promoted_default(self):
        from fixtures import negotiation_game
        game = negotiation_game(role="seller", game_state={"player_1_value": 80.0})
        promoted = self._decide(game)
        explicit = self._decide(game, neg_ii_prior_capture_frac=0.95)
        rollback = self._decide(game, neg_ii_prior_capture_frac=0.0)
        assert promoted == explicit
        assert promoted["product_price"] == pytest.approx(118.0)
        assert rollback["product_price"] != promoted["product_price"]


class TestNegotiationEndgame:
    @staticmethod
    def _stonewall_history(role):
        me = "player_1" if role == "seller" else "player_2"
        opp = "player_2" if role == "seller" else "player_1"
        if role == "seller":
            return [
                {"round": 1, "offer": {"price": 147.5, "from_player": me},
                 "counteroffer": 90.0},
                {"round": 2, "offer": {"price": 90.0, "from_player": opp},
                 "counteroffer": 146.0},
                {"round": 3, "offer": {"price": 146.0, "from_player": me},
                 "counteroffer": 90.0},
            ]
        return [
            {"round": 1, "offer": {"price": 130.0, "from_player": opp},
             "counteroffer": 81.0},
            {"round": 2, "offer": {"price": 81.0, "from_player": me},
             "counteroffer": 110.0},
            {"round": 3, "offer": {"price": 110.0, "from_player": opp},
             "counteroffer": 82.0},
            {"round": 4, "offer": {"price": 82.0, "from_player": me},
             "counteroffer": 110.0},
        ]

    @classmethod
    def _stonewall_game(cls, role, phase, rnd):
        state = {"round": rnd, "history": cls._stonewall_history(role)}
        if phase == "offer":
            return negotiation_game(role=role, game_state=state)
        offer = 90.0 if role == "seller" else 110.0
        return negotiation_decision(
            role=role, offer_price=offer, game_state=state
        )

    @pytest.mark.parametrize(
        "role,floor,capped_price",
        [("seller", 102.0, 145.54), ("buyer", 98.0, 82.17)],
    )
    @pytest.mark.parametrize("phase", ["offer", "decision"])
    def test_hidden_terminal_close_bypasses_reciprocity_at_round_nine(
        self, role, floor, capped_price, phase
    ):
        game = self._stonewall_game(role, phase, rnd=9)
        view = parse_game(game)
        capped = negotiation.decide(
            view, Knobs(llm_enabled=False, neg_terminal_close=False)
        )
        closing = negotiation.decide(
            view, Knobs(llm_enabled=False, neg_terminal_close=True)
        )
        assert capped["product_price"] == pytest.approx(capped_price)
        assert closing["product_price"] == pytest.approx(floor)

    @pytest.mark.parametrize(
        "role,floor,capped_price",
        [("seller", 102.0, 145.54), ("buyer", 98.0, 82.17)],
    )
    @pytest.mark.parametrize("phase", ["offer", "decision"])
    def test_hidden_terminal_close_preserves_reciprocity_at_round_eight(
        self, role, floor, capped_price, phase
    ):
        game = self._stonewall_game(role, phase, rnd=8)
        view = parse_game(game)
        capped = negotiation.decide(
            view, Knobs(llm_enabled=False, neg_terminal_close=False)
        )
        canary = negotiation.decide(
            view, Knobs(llm_enabled=False, neg_terminal_close=True)
        )
        assert capped["product_price"] == pytest.approx(capped_price)
        assert canary["product_price"] == pytest.approx(capped["product_price"])
        assert canary["product_price"] != pytest.approx(floor)

    @pytest.mark.parametrize("role", ["seller", "buyer"])
    @pytest.mark.parametrize("phase", ["offer", "decision"])
    def test_complete_info_terminal_close_is_inert(self, role, phase):
        game = self._stonewall_game(role, phase, rnd=9)
        game["game_state"].update({
            "complete_information": True,
            "player_1_value": 100.0 if role == "seller" else 80.0,
            "player_2_value": 200.0 if role == "seller" else 100.0,
        })
        view = parse_game(game)
        off = negotiation.decide(
            view, Knobs(llm_enabled=False, neg_terminal_close=False)
        )
        on = negotiation.decide(
            view, Knobs(llm_enabled=False, neg_terminal_close=True)
        )
        assert on == off

    @pytest.mark.parametrize(
        "role,floor", [("seller", 102.0), ("buyer", 98.0)]
    )
    @pytest.mark.parametrize("phase", ["offer", "decision"])
    def test_hidden_terminal_close_uses_floor_instead_of_optimizer(
        self, monkeypatch, role, floor, phase
    ):
        def unexpected_optimizer(*args, **kwargs):
            raise AssertionError("terminal floor was replaced by optimizer")

        monkeypatch.setattr(negotiation, "_optimized_price", unexpected_optimizer)
        game = self._stonewall_game(role, phase, rnd=9)
        closing = negotiation.decide(
            parse_game(game),
            Knobs(llm_enabled=False, neg_terminal_close=True),
        )
        assert closing["product_price"] == pytest.approx(floor)

    @pytest.mark.parametrize(
        "role,terminal,reciprocal,expected",
        [
            ("seller", 110.0, 105.0, 105.0),
            ("seller", 100.0, 105.0, 100.0),
            ("buyer", 90.0, 95.0, 95.0),
            ("buyer", 100.0, 95.0, 100.0),
        ],
    )
    def test_terminal_close_never_moves_away_from_opponent(
        self, role, terminal, reciprocal, expected
    ):
        class State:
            my_role = role

        assert negotiation._terminal_generosity_guard(
            State(), terminal, reciprocal
        ) == pytest.approx(expected)

    @pytest.mark.parametrize("role", ["seller", "buyer"])
    def test_terminal_close_does_not_change_one_round_optimizer(self, role):
        game = negotiation_game(role=role, game_state={"max_rounds": 1})
        off = negotiation.decide(
            parse_game(game), Knobs(llm_enabled=False, neg_terminal_close=False)
        )
        on = negotiation.decide(
            parse_game(game), Knobs(llm_enabled=False, neg_terminal_close=True)
        )
        assert on == off

    def test_finite_endgame_prices_floor(self):
        # Seller, T=10, round 9 (incomplete info): schedule used to leave the
        # ask at ~1.24x value; now the last offers go out at the floor.
        from fixtures import negotiation_game
        game = negotiation_game(role="seller", game_state={"round": 9})
        action = run(negotiation, game)
        assert action["product_price"] == pytest.approx(102.0)

    def test_no_finite_walkaway_counter_at_floor(self):
        # Buyer at round 9 of 10 facing a losing offer: never walk (that
        # forfeits our free round-10 floor offer) — reject and counter.
        from fixtures import negotiation_decision
        game = negotiation_decision(role="buyer", offer_price=150.0,
                                    game_state={"round": 9})
        action = run(negotiation, game)
        assert action["decision"] == "RejectOffer"
        assert action["product_price"] == pytest.approx(98.0)

    def test_trajectory_gate_blocks_low_percentile_crumb(self, monkeypatch):
        from fixtures import negotiation_decision
        from glee_agent.config import Knobs
        from glee_agent.families import negotiation as neg_mod
        from glee_agent.schema import parse_game
        # Unlimited horizon, round 12: schedule floor makes counter_payoff a
        # crumb, so the trajectory rule fires; the gate must consult the pool.
        game = negotiation_decision(role="seller", offer_price=103.0)
        del game["game_state"]["max_rounds"]
        game["game_state"]["horizon_known"] = False
        game["game_state"]["round"] = 12
        view = parse_game(game)
        monkeypatch.setattr(neg_mod, "_payoff_percentile", lambda *a, **k: 0.20)
        gated = neg_mod.decide(view, Knobs(llm_enabled=False, neg_traj_pct_gate=0.45))
        assert gated["decision"] == "RejectOffer"
        monkeypatch.setattr(neg_mod, "_payoff_percentile", lambda *a, **k: 0.80)
        good = neg_mod.decide(view, Knobs(llm_enabled=False, neg_traj_pct_gate=0.45))
        assert good["decision"] == "AcceptOffer"
        ungated = neg_mod.decide(view, Knobs(llm_enabled=False))
        assert ungated["decision"] == "AcceptOffer"

    def test_bare_float_counteroffer_parsed(self):
        from fixtures import negotiation_game
        from glee_agent.families import negotiation as neg_mod
        from glee_agent.schema import parse_game, parse_negotiation
        game = negotiation_game(role="seller", game_state={"history": [
            {"round": 1,
             "offer": {"price": 120.0, "from_player": "player_1"},
             "counteroffer": 95.0},
        ]})
        view = parse_game(game)
        n = parse_negotiation(view)
        assert neg_mod._opponent_best_price(view, n) == pytest.approx(95.0)


class TestNegotiationInvariants:
    """The feasibility and reciprocity invariants added after measuring that
    ~58% of our complete-info offers were ones the opponent could never
    profitably accept (0-0.6% acceptance)."""

    @staticmethod
    def _ci_game(role="seller", rnd=1, history=None, my_v=100.0, opp_v=200.0):
        from fixtures import negotiation_game
        st = {"complete_information": True, "round": rnd,
              "player_1_value": my_v if role == "seller" else opp_v,
              "player_2_value": opp_v if role == "seller" else my_v}
        if history is not None:
            st["history"] = history
        g = negotiation_game(role=role, game_state=st)
        return g

    def test_offer_always_leaves_known_opponent_a_profit(self):
        """Property: across roles, rounds and knob settings, a returned price
        never puts a known-value opponent under water."""
        from glee_agent.schema import parse_game, parse_negotiation
        for role in ("seller", "buyer"):
            for rnd in (1, 2, 5, 9, 10):
                for frac in (0.0, 0.4, 0.6):
                    my_value, opp_value = (
                        (100.0, 200.0) if role == "seller" else (200.0, 100.0)
                    )
                    g = self._ci_game(
                        role=role,
                        rnd=rnd,
                        my_v=my_value,
                        opp_v=opp_value,
                    )
                    view = parse_game(g)
                    n = parse_negotiation(view)
                    act = negotiation.decide(view, Knobs(llm_enabled=False,
                                                         neg_ci_floor_frac=frac))
                    price = act.get("product_price")
                    if price is None:
                        continue
                    opp_payoff = (n.opp_value - price if n.my_role == "seller"
                                  else price - n.opp_value)
                    assert opp_payoff > 0, (role, rnd, frac, price, n.opp_value)

    @pytest.mark.parametrize(
        ("role", "my_value", "opp_value"),
        [("seller", 1000.0, 1005.0), ("buyer", 1005.0, 1000.0)],
    )
    def test_thin_surplus_offer_stays_inside_both_reservations(
        self, role, my_value, opp_value
    ):
        game = self._ci_game(
            role=role, rnd=2, my_v=my_value, opp_v=opp_value
        )
        game["game_state"]["max_rounds"] = 2

        action = negotiation.decide(parse_game(game), Knobs(llm_enabled=False))

        price = action["product_price"]
        seller_value = my_value if role == "seller" else opp_value
        buyer_value = opp_value if role == "seller" else my_value
        assert price == pytest.approx(1002.5)
        assert price == round(price, 2)
        assert seller_value <= price <= buyer_value

    @pytest.mark.parametrize(
        ("role", "my_value", "opp_value", "offer"),
        [
            ("seller", 1000.0, 1005.0, 999.0),
            ("buyer", 1005.0, 1000.0, 1006.0),
        ],
    )
    def test_thin_surplus_counter_stays_inside_both_reservations(
        self, role, my_value, opp_value, offer
    ):
        from fixtures import negotiation_decision

        state = {
            "complete_information": True,
            "round": 1,
            "max_rounds": 2,
            "player_1_value": my_value if role == "seller" else opp_value,
            "player_2_value": opp_value if role == "seller" else my_value,
        }
        game = negotiation_decision(
            role=role, offer_price=offer, game_state=state
        )

        action = negotiation.decide(parse_game(game), Knobs(llm_enabled=False))

        assert action["decision"] == "RejectOffer"
        seller_value = my_value if role == "seller" else opp_value
        buyer_value = opp_value if role == "seller" else my_value
        assert action["product_price"] == pytest.approx(1002.5)
        assert action["product_price"] == round(action["product_price"], 2)
        assert seller_value <= action["product_price"] <= buyer_value

    @pytest.mark.parametrize("action_type", ["offer", "counter"])
    @pytest.mark.parametrize("role", ["seller", "buyer"])
    def test_subcent_surplus_retains_only_needed_precision(self, action_type, role):
        from fixtures import negotiation_decision

        seller_value, buyer_value = 1000.001, 1000.005
        state = {
            "complete_information": True,
            "round": 2 if action_type == "offer" else 1,
            "max_rounds": 2,
            "messages_allowed": False,
            "player_1_value": seller_value,
            "player_2_value": buyer_value,
        }
        if action_type == "offer":
            game = negotiation_game(role=role, game_state=state)
        else:
            incoming = (
                seller_value - 0.001
                if role == "seller"
                else buyer_value + 0.001
            )
            game = negotiation_decision(
                role=role, offer_price=incoming, game_state=state
            )

        action = negotiation.decide(parse_game(game), Knobs(llm_enabled=False))

        if action_type == "counter":
            assert action["decision"] == "RejectOffer"
        price = action["product_price"]
        assert seller_value < price < buyer_value
        assert price == pytest.approx(1000.003)
        assert price != round(price, 2)

    @pytest.mark.parametrize(
        ("role", "my_value", "opp_value"),
        [("seller", 1005.0, 1000.0), ("buyer", 1000.0, 1005.0)],
    )
    def test_incompatible_values_prioritize_own_non_loss(
        self, role, my_value, opp_value
    ):
        game = self._ci_game(
            role=role, rnd=2, my_v=my_value, opp_v=opp_value
        )

        action = negotiation.decide(parse_game(game), Knobs(llm_enabled=False))

        if role == "seller":
            assert action["product_price"] >= my_value
        else:
            assert 0.0 <= action["product_price"] <= my_value

    @pytest.mark.parametrize("role", ["seller", "buyer"])
    def test_hidden_value_feasibility_clamp_preserves_candidate(self, role):
        game = negotiation_game(role=role)
        state = parse_negotiation(parse_game(game))
        candidate = 175.0 if role == "seller" else 25.0

        assert negotiation._feasible_price(candidate, state) == candidate

    def test_concession_never_outruns_the_opponent(self):
        """A stonewalling opponent gets at most the drip, not the schedule."""
        from glee_agent.schema import parse_game
        # They repeat 120 twice (zero concession); we already offered 190.
        hist = [
            {"round": 1, "offer": {"price": 190.0, "from_player": "player_1"},
             "counteroffer": 120.0},
            {"round": 2, "offer": {"price": 185.0, "from_player": "player_1"},
             "counteroffer": 120.0},
        ]
        g = self._ci_game(rnd=3, history=hist)
        g["game_state"]["last_offer"] = {"price": 120.0, "from_player": "player_2",
                                         "round": 2, "message": None}
        g["phase"] = g["game_state"]["phase"] = "decision"
        g["game_state"]["current_player"] = "player_1"
        g["valid_actions"] = {"type": "decision", "fields": {
            "decision": ["AcceptOffer", "RejectOffer", "WalkAway"],
            "product_price": "number"}}
        act = negotiation.decide(parse_game(g), Knobs(llm_enabled=False))
        if act["decision"] == "RejectOffer":
            # my last was 185; a stonewaller buys at most the drip of movement
            assert act["product_price"] >= 185.0 - 0.02 * 200.0

    def test_never_counters_below_their_best_offer(self):
        """Walkback resistance: a price they already offered is banked."""
        from glee_agent.schema import parse_game
        hist = [
            {"round": 1, "offer": {"price": 190.0, "from_player": "player_1"},
             "counteroffer": 175.0},
            {"round": 2, "offer": {"price": 188.0, "from_player": "player_1"},
             "counteroffer": 130.0},
        ]
        g = self._ci_game(rnd=3, history=hist)
        g["game_state"]["last_offer"] = {"price": 130.0, "from_player": "player_2",
                                         "round": 2, "message": None}
        g["phase"] = g["game_state"]["phase"] = "decision"
        g["game_state"]["current_player"] = "player_1"
        g["valid_actions"] = {"type": "decision", "fields": {
            "decision": ["AcceptOffer", "RejectOffer", "WalkAway"],
            "product_price": "number"}}
        act = negotiation.decide(parse_game(g), Knobs(llm_enabled=False))
        if act["decision"] == "RejectOffer":
            assert act["product_price"] >= 175.0


class TestBargainingWalkback:
    """A share the opponent has already offered is conceded ground."""

    @staticmethod
    def _hist(shares_to_me, pot=1000):
        out = []
        for i, s in enumerate(shares_to_me, start=1):
            out.append({"round": i, "proposer": "player_2", "decision": "reject",
                        "offer": {"player_1_gain": pot * s,
                                  "player_2_gain": pot * (1 - s), "message": None}})
        return out

    def test_never_offers_self_less_than_they_already_gave(self):
        game = bargaining_game(game_state={
            "round": 5, "delta_1": 0.95, "delta_2": 0.95,
            "history": self._hist([0.30, 0.62])})
        act = run(bargaining, game)
        assert act["alice_gain"] >= 620.0 - 1e-6

    def test_rejects_an_offer_worse_than_their_own_best(self):
        game = bargaining_decision(my_gain=400, opp_gain=600, game_state={
            "round": 5, "delta_1": 0.95, "delta_2": 0.95,
            "history": self._hist([0.30, 0.62])})
        assert run(bargaining, game)["decision"] == "reject"

    def test_final_round_still_accepts_anything(self):
        game = bargaining_decision(my_gain=10, opp_gain=990, game_state={
            "round": 10, "history": self._hist([0.62])})
        assert run(bargaining, game)["decision"] == "accept"


class TestBargainingDisadvantageAnchor:
    def test_dis_anchor_knob_moves_the_opening(self):
        game = bargaining_game(game_state={"delta_1": 0.8, "delta_2": 1.0})
        from glee_agent.schema import parse_game as _pg
        hi = bargaining.decide(_pg(game), Knobs(llm_enabled=False, barg_dis_anchor=0.58))
        lo = bargaining.decide(_pg(game), Knobs(llm_enabled=False, barg_dis_anchor=0.50))
        assert hi["alice_gain"] == pytest.approx(580.0)
        assert lo["alice_gain"] == pytest.approx(500.0)


class TestNegotiationStalemateKnobs:
    @staticmethod
    def _marathon(offer_price, rnd):
        from fixtures import negotiation_decision
        g = negotiation_decision(role="buyer", offer_price=offer_price)
        del g["game_state"]["max_rounds"]
        g["game_state"]["horizon_known"] = False
        g["game_state"]["round"] = rnd
        return g

    def test_never_walk_knob_keeps_countering(self):
        game = self._marathon(150.0, 30)   # offered above our value, hopeless
        assert run(negotiation, game)["decision"] == "WalkAway"
        act = negotiation.decide(parse_game(game),
                                 Knobs(llm_enabled=False, neg_never_walk=True))
        assert act["decision"] == "RejectOffer"

    def test_stall_accept_knob_closes_earlier(self):
        game = self._marathon(95.0, 9)     # small profit on the table at round 9
        assert run(negotiation, game)["decision"] == "RejectOffer"
        act = negotiation.decide(parse_game(game),
                                 Knobs(llm_enabled=False, neg_stall_accept=8))
        assert act["decision"] == "AcceptOffer"


class TestNegotiationFloorLadder:
    def test_higher_floor_actually_raises_the_offer(self):
        """Regression: the anchor cap used to clamp the floor, so every floor
        above neg_ci_anchor_frac produced identical play."""
        from fixtures import negotiation_game
        from glee_agent.schema import parse_game, parse_negotiation
        prices = []
        for frac in (0.5, 0.7, 0.9):
            g = negotiation_game(role="seller", game_state={
                "complete_information": True, "round": 3,
                "player_1_value": 100.0, "player_2_value": 200.0})
            v = parse_game(g)
            act = negotiation.decide(v, Knobs(llm_enabled=False, neg_ci_floor_frac=frac))
            prices.append(act["product_price"])
        assert prices[0] < prices[1] < prices[2], prices
        n = parse_negotiation(parse_game(negotiation_game(role="seller", game_state={
            "complete_information": True, "round": 3,
            "player_1_value": 100.0, "player_2_value": 200.0})))
        assert prices[-1] < n.opp_value        # still feasible


class TestOpponentBook:
    """Per-opponent anchor shift: bounded, sample-gated, off by default."""

    def _game(self, name, monkey, entry):
        from glee_agent.theory import opponents as O
        monkey.setattr(O, "book_entry", lambda n, f: entry if n == name else None)
        return bargaining_game(opponent={"type": "agent", "name": name})

    def test_off_by_default(self, monkeypatch):
        g = self._game("Agent Smith", monkeypatch, {"n": 500, "share": 0.40})
        base = bargaining.decide(parse_game(g), Knobs(llm_enabled=False))
        assert base["alice_gain"] == pytest.approx(
            bargaining.decide(parse_game(bargaining_game()), Knobs(llm_enabled=False))["alice_gain"])

    def test_holds_harder_against_a_name_that_beats_us(self, monkeypatch):
        g = self._game("Agent Smith", monkeypatch, {"n": 500, "share": 0.40})
        k = Knobs(llm_enabled=False, barg_book_gain=1.0)
        tough = bargaining.decide(parse_game(g), k)
        neutral = bargaining.decide(parse_game(bargaining_game()), k)
        assert tough["alice_gain"] > neutral["alice_gain"]

    def test_thin_record_moves_nothing(self, monkeypatch):
        g = self._game("Newbie", monkeypatch, {"n": 5, "share": 0.40})
        k = Knobs(llm_enabled=False, barg_book_gain=1.0)
        assert bargaining.decide(parse_game(g), k)["alice_gain"] == pytest.approx(
            bargaining.decide(parse_game(bargaining_game()), k)["alice_gain"])

    def test_adjustment_is_bounded(self, monkeypatch):
        from glee_agent.families import bargaining as B
        g = self._game("Crusher", monkeypatch, {"n": 900, "share": 0.05})
        assert B._book_adjust(parse_game(g), Knobs(barg_book_gain=5.0)) <= 0.10
        g2 = self._game("Pushover", monkeypatch, {"n": 900, "share": 0.99})
        assert B._book_adjust(parse_game(g2), Knobs(barg_book_gain=5.0)) >= -0.05


class TestOpponentBookAccept:
    def _game(self, name, monkey, entry, my_gain=400):
        from glee_agent.theory import opponents as O
        monkey.setattr(O, "book_entry", lambda n, f: entry if n == name else None)
        return bargaining_decision(my_gain=my_gain, opp_gain=1000 - my_gain,
                                   opponent={"type": "agent", "name": name})

    def test_raises_the_bar_against_a_name_that_beats_us(self, monkeypatch):
        """The shift must bite on barg_accept_great -- the rule that actually
        decides most games. An offer of 0.66 clears the default 0.65 bar and is
        taken; against a name that has been beating us, the bar moves to 0.73
        and we hold out instead."""
        g = self._game("Agent Smith", monkeypatch, {"n": 500, "share": 0.40}, my_gain=660)
        off = bargaining.decide(parse_game(g), Knobs(llm_enabled=False))["decision"]
        on = bargaining.decide(parse_game(g),
                               Knobs(llm_enabled=False, barg_book_accept_gain=1.0))["decision"]
        assert (off, on) == ("accept", "reject")

    def test_no_shift_for_a_name_we_beat(self, monkeypatch):
        from glee_agent.families import bargaining as B
        g = self._game("Pushover", monkeypatch, {"n": 500, "share": 0.65})
        assert B._book_accept_shift(parse_game(g), Knobs(barg_book_accept_gain=1.0)) == 0.0

    def test_shift_is_bounded_and_gated(self, monkeypatch):
        from glee_agent.families import bargaining as B
        g = self._game("Crusher", monkeypatch, {"n": 900, "share": 0.05})
        assert B._book_accept_shift(parse_game(g), Knobs(barg_book_accept_gain=9.0)) <= 0.08
        thin = self._game("Newbie", monkeypatch, {"n": 5, "share": 0.05})
        assert B._book_accept_shift(parse_game(thin), Knobs(barg_book_accept_gain=1.0)) == 0.0


class TestSellerBuyerSurplus:
    """KG's x* is the buyer's indifference point; sitting on it earns them
    exactly zero and they stop buying. Measured live: in configs where
    p*v == price we ran a lie rate of 1.000 and earned 0.39-0.56 of max
    revenue while the field ran 0.28-0.43 and earned 0.70-0.77."""

    def test_lie_rate_leaves_the_buyer_a_real_surplus(self):
        from glee_agent.theory.bayes import lie_rate_for_surplus, posterior_high_given_rec
        p, v, price, s = 0.8, 12500.0, 10000.0, 0.10
        x = lie_rate_for_surplus(p, v, price, s)
        q = posterior_high_given_rec(p, x)
        assert q * v - price >= s * price - 1e-6

    def test_never_exceeds_kg(self):
        from glee_agent.theory.bayes import lie_rate_for_surplus, kg_lie_rate
        for p, v, price in ((0.5, 20000.0, 10000.0), (0.8, 12000.0, 10000.0),
                            (0.333, 20000.0, 10000.0)):
            assert lie_rate_for_surplus(p, v, price, 0.10) <= kg_lie_rate(p, v, price) + 1e-9

    def test_zero_when_product_cannot_clear_the_bar(self):
        from glee_agent.theory.bayes import lie_rate_for_surplus
        assert lie_rate_for_surplus(0.5, 10500.0, 10000.0, 0.10) == 0.0

    def test_indifference_config_no_longer_pools(self):
        """p*v == price exactly: the old short-circuit recommended every
        round, which is what the live logs showed."""
        from glee_agent.families import persuasion as P
        from glee_agent.schema import parse_game, parse_persuasion
        recs = 0
        for rnd in range(1, 21):
            g = persuasion_game(actor="seller", quality="low", game_state={
                "p": 0.8, "v": 12500.0, "product_price": 10000.0,
                "total_rounds": 20, "round": rnd})
            v = parse_game(g)
            recs += P._seller_wants_to_recommend(v, parse_persuasion(v),
                                                 Knobs(llm_enabled=False))
        assert recs < 20, "still pooling on every low-quality round"


class TestBlindSeller:
    """v hidden is HALF our seller games. The branch used to return True
    before trust repair, so a blind seller pooled forever with no recovery."""

    def _blind(self, history, rnd):
        g = persuasion_game(actor="seller", quality="low", game_state={
            "p": 0.5, "product_price": 100.0, "total_rounds": 20,
            "round": rnd, "history": history})
        g["game_state"]["v"] = None
        return g

    def test_trust_repair_now_applies_when_v_is_hidden(self):
        from glee_agent.families import persuasion as P
        from glee_agent.schema import parse_game, parse_persuasion
        hist = [{"round": r, "seller_message": "I recommend it",
                 "buyer_decision": "no", "bought": False} for r in (1, 2, 3)]
        v = parse_game(self._blind(hist, 4))
        assert P._seller_wants_to_recommend(v, parse_persuasion(v),
                                            Knobs(llm_enabled=False)) is False

    def test_default_still_pools_while_the_buyer_is_buying(self):
        from glee_agent.families import persuasion as P
        from glee_agent.schema import parse_game, parse_persuasion
        hist = [{"round": 1, "seller_message": "I recommend it",
                 "buyer_decision": "yes", "bought": True, "quality": "high"}]
        v = parse_game(self._blind(hist, 2))
        assert P._seller_wants_to_recommend(v, parse_persuasion(v),
                                            Knobs(llm_enabled=False)) is True

    def test_knob_reduces_the_blind_lie_rate(self):
        from glee_agent.families import persuasion as P
        from glee_agent.schema import parse_game, parse_persuasion
        recs = 0
        for rnd in range(1, 21):
            v = parse_game(self._blind([], rnd))
            recs += P._seller_wants_to_recommend(
                v, parse_persuasion(v), Knobs(llm_enabled=False, pers_blind_lie=0.4))
        assert 1 <= recs <= 14, recs
