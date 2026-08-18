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
