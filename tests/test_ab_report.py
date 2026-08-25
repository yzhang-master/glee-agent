"""Regression tests for the live A/B percentile report."""

import json

from scripts.ab_report import game_percentile


class _RecordingTargets:
    def __init__(self):
        self.calls = []

    def payoff_percentile(self, family, key, role, payoff):
        decoded = json.loads(key) if key is not None else None
        self.calls.append((family, decoded, role, payoff))
        if decoded and "seller_value" in decoded and "buyer_value" in decoded:
            return 0.73
        if decoded and "my_delta" in decoded:
            return 0.61
        if decoded and decoded.get("is_seller_know_cv") is False:
            return 0.62
        return 0.42


def test_complete_information_negotiation_uses_exact_pool():
    targets = _RecordingTargets()
    config = {
        "player_1_value": 80,
        "player_2_value": 120,
        "max_rounds": 10,
        "horizon_known": True,
        "messages_allowed": False,
        "complete_information": True,
    }
    row = {
        "family": "negotiation",
        "your_player": "player_1",
        "config_json": json.dumps(config),
        "my_payoff": 25,
    }

    assert game_percentile(targets, row) == 0.73
    assert len(targets.calls) == 1
    _, key, role, payoff = targets.calls[0]
    assert key["seller_value"] == 80.0
    assert key["buyer_value"] == 120.0
    assert role == "player_1"
    assert payoff == 25


def test_hidden_bargaining_uses_visible_role_marginal():
    targets = _RecordingTargets()
    config = {
        "money_to_divide": 100,
        "delta_1": 0.9,
        "delta_2": None,
        "max_rounds": 12,
        "horizon_known": True,
        "messages_allowed": False,
        "complete_information": False,
    }
    row = {
        "family": "bargaining",
        "your_player": "player_1",
        "config_json": json.dumps(config),
        "my_payoff": 55,
    }

    assert game_percentile(targets, row) == 0.61
    assert len(targets.calls) == 1
    _, key, role, _ = targets.calls[0]
    assert key["my_delta"] == 0.9
    assert key["role"] == role == "player_1"
    assert "delta_2" not in key


def test_exact_bargaining_pool_is_preferred_to_available_marginal():
    targets = _RecordingTargets()
    config = {
        "money_to_divide": 100,
        "delta_1": 0.9,
        "delta_2": 0.8,
        "max_rounds": 12,
        "horizon_known": True,
        "messages_allowed": True,
        "complete_information": False,
    }
    row = {
        "family": "bargaining",
        "your_player": "player_1",
        "config_json": json.dumps(config),
        "my_payoff": 55,
    }

    assert game_percentile(targets, row) == 0.42
    assert len(targets.calls) == 1
    _, key, _, _ = targets.calls[0]
    assert key["delta_1"] == 0.9
    assert key["delta_2"] == 0.8
    assert "my_delta" not in key


def test_blind_seller_uses_visible_information_marginal():
    targets = _RecordingTargets()
    config = {
        "product_price": 100,
        "p": 0.5,
        "v": None,
        "u": None,
        "total_rounds": 20,
        "seller_message_type": "text",
    }
    row = {
        "family": "persuasion",
        "your_player": "player_1",
        "config_json": json.dumps(config),
        "my_payoff": 900,
    }

    assert game_percentile(targets, row) == 0.62
    assert len(targets.calls) == 1
    _, key, role, _ = targets.calls[0]
    assert key["is_seller_know_cv"] is False
    assert key["role"] == role == "player_1"
    assert "v" not in key and "u" not in key


def test_redacted_buyer_does_not_use_seller_marginal():
    targets = _RecordingTargets()
    row = {
        "family": "persuasion",
        "your_player": "player_2",
        "config_json": json.dumps({
            "product_price": 100,
            "p": 0.5,
            "v": None,
            "u": None,
            "total_rounds": 20,
            "seller_message_type": "text",
        }),
        "my_payoff": 10,
    }

    assert game_percentile(targets, row) is None
    assert targets.calls == []


def test_buildable_exact_miss_is_not_hidden_by_marginal_fallback():
    class _MissingTargets(_RecordingTargets):
        def payoff_percentile(self, family, key, role, payoff):
            decoded = json.loads(key)
            self.calls.append((family, decoded, role, payoff))
            return None

    targets = _MissingTargets()
    row = {
        "family": "bargaining",
        "your_player": "player_1",
        "config_json": json.dumps({
            "money_to_divide": 100,
            "delta_1": 0.9,
            "delta_2": 0.8,
            "max_rounds": 12,
            "horizon_known": True,
            "messages_allowed": True,
            "complete_information": False,
        }),
        "my_payoff": 50,
    }

    assert game_percentile(targets, row) is None
    assert len(targets.calls) == 1
    assert "delta_2" in targets.calls[0][1]
    assert "my_delta" not in targets.calls[0][1]
