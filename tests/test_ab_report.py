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
