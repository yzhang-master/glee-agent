"""Results-capture layer: JSONL record shapes, LoggingGleeClient.move
pass-through + logging, and the never-raise guarantee on garbage input."""

import json
import threading

import pytest

from glee_agent import logging_
from glee_agent.capture import LoggingGleeClient


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(logging_, "LOG_DIR", tmp_path)
    return tmp_path


def read_records(log_dir, prefix: str) -> list[dict]:
    files = sorted(log_dir.glob(f"{prefix}-*.jsonl"))
    records = []
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            records.append(json.loads(line))
    return records


def test_log_result_game_over(log_dir):
    move_response = {
        "valid": True,
        "game_over": True,
        "result": {
            "player_1_payoff": 600,
            "player_2_payoff": 400,
            "outcome": "agreement",
            "agreed_round": 12,
        },
        "error": None,
        "attempts_left": None,
    }
    logging_.log_result("main", "game-123", move_response)

    records = read_records(log_dir, "main")
    assert len(records) == 1
    rec = records[0]
    assert rec["type"] == "result"
    assert isinstance(rec["ts"], float)
    assert rec["agent"] == "main"
    assert rec["game_id"] == "game-123"
    assert rec["valid"] is True
    assert rec["attempts_left"] is None
    assert rec["game_over"] is True
    assert rec["error"] is None
    assert rec["result"]["player_1_payoff"] == 600
    assert rec["result"]["outcome"] == "agreement"


def test_log_result_invalid_move(log_dir):
    move_response = {
        "valid": False,
        "game_over": False,
        "result": None,
        "error": "gains must sum to the pot",
        "attempts_left": 2,
    }
    logging_.log_result("test_a", "game-9", move_response)

    rec = read_records(log_dir, "test_a")[0]
    assert rec["valid"] is False
    assert rec["attempts_left"] == 2
    assert rec["game_over"] is False
    assert rec["error"] == "gains must sum to the pot"
    assert rec["result"] is None


def test_log_snapshot_shape(log_dir):
    stats = {
        "agent_id": "abc",
        "agent_name": "glee-main",
        "scores": {"bargaining": {"rating": 1502.3, "games_played": 41}},
        "active_games": 3,
    }
    logging_.log_snapshot("main", stats)

    rec = read_records(log_dir, "main")[0]
    assert rec["type"] == "snapshot"
    assert isinstance(rec["ts"], float)
    assert rec["agent"] == "main"
    assert rec["scores"] == {"bargaining": {"rating": 1502.3, "games_played": 41}}
    assert rec["active_games"] == 3


def test_log_lb_snapshot_truncates_and_finds_file(log_dir):
    entries = [{"rank": i + 1, "player_id": f"p{i}", "rating": 1500 - i} for i in range(60)]
    me = {"rank": 7, "player_id": "p6", "rating": 1494}
    logging_.log_lb_snapshot("bargaining", entries, me)

    rec = read_records(log_dir, "platform")[0]
    assert rec["type"] == "lb_snapshot"
    assert rec["family"] == "bargaining"
    assert len(rec["top"]) == 50
    assert rec["top"][0]["rank"] == 1
    assert rec["me"] == me


def test_logging_client_move_logs_and_returns(log_dir, monkeypatch):
    client = LoggingGleeClient(api_key="glee_test_key", agent_label="test_a")
    response = {
        "valid": True,
        "game_over": False,
        "result": None,
        "error": None,
        "attempts_left": None,
    }
    calls = []

    def fake_request(method, path, **kwargs):
        calls.append((method, path, kwargs))
        return response

    monkeypatch.setattr(client, "_request", fake_request)

    out = client.move("g-42", {"decision": "accept"})
    assert out is response  # returned unchanged
    assert calls == [("POST", "/games/g-42/move", {"json": {"action": {"decision": "accept"}}})]

    rec = read_records(log_dir, "test_a")[0]
    assert rec["type"] == "result"
    assert rec["agent"] == "test_a"
    assert rec["game_id"] == "g-42"
    assert rec["valid"] is True
    assert rec["game_over"] is False


def test_logging_client_move_logs_error_and_reraises(log_dir, monkeypatch):
    client = LoggingGleeClient(api_key="glee_test_key", agent_label="test_a")

    def boom(method, path, **kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(client, "_request", boom)

    with pytest.raises(RuntimeError, match="connection reset"):
        client.move("g-42", {"decision": "accept"})

    rec = read_records(log_dir, "test_a")[0]
    assert rec["type"] == "result"
    assert rec["game_id"] == "g-42"
    assert rec["valid"] is None
    assert rec["game_over"] is False
    assert rec["result"] is None
    assert rec["error"] == "connection reset"


def test_log_functions_never_raise_on_garbage(log_dir):
    # None / non-dict move responses.
    logging_.log_result("main", "g", None)
    logging_.log_result("main", "g", "not a dict")  # type: ignore[arg-type]
    logging_.log_result("main", None, None, error=object())  # type: ignore[arg-type]

    # Non-serializable values: default=str covers objects/sets; a circular
    # reference makes json.dumps raise, which must be swallowed.
    circular: dict = {}
    circular["self"] = circular
    logging_.log_result("main", "g", {"result": circular, "valid": object()})
    logging_.log_result("main", "g", {"result": {1, 2, 3}, "game_over": threading.Lock()})

    # Garbage stats / leaderboard payloads.
    logging_.log_snapshot("main", None)  # type: ignore[arg-type]
    logging_.log_snapshot("main", ["not", "a", "dict"])  # type: ignore[arg-type]
    logging_.log_snapshot("main", {"scores": object()})
    logging_.log_lb_snapshot("bargaining", None, None)  # type: ignore[arg-type]
    logging_.log_lb_snapshot("bargaining", 42, {"player_id": "x"})  # type: ignore[arg-type]
    logging_.log_lb_snapshot(object(), [{"rank": 1}], None)  # type: ignore[arg-type]

    # No assertion on contents — the contract under test is "never raise".
    # Everything parseable that did land must still be valid JSON.
    for rec in read_records(log_dir, "main") + read_records(log_dir, "platform"):
        assert isinstance(rec, dict)


def test_log_functions_never_raise_when_log_dir_unwritable(tmp_path, monkeypatch):
    # Point LOG_DIR at a path that cannot be created (parent is a file).
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setattr(logging_, "LOG_DIR", blocker / "logs")

    logging_.log_result("main", "g", {"valid": True})
    logging_.log_snapshot("main", {"scores": {}, "active_games": 0})
    logging_.log_lb_snapshot("negotiation", [], None)
    logging_.log_turn("main", {}, {}, [], 0.1)
