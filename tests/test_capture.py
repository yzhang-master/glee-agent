"""Results-capture layer: JSONL record shapes, LoggingGleeClient.move
pass-through + logging, and the never-raise guarantee on garbage input."""

import json
import threading

import pytest

import glee_agent.capture as capture
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
    logging_.log_result(
        "main",
        "game-123",
        move_response,
        result_source="move",
        reaped=False,
    )

    records = read_records(log_dir, "main")
    assert len(records) == 1
    rec = records[0]
    assert rec["type"] == "result"
    assert isinstance(rec["ts"], float)
    assert rec["agent"] == "main"
    assert rec["game_id"] == "game-123"
    assert rec["result_source"] == "move"
    assert rec["reaped"] is False
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
    logging_.log_result(
        "test_a",
        "game-9",
        move_response,
        result_source="move",
        reaped=False,
    )

    rec = read_records(log_dir, "test_a")[0]
    assert rec["valid"] is False
    assert rec["attempts_left"] == 2
    assert rec["game_over"] is False
    assert rec["error"] == "gains must sum to the pot"
    assert rec["result"] is None


def test_log_result_does_not_coerce_game_over_or_unstructured_result(log_dir):
    logging_.log_result(
        "test_a",
        "game-untrusted",
        {
            "valid": "false",
            "game_over": "false",
            "result": ["not", "a", "terminal", "object"],
        },
        result_source="move",
        reaped=False,
    )

    record = read_records(log_dir, "test_a")[0]
    assert record["valid"] == "false"  # preserved exactly, never truthified
    assert record["game_over"] is None
    assert record["result"] is None
    assert record["result_source"] == "move"
    assert record["reaped"] is False


def test_log_result_preserves_explicit_reaper_provenance(log_dir):
    result = {"outcome": "no_deal", "player_1_payoff": 0, "player_2_payoff": 0}
    logging_.log_result(
        "test_c",
        "game-reaped",
        {"valid": None, "game_over": True, "result": result, "reaped": True},
        result_source="reaper",
    )

    record = read_records(log_dir, "test_c")[0]
    assert record["result_source"] == "reaper"
    assert record["reaped"] is True
    assert record["game_over"] is True
    assert record["valid"] is None
    assert record["result"] == result


def test_log_turn_persists_canary_assignment_metadata(log_dir):
    assignment = {
        "status": "assigned",
        "plan_id": "rotation-1",
        "rule_id": "barg-1",
        "arm": "treatment",
        "assignment_sha256": "a" * 64,
        "enrollment": "new",
    }

    logging_.log_turn(
        "test_a",
        {"game_id": "g-1"},
        {"decision": "accept"},
        [],
        0.01,
        canary_assignment=assignment,
    )

    record = read_records(log_dir, "test_a")[0]
    assert record["canary_assignment"] == assignment


def test_only_assigned_canary_turns_are_fsynced_before_return(
    log_dir, monkeypatch
):
    real_fsync = logging_.os.fsync
    calls = []

    def record_fsync(descriptor):
        calls.append(descriptor)
        return real_fsync(descriptor)

    monkeypatch.setattr(logging_.os, "fsync", record_fsync)

    assert logging_.log_turn(
        "test_a",
        {"game_id": "base"},
        {"decision": "accept"},
        [],
        0.01,
        canary_assignment={"status": "unassigned", "reason": "plan_missing"},
    )
    assert calls == []

    assert logging_.log_turn(
        "test_a",
        {"game_id": "canary"},
        {"decision": "accept"},
        [],
        0.01,
        canary_assignment={"status": "assigned", "arm": "treatment"},
    )
    assert len(calls) >= 2  # turn file plus its directory entry


def test_assigned_turn_fsync_failure_is_reported(log_dir, monkeypatch):
    monkeypatch.setattr(
        logging_.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("injected")),
    )

    assert logging_.log_turn(
        "test_a",
        {"game_id": "canary"},
        {"decision": "accept"},
        [],
        0.01,
        canary_assignment={"status": "assigned", "arm": "treatment"},
    ) is False
    assert logging_.log_turn(
        "test_a",
        {"game_id": "base"},
        {"decision": "accept"},
        [],
        0.01,
        canary_assignment={"status": "unassigned"},
    ) is True


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
    assert rec["result_source"] == "move"
    assert rec["reaped"] is False


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
    assert rec["game_over"] is None
    assert rec["result"] is None
    assert rec["error"] == "connection reset"
    assert rec["result_source"] == "move_transport_error"
    assert rec["reaped"] is False


def test_reaper_callsite_emits_exact_provenance(monkeypatch):
    calls = []
    marked = []

    class GameClient:
        agent_label = "test_b"

        @staticmethod
        def unresolved_games():
            return ["g-reaper"]

        @staticmethod
        def mark_resolved(game_id):
            marked.append(game_id)

    class TelemetryClient:
        @staticmethod
        def game_state(game_id):
            assert game_id == "g-reaper"
            return {
                "status": "completed",
                "result": {
                    "outcome": "agreement",
                    "player_1_payoff": 60,
                    "player_2_payoff": 40,
                },
            }

    class StopLoop(BaseException):
        pass

    sleeps = [0]

    def one_iteration(_seconds):
        if sleeps:
            sleeps.pop()
            return
        raise StopLoop

    class ImmediateThread:
        def __init__(self, *, target, name, daemon):
            self.target = target
            self.name = name
            self.daemon = daemon

        def start(self):
            try:
                self.target()
            except StopLoop:
                pass

    monkeypatch.setattr(
        capture,
        "log_result",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    monkeypatch.setattr(capture.time, "sleep", one_iteration)
    monkeypatch.setattr(capture.threading, "Thread", ImmediateThread)

    capture.start_reaper_thread(GameClient(), TelemetryClient(), interval=1)

    assert marked == ["g-reaper"]
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0:2] == ("test_b", "g-reaper")
    assert args[2]["game_over"] is True
    assert kwargs == {"result_source": "reaper", "reaped": True}


def test_log_functions_never_raise_on_garbage(log_dir):
    # None / non-dict move responses.
    logging_.log_result(
        "main", "g", None, result_source="move_transport_error", reaped=False
    )
    logging_.log_result(  # type: ignore[arg-type]
        "main", "g", "not a dict", result_source="move", reaped=False
    )
    logging_.log_result(  # type: ignore[arg-type]
        "main",
        None,
        None,
        result_source="move_transport_error",
        reaped=False,
        error=object(),
    )

    # Non-serializable values: default=str covers objects/sets; a circular
    # reference makes json.dumps raise, which must be swallowed.
    circular: dict = {}
    circular["self"] = circular
    logging_.log_result(
        "main",
        "g",
        {"result": circular, "valid": object()},
        result_source="move",
        reaped=False,
    )
    logging_.log_result(
        "main",
        "g",
        {"result": {1, 2, 3}, "game_over": threading.Lock()},
        result_source="move",
        reaped=False,
    )

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

    logging_.log_result(
        "main", "g", {"valid": True}, result_source="move", reaped=False
    )
    logging_.log_snapshot("main", {"scores": {}, "active_games": 0})
    logging_.log_lb_snapshot("negotiation", [], None)
    assert logging_.log_turn("main", {}, {}, [], 0.1) is False
    assert logging_.log_runtime("main", {"pid": 1}) is False
