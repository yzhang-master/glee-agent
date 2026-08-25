"""Append-only JSONL logs: runtime, turn, result, and telemetry records.

One file is written per agent per day (platform-wide records go to
``platform-<day>.jsonl``); writes are thread-safe.  Every writer swallows all
exceptions because observability must never raise into the game loop.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

# log_turn sits on the pre-move path of every game thread; telemetry writers
# (results, snapshots, leaderboards) take a separate lock so a stalled
# telemetry write can never block a move past the turn clock.
_lock = threading.Lock()
_telemetry_lock = threading.Lock()


def log_turn(
    agent_label: str,
    game: dict,
    action: dict,
    corrections: list[str],
    elapsed_s: float,
    move_result: dict | None = None,
    error: str | None = None,
    canary_assignment: dict | None = None,
) -> bool:
    try:
        LOG_DIR.mkdir(exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = LOG_DIR / f"{agent_label}-{day}.jsonl"
        record = {
            "type": "turn",
            "ts": time.time(),
            "game": game,
            "action": action,
            "corrections": corrections,
            "elapsed_s": round(elapsed_s, 3),
            "move_result": move_result,
            "error": error,
            "canary_assignment": (
                canary_assignment if isinstance(canary_assignment, dict) else None
            ),
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        return True
    except Exception:  # noqa: BLE001 — logging must never break play
        return False


def _append(prefix: str, record: dict, *, durable: bool = False) -> bool:
    """Append one JSON record to logs/<prefix>-<YYYYMMDD>.jsonl. Never raises."""
    try:
        LOG_DIR.mkdir(exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = LOG_DIR / f"{prefix}-{day}.jsonl"
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _telemetry_lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            if durable:
                f.flush()
                os.fsync(f.fileno())
        if durable:
            flags = os.O_RDONLY
            if hasattr(os, "O_DIRECTORY"):
                flags |= os.O_DIRECTORY
            directory = os.open(LOG_DIR, flags)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        return True
    except Exception:  # noqa: BLE001 — logging must never break play
        return False


def log_result(
    agent_label: str,
    game_id: str,
    move_response: dict | None,
    error: str | None = None,
) -> None:
    """Log the server's response to a move as a "result" record.

    ``move_response`` is the dict returned by ``GleeClient.move`` (or None when
    the call raised, in which case ``error`` carries the exception text)."""
    try:
        resp = move_response if isinstance(move_response, dict) else {}
        record = {
            "type": "result",
            "ts": time.time(),
            "agent": agent_label,
            "game_id": game_id,
            "valid": resp.get("valid"),
            "attempts_left": resp.get("attempts_left"),
            "game_over": bool(resp.get("game_over")),
            "error": error if error is not None else resp.get("error"),
            "result": resp.get("result"),
        }
        _append(agent_label, record)
    except Exception:  # noqa: BLE001 — logging must never break play
        pass


def log_runtime(agent_label: str, manifest: dict) -> bool:
    """Append one startup policy manifest as a ``type=runtime`` record."""
    try:
        body = manifest if isinstance(manifest, dict) else {}
        # Reserved envelope keys are written last so malformed callers cannot
        # disguise the record or redirect it to another agent identity.
        record = {
            **body,
            "type": "runtime",
            "ts": time.time(),
            "agent": agent_label,
        }
        return _append(agent_label, record, durable=True)
    except Exception:  # noqa: BLE001 — logging must never break play
        return False


def log_snapshot(agent_label: str, stats: dict) -> None:
    """Log the agent's current scores/active-game count as a "snapshot" record."""
    try:
        s = stats if isinstance(stats, dict) else {}
        record = {
            "type": "snapshot",
            "ts": time.time(),
            "agent": agent_label,
            "scores": s.get("scores"),
            "active_games": s.get("active_games"),
        }
        _append(agent_label, record)
    except Exception:  # noqa: BLE001 — logging must never break play
        pass


def log_lb_snapshot(family: str, entries: list, me: dict | None) -> None:
    """Log one family's public leaderboard (top 50 entries plus our own entry,
    if present) as an "lb_snapshot" record in logs/platform-<day>.jsonl."""
    try:
        top = list(entries)[:50] if isinstance(entries, (list, tuple)) else []
        record = {
            "type": "lb_snapshot",
            "ts": time.time(),
            "family": family,
            "top": top,
            "me": me,
        }
        _append("platform", record)
    except Exception:  # noqa: BLE001 — logging must never break play
        pass
