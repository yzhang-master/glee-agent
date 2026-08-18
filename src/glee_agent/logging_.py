"""Append-only JSONL turn log: full game dict, chosen action, guard
corrections, timing. One file per agent per day; thread-safe via a lock
(appends are small; contention is negligible at concurrency ~8)."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parents[2] / "logs"

_lock = threading.Lock()


def log_turn(
    agent_label: str,
    game: dict,
    action: dict,
    corrections: list[str],
    elapsed_s: float,
    move_result: dict | None = None,
    error: str | None = None,
) -> None:
    try:
        LOG_DIR.mkdir(exist_ok=True)
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        path = LOG_DIR / f"{agent_label}-{day}.jsonl"
        record = {
            "ts": time.time(),
            "game": game,
            "action": action,
            "corrections": corrections,
            "elapsed_s": round(elapsed_s, 3),
            "move_result": move_result,
            "error": error,
        }
        line = json.dumps(record, ensure_ascii=False, default=str)
        with _lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001 — logging must never break play
        pass
