"""Results-capture layer: a GleeClient subclass that mirrors every move
response into the JSONL log as a "result" record, plus daemon threads that
periodically snapshot our agent stats and the public leaderboards. All
logging here is best-effort (the logging_ writers never raise); the threads
swallow every per-iteration exception so telemetry can never disturb play."""

from __future__ import annotations

import logging
import random
import threading
import time

import requests
from glee_sdk import GleeClient

from .logging_ import log_lb_snapshot, log_result, log_snapshot

logger = logging.getLogger(__name__)

FAMILIES = ("bargaining", "negotiation", "persuasion")
LEADERBOARD_URL = "https://glee-competition.com/api/leaderboard"


class LoggingGleeClient(GleeClient):
    """GleeClient that logs every move response (or transport error) as a
    "result" record before handing it back unchanged."""

    def __init__(self, api_key: str, agent_label: str, **kwargs) -> None:
        super().__init__(api_key, **kwargs)
        self.agent_label = agent_label

    def move(self, game_id: str, action: dict) -> dict:
        try:
            response = super().move(game_id, action)
        except Exception as e:
            # Transport/API failure: record it, then let the SDK run loop's
            # error handling see the original exception. str(e) is guarded so
            # a hostile __str__ can't replace the in-flight exception.
            try:
                error_text = str(e)
            except Exception:  # noqa: BLE001
                error_text = type(e).__name__
            log_result(self.agent_label, game_id, None, error=error_text)
            raise
        log_result(self.agent_label, game_id, response)
        return response


def start_snapshot_thread(
    client: GleeClient, agent_label: str, interval: float = 300
) -> threading.Thread:
    """Start a daemon thread that logs a "snapshot" record from client.stats()
    every ~interval seconds (jittered). Returns the started thread.

    Pass a DEDICATED client, never the one the game loop plays with:
    requests.Session is not thread-safe, and sharing the session that submits
    moves risks a corrupted cookie jar raising inside a move worker mid-game."""

    def _loop() -> None:
        while True:
            try:
                log_snapshot(agent_label, client.stats())
            except Exception as e:  # noqa: BLE001 — telemetry must never die
                logger.warning("Stats snapshot failed: %s", e)
            time.sleep(max(interval * random.uniform(0.8, 1.2), 1.0))

    thread = threading.Thread(target=_loop, name="glee-snapshot", daemon=True)
    thread.start()
    return thread


def start_leaderboard_thread(
    agent_id: str | None, interval: float = 900
) -> threading.Thread:
    """Start a daemon thread that logs an "lb_snapshot" record per family from
    the public leaderboard endpoint (no auth) every ~interval seconds
    (jittered). ``agent_id`` locates our own entry, which may be absent.
    Returns the started thread."""

    def _loop() -> None:
        while True:
            for family in FAMILIES:
                try:
                    resp = requests.get(
                        LEADERBOARD_URL, params={"family": family}, timeout=15
                    )
                    resp.raise_for_status()
                    entries = resp.json()
                    if not isinstance(entries, list):
                        entries = []
                    me = None
                    if agent_id is not None:
                        me = next(
                            (
                                e
                                for e in entries
                                if isinstance(e, dict)
                                and e.get("player_id") == agent_id
                            ),
                            None,
                        )
                    log_lb_snapshot(family, entries, me)
                except Exception as e:  # noqa: BLE001 — telemetry must never die
                    logger.warning("Leaderboard snapshot (%s) failed: %s", family, e)
            time.sleep(max(interval * random.uniform(0.8, 1.2), 1.0))

    thread = threading.Thread(target=_loop, name="glee-leaderboard", daemon=True)
    thread.start()
    return thread
