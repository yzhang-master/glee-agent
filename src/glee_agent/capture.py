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
    "result" record before handing it back unchanged, and tracks the games
    it has seen so the reaper thread can back-fill results for games that
    end on the OPPONENT's move (the server only pushes a result payload when
    our own move ends the game — without the reaper, every game where the
    opponent accepted our offer would have no recorded payoff)."""

    def __init__(self, api_key: str, agent_label: str, **kwargs) -> None:
        super().__init__(api_key, **kwargs)
        self.agent_label = agent_label
        self._tracked_lock = threading.Lock()
        # game_id -> {"last_seen": ts, "resolved": bool}
        self._tracked: dict[str, dict] = {}

    def pending_games(self) -> list[dict]:
        games = super().pending_games()
        try:
            now = time.time()
            with self._tracked_lock:
                for g in games:
                    gid = g.get("game_id")
                    if gid:
                        entry = self._tracked.setdefault(
                            gid, {"last_seen": now, "resolved": False}
                        )
                        entry["last_seen"] = now
        except Exception:  # noqa: BLE001 — tracking must never disturb play
            pass
        return games

    def unresolved_games(self, idle_for: float = 180.0, max_age: float = 86400.0) -> list[str]:
        """Game ids not seen in pending for >= idle_for seconds and without a
        recorded end. Entries older than max_age are dropped."""
        now = time.time()
        out: list[str] = []
        with self._tracked_lock:
            for gid in list(self._tracked):
                entry = self._tracked[gid]
                if now - entry["last_seen"] > max_age:
                    del self._tracked[gid]
                    continue
                if not entry["resolved"] and now - entry["last_seen"] >= idle_for:
                    out.append(gid)
        return out

    def mark_resolved(self, game_id: str) -> None:
        with self._tracked_lock:
            entry = self._tracked.get(game_id)
            if entry is not None:
                entry["resolved"] = True

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
            log_result(
                self.agent_label,
                game_id,
                None,
                result_source="move_transport_error",
                reaped=False,
                error=error_text,
            )
            raise
        log_result(
            self.agent_label,
            game_id,
            response,
            result_source="move",
            reaped=False,
        )
        if isinstance(response, dict) and response.get("game_over") is True:
            self.mark_resolved(game_id)
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


def start_reaper_thread(
    game_client: LoggingGleeClient,
    telemetry_client: GleeClient,
    interval: float = 120.0,
    per_cycle: int = 10,
) -> threading.Thread:
    """Back-fill results for games that ended on the opponent's move.

    Every ~interval seconds, take up to per_cycle idle unresolved games and
    fetch their full state via GET /games/{id} on the DEDICATED telemetry
    client (never the game session). A finished game's result is logged as a
    synthetic game-over "result" record so the store's rollup completes it."""

    def _loop() -> None:
        while True:
            time.sleep(max(interval * random.uniform(0.8, 1.2), 1.0))
            try:
                for gid in game_client.unresolved_games()[:per_cycle]:
                    try:
                        data = telemetry_client.game_state(gid)
                    except Exception as e:  # noqa: BLE001 — skip; retry next cycle
                        logger.warning("Reaper fetch failed for %s: %s", gid, e)
                        continue
                    if not isinstance(data, dict):
                        continue
                    status = data.get("status")
                    if status in ("completed", "no_deal"):
                        result = data.get("result")
                        log_result(
                            game_client.agent_label,
                            gid,
                            {"valid": None, "game_over": True, "result": result},
                            result_source="reaper",
                            reaped=True,
                        )
                        game_client.mark_resolved(gid)
                    elif status is None and data.get("detail"):
                        # Unknown game (404-ish body): stop tracking it.
                        game_client.mark_resolved(gid)
            except Exception as e:  # noqa: BLE001 — telemetry must never die
                logger.warning("Reaper cycle failed: %s", e)

    thread = threading.Thread(target=_loop, name="glee-reaper", daemon=True)
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
