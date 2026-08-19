"""Opponent profiles: dataset model stats merged with a live play book.

Two sources, both best-effort:

(a) the dataset ``models`` dict inside targets.json — per model_name_short
    behavioral stats (e.g. barg_accept_rate_when_offered_lt40pct). A profile
    matches when a model name appears (lowercase substring) inside the live
    opponent name; the longest matching model name wins.

(b) a live book aggregated ONCE per process from data/agent.db (read-only):
    per opp_name, the mean share of the pot conceded to us across accepted
    bargaining offers, and the sample count.

Nothing in this module ever raises into a caller: a missing, locked, or
corrupt database simply yields an empty live book, and ``profile`` returns
None when neither source knows the name.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path

from .targets import get_targets

logger = logging.getLogger("glee_agent")

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "agent.db"

_LOCK = threading.Lock()
_LIVE_BOOK: dict[str, dict] | None = None


def _load_live_book(path: Path) -> dict[str, dict]:
    """Aggregate agent.db -> {opp_name_lower: {live_n, live_share_to_me}}.

    Never raises; any failure (absent file, lock, corruption) yields {}.
    """
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0.5)
        try:
            cur = con.execute(
                "SELECT your_player, opp_name, config_json, result_json FROM games "
                "WHERE family='bargaining' AND outcome='agreement' "
                "AND opp_name IS NOT NULL"
            )
            acc: dict[str, list[float]] = {}
            for your_player, opp_name, config_json, result_json in cur:
                try:
                    if not isinstance(opp_name, str) or not opp_name.strip():
                        continue
                    cfg = json.loads(config_json) if config_json else {}
                    money = float(cfg.get("money_to_divide") or 0.0)
                    if money <= 0:
                        continue
                    res = json.loads(result_json) if result_json else {}
                    idx = 2 if your_player == "player_2" else 1
                    gain = res.get(f"agreed_player_{idx}_gain")
                    if isinstance(gain, bool) or not isinstance(gain, (int, float)):
                        continue
                    share = min(max(float(gain) / money, 0.0), 1.0)
                    slot = acc.setdefault(opp_name.strip().lower(), [0.0, 0.0])
                    slot[0] += 1.0
                    slot[1] += share
                except Exception:  # noqa: BLE001 — skip any malformed row
                    continue
            return {
                name: {"live_n": int(n), "live_share_to_me": total / n}
                for name, (n, total) in acc.items()
                if n > 0
            }
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001 — absent/locked/corrupt db is normal
        logger.debug("opponent live book unavailable (%s): %s", path, e)
        return {}


def _live_book() -> dict[str, dict]:
    """Lazy module-level cache; computed once per process."""
    global _LIVE_BOOK
    if _LIVE_BOOK is None:
        with _LOCK:
            if _LIVE_BOOK is None:
                _LIVE_BOOK = _load_live_book(DB_PATH)
    return _LIVE_BOOK


def reset_cache() -> None:
    """Drop the cached live book (tests / manual reloads)."""
    global _LIVE_BOOK
    with _LOCK:
        _LIVE_BOOK = None


def profile(name: str | None) -> dict | None:
    """Merged behavioral profile for a live opponent name, or None.

    Keys (when present): ``model`` plus the dataset stats for the matched
    model (barg_n, barg_accept_rate_when_offered_lt40pct, ...), and
    ``live_n`` / ``live_share_to_me`` from the live book. Never raises.
    """
    try:
        if not isinstance(name, str) or not name.strip():
            return None
        lower = name.strip().lower()
        merged: dict = {}

        models = get_targets().models
        if isinstance(models, dict):
            best_key = None
            for key in models:
                k = str(key).lower()
                if k and k in lower and (best_key is None or len(k) > len(str(best_key).lower())):
                    best_key = key
            if best_key is not None and isinstance(models.get(best_key), dict):
                merged.update(models[best_key])
                merged["model"] = str(best_key)

        live = _live_book().get(lower)
        if isinstance(live, dict):
            merged.update(live)

        return merged or None
    except Exception:  # noqa: BLE001 — lookups must never raise
        return None
