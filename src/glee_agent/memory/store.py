"""SQLite store over the JSONL logs: schema, connection helper, and an
idempotent ``ingest()`` that tails logs/*.jsonl from per-file byte offsets.

Tables mirror the JSONL record contract (turns / results / snapshots / lb)
plus a per-game ``games`` rollup maintained during ingest. ``ingest()``
swallows all exceptions — it may be called from the live loop and must
never raise into play."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from ..logging_ import LOG_DIR

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parents[3] / "data" / "agent.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_meta (
    path   TEXT PRIMARY KEY,
    offset INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    ts REAL, agent TEXT, game_id TEXT, family TEXT, your_player TEXT,
    round INTEGER, phase TEXT, action_type TEXT, opp_type TEXT, opp_name TEXT,
    action_json TEXT, n_corrections INTEGER, corrections_json TEXT,
    elapsed_s REAL, error TEXT, state_json TEXT
);
CREATE TABLE IF NOT EXISTS results (
    ts REAL, agent TEXT, game_id TEXT, valid INTEGER, attempts_left INTEGER,
    game_over INTEGER, error TEXT, result_json TEXT
);
CREATE TABLE IF NOT EXISTS snapshots (
    ts REAL, agent TEXT, family TEXT, rating REAL, games_played INTEGER,
    active_games INTEGER
);
CREATE TABLE IF NOT EXISTS lb (
    ts REAL, family TEXT, rank INTEGER, player_id TEXT, player_name TEXT,
    player_type TEXT, rating REAL, games_played INTEGER, is_owner_best INTEGER,
    is_me INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT, agent TEXT, family TEXT, your_player TEXT,
    opp_type TEXT, opp_name TEXT, config_json TEXT, config_key TEXT,
    first_ts REAL, last_ts REAL, n_turns INTEGER, n_corrections INTEGER,
    n_errors INTEGER, n_invalid INTEGER, outcome TEXT, my_payoff REAL,
    opp_payoff REAL, agreed_round INTEGER, result_json TEXT,
    PRIMARY KEY (game_id, agent)
);
CREATE INDEX IF NOT EXISTS idx_games_family ON games(family);
CREATE INDEX IF NOT EXISTS idx_games_last_ts ON games(last_ts);
CREATE INDEX IF NOT EXISTS idx_turns_game_id ON turns(game_id);
CREATE INDEX IF NOT EXISTS idx_results_game_id ON results(game_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_agent_ts ON snapshots(agent, ts);
CREATE INDEX IF NOT EXISTS idx_lb_family_ts ON lb(family, ts);
"""

_BARGAINING_CONFIG = (
    "money_to_divide", "delta_1", "delta_2", "max_rounds",
    "horizon_known", "messages_allowed", "complete_information",
)
_NEGOTIATION_CONFIG = (
    "max_rounds", "horizon_known", "messages_allowed", "complete_information",
)
_PERSUASION_CONFIG = (
    "product_price", "p", "v", "u", "total_rounds", "seller_message_type",
)


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite store with WAL mode, Row rows,
    and the schema applied."""
    path = Path(db_path) if db_path is not None else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def ingest(log_dir: Path | None = None, db_path: Path | None = None) -> dict:
    """Scan logs/*.jsonl into the store, resuming each file from its stored
    byte offset (reset to 0 if the file shrank). Lines without "type" are
    legacy turn records; unparseable lines are counted as skipped. Never
    raises. Returns {"turns","results","snapshots","lb","skipped"} counts of
    records ingested this call."""
    counts = {"turns": 0, "results": 0, "snapshots": 0, "lb": 0, "skipped": 0}
    conn: sqlite3.Connection | None = None
    try:
        directory = Path(log_dir) if log_dir is not None else LOG_DIR
        conn = connect(db_path)
        # Serialize the WHOLE pass (offset reads included) as one write
        # transaction: without this, two concurrent ingests both read the same
        # offset and double-ingest the same lines, permanently inflating the
        # additive rollup counters. A concurrent caller blocks here (up to the
        # 30s sqlite timeout) and then re-reads the advanced offsets.
        conn.execute("BEGIN IMMEDIATE")
        acc: dict[tuple[str, str], dict] = {}
        for path in sorted(directory.glob("*.jsonl")):
            try:
                _ingest_file(conn, path, counts, acc)
            except Exception:  # noqa: BLE001 — one bad file must not stop ingest
                logger.exception("ingest: failed reading %s", path)
        _apply_rollups(conn, acc)
        conn.commit()
    except Exception:  # noqa: BLE001 — ingest must never raise into the game loop
        logger.exception("ingest failed")
        try:
            if conn is not None:
                conn.rollback()
        except Exception:  # noqa: BLE001
            pass
    finally:
        try:
            if conn is not None:
                conn.close()
        except Exception:  # noqa: BLE001
            pass
    return counts


# --- per-file / per-line ingestion -----------------------------------------


def _ingest_file(
    conn: sqlite3.Connection, path: Path, counts: dict, acc: dict[str, dict]
) -> None:
    key = str(path.resolve())
    row = conn.execute(
        "SELECT offset FROM ingest_meta WHERE path = ?", (key,)
    ).fetchone()
    offset = int(row["offset"]) if row is not None else 0
    size = path.stat().st_size
    if size < offset:  # file shrank (rotated/truncated) — start over
        offset = 0
    consumed = offset
    agent_default = path.name.split("-", 1)[0]
    try:
        with path.open("rb") as f:
            f.seek(offset)
            for raw in f:
                if not raw.endswith(b"\n"):
                    break  # partial trailing line (mid-write) — retry next time
                _ingest_line(conn, raw, agent_default, counts, acc)
                consumed += len(raw)
    finally:
        conn.execute(
            "INSERT INTO ingest_meta(path, offset) VALUES(?, ?) "
            "ON CONFLICT(path) DO UPDATE SET offset = excluded.offset",
            (key, consumed),
        )


def _ingest_line(
    conn: sqlite3.Connection,
    raw: bytes,
    agent_default: str,
    counts: dict,
    acc: dict[str, dict],
) -> None:
    try:
        rec = json.loads(raw.decode("utf-8", errors="replace"))
        if not isinstance(rec, dict):
            raise ValueError("record is not an object")
        rtype = rec.get("type", "turn")  # legacy lines are turns
    except Exception:  # noqa: BLE001 — unparseable line: skip forever
        counts["skipped"] += 1
        return
    try:
        if rtype == "turn":
            _handle_turn(conn, rec, agent_default, acc)
            counts["turns"] += 1
        elif rtype == "result":
            _handle_result(conn, rec, agent_default, acc)
            counts["results"] += 1
        elif rtype == "snapshot":
            _handle_snapshot(conn, rec, agent_default)
            counts["snapshots"] += 1
        elif rtype == "lb_snapshot":
            _handle_lb_snapshot(conn, rec)
            counts["lb"] += 1
        else:
            counts["skipped"] += 1
    except sqlite3.Error:
        # Transient DB failure (e.g. lock timeout): propagate BEFORE the
        # caller advances the offset, so this line is retried next pass
        # instead of being lost forever.
        raise
    except Exception:  # noqa: BLE001 — malformed-but-parseable record: skip
        counts["skipped"] += 1


def _handle_turn(
    conn: sqlite3.Connection, rec: dict, agent_default: str, acc: dict[str, dict]
) -> None:
    game = rec.get("game") if isinstance(rec.get("game"), dict) else {}
    state = game.get("game_state") if isinstance(game.get("game_state"), dict) else {}
    opp = game.get("opponent") if isinstance(game.get("opponent"), dict) else {}
    va = game.get("valid_actions") if isinstance(game.get("valid_actions"), dict) else {}
    corrections = rec.get("corrections") if isinstance(rec.get("corrections"), list) else []
    ts = rec.get("ts")
    agent = rec.get("agent") or agent_default
    game_id = game.get("game_id")
    family = game.get("game_family")
    your_player = game.get("your_player")
    error = rec.get("error")
    action = rec.get("action")
    conn.execute(
        "INSERT INTO turns (ts, agent, game_id, family, your_player, round,"
        " phase, action_type, opp_type, opp_name, action_json, n_corrections,"
        " corrections_json, elapsed_s, error, state_json)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            ts, agent, game_id, family, your_player,
            state.get("round"),
            game.get("phase") or state.get("phase"),
            va.get("type"),
            opp.get("type"), opp.get("name"),
            _dumps(action),
            len(corrections),
            _dumps(corrections),
            rec.get("elapsed_s"),
            error,
            _dumps(state) if state else None,
        ),
    )
    if not game_id:
        return
    a = acc.setdefault((game_id, agent), _new_acc())
    _acc_ts(a, ts)
    a["n_turns"] += 1
    a["n_corrections"] += len(corrections)
    if error not in (None, ""):
        a["n_errors"] += 1
    tsv = ts if isinstance(ts, (int, float)) else None
    # A turn only wins "latest" with a real, newer timestamp (a malformed ts
    # must not overwrite the seat/config of a properly-stamped turn).
    if a["latest_turn_ts"] is None or (tsv is not None and tsv >= a["latest_turn_ts"]):
        if tsv is not None:
            a["latest_turn_ts"] = tsv
        a["has_turn"] = True
        a["agent"] = agent
        a["family"] = family
        a["your_player"] = your_player
        a["config_json"], a["config_key"] = _extract_config(family, state)
    if not _opp_known(a["opp_type"]) and opp.get("type") is not None:
        if _opp_known(opp.get("type")) or a["opp_type"] is None:
            a["opp_type"] = opp.get("type")
            a["opp_name"] = opp.get("name")
    # Future-proofing: a turn record may carry the move's server response
    # inline; treat it exactly like a "result" record for the rollup.
    mr = rec.get("move_result")
    if isinstance(mr, dict):
        if mr.get("valid") in (False, 0):
            a["n_invalid"] += 1
        if mr.get("game_over") and isinstance(mr.get("result"), dict):
            a["final_result"] = mr["result"]


def _handle_result(
    conn: sqlite3.Connection, rec: dict, agent_default: str, acc: dict[str, dict]
) -> None:
    ts = rec.get("ts")
    agent = rec.get("agent") or agent_default
    game_id = rec.get("game_id")
    valid = rec.get("valid")
    game_over = rec.get("game_over")
    result = rec.get("result")
    conn.execute(
        "INSERT INTO results (ts, agent, game_id, valid, attempts_left,"
        " game_over, error, result_json) VALUES (?,?,?,?,?,?,?,?)",
        (
            ts, agent, game_id,
            None if valid is None else int(bool(valid)),
            rec.get("attempts_left"),
            None if game_over is None else int(bool(game_over)),
            rec.get("error"),
            _dumps(result) if result is not None else None,
        ),
    )
    if not game_id:
        return
    a = acc.setdefault((game_id, agent), _new_acc())
    _acc_ts(a, ts)
    if valid is False or valid == 0:
        a["n_invalid"] += 1
    if a["agent_fallback"] is None:
        a["agent_fallback"] = agent
    if game_over and isinstance(result, dict):
        a["final_result"] = result


def _handle_snapshot(conn: sqlite3.Connection, rec: dict, agent_default: str) -> None:
    ts = rec.get("ts")
    agent = rec.get("agent") or agent_default
    active = rec.get("active_games")
    scores = rec.get("scores") if isinstance(rec.get("scores"), dict) else {}
    for family, score in scores.items():
        s = score if isinstance(score, dict) else {}
        conn.execute(
            "INSERT INTO snapshots (ts, agent, family, rating, games_played,"
            " active_games) VALUES (?,?,?,?,?,?)",
            (ts, agent, family, s.get("rating"), s.get("games_played"), active),
        )


def _handle_lb_snapshot(conn: sqlite3.Connection, rec: dict) -> None:
    ts = rec.get("ts")
    family = rec.get("family")
    top = rec.get("top") if isinstance(rec.get("top"), list) else []
    me = rec.get("me") if isinstance(rec.get("me"), dict) else None
    me_id = me.get("player_id") if me else None

    def _insert(entry: dict, is_me: bool) -> None:
        conn.execute(
            "INSERT INTO lb (ts, family, rank, player_id, player_name,"
            " player_type, rating, games_played, is_owner_best, is_me)"
            " VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                ts, family,
                entry.get("rank"), entry.get("player_id"),
                entry.get("player_name"), entry.get("player_type"),
                entry.get("rating"), entry.get("games_played"),
                int(bool(entry.get("is_owner_best"))),
                int(is_me),
            ),
        )

    seen_me = False
    for entry in top:
        if not isinstance(entry, dict):
            continue
        is_me = me_id is not None and entry.get("player_id") == me_id
        seen_me = seen_me or is_me
        _insert(entry, is_me)
    # Our entry survives even when we sit below the stored top-50 — it is the
    # only persisted source of our own rank/rating in that case.
    if me is not None and not seen_me:
        _insert(me, True)


# --- games rollup -----------------------------------------------------------


def _new_acc() -> dict:
    return {
        "has_turn": False, "latest_turn_ts": None,
        "agent": None, "agent_fallback": None,
        "family": None, "your_player": None,
        "opp_type": None, "opp_name": None,
        "config_json": None, "config_key": None,
        "n_turns": 0, "n_corrections": 0, "n_errors": 0, "n_invalid": 0,
        "first_ts": None, "last_ts": None,
        "final_result": None,
    }


def _acc_ts(a: dict, ts: object) -> None:
    if not isinstance(ts, (int, float)):
        return
    a["first_ts"] = ts if a["first_ts"] is None else min(a["first_ts"], ts)
    a["last_ts"] = ts if a["last_ts"] is None else max(a["last_ts"], ts)


def _opp_known(opp_type: object) -> bool:
    return opp_type not in (None, "hidden")


def _dumps(obj: object) -> str | None:
    if obj is None:
        return None
    return json.dumps(obj, ensure_ascii=False, default=str)


def _extract_config(family: object, state: dict) -> tuple[str | None, str | None]:
    """Per-family game-config dict from a turn's game_state; returns
    (config_json, canonical config_key) or (None, None) for unknown families."""
    if family == "bargaining":
        cfg = {k: state.get(k) for k in _BARGAINING_CONFIG}
    elif family == "negotiation":
        cfg = {
            k: state.get(k)
            for k in ("player_1_value", "player_2_value")
            if state.get(k) is not None
        }
        cfg.update({k: state.get(k) for k in _NEGOTIATION_CONFIG})
    elif family == "persuasion":
        cfg = {k: state.get(k) for k in _PERSUASION_CONFIG}
    else:
        return None, None
    return (
        json.dumps(cfg, ensure_ascii=False, default=str),
        json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str),
    )


def _payoffs(your_player: object, result: dict) -> tuple[object, object]:
    """(my_payoff, opp_payoff) per the contract: i = 1 if your_player ==
    "player_1" else 2; unknown player -> (None, None)."""
    if your_player is None:
        return None, None
    if your_player == "player_1":
        return result.get("player_1_payoff"), result.get("player_2_payoff")
    return result.get("player_2_payoff"), result.get("player_1_payoff")


def _apply_rollups(conn: sqlite3.Connection, acc: dict[tuple[str, str], dict]) -> None:
    for (game_id, acc_agent), a in acc.items():
        old = conn.execute(
            "SELECT * FROM games WHERE game_id = ? AND agent = ?",
            (game_id, acc_agent),
        ).fetchone()

        def _old(col: str, default: object = None) -> object:
            return old[col] if old is not None and old[col] is not None else default

        agent = acc_agent  # the accumulator key IS the row identity
        if a["has_turn"]:  # this run's turns are newer than any previously stored
            family, your_player = a["family"], a["your_player"]
            config_json, config_key = a["config_json"], a["config_key"]
        else:
            family, your_player = _old("family"), _old("your_player")
            config_json, config_key = _old("config_json"), _old("config_key")

        if _opp_known(_old("opp_type")):
            opp_type, opp_name = old["opp_type"], old["opp_name"]
        elif _opp_known(a["opp_type"]):
            opp_type, opp_name = a["opp_type"], a["opp_name"]
        elif _old("opp_type") is not None:
            opp_type, opp_name = old["opp_type"], old["opp_name"]
        else:
            opp_type, opp_name = a["opp_type"], a["opp_name"]

        firsts = [t for t in (_old("first_ts"), a["first_ts"]) if t is not None]
        lasts = [t for t in (_old("last_ts"), a["last_ts"]) if t is not None]
        first_ts = min(firsts) if firsts else None
        last_ts = max(lasts) if lasts else None

        if a["final_result"] is not None:
            res = a["final_result"]
            outcome = res.get("outcome")
            agreed_round = res.get("agreed_round")
            result_json = _dumps(res)
            my_payoff, opp_payoff = _payoffs(your_player, res)
        else:
            outcome, agreed_round = _old("outcome"), _old("agreed_round")
            result_json = _old("result_json")
            my_payoff, opp_payoff = _old("my_payoff"), _old("opp_payoff")
            if my_payoff is None and result_json and your_player is not None:
                try:  # your_player learned after the result: backfill payoffs
                    my_payoff, opp_payoff = _payoffs(
                        your_player, json.loads(result_json)
                    )
                except Exception:  # noqa: BLE001
                    pass

        conn.execute(
            "INSERT OR REPLACE INTO games (game_id, agent, family, your_player,"
            " opp_type, opp_name, config_json, config_key, first_ts, last_ts,"
            " n_turns, n_corrections, n_errors, n_invalid, outcome, my_payoff,"
            " opp_payoff, agreed_round, result_json)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                game_id, agent, family, your_player, opp_type, opp_name,
                config_json, config_key, first_ts, last_ts,
                _old("n_turns", 0) + a["n_turns"],
                _old("n_corrections", 0) + a["n_corrections"],
                _old("n_errors", 0) + a["n_errors"],
                _old("n_invalid", 0) + a["n_invalid"],
                outcome, my_payoff, opp_payoff, agreed_round, result_json,
            ),
        )
