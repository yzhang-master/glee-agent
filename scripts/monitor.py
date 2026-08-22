"""Terminal status monitor for the GLEE agent.

Usage:
    python scripts/monitor.py [--agent main] [--watch SECONDS] [--no-api]

Read-only: ingests fresh log lines via glee_agent.memory.store, optionally
GETs live stats via GleeClient.stats(), and prints a compact report. It never
queues games, never calls move(), and never writes to the platform.

Expected sqlite schema (the glee_agent.memory.store contract; column lookups
are defensive, so near-miss names degrade to a warning, not a traceback):
    turns(ts, agent, game_id, family, n_corrections|corrections, elapsed_s, error)
    results(ts, agent, game_id, valid, attempts_left, game_over, error, result)
    games(game_id, agent, family, opponent_name|opp_name, opponent_type|opp_type,
          first_ts, last_ts, n_turns, n_invalid, n_corrections, outcome,
          my_payoff, agreed_round)
    snapshots(ts, agent, family, rating, games_played, active_games)
    lb(ts, family, rank, player_id, player_name, player_type, rating,
       games_played, is_owner_best, is_baseline, is_benchmark)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

FAMILIES = ("bargaining", "negotiation", "persuasion")
DEFAULT_RATING = 1000.0
DECAY_RATING_THRESHOLD = 1800.0
DECAY_GAMES_PER_48H = 100
TOP100_GAMES_PER_DAY = 10

_STORE_IMPORT_ERROR: str | None = None
try:
    from glee_agent.memory import store  # noqa: E402
except Exception as exc:  # noqa: BLE001 — module may not exist yet
    store = None  # type: ignore[assignment]
    _STORE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# sqlite helpers (defensive: missing tables/columns degrade, never raise)
# ---------------------------------------------------------------------------

def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _pick(cols: set[str], candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in cols:
            return name
    return None


def _where(clauses: list[str]) -> str:
    return (" WHERE " + " AND ".join(clauses)) if clauses else ""


def _agent_clause(cols: set[str], agent: str | None) -> tuple[list[str], list[object]]:
    if agent and "agent" in cols:
        return (["agent = ?"], [agent])
    return ([], [])


def _scalar(conn: sqlite3.Connection, sql: str, params: list[object]) -> object:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


# ---------------------------------------------------------------------------
# data extraction
# ---------------------------------------------------------------------------

def _live_scores(stats: dict | None) -> dict[str, tuple[float, int]] | None:
    """{family: (rating, games_played)} from a GleeClient.stats() dict."""
    if not stats or not isinstance(stats.get("scores"), dict):
        return None
    out: dict[str, tuple[float, int]] = {}
    for fam, sc in stats["scores"].items():
        try:
            out[fam] = (float(sc.get("rating", DEFAULT_RATING)), int(sc.get("games_played", 0)))
        except (TypeError, ValueError, AttributeError):
            continue
    return out or None


def _snapshot_scores(
    conn: sqlite3.Connection | None, agent: str | None
) -> dict[str, tuple[float, int]] | None:
    """Latest per-family (rating, games_played) from the snapshots table."""
    if conn is None:
        return None
    cols = _table_columns(conn, "snapshots")
    if not {"ts", "family", "rating"} <= cols:
        return None
    clauses, params = _agent_clause(cols, agent)
    gp = "games_played" if "games_played" in cols else "NULL"
    rows = conn.execute(
        f"SELECT family, rating, {gp} FROM snapshots{_where(clauses)} ORDER BY ts",
        params,
    ).fetchall()
    out: dict[str, tuple[float, int]] = {}
    for fam, rating, games in rows:  # later ts overwrites earlier
        if rating is None:  # malformed snapshot row must not kill the report
            continue
        out[str(fam)] = (float(rating), int(games or 0))
    return out or None


def _completed_counts(
    conn: sqlite3.Connection, agent: str | None, since: float | None
) -> dict[str, int]:
    """{family: completed-game count} since `since` (None = all time)."""
    cols = _table_columns(conn, "games")
    ts_col = _pick(cols, ("last_ts", "ended_ts", "end_ts", "ts"))
    if "outcome" not in cols or "family" not in cols:
        raise ValueError("games table lacks outcome/family columns")
    clauses, params = _agent_clause(cols, agent)
    clauses.append("outcome IS NOT NULL")
    if since is not None:
        if ts_col is None:
            raise ValueError("games table has no timestamp column")
        clauses.append(f"{ts_col} >= ?")
        params.append(since)
    rows = conn.execute(
        f"SELECT family, COUNT(*) FROM games{_where(clauses)} GROUP BY family", params
    ).fetchall()
    return {str(fam): int(n) for fam, n in rows}


def _invalid_counts(
    conn: sqlite3.Connection, agent: str | None, cutoff_24h: float
) -> tuple[int, int]:
    """(total invalid moves, invalid moves in last 24h)."""
    rcols = _table_columns(conn, "results")
    if {"valid", "ts"} <= rcols:
        clauses, params = _agent_clause(rcols, agent)
        clauses.append("valid = 0")
        total = int(_scalar(conn, f"SELECT COUNT(*) FROM results{_where(clauses)}", params) or 0)
        clauses.append("ts >= ?")
        params.append(cutoff_24h)
        last24 = int(_scalar(conn, f"SELECT COUNT(*) FROM results{_where(clauses)}", params) or 0)
        return total, last24
    gcols = _table_columns(conn, "games")
    if "n_invalid" in gcols:
        clauses, params = _agent_clause(gcols, agent)
        total = int(
            _scalar(conn, f"SELECT COALESCE(SUM(n_invalid), 0) FROM games{_where(clauses)}", params)
            or 0
        )
        last24 = 0
        ts_col = _pick(gcols, ("last_ts", "ended_ts", "end_ts", "ts"))
        if ts_col is not None:
            clauses.append(f"{ts_col} >= ?")
            params.append(cutoff_24h)
            last24 = int(
                _scalar(
                    conn, f"SELECT COALESCE(SUM(n_invalid), 0) FROM games{_where(clauses)}", params
                )
                or 0
            )
        return total, last24
    raise ValueError("neither results.valid nor games.n_invalid available")


def _corrections_last24(conn: sqlite3.Connection, agent: str | None, cutoff: float) -> int:
    cols = _table_columns(conn, "turns")
    col = _pick(cols, ("n_corrections", "corrections"))
    if col is None or "ts" not in cols:
        raise ValueError("turns table lacks corrections/ts columns")
    clauses, params = _agent_clause(cols, agent)
    clauses.append("ts >= ?")
    params.append(cutoff)
    total = 0
    for (value,) in conn.execute(f"SELECT {col} FROM turns{_where(clauses)}", params):
        if value is None:
            continue
        if isinstance(value, (int, float)):
            total += int(value)
        else:
            try:
                parsed = json.loads(value)
                total += len(parsed) if isinstance(parsed, list) else int(parsed)
            except (ValueError, TypeError):
                pass
    return total


def _latest_lb(
    conn: sqlite3.Connection, family: str
) -> list[tuple[int, str, str, float]] | None:
    """[(rank, player_id, player_name, rating)] at the latest ts for family."""
    cols = _table_columns(conn, "lb")
    if not {"ts", "family", "rank", "rating"} <= cols:
        return None
    latest = _scalar(conn, "SELECT MAX(ts) FROM lb WHERE family = ?", [family])
    if latest is None:
        return None
    pid = "player_id" if "player_id" in cols else "NULL"
    pname = "player_name" if "player_name" in cols else "NULL"
    is_me = "is_me" if "is_me" in cols else "0"
    rows = conn.execute(
        f"SELECT rank, {pid}, {pname}, rating, {is_me} FROM lb"
        f" WHERE family = ? AND ts = ?"
        f" ORDER BY CASE WHEN rank IS NULL THEN 1 ELSE 0 END, rank",
        [family, latest],
    ).fetchall()
    # Null-safe: unranked entries (benchmarks, non-best agents, our own
    # below-top row) have rank NULL; keep them rather than crashing.
    return [
        (
            int(r) if r is not None else None,
            str(i) if i is not None else "",
            str(n) if n is not None else "",
            float(g) if g is not None else None,
            bool(m),
        )
        for r, i, n, g, m in rows
    ]


# ---------------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------------

def _fmt_num(value: object, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "-"


def _ratings_section(
    scores: dict[str, tuple[float, int]] | None, source: str
) -> list[str]:
    lines = [f"Ratings ({source})"]
    scores = scores or {}
    for fam in FAMILIES:
        if fam in scores:
            rating, games = scores[fam]
            lines.append(f"  {fam:<12} {rating:8.1f}   games {games}")
        else:
            lines.append(f"  {fam:<12} {DEFAULT_RATING:8.1f}   (no data)")
    overall = sum(scores.get(fam, (DEFAULT_RATING, 0))[0] for fam in FAMILIES) / len(FAMILIES)
    lines.append(f"  {'overall':<12} {overall:8.1f}")
    return lines


def _volume_section(conn: sqlite3.Connection, agent: str | None, now: float) -> list[str]:
    c24 = _completed_counts(conn, agent, now - 24 * 3600.0)
    c48 = _completed_counts(conn, agent, now - 48 * 3600.0)
    c6 = _completed_counts(conn, agent, now - 6 * 3600.0)
    lines = ["Volume (completed games)"]
    for fam in FAMILIES:
        lines.append(f"  {fam:<12} 24h {c24.get(fam, 0):<4} 48h {c48.get(fam, 0)}")
    lines.append(f"  games/hour (last 6h): {sum(c6.values()) / 6.0:.2f}")
    return lines



AUTH_MARKERS = ("Invalid API key", "[401]", "[403]")
RUN_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"


def _auth_alarm(agent: str | None) -> list[str]:
    """Surface a dead API key immediately.

    An agent whose key stops authenticating crash-loops silently: the tmux
    wrapper retries, the log fills with 401s, and nothing else notices until
    someone runs a manual check. That has cost us two outages, so the health
    section reads the tail of each run log directly rather than the DB (a
    non-authenticating agent writes no turns at all, so the DB cannot see it).
    """
    labels = [agent] if agent else ["main", "test_a", "test_b", "test_c", "test_d"]
    out = []
    for label in labels:
        path = RUN_LOG_DIR / f"run-{label}.log"
        try:
            with open(path, "rb") as fh:
                fh.seek(0, 2)
                fh.seek(max(0, fh.tell() - 20000))
                tail = fh.read().decode("utf-8", "replace")
        except OSError:
            continue
        recent = tail.splitlines()[-60:]
        if any(any(m in line for m in AUTH_MARKERS) for line in recent):
            out.append(f"  !! AUTH FAILURE   {label}: key rejected -- agent is NOT playing")
    return out or ["  auth               all keys authenticating"]


def _health_section(conn: sqlite3.Connection, agent: str | None, now: float) -> list[str]:
    cutoff = now - 24 * 3600.0
    lines = ["Health"]
    lines.extend(_auth_alarm(agent))
    try:
        total, last24 = _invalid_counts(conn, agent, cutoff)
        lines.append(f"  invalid moves      total {total}, last 24h {last24}")
    except (sqlite3.Error, ValueError) as exc:
        lines.append(f"  invalid moves      unavailable ({exc})")
    tcols = _table_columns(conn, "turns")
    clauses, params = _agent_clause(tcols, agent)
    clauses.append("ts >= ?")
    params.append(cutoff)
    if {"ts", "error"} <= tcols:
        err_clauses = clauses + ["error IS NOT NULL", "error != ''"]
        n_err = _scalar(conn, f"SELECT COUNT(*) FROM turns{_where(err_clauses)}", params)
        lines.append(f"  turn errors 24h    {int(n_err or 0)}")
    else:
        lines.append("  turn errors 24h    unavailable")
    try:
        lines.append(f"  corrections 24h    {_corrections_last24(conn, agent, cutoff)}")
    except (sqlite3.Error, ValueError) as exc:
        lines.append(f"  corrections 24h    unavailable ({exc})")
    if {"ts", "elapsed_s"} <= tcols:
        max_el = _scalar(conn, f"SELECT MAX(elapsed_s) FROM turns{_where(clauses)}", params)
        lines.append(f"  max elapsed_s 24h  {_fmt_num(max_el, 2) if max_el is not None else '-'}")
    else:
        lines.append("  max elapsed_s 24h  unavailable")
    return lines


def _decay_section(
    conn: sqlite3.Connection | None,
    agent: str | None,
    scores: dict[str, tuple[float, int]] | None,
    now: float,
) -> list[str]:
    lines = ["Decay quota"]
    c48: dict[str, int] = {}
    today: dict[str, int] = {}
    if conn is not None:
        try:
            c48 = _completed_counts(conn, agent, now - 48 * 3600.0)
            midnight = (
                datetime.fromtimestamp(now, tz=timezone.utc)
                .replace(hour=0, minute=0, second=0, microsecond=0)
                .timestamp()
            )
            today = _completed_counts(conn, agent, midnight)
        except (sqlite3.Error, ValueError) as exc:
            lines.append(f"  game counts unavailable ({exc})")
    hot = [fam for fam in FAMILIES if (scores or {}).get(fam, (0.0, 0))[0] > DECAY_RATING_THRESHOLD]
    if hot:
        for fam in hot:
            played = c48.get(fam, 0)
            if played >= DECAY_GAMES_PER_48H:
                status = "OK"
            else:
                status = f"BEHIND by {DECAY_GAMES_PER_48H - played}"
            rating = (scores or {})[fam][0]
            lines.append(
                f"  {fam} ({rating:.1f} > {DECAY_RATING_THRESHOLD:.0f}): "
                f"{played}/{DECAY_GAMES_PER_48H} in 48h  {status}"
            )
    else:
        lines.append(f"  no family above {DECAY_RATING_THRESHOLD:.0f} — no 48h quota")
    parts = " | ".join(f"{fam} {today.get(fam, 0)}" for fam in FAMILIES)
    lines.append(f"  games today ({TOP100_GAMES_PER_DAY}/day rule): {parts}")
    return lines


def _leaderboard_section(
    conn: sqlite3.Connection, agent_id: str | None
) -> list[str]:
    lines = ["Leaderboard (latest lb snapshot)"]
    for fam in FAMILIES:
        rows = _latest_lb(conn, fam)
        if not rows:
            lines.append(f"  {fam:<12} no data")
            continue
        by_rank = {rank: rating for rank, _pid, _name, rating, _m in rows if rank is not None}
        # Match by live agent_id when we have it, else the persisted is_me flag
        # (works with --no-api and when we sit below the stored top-50).
        me_row = next(
            (r for r in rows if (agent_id and r[1] == agent_id) or r[4]), None
        )
        if me_row is not None:
            rank_txt = f"#{me_row[0]}" if me_row[0] is not None else "unranked"
            me_txt = f"me {rank_txt} ({_fmt_num(me_row[3])})"
        else:
            me_txt = "me not in top"
        r1 = _fmt_num(by_rank.get(1))
        r5 = _fmt_num(by_rank.get(5))
        lines.append(f"  {fam:<12} {me_txt} | #1 {r1} | #5 {r5}")
    return lines


def _recent_section(conn: sqlite3.Connection, agent: str | None) -> list[str]:
    cols = _table_columns(conn, "games")
    if "outcome" not in cols:
        raise ValueError("games table lacks outcome column")
    ts_col = _pick(cols, ("last_ts", "ended_ts", "end_ts", "ts"))
    order = f" ORDER BY {ts_col} DESC" if ts_col else ""
    candidates = {
        "family": ("family",),
        "opponent_name": ("opponent_name", "opp_name"),
        "opponent_type": ("opponent_type", "opp_type"),
        "my_payoff": ("my_payoff", "payoff"),
        "agreed_round": ("agreed_round",),
    }
    sel = {name: (_pick(cols, cands) or "NULL") for name, cands in candidates.items()}
    clauses, params = _agent_clause(cols, agent)
    clauses.append("outcome IS NOT NULL")
    rows = conn.execute(
        f"SELECT {sel['family']}, {sel['opponent_name']}, {sel['opponent_type']}, outcome, "
        f"{sel['my_payoff']}, {sel['agreed_round']} FROM games{_where(clauses)}{order} LIMIT 8",
        params,
    ).fetchall()
    lines = ["Recent games (last 8)"]
    if not rows:
        lines.append("  none")
    for fam, opp_name, opp_type, outcome, payoff, agreed_round in rows:
        opp = f"{opp_name or '?'} ({opp_type or '?'})"
        rnd = f"round {agreed_round}" if agreed_round is not None else "round -"
        lines.append(
            f"  {str(fam):<12} vs {opp:<24} {str(outcome):<12} "
            f"payoff {_fmt_num(payoff)}  {rnd}"
        )
    return lines


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def build_report(
    conn: sqlite3.Connection | None,
    stats: dict | None,
    now: float,
    agent: str | None = None,
) -> str:
    """Build the full monitor report as a string. Pure: no network, no ingest.

    `conn` is a sqlite connection to the memory store (or None if missing),
    `stats` a GleeClient.stats() dict (or None if --no-api / API failure).
    """
    when = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [f"GLEE monitor — agent {agent or '?'} — {when} UTC"]
    warnings: list[str] = []

    scores = _live_scores(stats)
    source = "live"
    if scores is None:
        scores = _snapshot_scores(conn, agent)
        source = "snapshots" if scores is not None else "no data"

    def section(name: str, fn) -> None:
        try:
            lines.append("")
            lines.extend(fn())
        except Exception as exc:  # noqa: BLE001 — never let a section traceback
            lines.append(f"{name}")
            warnings.append(f"warning: {name} section unavailable ({exc})")

    section("Ratings", lambda: _ratings_section(scores, source))
    if conn is None:
        warnings.append("warning: no local db — volume/health/leaderboard/recent skipped")
        section("Decay quota", lambda: _decay_section(None, agent, scores, now))
    else:
        section("Volume", lambda: _volume_section(conn, agent, now))
        section("Health", lambda: _health_section(conn, agent, now))
        section("Decay quota", lambda: _decay_section(conn, agent, scores, now))
        agent_id = stats.get("agent_id") if isinstance(stats, dict) else None
        section("Leaderboard", lambda: _leaderboard_section(conn, agent_id))
        section("Recent games", lambda: _recent_section(conn, agent))

    if warnings:
        lines.append("")
        lines.extend(warnings)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_once(agent: str, no_api: bool) -> str:
    """One monitor pass: ingest, fetch stats, build the report. Never raises."""
    notes: list[str] = []
    conn: sqlite3.Connection | None = None
    stats: dict | None = None

    if store is None:
        notes.append(
            "warning: glee_agent.memory.store not importable "
            f"({_STORE_IMPORT_ERROR}) — local db sections skipped"
        )
    else:
        try:
            counts = store.ingest()
            notes.append(f"ingest: {counts}")
        except Exception as exc:  # noqa: BLE001
            notes.append(f"warning: store.ingest() failed ({exc})")
        try:
            conn = store.connect()
        except Exception as exc:  # noqa: BLE001
            notes.append(f"warning: store.connect() failed ({exc})")

    if not no_api:
        try:
            from glee_sdk import GleeClient

            from glee_agent.config import load_settings

            settings = load_settings(agent)
            stats = GleeClient(api_key=settings.glee_api_key).stats()
        except Exception as exc:  # noqa: BLE001
            notes.append(f"warning: live stats unavailable ({exc})")

    try:
        report = build_report(conn, stats, time.time(), agent=agent)
    except Exception as exc:  # noqa: BLE001
        report = f"warning: report build failed ({exc})"
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    return "\n".join([report, *notes]) if notes else report


def main() -> None:
    parser = argparse.ArgumentParser(description="GLEE agent status monitor (read-only)")
    parser.add_argument(
        "--agent", default="main",
        choices=["main", "test_a", "test_b", "test_c", "test_d"],
    )
    parser.add_argument("--watch", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--no-api", action="store_true", help="skip the live stats() call")
    args = parser.parse_args()

    # The stats() call spends the LIVE agent's 60 req/min budget (same API
    # key), so in watch mode it fires at most once per minute no matter how
    # tight the loop; between calls the report runs from the local db.
    api_min_interval = 60.0
    last_api: float | None = None
    try:
        while True:
            skip_api = args.no_api or (
                last_api is not None
                and time.monotonic() - last_api < api_min_interval
            )
            if not skip_api:
                last_api = time.monotonic()
            output = _run_once(args.agent, skip_api)
            if args.watch:
                print("\x1b[2J\x1b[H", end="")
            print(output)
            if not args.watch:
                break
            time.sleep(max(args.watch, 1.0))
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
