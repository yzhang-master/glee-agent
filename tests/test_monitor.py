"""Tests for scripts/monitor.py — build_report against an in-memory sqlite db.

No network: build_report is pure given a connection and a stats dict.
"""

import importlib.util
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location("monitor", ROOT / "scripts" / "monitor.py")
monitor = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(monitor)

NOW = 1_787_000_000.0  # fixed "now" so window math is deterministic

SCHEMA = """
CREATE TABLE turns (
    ts REAL, agent TEXT, game_id TEXT, family TEXT,
    n_corrections INTEGER, elapsed_s REAL, error TEXT
);
CREATE TABLE results (
    ts REAL, agent TEXT, game_id TEXT, valid INTEGER, attempts_left INTEGER,
    game_over INTEGER, error TEXT, result TEXT
);
CREATE TABLE games (
    game_id TEXT PRIMARY KEY, agent TEXT, family TEXT,
    opponent_name TEXT, opponent_type TEXT, first_ts REAL, last_ts REAL,
    n_turns INTEGER, n_invalid INTEGER, n_corrections INTEGER,
    outcome TEXT, my_payoff REAL, agreed_round INTEGER
);
CREATE TABLE snapshots (
    ts REAL, agent TEXT, family TEXT, rating REAL,
    games_played INTEGER, active_games INTEGER
);
CREATE TABLE lb (
    ts REAL, family TEXT, rank INTEGER, player_id TEXT, player_name TEXT,
    player_type TEXT, rating REAL, games_played INTEGER,
    is_owner_best INTEGER, is_baseline INTEGER, is_benchmark INTEGER
);
"""


def make_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    return conn


def add_game(conn, game_id, family, ts, outcome="agreement", payoff=600.0, rnd=3,
             opp_name="Alice", opp_type="human", n_invalid=0, agent="main"):
    conn.execute(
        "INSERT INTO games (game_id, agent, family, opponent_name, opponent_type,"
        " first_ts, last_ts, n_turns, n_invalid, n_corrections, outcome, my_payoff,"
        " agreed_round) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (game_id, agent, family, opp_name, opp_type, ts - 60, ts, 5, n_invalid, 0,
         outcome, payoff, rnd),
    )


class TestOverallAverage:
    def test_missing_families_count_as_1000(self):
        stats = {"agent_id": "me", "scores": {"bargaining": {"rating": 1600.0, "games_played": 10}}}
        report = monitor.build_report(make_db(), stats, NOW)
        # (1600 + 1000 + 1000) / 3 = 1200.0
        assert "1200.0" in report
        assert "1600.0" in report

    def test_all_families_present(self):
        stats = {
            "agent_id": "me",
            "scores": {
                "bargaining": {"rating": 1500.0, "games_played": 1},
                "negotiation": {"rating": 1290.0, "games_played": 2},
                "persuasion": {"rating": 900.0, "games_played": 3},
            },
        }
        report = monitor.build_report(make_db(), stats, NOW)
        # (1500 + 1290 + 900) / 3 = 1230.0
        assert "1230.0" in report

    def test_no_stats_no_snapshots_defaults_to_1000(self):
        report = monitor.build_report(make_db(), None, NOW)
        assert "1000.0" in report

    def test_snapshot_fallback_used_when_no_stats(self):
        conn = make_db()
        conn.execute(
            "INSERT INTO snapshots (ts, agent, family, rating, games_played, active_games)"
            " VALUES (?,?,?,?,?,?)",
            (NOW - 100, "main", "negotiation", 1450.0, 7, 2),
        )
        report = monitor.build_report(conn, None, NOW)
        assert "snapshots" in report
        assert "1450.0" in report
        # (1450 + 1000 + 1000) / 3 = 1150.0
        assert "1150.0" in report


class TestDecayQuota:
    def test_behind_when_hot_family_lacks_games(self):
        conn = make_db()
        add_game(conn, "g1", "bargaining", NOW - 3600)
        add_game(conn, "g2", "bargaining", NOW - 7200)
        stats = {"agent_id": "me", "scores": {"bargaining": {"rating": 1850.0, "games_played": 200}}}
        report = monitor.build_report(conn, stats, NOW)
        assert "BEHIND by 98" in report
        assert "2/100" in report

    def test_ok_when_quota_met(self):
        conn = make_db()
        for i in range(100):
            add_game(conn, f"g{i}", "bargaining", NOW - 1000 - i)
        stats = {"agent_id": "me", "scores": {"bargaining": {"rating": 1850.0, "games_played": 500}}}
        report = monitor.build_report(conn, stats, NOW)
        assert "100/100" in report
        assert "BEHIND" not in report

    def test_no_quota_below_threshold(self):
        conn = make_db()
        stats = {"agent_id": "me", "scores": {"bargaining": {"rating": 1799.0, "games_played": 50}}}
        report = monitor.build_report(conn, stats, NOW)
        assert "BEHIND" not in report
        assert "no 48h quota" in report

    def test_games_today_line_always_present(self):
        report = monitor.build_report(make_db(), None, NOW)
        assert "games today" in report


class TestGracefulDegradation:
    def test_empty_db_builds(self):
        report = monitor.build_report(make_db(), None, NOW)
        assert isinstance(report, str)
        assert "Ratings" in report
        assert "Volume" in report
        assert "Traceback" not in report

    def test_no_db_builds(self):
        report = monitor.build_report(None, None, NOW)
        assert isinstance(report, str)
        assert "warning" in report

    def test_db_without_tables_builds(self):
        conn = sqlite3.connect(":memory:")
        report = monitor.build_report(conn, None, NOW)
        assert isinstance(report, str)
        assert "Traceback" not in report

    def test_current_time_works(self):
        report = monitor.build_report(make_db(), None, time.time())
        assert isinstance(report, str)


class TestSections:
    def test_volume_counts_windows(self):
        conn = make_db()
        add_game(conn, "g1", "bargaining", NOW - 3600)          # in 24h
        add_game(conn, "g2", "bargaining", NOW - 30 * 3600)     # in 48h only
        add_game(conn, "g3", "negotiation", NOW - 100 * 3600)   # outside both
        report = monitor.build_report(conn, None, NOW)
        assert "bargaining   24h 1    48h 2" in report

    def test_health_counts(self):
        conn = make_db()
        conn.execute(
            "INSERT INTO results (ts, agent, game_id, valid, attempts_left, game_over,"
            " error, result) VALUES (?,?,?,?,?,?,?,?)",
            (NOW - 100, "main", "g1", 0, 2, 0, "bad json", None),
        )
        conn.execute(
            "INSERT INTO turns (ts, agent, game_id, family, n_corrections, elapsed_s, error)"
            " VALUES (?,?,?,?,?,?,?)",
            (NOW - 200, "main", "g1", "bargaining", 3, 12.5, "timeout"),
        )
        report = monitor.build_report(conn, None, NOW)
        assert "invalid moves      total 1, last 24h 1" in report
        assert "turn errors 24h    1" in report
        assert "corrections 24h    3" in report
        assert "max elapsed_s 24h  12.50" in report

    def test_leaderboard_rank_match(self):
        conn = make_db()
        for rank, pid, rating in [(1, "p1", 1900.0), (2, "me", 1880.0), (5, "p5", 1700.0)]:
            conn.execute(
                "INSERT INTO lb (ts, family, rank, player_id, player_name, player_type,"
                " rating, games_played, is_owner_best, is_baseline, is_benchmark)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (NOW - 50, "bargaining", rank, pid, f"name-{pid}", "agent", rating, 10, 0, 0, 0),
            )
        stats = {"agent_id": "me", "scores": {}}
        report = monitor.build_report(conn, stats, NOW)
        assert "me #2" in report
        assert "#1 1900.0" in report
        assert "#5 1700.0" in report

    def test_recent_games_listed(self):
        conn = make_db()
        add_game(conn, "g1", "persuasion", NOW - 500, outcome="no_sale", payoff=0.0, rnd=None,
                 opp_name="BuyerBot", opp_type="agent")
        report = monitor.build_report(conn, None, NOW)
        assert "BuyerBot (agent)" in report
        assert "no_sale" in report

    def test_recent_games_store_column_variant(self):
        # The real memory store names the opponent columns opp_name/opp_type.
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE games (game_id TEXT PRIMARY KEY, agent TEXT, family TEXT,"
            " opp_name TEXT, opp_type TEXT, last_ts REAL, n_invalid INTEGER,"
            " outcome TEXT, my_payoff REAL, agreed_round INTEGER)"
        )
        conn.execute(
            "INSERT INTO games VALUES ('g1', 'main', 'bargaining', 'Zed', 'hidden',"
            " ?, 0, 'agreement', 55.0, 4)",
            (NOW - 10,),
        )
        report = monitor.build_report(conn, None, NOW)
        assert "Zed (hidden)" in report
