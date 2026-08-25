"""Opponent profile book: dataset model matching + live agent.db aggregation."""

from __future__ import annotations

import json
import sqlite3
from types import SimpleNamespace

import pytest

from glee_agent.theory import opponents as O
from glee_agent.theory import targets as T

MODELS = {
    "gpt-4o": {"barg_n": 100, "barg_accept_rate_when_offered_lt40pct": 0.25},
    "gpt-4o-mini": {"barg_n": 50, "barg_accept_rate_when_offered_lt40pct": 0.03},
    "gemini-1.5-flash": {"barg_n": 10},
}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """Point the live book at a (by default absent) db and use fixed models."""
    T.set_targets(T.Targets({"models": MODELS}))
    monkeypatch.setattr(O, "DB_PATH", tmp_path / "absent.db")
    O.reset_cache()
    yield tmp_path
    O.reset_cache()
    T.set_targets(None)


def _make_db(path, rows):
    con = sqlite3.connect(str(path))
    con.execute(
        "CREATE TABLE games (game_id TEXT, family TEXT, your_player TEXT,"
        " opp_name TEXT, config_json TEXT, outcome TEXT, result_json TEXT)"
    )
    con.executemany("INSERT INTO games VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()


def _row(gid, opp, player="player_1", money=100.0, gain=60.0,
         outcome="agreement", family="bargaining"):
    cfg = json.dumps({"money_to_divide": money})
    res = json.dumps({f"agreed_{player}_gain": gain})
    return (gid, family, player, opp, cfg, outcome, res)


class TestDatasetMatch:
    def test_exact_name_match(self):
        prof = O.profile("gpt-4o")
        assert prof is not None
        assert prof["model"] == "gpt-4o"
        assert prof["barg_n"] == 100

    def test_match_inside_longer_name(self):
        prof = O.profile("openai/GPT-4o (organizer bench)")
        assert prof is not None and prof["model"] == "gpt-4o"

    def test_longest_model_name_wins(self):
        prof = O.profile("run-gpt-4o-mini-2024")
        assert prof is not None and prof["model"] == "gpt-4o-mini"

    def test_unknown_name_is_none(self):
        assert O.profile("Rubinstein") is None

    def test_missing_or_empty_name_is_none(self):
        assert O.profile(None) is None
        assert O.profile("") is None
        assert O.profile("   ") is None

    def test_no_models_dict_is_none(self):
        T.set_targets(T.Targets.null())
        assert O.profile("gpt-4o") is None

    def test_dataset_only_lookup_skips_live_book(self, monkeypatch):
        loads = []
        monkeypatch.setattr(O, "_load_live_book", lambda path: loads.append(path) or {})

        prof = O.profile("gpt-4o", include_live=False)

        assert prof is not None
        assert prof["barg_n"] == 100
        assert loads == []

    def test_bargaining_hot_path_skips_live_book_and_uses_dataset(self, monkeypatch):
        from glee_agent.families import bargaining

        T.set_targets(T.Targets({"models": {
            "gpt-4o": {
                "barg_n": 5000,
                "barg_accept_rate_when_offered_lt40pct": 0.25,
            },
        }}))
        loads = []
        monkeypatch.setattr(O, "_load_live_book", lambda path: loads.append(path) or {})

        adj = bargaining._profile_adjust(SimpleNamespace(opponent_name="gpt-4o"))

        assert adj == {
            "floor": pytest.approx(0.05),
            "anchor": 0.0,
            "give": pytest.approx(-0.05),
            "patient": False,
        }
        assert loads == []


class TestLiveBook:
    def test_db_absent_gives_dataset_only(self):
        # Live book silently empty; dataset half still answers.
        prof = O.profile("gemini-1.5-flash")
        assert prof == {"barg_n": 10, "model": "gemini-1.5-flash"}
        assert O.profile("SoftBot") is None  # db absent + unknown name -> None

    def test_aggregates_accepted_shares(self, isolated, monkeypatch):
        db = isolated / "agent.db"
        _make_db(db, [
            _row("g1", "SoftBot", player="player_1", gain=60.0),          # 0.60
            _row("g2", "SoftBot", player="player_2", gain=70.0),          # 0.70
            _row("g3", "SoftBot", outcome="no_deal", gain=90.0),          # excluded
            _row("g4", "SoftBot", family="negotiation", gain=90.0),       # excluded
            _row("g5", "SoftBot", money=0.0, gain=90.0),                  # excluded
            _row("g6", "other", gain=10.0),                               # other name
        ])
        monkeypatch.setattr(O, "DB_PATH", db)
        O.reset_cache()
        prof = O.profile("SoftBot")
        assert prof is not None
        assert prof["live_n"] == 2
        assert prof["live_share_to_me"] == pytest.approx(0.65)

    def test_live_merges_with_dataset(self, isolated, monkeypatch):
        db = isolated / "agent.db"
        _make_db(db, [_row("g1", "gpt-4o", gain=55.0)])
        monkeypatch.setattr(O, "DB_PATH", db)
        O.reset_cache()
        prof = O.profile("gpt-4o")
        assert prof["model"] == "gpt-4o"
        assert prof["barg_n"] == 100
        assert prof["live_n"] == 1
        assert prof["live_share_to_me"] == pytest.approx(0.55)

    def test_corrupt_db_never_raises(self, isolated, monkeypatch):
        bad = isolated / "corrupt.db"
        bad.write_bytes(b"this is definitely not a sqlite database")
        monkeypatch.setattr(O, "DB_PATH", bad)
        O.reset_cache()
        assert O.profile("SoftBot") is None
        assert O.profile("gpt-4o")["barg_n"] == 100  # dataset half unharmed

    def test_book_cached_once(self, isolated, monkeypatch):
        db = isolated / "agent.db"
        _make_db(db, [_row("g1", "SoftBot", gain=60.0)])
        monkeypatch.setattr(O, "DB_PATH", db)
        O.reset_cache()
        assert O.profile("SoftBot")["live_n"] == 1
        # A second call must not re-read the file: point the path at garbage
        # and confirm the cached book still answers.
        monkeypatch.setattr(O, "DB_PATH", isolated / "gone.db")
        assert O.profile("SoftBot")["live_n"] == 1
