"""SQLite store + JSONL ingest: counts, idempotency, offset resume,
games rollup payoff mapping, and file-shrink recovery."""

import json
from pathlib import Path

from glee_agent.memory.store import connect, ingest

TS0 = 1787076848.0


def _bargaining_game(game_id: str, round_: int, opponent: dict) -> dict:
    """Mimic the production game dict (trimmed prompt/valid_actions)."""
    return {
        "game_id": game_id,
        "game_family": "bargaining",
        "your_player": "player_1",
        "phase": "proposal",
        "opponent": opponent,
        "game_state": {
            "game_family": "bargaining",
            "round": round_,
            "max_rounds": 12,
            "horizon_known": True,
            "phase": "proposal",
            "current_player": "player_1",
            "proposer": "player_1",
            "money_to_divide": 100,
            "delta_1": 0.95,
            "delta_2": 0.9,
            "complete_information": False,
            "messages_allowed": False,
            "history": [],
        },
        "valid_actions": {"type": "offer", "fields": {"alice_gain": "number"}},
        "prompt": "You are Alice...",
    }


def _legacy_turn(ts: float, game: dict, action: dict) -> dict:
    """Current production format: log_turn record WITHOUT a "type" key."""
    return {
        "ts": ts,
        "game": game,
        "action": action,
        "corrections": [],
        "elapsed_s": 0.0,
        "move_result": None,
        "error": None,
    }


def _write_jsonl(path: Path, records: list, garbage: bool = False) -> None:
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if garbage:
            f.write("this is {not json%%%\n")


def _fixture_log(log_dir: Path) -> None:
    game_hidden = _bargaining_game("g1", 1, {"type": "hidden", "name": None})
    game_shown = _bargaining_game("g1", 2, {"type": "agent", "name": "Rival"})
    records = [
        _legacy_turn(TS0, game_hidden, {"alice_gain": 80, "bob_gain": 20}),
        _legacy_turn(TS0 + 1, game_hidden, {"decision": "reject"}),
        {  # typed turn, one correction, opponent now revealed
            "type": "turn",
            "ts": TS0 + 2,
            "game": game_shown,
            "action": {"alice_gain": 60, "bob_gain": 40},
            "corrections": ["clamped alice_gain"],
            "elapsed_s": 0.012,
            "move_result": None,
            "error": None,
        },
        {
            "type": "result",
            "ts": TS0 + 3,
            "agent": "main",
            "game_id": "g1",
            "valid": True,
            "attempts_left": 3,
            "game_over": True,
            "error": None,
            "result": {
                "player_1_payoff": 600,
                "player_2_payoff": 400,
                "outcome": "agreement",
                "agreed_round": 2,
            },
        },
        {
            "type": "snapshot",
            "ts": TS0 + 4,
            "agent": "main",
            "scores": {
                "bargaining": {"rating": 1512.5, "games_played": 10},
                "persuasion": {"rating": 1480.0, "games_played": 4},
            },
            "active_games": 3,
        },
    ]
    _write_jsonl(log_dir / "main-20260818.jsonl", records, garbage=True)
    _write_jsonl(
        log_dir / "platform-20260818.jsonl",
        [
            {
                "type": "lb_snapshot",
                "ts": TS0 + 5,
                "family": "bargaining",
                "top": [
                    {
                        "rank": 1, "player_id": "p1", "player_name": "Best",
                        "player_type": "agent", "rating": 1600.0,
                        "games_played": 50, "is_owner_best": True,
                        "is_baseline": False, "is_benchmark": False,
                    },
                    {
                        "rank": 2, "player_id": "p2", "player_name": "Us",
                        "player_type": "agent", "rating": 1512.5,
                        "games_played": 10, "is_owner_best": False,
                        "is_baseline": False, "is_benchmark": False,
                    },
                ],
                "me": {"rank": 2, "player_id": "p2"},
            }
        ],
    )


def _counts(db_path: Path) -> dict:
    conn = connect(db_path)
    try:
        return {
            t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
            for t in ("turns", "results", "snapshots", "lb", "games")
        }
    finally:
        conn.close()


def test_ingest_counts_and_rollup(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    db_path = tmp_path / "agent.db"
    _fixture_log(log_dir)

    counts = ingest(log_dir=log_dir, db_path=db_path)
    assert counts == {"turns": 3, "results": 1, "snapshots": 1, "lb": 1, "skipped": 1}
    rows = _counts(db_path)
    assert rows == {"turns": 3, "results": 1, "snapshots": 2, "lb": 2, "games": 1}

    conn = connect(db_path)
    try:
        # legacy lines: agent inferred from the filename prefix
        agents = {r["agent"] for r in conn.execute("SELECT agent FROM turns")}
        assert agents == {"main"}

        g = conn.execute("SELECT * FROM games WHERE game_id = 'g1'").fetchone()
        assert g["agent"] == "main"
        assert g["family"] == "bargaining"
        assert g["your_player"] == "player_1"
        assert g["opp_type"] == "agent" and g["opp_name"] == "Rival"
        assert g["n_turns"] == 3
        assert g["n_corrections"] == 1
        assert g["n_errors"] == 0
        assert g["n_invalid"] == 0
        assert g["first_ts"] == TS0 and g["last_ts"] == TS0 + 3
        assert g["outcome"] == "agreement"
        assert g["agreed_round"] == 2
        assert g["my_payoff"] == 600 and g["opp_payoff"] == 400
        expected_cfg = {
            "money_to_divide": 100,
            "delta_1": 0.95,
            "delta_2": 0.9,
            "max_rounds": 12,
            "horizon_known": True,
            "messages_allowed": False,
            "complete_information": False,
        }
        assert g["config_key"] == json.dumps(
            expected_cfg, sort_keys=True, separators=(",", ":")
        )
        assert json.loads(g["config_json"]) == expected_cfg

        lb = conn.execute("SELECT * FROM lb ORDER BY rank").fetchall()
        assert lb[0]["player_name"] == "Best" and lb[0]["is_owner_best"] == 1
        assert lb[1]["rating"] == 1512.5 and lb[1]["is_owner_best"] == 0

        snaps = conn.execute("SELECT * FROM snapshots ORDER BY family").fetchall()
        assert [s["family"] for s in snaps] == ["bargaining", "persuasion"]
        assert snaps[0]["rating"] == 1512.5 and snaps[0]["active_games"] == 3
    finally:
        conn.close()


def test_ingest_idempotent(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    db_path = tmp_path / "agent.db"
    _fixture_log(log_dir)

    ingest(log_dir=log_dir, db_path=db_path)
    first = _counts(db_path)
    counts = ingest(log_dir=log_dir, db_path=db_path)
    assert counts == {"turns": 0, "results": 0, "snapshots": 0, "lb": 0, "skipped": 0}
    assert _counts(db_path) == first


def test_offset_resume(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    db_path = tmp_path / "agent.db"
    _fixture_log(log_dir)
    ingest(log_dir=log_dir, db_path=db_path)

    _write_jsonl(
        log_dir / "main-20260818.jsonl",
        [
            {
                "type": "result",
                "ts": TS0 + 10,
                "agent": "main",
                "game_id": "g1",
                "valid": False,
                "attempts_left": 2,
                "game_over": False,
                "error": "invalid action",
                "result": None,
            }
        ],
    )
    counts = ingest(log_dir=log_dir, db_path=db_path)
    assert counts == {"turns": 0, "results": 1, "snapshots": 0, "lb": 0, "skipped": 0}
    conn = connect(db_path)
    try:
        assert conn.execute("SELECT COUNT(*) AS n FROM results").fetchone()["n"] == 2
        g = conn.execute("SELECT * FROM games WHERE game_id = 'g1'").fetchone()
        assert g["n_invalid"] == 1
        assert g["last_ts"] == TS0 + 10  # results extend the ts range
        # game-over result from the first ingest is preserved
        assert g["outcome"] == "agreement" and g["my_payoff"] == 600
    finally:
        conn.close()


def test_rollup_payoffs_player_2(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    db_path = tmp_path / "agent.db"
    game = _bargaining_game("g2", 1, {"type": "human", "name": "Ann"})
    game["your_player"] = "player_2"
    _write_jsonl(
        log_dir / "test_a-20260818.jsonl",
        [
            _legacy_turn(TS0, game, {"decision": "accept"}),
            {
                "type": "result",
                "ts": TS0 + 1,
                "agent": "test_a",
                "game_id": "g2",
                "valid": True,
                "attempts_left": 3,
                "game_over": True,
                "error": None,
                "result": {
                    "player_1_payoff": 100,
                    "player_2_payoff": 900,
                    "outcome": "agreement",
                    "agreed_round": 1,
                },
            },
        ],
    )
    ingest(log_dir=log_dir, db_path=db_path)
    conn = connect(db_path)
    try:
        g = conn.execute("SELECT * FROM games WHERE game_id = 'g2'").fetchone()
        assert g["agent"] == "test_a"  # legacy prefix before the first "-"
        assert g["your_player"] == "player_2"
        assert g["my_payoff"] == 900 and g["opp_payoff"] == 100
        assert g["opp_type"] == "human" and g["opp_name"] == "Ann"
    finally:
        conn.close()


def test_file_shrink_resets_offset(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    db_path = tmp_path / "agent.db"
    _fixture_log(log_dir)
    ingest(log_dir=log_dir, db_path=db_path)
    before = _counts(db_path)

    # Rewrite the log much shorter than the stored offset.
    log_path = log_dir / "main-20260818.jsonl"
    log_path.write_text("", encoding="utf-8")
    _write_jsonl(
        log_path,
        [
            {
                "type": "result",
                "ts": TS0 + 20,
                "agent": "main",
                "game_id": "g9",
                "valid": True,
                "attempts_left": 3,
                "game_over": False,
                "error": None,
                "result": None,
            }
        ],
    )
    counts = ingest(log_dir=log_dir, db_path=db_path)
    assert counts == {"turns": 0, "results": 1, "snapshots": 0, "lb": 0, "skipped": 0}
    after = _counts(db_path)
    assert after["results"] == before["results"] + 1
    assert after["turns"] == before["turns"]  # nothing double-ingested
