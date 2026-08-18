"""Run the GLEE agent.

Usage:
    python scripts/run_agent.py --agent main --concurrency 8
    python scripts/run_agent.py --agent test_a --max-games 50
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glee_sdk import GleeClient  # noqa: E402

from glee_agent.capture import (  # noqa: E402
    LoggingGleeClient,
    start_leaderboard_thread,
    start_reaper_thread,
    start_snapshot_thread,
)
from glee_agent.config import load_settings  # noqa: E402
from glee_agent.dispatcher import build_strategy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_agent")


def acquire_instance_lock(agent_label: str):
    """Refuse to start when another process is already playing this agent.

    Two processes sharing one API key poll the same pending games and race
    each other's moves — the server rejects the loser with "It is not your
    turn", burning attempts and rate budget. flock releases automatically on
    process death, so a crashed run never wedges the lock."""
    import fcntl

    lock_dir = Path(__file__).resolve().parents[1] / "logs"
    lock_dir.mkdir(exist_ok=True)
    lock_file = (lock_dir / f".agent-{agent_label}.lock").open("w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        logger.error(
            "Another process is already running agent %r (probably the tmux "
            "session 'glee-main'). Two copies on one API key race each "
            "other's moves and burn attempts. For manual experiments, stop "
            "the session first (tmux send-keys -t glee-main C-c) or use a "
            "test agent key (--agent test_a).",
            agent_label,
        )
        raise SystemExit(1)
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    return lock_file  # keep the handle alive for the process lifetime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", default="main", choices=["main", "test_a", "test_b"])
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument(
        "--families",
        default=None,
        help="Comma-separated subset, e.g. bargaining,persuasion (default: all)",
    )
    args = parser.parse_args()

    settings = load_settings(args.agent)
    _lock = acquire_instance_lock(settings.agent_label)  # noqa: F841 — held for process lifetime
    concurrency = args.concurrency or settings.concurrency
    families = args.families.split(",") if args.families else None

    client = LoggingGleeClient(
        api_key=settings.glee_api_key, agent_label=settings.agent_label
    )
    stats = client.stats()
    logger.info("Agent %s (%s): %s", settings.agent_label, stats.get("agent_name"), stats)

    # Telemetry gets its own client: requests.Session is not thread-safe, and
    # the game loop's session must never be shared across threads.
    telemetry = GleeClient(api_key=settings.glee_api_key)
    start_snapshot_thread(telemetry, settings.agent_label)
    start_leaderboard_thread(stats.get("agent_id"))
    start_reaper_thread(client, GleeClient(api_key=settings.glee_api_key))

    strategy = build_strategy(settings)
    client.run(
        strategy,
        game_families=families,
        concurrency=concurrency,
        max_games=args.max_games,
        max_time=args.max_time,
    )


if __name__ == "__main__":
    main()
