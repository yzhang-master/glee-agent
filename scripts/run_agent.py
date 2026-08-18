"""Run the GLEE agent.

Usage:
    python scripts/run_agent.py --agent main --concurrency 8
    python scripts/run_agent.py --agent test_a --max-games 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glee_sdk import GleeClient  # noqa: E402

from glee_agent.capture import (  # noqa: E402
    LoggingGleeClient,
    start_leaderboard_thread,
    start_snapshot_thread,
)
from glee_agent.config import load_settings  # noqa: E402
from glee_agent.dispatcher import build_strategy  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("run_agent")


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
    concurrency = args.concurrency or settings.concurrency
    families = args.families.split(",") if args.families else None

    client = LoggingGleeClient(
        api_key=settings.glee_api_key, agent_label=settings.agent_label
    )
    stats = client.stats()
    logger.info("Agent %s (%s): %s", settings.agent_label, stats.get("agent_name"), stats)

    # Telemetry gets its own client: requests.Session is not thread-safe, and
    # the game loop's session must never be shared across threads.
    start_snapshot_thread(
        GleeClient(api_key=settings.glee_api_key), settings.agent_label
    )
    start_leaderboard_thread(stats.get("agent_id"))

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
