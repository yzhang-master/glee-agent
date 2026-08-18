"""Compare agents (main vs test arms) on percentile-scored performance.

For every completed game in the window, look up where its realized payoff
sits in the dataset pool for that (config, role) — the same currency the
live rating uses — and average per (agent, family). This makes a 50-game
test run directly comparable to main's concurrent play, without waiting
for ratings to converge.

Usage:
    .venv/bin/python scripts/ab_report.py [--hours 24]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glee_agent.memory import store  # noqa: E402
from glee_agent.theory import targets as targets_mod  # noqa: E402

FAMILIES = ["bargaining", "negotiation", "persuasion"]


def game_percentile(tg, row) -> float | None:
    """Percentile of this game's payoff vs the dataset pool, or None."""
    family = row["family"]
    config_key = row["config_key"]
    payoff = row["my_payoff"]
    if payoff is None or not config_key:
        return None
    role = row["your_player"] or "player_1"
    try:
        return tg.payoff_percentile(family, config_key, role, payoff)
    except Exception:  # noqa: BLE001 — a bad lookup must not kill the report
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=24.0)
    args = parser.parse_args()

    store.ingest()
    conn = store.connect()
    tg = targets_mod.get_targets()
    since = time.time() - args.hours * 3600

    rows = conn.execute(
        "SELECT agent, family, your_player, config_key, my_payoff, outcome "
        "FROM games WHERE outcome IS NOT NULL AND last_ts >= ?",
        (since,),
    ).fetchall()

    # stats[agent][family] -> dict of accumulators
    stats: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "pct_sum": 0.0, "pct_n": 0,
                                     "payoff_sum": 0.0, "no_deal": 0})
    )
    for row in rows:
        s = stats[row["agent"]][row["family"]]
        s["n"] += 1
        if row["my_payoff"] is not None:
            s["payoff_sum"] += row["my_payoff"]
        if row["outcome"] == "no_deal":
            s["no_deal"] += 1
        pct = game_percentile(tg, row)
        if pct is not None:
            s["pct_sum"] += pct
            s["pct_n"] += 1

    print(f"A/B report — completed games, last {args.hours:g}h "
          f"(percentiles vs dataset pool; rating needs avg > ~0.55)\n")
    header = f"{'agent':<10} {'family':<12} {'games':>5} {'avg pct':>8} " \
             f"{'scored':>6} {'avg payoff':>11} {'no-deal':>7}"
    print(header)
    print("-" * len(header))
    for agent in sorted(stats):
        for fam in FAMILIES:
            s = stats[agent].get(fam)
            if not s or s["n"] == 0:
                continue
            avg_pct = f"{s['pct_sum'] / s['pct_n']:.3f}" if s["pct_n"] else "—"
            avg_pay = s["payoff_sum"] / s["n"]
            print(f"{agent:<10} {fam:<12} {s['n']:>5} {avg_pct:>8} "
                  f"{s['pct_n']:>6} {avg_pay:>11,.1f} {s['no_deal']:>7}")
        print()

    # Coverage note: how many games couldn't be scored (config unmatched).
    total = sum(s["n"] for a in stats.values() for s in a.values())
    scored = sum(s["pct_n"] for a in stats.values() for s in a.values())
    if total:
        print(f"coverage: {scored}/{total} games matched a dataset config "
              f"({scored / total * 100:.0f}%) — unmatched games likely have "
              f"config-key drift; investigate if low.")


if __name__ == "__main__":
    main()
