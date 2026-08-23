"""Per-opponent behavioural book, compiled from our own game logs.

Why this beats tuning global knobs: the competition scores you against what
the FIELD earns versus the same opponent, and opponent identity is disclosed
in half our games. Measured over ~22k bargaining games, our head-to-head
share ranges from 0.407 (Agent Smith) to 0.581 (codex) -- a spread of 0.17,
where the global knobs we have been tuning move things by 0.02-0.04. The
opponents beating us are not subtler, they simply anchor harder: they open
by offering us 0.347 where the ones we beat open at 0.451.

Reads the JSONL logs directly rather than agent.db, so it stays fresh
without a multi-GB ingest. Output is small and loaded read-only by
theory/opponents.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from collections import defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "opponent_book.json"
MIN_N = 40           # games before a name gets a profile at all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--days", type=int, default=3)
    args = ap.parse_args()

    share = defaultdict(lambda: defaultdict(list))   # name -> family -> [share]
    openings = defaultdict(lambda: defaultdict(list))

    for agent in ("main", "test_a", "test_b", "test_c", "test_d"):
        meta = {}
        for path in sorted(glob.glob(f"logs/{agent}-2026*.jsonl"))[-args.days:]:
            try:
                fh = open(path, encoding="utf-8", errors="replace")
            except OSError:
                continue
            with fh:
                for line in fh:
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    if d.get("type") == "turn":
                        g = d.get("game") or {}
                        name = (g.get("opponent") or {}).get("name")
                        gid = g.get("game_id")
                        if not gid or not name:
                            continue
                        fam = g.get("game_family")
                        me = g.get("your_player")
                        meta[gid] = (fam, me, name)
                        # their opening offer to us, as a share of the pot
                        if fam == "bargaining":
                            st = g.get("game_state") or {}
                            M = st.get("money_to_divide")
                            idx = 1 if me == "player_1" else 2
                            if M:
                                for e in (st.get("history") or []):
                                    if not isinstance(e, dict) or e.get("proposer") == me:
                                        continue
                                    v = (e.get("offer") or {}).get(f"player_{idx}_gain")
                                    if isinstance(v, (int, float)):
                                        openings[name][fam].append(v / M)
                                        break
                    elif d.get("type") == "result":
                        info = meta.get(d.get("game_id"))
                        if not info:
                            continue
                        fam, me, name = info
                        r = d.get("result") or {}
                        if not r:
                            continue
                        idx = 1 if me == "player_1" else 2
                        mine = r.get(f"player_{idx}_payoff")
                        theirs = r.get(f"player_{3 - idx}_payoff")
                        if not isinstance(mine, (int, float)) or not isinstance(theirs, (int, float)):
                            continue
                        tot = mine + theirs
                        if tot > 0:
                            share[name][fam].append(mine / tot)

    book: dict = {}
    for name, fams in share.items():
        rec = {}
        for fam, vals in fams.items():
            if len(vals) < MIN_N:
                continue
            entry = {"n": len(vals), "share": round(statistics.mean(vals), 4)}
            op = openings.get(name, {}).get(fam)
            if op and len(op) >= 20:
                entry["their_open_to_me"] = round(statistics.mean(op), 4)
            rec[fam] = entry
        if rec:
            book[name] = rec

    Path(args.out).write_text(json.dumps({"opponents": book}))
    tough = sorted(((v["bargaining"]["share"], k) for k, v in book.items()
                    if "bargaining" in v))[:5]
    print(f"wrote {args.out}: {len(book)} opponents")
    print("toughest in bargaining:", [f"{k} {s:.3f}" for s, k in tough])


if __name__ == "__main__":
    main()
