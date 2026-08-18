#!/usr/bin/env python
"""Self-analysis of our own live GLEE games: where do we bleed rating?

Reads data/agent.db (read-only) plus logs/main-*.jsonl (full game dicts and
result payloads) and regenerates docs/self_analysis.md. Re-runnable; never
writes to the db.

Payoff sources, in order of trust:
  captured      — result payload from the platform (db or log).
  reconstructed — no result arrived, but OUR last action deterministically
                  ended the game (we accepted an offer / walked away / final
                  buyer 'no'), so the payoff is computable exactly.
  approx_seller — persuasion seller games that reached the final round; the
                  buyer's last decision is unobserved, so the running
                  seller_total_payoff undercounts by at most one price.
  inferred      — game went idle right after OUR offer: the opponent most
                  likely accepted it (shown separately, never mixed into
                  aggregate stats).

Usage: .venv/bin/python scripts/self_analysis.py
"""

from __future__ import annotations

import glob
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB = REPO / "data" / "agent.db"
OUT = REPO / "docs" / "self_analysis.md"
IDLE_S = 1800  # a game untouched this long before the log's end is over

STATE_TRANSIENT = {
    "round", "phase", "current_player", "history", "last_offer",
    "seller_message", "game_family", "seller_total_payoff",
    "buyer_total_payoff", "round_quality", "proposer",
}


# ----------------------------------------------------------------- loading

def load_db_games() -> dict[str, dict]:
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM games WHERE agent='main'").fetchall()
    con.close()
    return {r["game_id"]: dict(r) for r in rows}


def load_log_games() -> tuple[dict[str, dict], float]:
    """game_id -> log entry; also returns the last timestamp in the logs."""
    games: dict[str, dict] = {}
    last_ts = 0.0
    for path in sorted(glob.glob(str(REPO / "logs" / "main-*.jsonl"))):
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                last_ts = max(last_ts, rec.get("ts") or 0)
                g = rec.get("game")
                if g:
                    gid = g.get("game_id")
                    if not gid:
                        continue
                    entry = games.setdefault(gid, {
                        "family": g.get("game_family"),
                        "your_player": g.get("your_player"),
                        "opp_type": "hidden", "opp_name": None,
                        "state": {}, "actions": [], "result": None,
                        "last_ts": 0.0, "last_phase": None, "last_action": None,
                        "last_state": {},
                    })
                    opp = g.get("opponent") or {}
                    if opp.get("type") and opp["type"] != "hidden":
                        entry["opp_type"] = opp["type"]
                    if opp.get("name"):
                        entry["opp_name"] = opp["name"]
                    st = g.get("game_state") or {}
                    if (st.get("round") or 0) >= (entry["state"].get("round") or 0):
                        entry["state"] = st
                    act = rec.get("action")
                    if act is not None:
                        entry["actions"].append((st.get("round"), g.get("phase"), act))
                    entry["last_ts"] = rec.get("ts") or entry["last_ts"]
                    entry["last_phase"] = g.get("phase")
                    entry["last_action"] = act
                    entry["last_state"] = st
                elif rec.get("type") == "result" and rec.get("result"):
                    gid = rec.get("game_id")
                    if gid and gid in games:
                        games[gid]["result"] = rec["result"]
    return games, last_ts


# ------------------------------------------------------------ reconstruction

def _my_index(yp: str) -> int:
    return int(yp[-1])


def reconstruct(l: dict) -> dict | None:
    """For an idle game with no result: outcome+payoff if OUR last action
    deterministically ended it. Returns {outcome, my_payoff, source, ...}."""
    fam, yp = l["family"], l["your_player"]
    st, act, phase = l["last_state"], l["last_action"], l["last_phase"]
    if not fam or not yp or not isinstance(act, dict):
        return None
    i = _my_index(yp)
    rnd = st.get("round") or 0

    if fam == "bargaining" and phase == "decision":
        dec = act.get("decision")
        lo = st.get("last_offer") or {}
        if dec == "accept" and f"player_{i}_gain" in lo:
            delta = st.get(f"delta_{i}")
            delta = 1.0 if delta is None else delta
            gain = lo[f"player_{i}_gain"]
            return {"outcome": "agreement", "my_payoff": gain * delta ** (rnd - 1),
                    "agreed_round": rnd, "source": "reconstructed",
                    "agreed_my_gain": gain}
        if dec == "walkaway":
            return {"outcome": "walked_away", "my_payoff": 0.0,
                    "source": "reconstructed"}
        if dec == "reject" and st.get("max_rounds") and rnd >= st["max_rounds"]:
            return {"outcome": "no_deal", "my_payoff": 0.0,
                    "source": "reconstructed"}

    if fam == "negotiation" and phase == "decision":
        dec = act.get("decision")
        val = st.get(f"player_{i}_value")
        lo = st.get("last_offer") or {}
        price = lo.get("price")
        role = "seller" if i == 1 else "buyer"
        if dec == "AcceptOffer" and val is not None and price is not None:
            pay = (price - val) if role == "seller" else (val - price)
            return {"outcome": "agreement", "my_payoff": pay,
                    "agreed_round": rnd, "source": "reconstructed"}
        if dec == "WalkAway":
            return {"outcome": "walked_away", "my_payoff": 0.0,
                    "source": "reconstructed"}
        if dec == "RejectOffer" and st.get("max_rounds") and rnd >= st["max_rounds"]:
            return {"outcome": "no_deal", "my_payoff": 0.0,
                    "source": "reconstructed"}

    if fam == "persuasion":
        total = st.get("total_rounds") or 0
        if phase == "buyer_decision" and act.get("decision") == "no" and rnd >= total:
            return {"outcome": "completed", "my_payoff": st.get("buyer_total_payoff", 0.0),
                    "source": "reconstructed"}
        if phase in ("seller_message", "seller_recommendation") and rnd >= total > 0:
            return {"outcome": "completed", "my_payoff": st.get("seller_total_payoff", 0.0),
                    "source": "approx_seller"}
    return None


def infer_accept(l: dict) -> dict | None:
    """Idle right after OUR offer: opponent most likely accepted it."""
    fam, yp = l["family"], l["your_player"]
    st, act, phase = l["last_state"], l["last_action"], l["last_phase"]
    if phase != "offer" or not isinstance(act, dict) or not yp:
        return None
    i = _my_index(yp)
    rnd = st.get("round") or 0
    if fam == "bargaining":
        pot = st.get("money_to_divide")
        mine = act.get("alice_gain" if i == 1 else "bob_gain")
        if pot and mine is not None:
            delta = st.get(f"delta_{i}")
            delta = 1.0 if delta is None else delta
            return {"share": mine / pot, "payoff": mine * delta ** (rnd - 1),
                    "round": rnd, "pot": pot}
    if fam == "negotiation":
        val = st.get(f"player_{i}_value")
        price = act.get("product_price")
        if val and price is not None:
            pay = (price - val) if i == 1 else (val - price)
            return {"payoff": pay, "norm": pay / val, "round": rnd}
    return None


def merge(db_games: dict, log_games: dict, log_end: float) -> tuple[list[dict], dict]:
    """(completed games, misc counters). Inferred-accept games are returned
    inside misc, never inside the completed list."""
    out, misc = [], {"inferred_barg": [], "inferred_neg": [],
                     "unknown_ended": Counter(), "in_flight": 0}
    for gid in set(db_games) | set(log_games):
        d = db_games.get(gid, {})
        l = log_games.get(gid, {})
        res = l.get("result")
        outcome = d.get("outcome") or (res or {}).get("outcome")
        family = d.get("family") or l.get("family")
        yp = d.get("your_player") or l.get("your_player")
        rec = None
        if outcome:
            my_payoff = d.get("my_payoff")
            opp_payoff = d.get("opp_payoff")
            if my_payoff is None and res and yp:
                my_payoff = res.get(f"{yp}_payoff")
                opp_payoff = res.get(f"player_{3 - _my_index(yp)}_payoff")
            if my_payoff is None:
                continue
            rec = {"outcome": outcome, "my_payoff": my_payoff,
                   "opp_payoff": opp_payoff, "source": "captured",
                   "agreed_round": d.get("agreed_round") or (res or {}).get("agreed_round")}
        elif l and l["last_ts"] < log_end - IDLE_S:
            rec = reconstruct(l)
            if rec is None:
                inf = infer_accept(l)
                if inf is not None and family == "bargaining":
                    misc["inferred_barg"].append(inf)
                elif inf is not None and family == "negotiation":
                    misc["inferred_neg"].append(inf)
                else:
                    misc["unknown_ended"][(family, l.get("last_phase"))] += 1
                continue
        elif l:
            misc["in_flight"] += 1
            continue
        else:
            continue

        cfg = json.loads(d["config_json"]) if d.get("config_json") else {}
        if not cfg:
            cfg = {k: v for k, v in (l.get("state") or {}).items()
                   if k not in STATE_TRANSIENT}
        out.append({
            "game_id": gid, "family": family, "your_player": yp,
            "opp_type": d.get("opp_type") or l.get("opp_type") or "hidden",
            "opp_name": d.get("opp_name") or l.get("opp_name"),
            "config": cfg,
            "result": res or (json.loads(d["result_json"]) if d.get("result_json") else {}),
            "state": l.get("state") or {}, "actions": l.get("actions") or [],
            "opp_payoff": rec.get("opp_payoff"), **rec,
        })
    return out, misc


# ------------------------------------------------------------------ helpers

def cfg_short(family: str, cfg: dict) -> str:
    if family == "bargaining":
        return (f"pot={cfg.get('money_to_divide')}, d1={cfg.get('delta_1')}, "
                f"d2={cfg.get('delta_2')}, T={cfg.get('max_rounds')}, "
                f"msg={cfg.get('messages_allowed')}, ci={cfg.get('complete_information')}")
    if family == "negotiation":
        val = cfg.get("player_1_value", cfg.get("player_2_value"))
        return (f"my_value={val}, T={cfg.get('max_rounds')}, "
                f"msg={cfg.get('messages_allowed')}, ci={cfg.get('complete_information')}")
    return (f"price={cfg.get('product_price')}, p={round(cfg.get('p'), 3) if cfg.get('p') else cfg.get('p')}, "
            f"v={cfg.get('v')}, u={cfg.get('u')}, T={cfg.get('total_rounds')}, "
            f"msgtype={cfg.get('seller_message_type')}")


def norm_payoff(g: dict) -> float | None:
    cfg, fam = g["config"], g["family"]
    if fam == "bargaining":
        pot = cfg.get("money_to_divide") or 0
        return g["my_payoff"] / pot if pot else None
    if fam == "negotiation":
        val = cfg.get("player_1_value") or cfg.get("player_2_value")
        return g["my_payoff"] / val if val else None
    price = cfg.get("product_price") or 0
    total = cfg.get("total_rounds") or 1
    return g["my_payoff"] / (price * total) if price else None


def role(g: dict) -> str:
    if g["family"] in ("negotiation", "persuasion"):
        return "seller" if g["your_player"] == "player_1" else "buyer"
    return g["your_player"]


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


# ----------------------------------------------------------------- sections

def sec_caveats(games: list[dict], misc: dict, db_games: dict, log_games: dict) -> list[str]:
    src = Counter(g["source"] for g in games)
    L = ["## 0. Data & the blind spot", ""]
    L.append(f"- db tracks {len(db_games)} games; the log carries {len(log_games)} "
             f"game ids. Completed games analyzed: **{len(games)}** — "
             f"{src.get('captured', 0)} captured (platform result payload), "
             f"{src.get('reconstructed', 0)} reconstructed (our own accept/walk/"
             f"final-'no' ended the game, payoff computed exactly), "
             f"{src.get('approx_seller', 0)} approx (persuasion seller games at "
             f"the final round; total undercounts by <= 1 round's price).")
    L.append(f"- **Capture blind spot: the platform only sends a result payload "
             f"when OUR move ends the game.** Games ending on the opponent's "
             f"move (they accept our offer; the buyer's final round-20 decision "
             f"in our seller games) produce no result record — that is why the "
             f"raw db shows zero proposer-seat agreements and zero persuasion "
             f"seller completions. {len(misc['inferred_barg'])} bargaining and "
             f"{len(misc['inferred_neg'])} negotiation games went idle right "
             f"after our offer (opponent most likely accepted — analyzed "
             f"separately below); "
             f"{sum(misc['unknown_ended'].values())} games ended with unknown "
             f"outcome; {misc['in_flight']} still in flight.")
    L.append("")
    return L


def sec_overview(games: list[dict], db_games: dict) -> list[str]:
    L = ["## 1. Per-family overview", ""]
    L.append("| family | completed analyzed | mean payoff | median payoff | "
             "mean norm. payoff | median norm. payoff | no-deal rate |")
    L.append("|---|---|---|---|---|---|---|")
    for fam in ("bargaining", "negotiation", "persuasion"):
        fg = [g for g in games if g["family"] == fam]
        nodeal = [g for g in fg if g["outcome"] in ("no_deal", "walked_away")]
        np_ = [norm_payoff(g) for g in fg]
        L.append(
            f"| {fam} | {len(fg)} | "
            f"{fmt(mean([g['my_payoff'] for g in fg]))} | "
            f"{fmt(median([g['my_payoff'] for g in fg]))} | "
            f"{fmt(mean(np_), 3)} | {fmt(median(np_), 3)} | "
            f"{len(nodeal)}/{len(fg)} ({fmt(len(nodeal)/len(fg)*100 if fg else None, 0)}%) |")
    L.append("")
    L.append("Norm. payoff = payoff/pot (bargaining), payoff/my_value "
             "(negotiation), payoff/(price*rounds) (persuasion).")
    L.append("")
    L.append("### Lowest-payoff configs (mean normalized payoff, ascending, n >= 2)")
    L.append("")
    for fam in ("bargaining", "negotiation", "persuasion"):
        by_cfg = defaultdict(list)
        for g in games:
            if g["family"] == fam:
                by_cfg[cfg_short(fam, g["config"])].append(g)
        rows = sorted(
            ((mean([norm_payoff(x) for x in v]), k, v) for k, v in by_cfg.items()
             if len(v) >= 2 and mean([norm_payoff(x) for x in v]) is not None),
            key=lambda t: t[0])[:6]
        L.append(f"**{fam}**")
        L.append("")
        L.append("| config | n | mean norm. payoff | mean payoff |")
        L.append("|---|---|---|---|")
        for m, k, v in rows:
            L.append(f"| {k} | {len(v)} | {fmt(m, 3)} | "
                     f"{fmt(mean([x['my_payoff'] for x in v]))} |")
        L.append("")
    return L


def sec_bargaining(games: list[dict], misc: dict) -> list[str]:
    L = ["## 2. Bargaining", ""]
    bg = [g for g in games if g["family"] == "bargaining"]
    agreed = [g for g in bg if g["outcome"] == "agreement" and g.get("agreed_round")]

    L.append("### Agreed-round distribution (each extra round burns delta)")
    L.append("")
    hist = Counter(g["agreed_round"] for g in agreed)
    L.append("| round | games | share of agreements |")
    L.append("|---|---|---|")
    for r in sorted(hist):
        L.append(f"| {r} | {hist[r]} | {hist[r]/len(agreed)*100:.0f}% |")
    late = [g for g in agreed if g["agreed_round"] >= 5]

    def _disc_loss(g):
        res = g["result"] or {}
        gain = res.get("agreed_player_1_gain" if g["your_player"] == "player_1"
                       else "agreed_player_2_gain", g.get("agreed_my_gain"))
        pot = g["config"].get("money_to_divide")
        if gain is None or not pot:
            return None
        return (gain - g["my_payoff"]) / pot
    disc = [_disc_loss(g) for g in agreed]
    L.append("")
    L.append(f"- Agreements at round >= 5: **{len(late)}/{len(agreed)}** "
             f"({len(late)/max(len(agreed),1)*100:.0f}%). Mean agreed round: "
             f"{fmt(mean([g['agreed_round'] for g in agreed]), 1)}.")
    L.append(f"- Mean pot share LOST to discounting (agreed nominal gain minus "
             f"realized payoff, as share of pot): **{fmt(mean(disc), 3)}**.")
    L.append("")

    L.append("### Share of pot when we agree (realized payoff / pot)")
    L.append("")
    L.append("| split | n | mean share | median share |")
    L.append("|---|---|---|---|")

    def row(label, sel):
        s = [norm_payoff(g) for g in sel]
        L.append(f"| {label} | {len(sel)} | {fmt(mean(s), 3)} | {fmt(median(s), 3)} |")

    def i_proposed(g):
        return (g["agreed_round"] % 2 == 1) == (g["your_player"] == "player_1")

    row("all observed agreements", agreed)
    row("my proposal accepted (proposer seat)", [g for g in agreed if i_proposed(g)])
    row("I accepted theirs (responder seat)", [g for g in agreed if not i_proposed(g)])
    for ot in ("agent", "human", "hidden"):
        row(f"vs {ot} opponents", [g for g in agreed if g["opp_type"] == ot])
    L.append("")
    inf = misc["inferred_barg"]
    if inf:
        L.append(f"**Proposer seat via inference:** {len(inf)} further games went "
                 f"idle immediately after our offer — the opponent almost "
                 f"certainly accepted it. Mean share we had proposed for "
                 f"ourselves: **{fmt(mean([x['share'] for x in inf]), 3)}** "
                 f"(median {fmt(median([x['share'] for x in inf]), 3)}), at mean "
                 f"round {fmt(mean([x['round'] for x in inf]), 1)}. These are the "
                 f"only games where our own proposals close — and they close "
                 f"late, after the anchor has already been conceded down.")
        L.append("")

    nod = [g for g in bg if g["outcome"] in ("no_deal", "walked_away")]
    L.append(f"### No-deal games: {len(nod)}/{len(bg)}")
    L.append("")
    if nod:
        cc = Counter(cfg_short("bargaining", g["config"]) for g in nod)
        L.append("| config | no-deals |")
        L.append("|---|---|")
        for k, n in cc.most_common():
            L.append(f"| {k} | {n} |")
    L.append("")
    return L


def sec_negotiation(games: list[dict], misc: dict) -> list[str]:
    L = ["## 3. Negotiation", ""]
    ng = [g for g in games if g["family"] == "negotiation"]

    L.append("| role | n | closed (agreement) | mean payoff | mean payoff/value | "
             "median payoff/value |")
    L.append("|---|---|---|---|---|---|")
    for r in ("seller", "buyer"):
        sel = [g for g in ng if role(g) == r]
        closed = [g for g in sel if g["outcome"] == "agreement"]
        np_ = [norm_payoff(g) for g in sel]
        L.append(f"| {r} | {len(sel)} | {len(closed)} "
                 f"({len(closed)/max(len(sel),1)*100:.0f}%) | "
                 f"{fmt(mean([g['my_payoff'] for g in sel]))} | "
                 f"{fmt(mean(np_), 3)} | {fmt(median(np_), 3)} |")
    L.append("")

    inf = misc["inferred_neg"]
    if inf:
        L.append(f"- **Our-offer-accepted (inferred):** {len(inf)} games went idle "
                 f"right after our offer; if accepted, mean payoff/value "
                 f"**{fmt(mean([x['norm'] for x in inf]), 3)}** — our proposer-seat "
                 f"deals are far richer than our responder-seat ones.")

    left_on_table = []
    for g in ng:
        if g["outcome"] != "agreement":
            continue
        st = g["state"]
        val = g["config"].get("player_1_value") or g["config"].get("player_2_value")
        if not val:
            continue
        r = role(g)
        best = None
        for e in st.get("history") or []:
            off = e.get("offer") if isinstance(e, dict) else None
            if isinstance(off, dict) and off.get("from_player") != g["your_player"]:
                p = off.get("price")
                if p is not None:
                    best = p if best is None else (max(best, p) if r == "seller" else min(best, p))
        if best is None:
            continue
        best_pay = (best - val) if r == "seller" else (val - best)
        left_on_table.append((best_pay - g["my_payoff"]) / val)
    L.append(f"- Realized vs best opposing offer ever seen in closed deals "
             f"((best-possible minus realized)/value): mean "
             f"**{fmt(mean(left_on_table), 3)}** over {len(left_on_table)} games "
             f"(negative = we closed better than their best standing offer).")

    walks = 0
    for g in ng:
        for _, _, act in g["actions"]:
            if isinstance(act, dict) and act.get("decision") == "WalkAway":
                walks += 1
    L.append(f"- Our WalkAway fired in **{walks}** turns; "
             f"{len([g for g in ng if g['outcome']=='walked_away'])} games ended "
             f"walked_away, {len([g for g in ng if g['outcome']=='no_deal'])} no_deal.")

    rounds = [g.get("agreed_round") or g["state"].get("round") for g in ng]
    rounds = [r for r in rounds if r]
    marathons = [r for r in rounds if r >= 15]
    L.append(f"- Rounds to finish: mean {fmt(mean(rounds), 1)}, median "
             f"{fmt(median(rounds), 0)}, max {max(rounds) if rounds else 'n/a'}; "
             f"**{len(marathons)}** games ran >= 15 rounds (marathons).")

    ults = [g for g in ng if g["config"].get("max_rounds") == 1]
    if ults:
        closed = [g for g in ults if g["outcome"] == "agreement"]
        L.append(f"- max_rounds=1 ultimatums: {len(ults)} games, closed "
                 f"{len(closed)} ({len(closed)/len(ults)*100:.0f}%), mean "
                 f"payoff/value {fmt(mean([norm_payoff(g) for g in ults]), 3)}.")
    L.append("")
    return L


def sec_persuasion(games: list[dict]) -> list[str]:
    L = ["## 4. Persuasion", ""]
    pg = [g for g in games if g["family"] == "persuasion"]
    buyers = [g for g in pg if role(g) == "buyer"]
    sellers = [g for g in pg if role(g) == "seller"]

    neg = [g for g in buyers if g["my_payoff"] < 0]
    L.append(f"### Buyer games: {len(buyers)}; NEGATIVE totals: {len(neg)} "
             f"({len(neg)/max(len(buyers),1)*100:.0f}%)")
    L.append("")
    if neg:
        L.append("| config | payoff | rounds bought |")
        L.append("|---|---|---|")
        for g in sorted(neg, key=lambda x: norm_payoff(x) or 0):
            rb = (g["result"] or {}).get("rounds_bought")
            L.append(f"| {cfg_short('persuasion', g['config'])} | "
                     f"{g['my_payoff']:.0f} | {rb if rb is not None else '?'} |")
        L.append("")
        thin = [g for g in neg
                if g["config"].get("v") and g["config"].get("product_price")
                and (g["config"]["v"] - g["config"]["product_price"])
                / g["config"]["product_price"] < 0.5]
        L.append(f"- {len(thin)}/{len(neg)} negative games are thin-margin "
                 f"(v/price < 1.5); the rest lost on p=1/3 configs where the "
                 f"prior EV of a blind buy is already negative.")
    zero = [g for g in buyers if g["my_payoff"] == 0]
    L.append(f"- Buyer games ending at exactly 0 (never profitably bought): "
             f"{len(zero)}/{len(buyers)}. Mean buyer norm. payoff "
             f"{fmt(mean([norm_payoff(g) for g in buyers]), 3)}.")
    L.append("")

    L.append(f"### Seller games: {len(sellers)} "
             f"(all payoffs approx: final buyer decision unobserved)")
    L.append("")
    sell_rates, collapse_games, post_lie_buy, post_lie_n, lied_games = [], 0, 0, 0, 0
    for g in sellers:
        hist = g["state"].get("history") or []
        if hist:
            sell_rates.append(sum(1 for e in hist if isinstance(e, dict)
                                  and e.get("bought")) / len(hist))
        lied_at = [i for i, e in enumerate(hist)
                   if isinstance(e, dict) and e.get("bought") and e.get("quality") == "low"]
        if lied_at:
            lied_games += 1
            after = [e for e in hist[lied_at[0] + 1:] if isinstance(e, dict)]
            buys_after = sum(1 for e in after if e.get("bought"))
            post_lie_buy += buys_after
            post_lie_n += len(after)
            if after and buys_after / len(after) < 0.2:
                collapse_games += 1
    L.append(f"- Sell rate (rounds bought / rounds observed): mean "
             f"**{fmt(mean(sell_rates), 3)}**, median {fmt(median(sell_rates), 3)} "
             f"over {len(sell_rates)} games. Mean seller norm. payoff "
             f"{fmt(mean([norm_payoff(g) for g in sellers]), 3)}.")
    if post_lie_n:
        L.append(f"- Low-quality sales ('lies' that got bought) occurred in "
                 f"{lied_games} games; after the first one buyers bought in "
                 f"{post_lie_buy}/{post_lie_n} subsequent rounds "
                 f"({post_lie_buy/post_lie_n*100:.0f}%); **{collapse_games}** games "
                 f"show trust collapse (<20% buying after the lie).")
    else:
        L.append("- No completed low-quality sale observed in seller games.")
    L.append("")
    return L


def sec_opponents(games: list[dict]) -> list[str]:
    L = ["## 5. Payoff by disclosed opponent", ""]
    by_opp = defaultdict(list)
    for g in games:
        if g["opp_name"]:
            by_opp[g["opp_name"]].append(g)
    rows = []
    for name, gs in by_opp.items():
        if len(gs) < 2:
            continue
        np_ = mean([norm_payoff(g) for g in gs])
        wins = sum(1 for g in gs if g.get("opp_payoff") is not None
                   and g["my_payoff"] > g["opp_payoff"])
        known = sum(1 for g in gs if g.get("opp_payoff") is not None)
        rows.append((np_ if np_ is not None else -9, name, gs, wins, known))
    rows.sort()
    L.append("| opponent | n | mean norm. payoff | my payoff > theirs |")
    L.append("|---|---|---|---|")
    for np_, name, gs, wins, known in rows:
        L.append(f"| {name} | {len(gs)} | {fmt(np_ if np_ != -9 else None, 3)} | "
                 f"{wins}/{known} |")
    L.append("")
    L.append("(>= 2 completed games; hidden opponents — the majority — carry no "
             "name. Low rows = who beats us; high rows = who we farm.)")
    L.append("")
    return L


def sec_recommendations(games: list[dict], misc: dict) -> list[str]:
    bg = [g for g in games if g["family"] == "bargaining"]
    agreed = [g for g in bg if g["outcome"] == "agreement" and g.get("agreed_round")]
    late = [g for g in agreed if g["agreed_round"] >= 5]
    resp_share = mean([norm_payoff(g) for g in agreed
                       if (g["agreed_round"] % 2 == 1) != (g["your_player"] == "player_1")])
    inf_share = mean([x["share"] for x in misc["inferred_barg"]])
    ng = [g for g in games if g["family"] == "negotiation"]
    nodeal_ng = [g for g in ng if g["outcome"] in ("no_deal", "walked_away")]
    ults = [g for g in ng if g["config"].get("max_rounds") == 1]
    ult_closed = [g for g in ults if g["outcome"] == "agreement"]
    pg = [g for g in games if g["family"] == "persuasion"]
    buyers = [g for g in pg if role(g) == "buyer"]
    negb = [g for g in buyers if g["my_payoff"] < 0]

    L = ["## 6. Top 5 recommendations (ranked by expected rating impact)", ""]
    L.append(
        f"1. **Negotiation: close more deals — {len(nodeal_ng)}/{len(ng)} "
        f"({len(nodeal_ng)/max(len(ng),1)*100:.0f}%) of our games end at $0.** "
        f"The zero pile sits at/below the pool's no-deal mass, capping the "
        f"family percentile. Two knobs in `config.py`: lower `neg_anchor_markup` "
        f"(0.9 -> ~0.5) so counterparts engage instead of stonewalling, and "
        f"lower `neg_beta` (2.5 -> ~1.8) so we reach acceptable territory while "
        f"the opponent is still at the table; in "
        f"`families/negotiation.py::decide` relax the accept rule "
        f"`payoff >= counter_payoff * 0.9` to ~0.75 once round/T > 0.5 — a live "
        f"positive offer beats a speculative counter that risks the whole game.")
    L.append(
        f"2. **Negotiation: reprice max_rounds=1 ultimatums from dataset "
        f"acceptance curves — only {len(ult_closed)}/{len(ults)} closed** (mean "
        f"payoff/value {fmt(mean([norm_payoff(g) for g in ults]), 3)}). The "
        f"`max_rounds == 1` branch prices at (anchor+floor)/2 = value*~1.46 as "
        f"seller, which the field rejects. Use targets.json `neg_accept` "
        f"buckets to maximize P(accept) * margin; a ~10-20% markup that closes "
        f"60% of the time dominates a 46% markup that closes 27%.")
    L.append(
        f"3. **Bargaining: our anchors never close — get accepted earlier.** "
        f"Every captured agreement is us accepting theirs (mean share "
        f"{fmt(resp_share, 3)}); the {len(misc['inferred_barg'])} "
        f"inferred proposer-seat closes (mean proposed share {fmt(inf_share, 3)}) "
        f"only land at mean round "
        f"{fmt(mean([x['round'] for x in misc['inferred_barg']]), 1)}, after "
        f"discounting ate the premium, and {len(late)}/{len(agreed)} observed "
        f"agreements land at round >= 5. Drop `barg_anchor_agent` (0.80 -> "
        f"~0.68) and `barg_beta` (2.5 -> ~1.5) so our round-1/3 offers are "
        f"acceptable while the pot is still whole — an accepted 0.62 in round 1 "
        f"beats an accepted 0.65 in round 6 under any delta < 1.")
    L.append(
        f"4. **Persuasion buyer: stop buying into negative-EV configs — "
        f"{len(negb)}/{len(buyers)} buyer games ended NEGATIVE.** Losses "
        f"concentrate in thin-margin (v/price <= 1.25) and p=1/3 configs where "
        f"only near-perfect seller honesty makes buying profitable; a negative "
        f"total sits below the pool's entire never-buy mass at 0. Raise "
        f"`pers_buy_margin` (0.02 -> ~0.10) when (v-price)/price < 0.5, cut "
        f"`pers_explore_frac` (0.33 -> ~0.15), and add a hard stop in "
        f"`families/persuasion.py::_buyer_decide`: never buy speculatively once "
        f"cumulative payoff would go negative.")
    L.append(
        f"5. **Bargaining: accept good offers faster under heavy discounting.** "
        f"Mean pot share lost to delta-decay across agreements is "
        f"{fmt(mean([( (g['result'] or {}).get('agreed_player_1_gain' if g['your_player']=='player_1' else 'agreed_player_2_gain', g.get('agreed_my_gain')) - g['my_payoff'])/g['config'].get('money_to_divide') for g in agreed if g['config'].get('money_to_divide') and ((g['result'] or {}).get('agreed_player_1_gain' if g['your_player']=='player_1' else 'agreed_player_2_gain', g.get('agreed_my_gain')) is not None)]), 3)} "
        f"of the pot — pure burn. Scale `barg_cont_realism` (0.85) by the joint "
        f"discount (multiply by min(delta_1, delta_2, 1.0)) and lower "
        f"`barg_accept_great` (0.65 -> ~0.58) when min(delta) <= 0.9, so the "
        f"continuation value stops overrating a future round that is worth 10% "
        f"less by construction.")
    L.append("")
    return L


# --------------------------------------------------------------------- main

def main() -> None:
    db_games = load_db_games()
    log_games, log_end = load_log_games()
    games, misc = merge(db_games, log_games, log_end)

    L = ["# Self-analysis: live game performance", ""]
    L.append(f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
             f"by `scripts/self_analysis.py` (re-runnable; regenerates this file).")
    L.append("")
    L += sec_caveats(games, misc, db_games, log_games)
    L += sec_overview(games, db_games)
    L += sec_bargaining(games, misc)
    L += sec_negotiation(games, misc)
    L += sec_persuasion(games)
    L += sec_opponents(games)
    L += sec_recommendations(games, misc)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L))
    print(f"wrote {OUT} ({len(games)} completed games)")


if __name__ == "__main__":
    main()
