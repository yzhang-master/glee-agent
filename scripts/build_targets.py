#!/usr/bin/env python3
"""Build data/targets.json from the GLEE research dataset in data/glee_raw/Data.

Aggregates per-config realized-payoff quantiles, deal rates, negotiation
role-marginals, acceptance-curve counters, persuasion sell rates and
per-model profiles. See repo task notes for the schema contract.
"""
import argparse
import csv
import datetime
import json
import math
import os
import sys
from collections import defaultdict

csv.field_size_limit(1 << 30)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "glee_raw", "Data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "targets.json")

GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95

LLM_TYPE = "litellm"
HUMAN_SIDE_LLM_TYPE = "otree_LLM"  # LLM side inside human_vs_llm games


def dumps(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":"), default=str)


def quantiles(vals):
    s = sorted(vals)
    n = len(s)
    out = []
    for p in GRID:
        pos = (n - 1) * p
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out.append(round(s[lo] * (1 - frac) + s[hi] * frac, 4))
    return out


def rl_bucket(max_rounds, r):
    if max_rounds is None:
        return "inf"
    left = max_rounds - r + 1
    if left <= 1:
        return "1"
    if left <= 3:
        return "2-3"
    return "4+"


def ffloat(x):
    if x is None:
        return None
    x = str(x).strip()
    if x == "" or x.lower() == "none":
        return None
    return float(x)


def fint(x):
    v = ffloat(x)
    return None if v is None else int(v)


def is_human_type(ptype):
    return ptype not in (LLM_TYPE, HUMAN_SIDE_LLM_TYPE)


def model_short(pargs, ptype):
    if ptype != LLM_TYPE:
        return None
    mn = (pargs or {}).get("model_name") or ""
    if not mn:
        return None
    return mn.split("/")[-1].lower()


class Agg:
    def __init__(self):
        # family -> config_key -> record
        self.cfg = {
            "bargaining": defaultdict(lambda: {"n": 0, "deals": 0, "p1": [], "p2": []}),
            "negotiation": defaultdict(lambda: {"n": 0, "deals": 0, "p1": [], "p2": []}),
            "persuasion": defaultdict(lambda: {"n": 0, "p1": [], "p2": [], "yes": 0, "dec": 0}),
        }
        self.neg_by_role = defaultdict(lambda: {"n": 0, "deals": 0, "vals": []})
        self.barg_accept = defaultdict(lambda: [0, 0])
        self.neg_accept = defaultdict(lambda: [0, 0])
        self.models = defaultdict(lambda: {"barg_n": 0, "barg_lt40": [0, 0],
                                           "neg_n": 0, "pers_buyer": [0, 0]})
        self.counts = defaultdict(int)
        self.skips = defaultdict(int)
        self.unterminated = defaultdict(int)


def read_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as fh:
        rdr = csv.DictReader(fh)
        if not rdr.fieldnames or "player" not in rdr.fieldnames:
            raise ValueError("bad header")
        return list(rdr)


def decision_of(row):
    d = (row.get("decision") or "").strip()
    if d == "" or d == "None":
        return None
    return d


# --------------------------------------------------------------------------- barga

def do_bargaining(cfg_all, ga, rows, agg, human_p1, human_p2, m1, m2):
    money = float(ga["money_to_divide"])
    d1 = float(ga["delta_1"])
    d2 = float(ga["delta_2"])
    mr = ga.get("max_rounds")
    mr = int(mr) if mr is not None else None
    cfg = {"money_to_divide": money, "delta_1": d1, "delta_2": d2,
           "max_rounds": mr, "horizon_known": mr is not None,
           "messages_allowed": bool(ga["messages_allowed"]),
           "complete_information": bool(ga["complete_information"])}
    key = dumps(cfg)

    last_offer = None  # (a_gain, b_gain, round)
    p1 = p2 = 0.0
    deal = False
    ended = False
    last_r = 0
    for row in rows:
        a = ffloat(row.get("alice_gain"))
        b = ffloat(row.get("bob_gain"))
        dec = decision_of(row)
        if a is not None and b is not None:
            try:
                last_offer = (a, b, max(int(float(row.get("round") or 0)), 1))
            except ValueError:
                pass
            continue
        if dec is None:
            continue
        # decision row
        responder = (row.get("player") or "").strip()
        try:
            r = int(float(row.get("round") or 0))
        except ValueError:
            r = 0
        if r < 1:  # empty/garbage round parses to 0 (no ValueError) — r=0
            r = last_offer[2] if last_offer else 1  # would INFLATE d**(r-1)
        is_accept = dec.lower() == "accept"
        is_reject = dec.lower() == "reject"
        if last_offer is not None and money > 0:
            share = (last_offer[0] if responder == "Alice" else last_offer[1]) / money
            # +1e-6 matches theory/targets.py::_bucket_005 exactly — without it,
            # exact multiples (0.30, 0.60...) bin one bucket low here but
            # on-boundary at lookup time, skewing the curve at precisely the
            # candidate shares the live optimizer scans.
            bucket = round(min(max(math.floor(share / 0.05 + 1e-6) * 0.05, 0.0), 0.95), 2)
            human = human_p1 if responder == "Alice" else human_p2
            akey = dumps({"share_bucket": bucket, "rounds_left_bucket": rl_bucket(mr, r),
                          "human": bool(human)})
            c = agg.barg_accept[akey]
            c[0] += 1
            c[1] += int(is_accept)
            rm = m1 if responder == "Alice" else m2
            if rm is not None and share < 0.40:
                mm = agg.models[rm]["barg_lt40"]
                mm[0] += 1
                mm[1] += int(is_accept)
        if is_accept and last_offer is not None:
            p1 = last_offer[0] * (d1 ** (r - 1))
            p2 = last_offer[1] * (d2 ** (r - 1))
            deal = True
            ended = True
            break
        if not is_reject and not is_accept:
            # walkaway-ish terminal decision
            ended = True
            break
        last_r = r
    if not ended and not (mr is not None and last_r >= mr):
        agg.unterminated["bargaining"] += 1

    rec = agg.cfg["bargaining"][key]
    rec["n"] += 1
    rec["deals"] += int(deal)
    rec["p1"].append(p1)
    rec["p2"].append(p2)
    for m in (m1, m2):
        if m is not None:
            agg.models[m]["barg_n"] += 1


# --------------------------------------------------------------------------- nego

def do_negotiation(cfg_all, ga, rows, agg, human_p1, human_p2, m1, m2):
    order = float(ga["product_price_order"])
    sv = ga.get("seller_value")
    bv = ga.get("buyer_value")
    s_abs = float(sv) * order if sv is not None else None
    b_abs = float(bv) * order if bv is not None else None
    mr = ga.get("max_rounds")
    mr = int(mr) if mr is not None else None
    base = {"max_rounds": mr, "horizon_known": mr is not None,
            "messages_allowed": bool(ga["messages_allowed"]),
            "complete_information": bool(ga["complete_information"])}
    cfg = dict(base)
    cfg["seller_value"] = s_abs
    cfg["buyer_value"] = b_abs
    key = dumps(cfg)

    last_price = None
    p1 = p2 = 0.0
    deal = False
    ended = False
    last_r = 0
    for row in rows:
        dec = decision_of(row)
        price = ffloat(row.get("product_price"))
        if dec is None:
            if price is not None:
                last_price = price
            continue
        responder = (row.get("player") or "").strip()
        try:
            r = int(float(row.get("round") or 0))
        except ValueError:
            r = 0
        if r < 1:  # empty round parses to 0, not ValueError
            r = 1
        role = "seller" if responder == "Alice" else "buyer"
        my_val = s_abs if role == "seller" else b_abs
        is_accept = dec == "AcceptOffer"
        is_reject = dec == "RejectOffer"
        if last_price is not None and my_val is not None and my_val > 0:
            rel = last_price / my_val
            # Same epsilon as the live lookup; hi clamp 1.95 matches
            # theory/targets.py (rel >= 1.95 pools into one top bucket both sides).
            bucket = round(min(max(math.floor(rel / 0.05 + 1e-6) * 0.05, 0.0), 1.95), 2)
            human = human_p1 if responder == "Alice" else human_p2
            akey = dumps({"rel_bucket": bucket, "role": role,
                          "rounds_left_bucket": rl_bucket(mr, r), "human": bool(human)})
            c = agg.neg_accept[akey]
            c[0] += 1
            c[1] += int(is_accept)
        if is_accept and last_price is not None:
            if s_abs is not None:
                p1 = last_price - s_abs
            if b_abs is not None:
                p2 = b_abs - last_price
            deal = True
            ended = True
            break
        if not is_reject and not is_accept:
            # SellToJhon / BuyFromJhon / other walkaway
            ended = True
            break
        last_r = r
    if not ended and not (mr is not None and last_r >= mr):
        agg.unterminated["negotiation"] += 1

    rec = agg.cfg["negotiation"][key]
    rec["n"] += 1
    rec["deals"] += int(deal)
    rec["p1"].append(p1)
    rec["p2"].append(p2)
    for role, my_val, pay in (("seller", s_abs, p1), ("buyer", b_abs, p2)):
        if my_val is None:
            continue
        rk = dumps(dict(base, role=role, my_value=my_val))
        rr = agg.neg_by_role[rk]
        rr["n"] += 1
        rr["deals"] += int(deal)
        rr["vals"].append(pay)
    for m in (m1, m2):
        if m is not None:
            agg.models[m]["neg_n"] += 1


# --------------------------------------------------------------------------- pers

def do_persuasion(cfg_all, ga, rows, agg, human_p1, human_p2, m1, m2):
    price = float(ga["product_price"])
    cfg = {"product_price": price, "p": float(ga["p"]),
           "v": float(ga["v"]) * price, "u": float(ga["c"]) * price,
           "total_rounds": int(ga["total_rounds"]),
           "seller_message_type": str(ga["seller_message_type"])}
    key = dumps(cfg)

    # In binary-message games the SELLER's yes/no recommendation is logged in
    # the same `decision` column as the buyer's decision (there is no
    # `message` column), so payoffs must be computed from the buyer's rows
    # only — counting every yes row doubles both pools.
    buyer_name = ((cfg_all.get("player_2_args") or {}).get("public_name") or "Bob").strip()

    worth = None
    p1 = p2 = 0.0
    yes = dec_n = 0
    for row in rows:
        player = (row.get("player") or "").strip()
        if player == "Nature":
            w = ffloat(row.get("product_worth"))
            if w is not None:
                worth = w
            continue
        if player != buyer_name:
            continue
        dec = decision_of(row)
        if dec is None:
            continue
        d = dec.lower()
        if d not in ("yes", "no"):
            continue
        dec_n += 1
        if d == "yes":
            yes += 1
            p1 += price
            if worth is not None:
                p2 += worth - price
        if m2 is not None:
            agg.models[m2]["pers_buyer"][0] += 1
            agg.models[m2]["pers_buyer"][1] += int(d == "yes")

    rec = agg.cfg["persuasion"][key]
    rec["n"] += 1
    rec["p1"].append(round(p1, 6))
    rec["p2"].append(round(p2, 6))
    rec["yes"] += yes
    rec["dec"] += dec_n


HANDLERS = {"bargaining": do_bargaining, "negotiation": do_negotiation,
            "persuasion": do_persuasion}


def iter_game_dirs(root, families):
    for src in ("llm_vs_llm", "human_vs_llm"):
        for fam in families:
            base = os.path.join(root, src, fam)
            if not os.path.isdir(base):
                continue
            for dirpath, dirnames, filenames in os.walk(base):
                if "config.json" in filenames:
                    dirnames[:] = []
                    yield fam, dirpath


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["bargaining", "negotiation", "persuasion"])
    ap.add_argument("--limit", type=int, default=0, help="max games per family (0 = all)")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    families = [args.family] if args.family else ["bargaining", "negotiation", "persuasion"]
    agg = Agg()
    total = 0
    for fam, gd in iter_game_dirs(args.root, families):
        if args.limit and agg.counts[fam] + agg.skips[fam] >= args.limit:
            continue
        try:
            with open(os.path.join(gd, "config.json"), encoding="utf-8") as fh:
                conf = json.load(fh)
            ga = conf["game_args"]
            rows = read_rows(os.path.join(gd, "game.csv"))
        except Exception:
            agg.skips[fam] += 1
            continue
        if fam == "persuasion" and ga.get("is_myopic"):
            # Myopic games give the seller a FRESH buyer each round (armed
            # only with aggregate sales stats) — a systematically easier
            # selling regime. The live platform is never myopic (no
            # is_myopic in any live game_state), so myopic games would
            # contaminate the comparison pool: skip them (~half the split).
            agg.skips[fam] += 1
            continue
        t1 = conf.get("player_1_type") or ""
        t2 = conf.get("player_2_type") or ""
        m1 = model_short(conf.get("player_1_args"), t1)
        m2 = model_short(conf.get("player_2_args"), t2)
        try:
            HANDLERS[fam](conf, ga, rows, agg, is_human_type(t1), is_human_type(t2), m1, m2)
        except Exception:
            agg.skips[fam] += 1
            continue
        agg.counts[fam] += 1
        total += 1
        if total % 5000 == 0:
            print(f"... {total} games parsed {dict(agg.counts)}", flush=True)

    # ---- assemble output
    out = {
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "n_games": {f: agg.counts[f] for f in families},
        "quantile_grid": GRID,
    }
    barg = {}
    for k, r in agg.cfg["bargaining"].items():
        barg[k] = {"n": r["n"], "deal_rate": round(r["deals"] / r["n"], 4),
                   "payoff_q": {"player_1": quantiles(r["p1"]), "player_2": quantiles(r["p2"])}}
    out["bargaining"] = barg
    neg = {}
    for k, r in agg.cfg["negotiation"].items():
        neg[k] = {"n": r["n"], "deal_rate": round(r["deals"] / r["n"], 4),
                  "payoff_q": {"player_1": quantiles(r["p1"]), "player_2": quantiles(r["p2"])}}
    out["negotiation"] = neg
    out["neg_by_role"] = {k: {"n": r["n"], "deal_rate": round(r["deals"] / r["n"], 4),
                              "payoff_q": quantiles(r["vals"])}
                          for k, r in agg.neg_by_role.items()}
    pers = {}
    for k, r in agg.cfg["persuasion"].items():
        pers[k] = {"n": r["n"],
                   "payoff_q": {"player_1": quantiles(r["p1"]), "player_2": quantiles(r["p2"])},
                   "seller_sell_rate": round(r["yes"] / r["dec"], 4) if r["dec"] else 0.0}
    out["persuasion"] = pers
    out["barg_accept"] = dict(agg.barg_accept)
    out["neg_accept"] = dict(agg.neg_accept)
    models = {}
    for m, r in sorted(agg.models.items()):
        lt = r["barg_lt40"]
        pb = r["pers_buyer"]
        models[m] = {"barg_n": r["barg_n"],
                     "barg_accept_rate_when_offered_lt40pct":
                         round(lt[1] / lt[0], 4) if lt[0] else None,
                     "neg_n": r["neg_n"],
                     "pers_buyer_trust_rate": round(pb[1] / pb[0], 4) if pb[0] else None}
    out["models"] = models

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    size = os.path.getsize(args.out)

    print("\n=== DONE ===")
    print(f"parsed per family: {dict(agg.counts)}")
    print(f"skips per family: {dict(agg.skips)}")
    print(f"unterminated (counted as no-deal 0): {dict(agg.unterminated)}")
    print(f"{args.out}: {size / 1e6:.2f} MB")
    print(f"distinct configs: barg={len(barg)} neg={len(neg)} pers={len(pers)} "
          f"neg_by_role={len(out['neg_by_role'])} barg_accept={len(out['barg_accept'])} "
          f"neg_accept={len(out['neg_accept'])} models={len(models)}")
    for fam, table in (("bargaining", barg), ("negotiation", neg), ("persuasion", pers)):
        top = sorted(table.items(), key=lambda kv: -kv[1]["n"])[:3]
        print(f"top {fam} configs:")
        for k, v in top:
            q = v["payoff_q"]["player_1"]
            print(f"  n={v['n']} p1_q50={q[9]} p1_q90={q[17]} key={k}")


if __name__ == "__main__":
    main()
