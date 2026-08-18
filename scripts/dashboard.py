"""Generate the Pit Wall — a self-contained HTML ops dashboard for the agent.

Reads data/agent.db (ingesting fresh JSONL first), writes dashboard/index.html.
No external assets: all CSS/JS/SVG inline, theme-aware (light/dark tokens).

Usage:
    .venv/bin/python scripts/dashboard.py [--out dashboard/index.html] [--no-ingest]
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from glee_agent.memory import store  # noqa: E402

FAMILIES = ["bargaining", "negotiation", "persuasion"]
FAMILY_LABEL = {"bargaining": "Bargaining", "negotiation": "Negotiation", "persuasion": "Persuasion"}
COMP_START = datetime(2026, 8, 1, tzinfo=timezone.utc)
COMP_END = datetime(2026, 8, 30, tzinfo=timezone.utc)

# Validated reference palette (dataviz skill): families on categorical slots
# 1-3 (the all-pairs-safe trio), status colors reserved for health chips.
SERIES = {
    "bargaining": ("#2a78d6", "#3987e5"),   # (light, dark)
    "negotiation": ("#eb6834", "#d95926"),
    "persuasion": ("#1baf7a", "#199e70"),
}


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def fmt_num(x, digits=0) -> str:
    if x is None:
        return "—"
    if abs(x) >= 10000 and digits == 0:
        return f"{x:,.0f}"
    return f"{x:,.{digits}f}"


# --------------------------------------------------------------- queries

def q_latest_ratings(conn):
    """{family: {rating, games_played, ts}} from the latest snapshot rows."""
    out = {}
    for fam in FAMILIES:
        row = conn.execute(
            "SELECT ts, rating, games_played FROM snapshots WHERE family=? "
            "ORDER BY ts DESC LIMIT 1", (fam,)
        ).fetchone()
        if row:
            out[fam] = {"rating": row["rating"], "games": row["games_played"], "ts": row["ts"]}
    return out


def q_rating_at(conn, fam: str, before_ts: float):
    row = conn.execute(
        "SELECT rating FROM snapshots WHERE family=? AND ts<=? ORDER BY ts DESC LIMIT 1",
        (fam, before_ts),
    ).fetchone()
    return row["rating"] if row else None


def q_rating_series(conn, fam: str, limit=400):
    rows = conn.execute(
        "SELECT ts, rating FROM snapshots WHERE family=? ORDER BY ts", (fam,)
    ).fetchall()
    if len(rows) > limit:  # thin evenly, keep the endpoints
        step = len(rows) / limit
        rows = [rows[int(i * step)] for i in range(limit - 1)] + [rows[-1]]
    return [(r["ts"], r["rating"]) for r in rows]


def q_games_per_day(conn, days=10):
    since = time.time() - days * 86400
    rows = conn.execute(
        "SELECT date(last_ts,'unixepoch') AS day, family, COUNT(*) AS n FROM games "
        "WHERE outcome IS NOT NULL AND last_ts>=? GROUP BY day, family ORDER BY day",
        (since,),
    ).fetchall()
    by_day: dict[str, dict[str, int]] = {}
    for r in rows:
        by_day.setdefault(r["day"], {})[r["family"]] = r["n"]
    return by_day


def q_health(conn):
    now = time.time()
    day_ago = now - 86400
    h = {}
    h["invalid_total"] = conn.execute(
        "SELECT COUNT(*) c FROM results WHERE valid=0").fetchone()["c"]
    h["invalid_24h"] = conn.execute(
        "SELECT COUNT(*) c FROM results WHERE valid=0 AND ts>=?", (day_ago,)).fetchone()["c"]
    h["errors_24h"] = conn.execute(
        "SELECT COUNT(*) c FROM turns WHERE error IS NOT NULL AND ts>=?", (day_ago,)).fetchone()["c"]
    h["corrections_24h"] = conn.execute(
        "SELECT COALESCE(SUM(n_corrections),0) c FROM turns WHERE ts>=?", (day_ago,)).fetchone()["c"]
    row = conn.execute(
        "SELECT MAX(elapsed_s) m FROM turns WHERE ts>=?", (day_ago,)).fetchone()
    h["max_latency_24h"] = row["m"] or 0.0
    for fam in FAMILIES:
        h[f"games48_{fam}"] = conn.execute(
            "SELECT COUNT(*) c FROM games WHERE family=? AND outcome IS NOT NULL AND last_ts>=?",
            (fam, now - 172800),
        ).fetchone()["c"]
        h[f"games24_{fam}"] = conn.execute(
            "SELECT COUNT(*) c FROM games WHERE family=? AND outcome IS NOT NULL AND last_ts>=?",
            (fam, day_ago),
        ).fetchone()["c"]
    return h


def q_leaderboard(conn, agent_id: str | None):
    out = {}
    for fam in FAMILIES:
        latest = conn.execute(
            "SELECT MAX(ts) m FROM lb WHERE family=?", (fam,)).fetchone()["m"]
        if latest is None:
            continue
        rows = conn.execute(
            "SELECT * FROM lb WHERE family=? AND ts=? ORDER BY "
            "CASE WHEN rank IS NULL THEN 1 ELSE 0 END, rank", (fam, latest),
        ).fetchall()
        me = next((r for r in rows if agent_id and r["player_id"] == agent_id), None)
        ranked = [r for r in rows if r["rank"] is not None]
        out[fam] = {
            "top1": ranked[0] if ranked else None,
            "top5": ranked[4] if len(ranked) >= 5 else None,
            "me": me,
            "ts": latest,
        }
    return out


def q_config_table(conn, fam: str, min_n=2, limit=6):
    rows = conn.execute(
        "SELECT config_key, COUNT(*) n, AVG(my_payoff) avg_payoff, "
        "SUM(CASE WHEN outcome='agreement' OR my_payoff>0 THEN 1 ELSE 0 END)*1.0/COUNT(*) deal_rate "
        "FROM games WHERE family=? AND outcome IS NOT NULL AND my_payoff IS NOT NULL "
        "GROUP BY config_key HAVING n>=? ORDER BY avg_payoff DESC", (fam, min_n),
    ).fetchall()
    return rows[:limit], rows[-limit:] if len(rows) > limit else []


def q_opponents(conn, limit=12):
    return conn.execute(
        "SELECT opp_name, opp_type, family, COUNT(*) n, AVG(my_payoff) avg_payoff "
        "FROM games WHERE opp_name IS NOT NULL AND outcome IS NOT NULL "
        "GROUP BY opp_name, family ORDER BY n DESC LIMIT ?", (limit,),
    ).fetchall()


def q_recent_games(conn, limit=15):
    return conn.execute(
        "SELECT game_id, family, opp_type, opp_name, outcome, my_payoff, opp_payoff, "
        "agreed_round, n_turns, n_invalid, last_ts FROM games "
        "WHERE outcome IS NOT NULL ORDER BY last_ts DESC LIMIT ?", (limit,),
    ).fetchall()


# --------------------------------------------------------------- SVG charts

def _scale(vals, lo_px, hi_px, pad_frac=0.08):
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-9:
        vmin, vmax = vmin - 1, vmax + 1
    pad = (vmax - vmin) * pad_frac
    vmin, vmax = vmin - pad, vmax + pad

    def f(v):
        return lo_px + (v - vmin) / (vmax - vmin) * (hi_px - lo_px)

    return f, vmin, vmax


def sparkline(points, w=120, h=34) -> str:
    """12-point sparkline: de-emphasis line, accent endpoint."""
    if len(points) < 2:
        return f'<svg width="{w}" height="{h}" aria-hidden="true"></svg>'
    pts = points[-12:]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    fx, *_ = _scale(xs, 4, w - 6)
    fy, *_ = _scale(ys, h - 5, 5)
    path = " ".join(f"{'M' if i == 0 else 'L'}{fx(x):.1f},{fy(y):.1f}" for i, (x, y) in enumerate(pts))
    ex, ey = fx(xs[-1]), fy(ys[-1])
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" aria-hidden="true">'
        f'<path d="{path}" fill="none" stroke="var(--muted)" stroke-width="1.5" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3.5" fill="var(--accent)" '
        f'stroke="var(--surface)" stroke-width="2"/></svg>'
    )


def rating_chart(series: dict[str, list], w=920, h=280) -> str:
    """Multi-line rating history; one series per family, direct end labels."""
    all_pts = [p for pts in series.values() for p in pts]
    if len(all_pts) < 2:
        return ('<div class="empty">Rating history appears here once snapshots accumulate '
                '(the agent records one every 5 minutes).</div>')
    ml, mr, mt, mb = 46, 96, 14, 24
    fx, *_ = _scale([p[0] for p in all_pts], ml, w - mr, 0.02)
    fy, ymin, ymax = _scale([p[1] for p in all_pts], h - mb, mt)

    # Clean-number gridlines.
    grid, labels = [], []
    step = max(round((ymax - ymin) / 4 / 50) * 50, 50)
    y = math.ceil(ymin / step) * step
    while y < ymax:
        py = fy(y)
        grid.append(f'<line x1="{ml}" y1="{py:.1f}" x2="{w - mr}" y2="{py:.1f}" class="grid"/>')
        labels.append(f'<text x="{ml - 8}" y="{py + 4:.1f}" class="tick" text-anchor="end">{y:,.0f}</text>')
        y += step

    lines, endcaps, hover_meta = [], [], []
    for fam, pts in series.items():
        if len(pts) < 2:
            continue
        color = f"var(--s-{fam})"
        path = " ".join(f"{'M' if i == 0 else 'L'}{fx(x):.1f},{fy(v):.1f}" for i, (x, v) in enumerate(pts))
        lines.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        ex, ey = fx(pts[-1][0]), fy(pts[-1][1])
        endcaps.append(
            f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="{color}" '
            f'stroke="var(--surface)" stroke-width="2"/>'
            f'<text x="{ex + 9:.1f}" y="{ey + 4:.1f}" class="endlab">'
            f'{FAMILY_LABEL[fam]} {pts[-1][1]:,.0f}</text>'
        )
        hover_meta.append({
            "name": FAMILY_LABEL[fam],
            "pts": [[round(fx(x), 1), round(fy(v), 1), x, round(v, 1)] for x, v in pts],
        })

    meta = esc(json.dumps(hover_meta))
    return (
        f'<svg class="linechart" width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
        f'data-series="{meta}" role="img" aria-label="Rating history per family">'
        f'{"".join(grid)}{"".join(labels)}'
        f'<line x1="{ml}" y1="{h - mb}" x2="{w - mr}" y2="{h - mb}" class="axis"/>'
        f'{"".join(lines)}{"".join(endcaps)}'
        f'<line class="xhair" x1="0" y1="{mt}" x2="0" y2="{h - mb}" style="display:none"/>'
        f'</svg>'
    )


def volume_chart(by_day: dict, w=460, h=200) -> str:
    """Games/day stacked columns, family colors, 2px surface gaps."""
    if not by_day:
        return '<div class="empty">No completed games yet today.</div>'
    days = sorted(by_day.keys())[-10:]
    totals = [sum(by_day[d].values()) for d in days]
    vmax = max(totals) or 1
    ml, mb, mt = 34, 30, 10
    plot_h = h - mb - mt
    slot = (w - ml - 8) / len(days)
    bar_w = min(24, slot * 0.6)
    cols = []
    for i, day in enumerate(days):
        x = ml + i * slot + (slot - bar_w) / 2
        y = h - mb
        segs = []
        stack = [(fam, by_day[day].get(fam, 0)) for fam in FAMILIES]
        nonzero = [s for s in stack if s[1] > 0]
        for j, (fam, n) in enumerate(nonzero):
            seg_h = n / vmax * plot_h
            y -= seg_h
            top = j == len(nonzero) - 1
            gap = 0 if top else 2
            rx = 4 if top else 0
            title = f"{day} · {FAMILY_LABEL[fam]}: {n}"
            segs.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
                f'height="{max(seg_h - gap, 1):.1f}" rx="{rx}" fill="var(--s-{fam})" '
                f'class="seg"><title>{esc(title)}</title></rect>'
            )
        label = day[5:].replace("-", "/")
        cols.append(
            "".join(segs)
            + f'<text x="{x + bar_w / 2:.1f}" y="{h - mb + 16}" class="tick" text-anchor="middle">{label}</text>'
            + (f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" class="collab" '
               f'text-anchor="middle">{totals[i]}</text>')
        )
    return (
        f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" '
        f'aria-label="Completed games per day, stacked by family">'
        f'<line x1="{ml}" y1="{h - mb}" x2="{w - 8}" y2="{h - mb}" class="axis"/>'
        f'{"".join(cols)}</svg>'
    )


# --------------------------------------------------------------- HTML

CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e;
  --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --border: rgba(11,11,11,0.10); --accent: #2a78d6;
  --s-bargaining: #2a78d6; --s-negotiation: #eb6834; --s-persuasion: #1baf7a;
  --good: #0ca30c; --warn: #fab219; --crit: #d03b3b;
  --good-text: #006300; --up: #006300; --down: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
    --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --border: rgba(255,255,255,0.10); --accent: #3987e5;
    --s-bargaining: #3987e5; --s-negotiation: #d95926; --s-persuasion: #199e70;
    --good-text: #0ca30c; --up: #0ca30c;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #ffffff; --ink-2: #c3c2b7;
  --muted: #898781; --grid: #2c2c2a; --axis: #383835;
  --border: rgba(255,255,255,0.10); --accent: #3987e5;
  --s-bargaining: #3987e5; --s-negotiation: #d95926; --s-persuasion: #199e70;
  --good-text: #0ca30c; --up: #0ca30c;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1040px; margin: 0 auto; padding: 24px 20px 56px; }
header { display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap; margin-bottom: 18px; }
header h1 { font-size: 21px; font-weight: 650; margin: 0; letter-spacing: -0.01em; }
header .sub { color: var(--ink-2); font-size: 13px; }
.kpis { display: grid; grid-template-columns: 1.3fr 1fr 1fr 1fr; gap: 12px; margin-bottom: 12px; }
.tile {
  background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 16px 12px; display: flex; flex-direction: column; gap: 2px; min-width: 0;
}
.tile .label { font-size: 12px; color: var(--ink-2); display: flex; align-items: center; gap: 7px; }
.tile .label .key { width: 10px; height: 10px; border-radius: 3px; flex: none; }
.tile .value { font-size: 26px; font-weight: 650; }
.tile.hero .value { font-size: 48px; line-height: 1.1; font-weight: 650; }
.tile .delta { font-size: 12px; }
.tile .delta.up { color: var(--up); } .tile .delta.down { color: var(--down); }
.tile .meta { font-size: 12px; color: var(--muted); }
.tile .spark { margin-top: auto; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin: 4px 0 20px; }
.chip {
  display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px;
  border: 1px solid var(--border); border-radius: 999px; padding: 4px 12px;
  background: var(--surface); color: var(--ink-2);
}
.chip .dot { font-weight: 700; }
.chip.ok .dot { color: var(--good); } .chip.warn .dot { color: var(--warn); }
.chip.crit .dot { color: var(--crit); }
.chip.warn, .chip.crit { color: var(--ink); }
section { margin-bottom: 26px; }
h2 { font-size: 14px; font-weight: 650; margin: 0 0 10px; color: var(--ink); }
h2 .note { font-weight: 400; color: var(--muted); font-size: 12.5px; margin-left: 8px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 14px; }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
svg text { font: 11px system-ui, sans-serif; fill: var(--muted); }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
svg .tick { font-variant-numeric: tabular-nums; }
svg .endlab { fill: var(--ink-2); font-size: 11.5px; font-weight: 600; }
svg .collab { fill: var(--ink-2); font-variant-numeric: tabular-nums; }
svg .xhair { stroke: var(--axis); stroke-width: 1; }
.seg:hover { opacity: 0.85; }
.legend { display: flex; gap: 16px; margin-top: 8px; font-size: 12.5px; color: var(--ink-2); }
.legend .key { display: inline-block; width: 10px; height: 10px; border-radius: 3px; margin-right: 5px; }
.tablewrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th { text-align: left; color: var(--ink-2); font-weight: 600; font-size: 12px;
     border-bottom: 1px solid var(--grid); padding: 6px 10px 6px 0; white-space: nowrap; }
td { border-bottom: 1px solid var(--grid); padding: 6px 10px 6px 0;
     font-variant-numeric: tabular-nums; white-space: nowrap; }
td.name { max-width: 260px; overflow: hidden; text-overflow: ellipsis; }
tr:last-child td { border-bottom: none; }
.pos { color: var(--good-text); } .neg { color: var(--down); }
.fam { display: inline-flex; align-items: center; gap: 6px; }
.fam .key { width: 9px; height: 9px; border-radius: 3px; }
.empty { color: var(--muted); font-size: 13px; padding: 20px 4px; }
.tooltip {
  position: fixed; pointer-events: none; z-index: 10; display: none;
  background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 7px 10px; font-size: 12px; color: var(--ink);
  box-shadow: 0 4px 14px rgba(0,0,0,0.12);
}
footer { color: var(--muted); font-size: 12px; margin-top: 30px; }
@media (max-width: 800px) { .kpis { grid-template-columns: 1fr 1fr; } .cols { grid-template-columns: 1fr; } }
"""

JS = """
(function () {
  var tip = document.createElement('div');
  tip.className = 'tooltip';
  document.body.appendChild(tip);
  document.querySelectorAll('svg.linechart').forEach(function (svg) {
    var series = JSON.parse(svg.getAttribute('data-series') || '[]');
    if (!series.length) return;
    var xhair = svg.querySelector('.xhair');
    svg.addEventListener('mousemove', function (ev) {
      var pt = svg.createSVGPoint();
      pt.x = ev.clientX; pt.y = ev.clientY;
      var loc = pt.matrixTransform(svg.getScreenCTM().inverse());
      var best = null;
      series.forEach(function (s) {
        s.pts.forEach(function (p) {
          var d = Math.abs(p[0] - loc.x);
          if (!best || d < best.d) best = { d: d, x: p[0] };
        });
      });
      if (!best) return;
      xhair.setAttribute('x1', best.x); xhair.setAttribute('x2', best.x);
      xhair.style.display = '';
      var rows = [];
      var when = null;
      series.forEach(function (s) {
        var near = null;
        s.pts.forEach(function (p) {
          if (!near || Math.abs(p[0] - best.x) < Math.abs(near[0] - best.x)) near = p;
        });
        if (near && Math.abs(near[0] - best.x) < 1) {
          rows.push(s.name + ': <b>' + near[3].toLocaleString() + '</b>');
          when = near[2];
        }
      });
      if (when) {
        var d = new Date(when * 1000);
        rows.unshift(d.toLocaleString([], {month:'short', day:'numeric', hour:'2-digit', minute:'2-digit'}));
      }
      tip.innerHTML = rows.join('<br>');
      tip.style.display = 'block';
      tip.style.left = (ev.clientX + 14) + 'px';
      tip.style.top = (ev.clientY + 12) + 'px';
    });
    svg.addEventListener('mouseleave', function () {
      tip.style.display = 'none';
      xhair.style.display = 'none';
    });
  });
})();
"""


def _tile(fam: str, ratings: dict, conn) -> str:
    info = ratings.get(fam)
    spark = sparkline(q_rating_series(conn, fam))
    if not info:
        return (
            f'<div class="tile"><div class="label"><span class="key" '
            f'style="background:var(--s-{fam})"></span>{FAMILY_LABEL[fam]}</div>'
            f'<div class="value">—</div><div class="meta">no games yet '
            f'(counts as 1000)</div><div class="spark">{spark}</div></div>'
        )
    prev = q_rating_at(conn, fam, time.time() - 86400)
    delta_html = ""
    if prev is not None:
        d = info["rating"] - prev
        cls = "up" if d >= 0 else "down"
        delta_html = f'<div class="delta {cls}">{"+" if d >= 0 else ""}{d:,.1f} vs 24h ago</div>'
    return (
        f'<div class="tile"><div class="label"><span class="key" '
        f'style="background:var(--s-{fam})"></span>{FAMILY_LABEL[fam]}</div>'
        f'<div class="value">{info["rating"]:,.1f}</div>{delta_html}'
        f'<div class="meta">{info["games"]:,} games</div>'
        f'<div class="spark">{spark}</div></div>'
    )


def _chip(status: str, icon: str, text: str) -> str:
    return f'<span class="chip {status}"><span class="dot">{icon}</span>{esc(text)}</span>'


def build_html(conn, agent_id: str | None, agent_name: str) -> str:
    now = datetime.now(timezone.utc)
    ratings = q_latest_ratings(conn)
    overall = sum(ratings.get(f, {}).get("rating", 1000.0) for f in FAMILIES) / 3
    health = q_health(conn)
    lb = q_leaderboard(conn, agent_id)
    day_n = (now - COMP_START).days + 1
    days_left = max((COMP_END - now).days, 0)

    # Health chips: status color + icon + label, never color alone.
    chips = []
    inv = health["invalid_24h"]
    chips.append(_chip("ok" if inv == 0 else "crit", "✓" if inv == 0 else "✕",
                       f"invalid moves 24h: {inv} (total {health['invalid_total']})"))
    err = health["errors_24h"]
    chips.append(_chip("ok" if err == 0 else "warn", "✓" if err == 0 else "▲",
                       f"strategy errors 24h: {err}"))
    corr = health["corrections_24h"]
    chips.append(_chip("ok" if corr == 0 else "warn", "✓" if corr == 0 else "▲",
                       f"guard corrections 24h: {corr}"))
    lat = health["max_latency_24h"]
    chips.append(_chip("ok" if lat < 30 else "warn", "✓" if lat < 30 else "▲",
                       f"max turn latency 24h: {lat:.1f}s"))
    for fam in FAMILIES:
        n48 = health[f"games48_{fam}"]
        rating = ratings.get(fam, {}).get("rating", 0)
        if rating > 1800:
            ok = n48 >= 100
            chips.append(_chip("ok" if ok else "crit", "✓" if ok else "✕",
                               f"{FAMILY_LABEL[fam]} decay quota: {n48}/100 in 48h"))

    series = {fam: q_rating_series(conn, fam) for fam in FAMILIES}
    series = {k: v for k, v in series.items() if len(v) >= 2}
    legend = "".join(
        f'<span><span class="key" style="background:var(--s-{f})"></span>{FAMILY_LABEL[f]}</span>'
        for f in FAMILIES
    )

    # Leaderboard panel rows.
    lb_rows = []
    for fam in FAMILIES:
        entry = lb.get(fam)
        if not entry:
            lb_rows.append(f"<tr><td>{FAMILY_LABEL[fam]}</td><td colspan='4' class='empty'>no snapshot yet</td></tr>")
            continue
        me = entry["me"]
        my_rank = me["rank"] if me and me["rank"] is not None else "—"
        my_rating = f'{me["rating"]:,.1f}' if me else "—"
        top1 = f'{entry["top1"]["rating"]:,.0f}' if entry["top1"] else "—"
        top5 = f'{entry["top5"]["rating"]:,.0f}' if entry["top5"] else "—"
        gap = ""
        if me and entry["top5"]:
            g = entry["top5"]["rating"] - me["rating"]
            gap = f'{g:+,.0f}'
        lb_rows.append(
            f'<tr><td><span class="fam"><span class="key" style="background:var(--s-{fam})"></span>'
            f'{FAMILY_LABEL[fam]}</span></td><td>{my_rank}</td><td>{my_rating}</td>'
            f'<td>{top1} / {top5}</td><td>{gap}</td></tr>'
        )

    # Config tables.
    config_sections = []
    for fam in FAMILIES:
        best, worst = q_config_table(conn, fam)
        if not best:
            continue
        rows = []
        for r in best:
            cfg = json.loads(r["config_key"]) if r["config_key"] else {}
            desc = ", ".join(f"{k}={v}" for k, v in cfg.items() if v is not None)
            rows.append(
                f'<tr><td class="name" title="{esc(desc)}">{esc(desc[:70]) or "—"}</td>'
                f'<td>{r["n"]}</td><td>{fmt_num(r["avg_payoff"], 1)}</td>'
                f'<td>{r["deal_rate"] * 100:.0f}%</td></tr>'
            )
        config_sections.append(
            f'<h2><span class="fam"><span class="key" style="background:var(--s-{fam})"></span>'
            f'{FAMILY_LABEL[fam]}</span><span class="note">best configs by mean payoff (n≥2)</span></h2>'
            f'<div class="tablewrap"><table><tr><th>Configuration</th><th>Games</th>'
            f'<th>Mean payoff</th><th>Deal rate</th></tr>{"".join(rows)}</table></div>'
        )

    opp_rows = []
    for r in q_opponents(conn):
        opp_rows.append(
            f'<tr><td class="name">{esc(r["opp_name"])}</td><td>{esc(r["opp_type"])}</td>'
            f'<td><span class="fam"><span class="key" style="background:var(--s-{r["family"]})"></span>'
            f'{FAMILY_LABEL.get(r["family"], r["family"])}</span></td>'
            f'<td>{r["n"]}</td><td>{fmt_num(r["avg_payoff"], 1)}</td></tr>'
        )

    recent_rows = []
    for r in q_recent_games(conn):
        when = datetime.fromtimestamp(r["last_ts"], tz=timezone.utc).strftime("%m/%d %H:%M")
        payoff = r["my_payoff"]
        cls = "pos" if (payoff or 0) > 0 else "neg"
        opp = r["opp_name"] or r["opp_type"] or "hidden"
        recent_rows.append(
            f'<tr><td>{when}</td>'
            f'<td><span class="fam"><span class="key" style="background:var(--s-{r["family"]})"></span>'
            f'{FAMILY_LABEL.get(r["family"], r["family"])}</span></td>'
            f'<td class="name">{esc(opp)}</td><td>{esc(r["outcome"] or "—")}</td>'
            f'<td class="{cls}">{fmt_num(payoff, 1)}</td>'
            f'<td>{r["agreed_round"] or "—"}</td><td>{r["n_invalid"] or 0}</td></tr>'
        )

    return f"""<title>Glee Pit Wall</title>
<style>{CSS}</style>
<div class="wrap">
<header>
  <h1>Glee Pit Wall</h1>
  <span class="sub">{esc(agent_name)} · day {day_n} of 29 · {days_left} days left ·
  generated {now.strftime("%Y-%m-%d %H:%M UTC")}</span>
</header>

<div class="kpis">
  <div class="tile hero">
    <div class="label">Overall rating <span class="note">(3-family average; unplayed = 1000)</span></div>
    <div class="value">{overall:,.1f}</div>
    <div class="meta">display rating — shrunk toward 1000 by g/(g+30)</div>
  </div>
  {_tile("bargaining", ratings, conn)}
  {_tile("negotiation", ratings, conn)}
  {_tile("persuasion", ratings, conn)}
</div>

<div class="chips">{"".join(chips)}</div>

<section>
  <h2>Rating history<span class="note">display rating per family, one snapshot every 5 min</span></h2>
  <div class="card">{rating_chart(series)}
  <div class="legend">{legend}</div></div>
</section>

<div class="cols">
  <section>
    <h2>Games per day<span class="note">completed, stacked by family</span></h2>
    <div class="card">{volume_chart(q_games_per_day(conn))}
    <div class="legend">{legend}</div></div>
  </section>
  <section>
    <h2>Leaderboard position<span class="note">latest snapshot per family</span></h2>
    <div class="card"><div class="tablewrap"><table>
      <tr><th>Family</th><th>My rank</th><th>My rating</th><th>#1 / #5 rating</th><th>Gap to #5</th></tr>
      {"".join(lb_rows)}
    </table></div></div>
  </section>
</div>

<section>
  {"".join(config_sections) or '<h2>Configurations</h2><div class="empty">Config analysis appears after enough completed games per configuration.</div>'}
</section>

<div class="cols">
  <section>
    <h2>Opponents<span class="note">disclosed names only (half of games)</span></h2>
    <div class="card"><div class="tablewrap"><table>
      <tr><th>Opponent</th><th>Type</th><th>Family</th><th>Games</th><th>Mean payoff</th></tr>
      {"".join(opp_rows) or '<tr><td colspan="5" class="empty">No disclosed opponents yet.</td></tr>'}
    </table></div></div>
  </section>
  <section>
    <h2>Recent games</h2>
    <div class="card"><div class="tablewrap"><table>
      <tr><th>When (UTC)</th><th>Family</th><th>Opponent</th><th>Outcome</th>
      <th>My payoff</th><th>Round</th><th>Invalid</th></tr>
      {"".join(recent_rows) or '<tr><td colspan="7" class="empty">No completed games yet.</td></tr>'}
    </table></div></div>
  </section>
</div>

<footer>Data: logs/*.jsonl → data/agent.db. Refresh: <code>.venv/bin/python scripts/dashboard.py</code>.
Competition: Aug 1–29, 2026 (AoE) · glee-competition.com</footer>
</div>
<script>{JS}</script>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="dashboard/index.html")
    parser.add_argument("--no-ingest", action="store_true")
    args = parser.parse_args()

    if not args.no_ingest:
        counts = store.ingest()
        print(f"Ingested: {counts}")

    conn = store.connect()
    # Agent identity from the latest snapshot's stats if available.
    agent_id, agent_name = None, "puss-glee-agent"
    try:
        from dotenv import load_dotenv
        load_dotenv()
        import os
        from glee_sdk import GleeClient
        key = os.environ.get("GLEE_API_KEY_MAIN", "")
        if key:
            s = GleeClient(api_key=key).stats()
            agent_id, agent_name = s.get("agent_id"), s.get("agent_name", agent_name)
    except Exception as e:  # noqa: BLE001 — dashboard must render offline too
        print(f"(offline mode: {e})")

    html_out = build_html(conn, agent_id, agent_name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_out, encoding="utf-8")
    print(f"Wrote {out} ({len(html_out):,} bytes)")


if __name__ == "__main__":
    main()
