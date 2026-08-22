"""Empirical percentile targets from the GLEE research dataset.

Loads data/targets.json (built offline from ~80k dataset games) and answers:
- where does a payoff sit vs the pool of ALL payoffs on the same config+role
  (exactly the live scoring pool), and
- how likely is the field to accept a given offer (bargaining share buckets /
  negotiation price-relative-to-value buckets).

Everything here is best-effort: a missing or corrupt targets file yields a
Null object whose lookups all return None, so strategies silently fall back
to pure theory. No function in this module ever raises into a caller.
"""

from __future__ import annotations

import json
import logging
import math
from bisect import bisect_right
from pathlib import Path
from typing import Any

logger = logging.getLogger("glee_agent")

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "targets.json"

DEFAULT_GRID = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 .. 0.95

MIN_BUCKET_N = 20  # accept-curve buckets thinner than this are noise


def _dumps(obj: dict) -> str:
    """Canonical config-key serialization. MUST match the builder byte-exact."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace("$", "").replace(",", "").strip())
        except ValueError:
            return None
    return None


def _int_or_none(value: Any) -> int | None:
    v = _num(value)
    if v is None or v < 1:
        return None
    return int(v)


def _bucket_005(x: float, lo: float, hi: float) -> float:
    """Floor to a 0.05-wide bucket, clamped into [lo, hi], canonical 2-dp float."""
    # 1e-6 absorbs float noise from arithmetic like 1-share right at a
    # bucket boundary (0.199999... must land in the 0.20 bucket).
    b = math.floor(x / 0.05 + 1e-6) * 0.05
    return round(min(max(b, lo), hi), 2)


def _rounds_left_bucket(rounds_left: int | None) -> str:
    # Every dataset config has a horizon (mr 12/99), so no "inf" bucket
    # exists; live unlimited-horizon turns behave like far-from-deadline
    # ones — map them to "4+" instead of a key that can never match.
    if rounds_left is None:
        return "4+"
    if rounds_left <= 1:
        return "1"
    if rounds_left <= 3:
        return "2-3"
    return "4+"


# ------------------------------------------------------------ config keys

def config_key_bargaining(state: dict) -> str | None:
    """Live bargaining game_state -> dataset config key (None if unbuildable,
    e.g. opponent's delta hidden under incomplete information)."""
    try:
        money = _num(state.get("money_to_divide"))
        d1 = _num(state.get("delta_1"))
        d2 = _num(state.get("delta_2"))
        if money is None or d1 is None or d2 is None:
            return None
        max_rounds = _int_or_none(state.get("max_rounds"))
        # The dataset has no unlimited-horizon configs; its max_rounds=99
        # games are de-facto unlimited (nothing reaches round 99), so live
        # no-limit games alias to those pools instead of never matching.
        if max_rounds is None:
            max_rounds, horizon_known = 99, True
        else:
            horizon_known = bool(state.get("horizon_known", True))
        return _dumps({
            "money_to_divide": money,
            "delta_1": d1,
            "delta_2": d2,
            "max_rounds": max_rounds,
            "horizon_known": horizon_known,
            "messages_allowed": bool(state.get("messages_allowed", True)),
            "complete_information": bool(state.get("complete_information", False)),
        })
    except Exception:  # noqa: BLE001 — lookups must never raise
        return None


def config_key_negotiation(
    state: dict, my_role: str, my_value: float | None
) -> tuple[str | None, str | None]:
    """Live negotiation game_state -> (full_config_key_or_None, role_key).

    The full key needs BOTH values (hidden under incomplete information);
    the role key marginalizes the opponent's value out and always builds
    as long as my own value is known.
    """
    try:
        max_rounds = _int_or_none(state.get("max_rounds"))
        # Alias live unlimited-horizon games to the dataset's de-facto
        # unlimited pools (max_rounds=99) — no null-horizon keys exist there.
        if max_rounds is None:
            max_rounds, horizon_known = 99, True
        else:
            horizon_known = bool(state.get("horizon_known", True))
        common = {
            "max_rounds": max_rounds,
            "horizon_known": horizon_known,
            "messages_allowed": bool(state.get("messages_allowed", True)),
            "complete_information": bool(state.get("complete_information", False)),
        }
        # Map player values onto seller/buyer via the role fields.
        values: dict[str, float | None] = {"seller": None, "buyer": None}
        for player in ("player_1", "player_2"):
            role = state.get(f"{player}_role")
            if role in ("seller", "buyer"):
                values[role] = _num(state.get(f"{player}_value"))
        full_key = None
        if values["seller"] is not None and values["buyer"] is not None:
            full_key = _dumps({
                "seller_value": values["seller"],
                "buyer_value": values["buyer"],
                **common,
            })
        role_key = None
        my_val = _num(my_value)
        if my_role in ("seller", "buyer") and my_val is not None:
            role_key = _dumps({"role": my_role, "my_value": my_val, **common})
        return full_key, role_key
    except Exception:  # noqa: BLE001
        return None, None


def config_key_persuasion(state: dict) -> str | None:
    """Live persuasion game_state -> dataset config key (v, u absolute)."""
    try:
        price = _num(state.get("product_price"))
        p = _num(state.get("p"))
        v = _num(state.get("v"))
        u = _num(state.get("u"))
        total_rounds = _int_or_none(state.get("total_rounds"))
        if None in (price, p, v, u) or total_rounds is None:
            return None
        return _dumps({
            "product_price": price,
            "p": p,
            "v": v,
            "u": u,
            "total_rounds": total_rounds,
            "seller_message_type": str(state.get("seller_message_type", "text")),
        })
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------ Targets

class Targets:
    """Loaded targets.json. All lookups return None on any miss/malformation."""

    def __init__(self, data: dict | None = None):
        data = data if isinstance(data, dict) else {}
        grid = data.get("quantile_grid")
        self.grid: list[float] = (
            [float(g) for g in grid]
            if isinstance(grid, list) and len(grid) >= 2
            else list(DEFAULT_GRID)
        )
        self.families: dict[str, dict] = {
            fam: (data.get(fam) if isinstance(data.get(fam), dict) else {})
            for fam in ("bargaining", "negotiation", "persuasion")
        }
        self.neg_by_role: dict = (
            data.get("neg_by_role") if isinstance(data.get("neg_by_role"), dict) else {}
        )
        self.barg_accept: dict = (
            data.get("barg_accept") if isinstance(data.get("barg_accept"), dict) else {}
        )
        self.neg_accept: dict = (
            data.get("neg_accept") if isinstance(data.get("neg_accept"), dict) else {}
        )
        self.models: dict = data.get("models") if isinstance(data.get("models"), dict) else {}
        # Marginal negotiation accept counts pooled over rel buckets, keyed
        # (role, rounds_left_bucket, human) — for opponents whose value we
        # cannot see (their rel bucket is unknowable live).
        self._neg_marginal: dict[tuple, list[float]] = {}
        for key, counts in self.neg_accept.items():
            try:
                kd = json.loads(key)
                mk = (kd.get("role"), kd.get("rounds_left_bucket"), bool(kd.get("human")))
                acc = self._neg_marginal.setdefault(mk, [0.0, 0.0])
                acc[0] += float(counts[0])
                acc[1] += float(counts[1])
            except Exception:  # noqa: BLE001
                continue

    @classmethod
    def null(cls) -> "Targets":
        return cls({})

    @property
    def is_null(self) -> bool:
        return not any(self.families.values()) and not self.neg_by_role

    # ---------------- payoff quantiles / percentiles

    def _quantile_row(self, family: str, config_key: str | None, role: str) -> list[float] | None:
        if not config_key:
            return None
        entry = self.families.get(family, {}).get(config_key)
        row: Any = None
        if isinstance(entry, dict):
            pq = entry.get("payoff_q")
            if isinstance(pq, dict):
                row = pq.get(role)
            elif isinstance(pq, list):
                row = pq
        elif family == "negotiation":
            by_role = self.neg_by_role.get(config_key)
            if isinstance(by_role, dict):
                row = by_role.get("payoff_q")
        if not isinstance(row, list) or len(row) != len(self.grid):
            return None
        try:
            vals = [float(v) for v in row]
        except (TypeError, ValueError):
            return None
        return vals

    def payoff_quantile(
        self, family: str, config_key: str | None, role: str, q: float
    ) -> float | None:
        """The pool payoff at quantile q (linear interpolation on the grid)."""
        vals = self._quantile_row(family, config_key, role)
        if vals is None:
            return None
        grid = self.grid
        q = min(max(float(q), grid[0]), grid[-1])
        for i in range(len(grid) - 1):
            if grid[i] <= q <= grid[i + 1]:
                span = grid[i + 1] - grid[i]
                frac = 0.0 if span <= 0 else (q - grid[i]) / span
                return vals[i] + (vals[i + 1] - vals[i]) * frac
        return vals[-1]

    def payoff_percentile(
        self, family: str, config_key: str | None, role: str, payoff: float
    ) -> float | None:
        """Where `payoff` sits vs the pool (realized payoffs incl. no-deal
        zeros). Monotone in payoff, clamped to [0.02, 0.98]."""
        vals = self._quantile_row(family, config_key, role)
        if vals is None:
            return None
        try:
            x = float(payoff)
        except (TypeError, ValueError):
            return None
        grid = self.grid
        if x < vals[0]:
            return 0.02
        if x > vals[-1]:
            return 0.98
        i = bisect_right(vals, x) - 1  # rightmost index with vals[i] <= x
        if i >= len(vals) - 1:
            pct = grid[-1]
        elif x <= vals[i]:
            pct = grid[i]
        else:
            span = vals[i + 1] - vals[i]
            frac = 0.0 if span <= 0 else (x - vals[i]) / span
            pct = grid[i] + (grid[i + 1] - grid[i]) * frac
        return min(max(pct, 0.02), 0.98)

    def deal_rate(self, family: str, config_key: str | None) -> float | None:
        if not config_key:
            return None
        entry = self.families.get(family, {}).get(config_key)
        if not isinstance(entry, dict) and family == "negotiation":
            entry = self.neg_by_role.get(config_key)
        if isinstance(entry, dict):
            return _num(entry.get("deal_rate"))
        return None

    # ---------------- empirical accept curves

    @staticmethod
    def _counts_ok(counts: Any) -> tuple[float, float] | None:
        try:
            n, k = float(counts[0]), float(counts[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return None
        if n < MIN_BUCKET_N or n <= 0:
            return None
        return n, k

    def barg_accept_prob(
        self, share_to_responder: float, rounds_left: int | None, human: bool
    ) -> float | None:
        """P(field responder accepts | share offered TO them). Backs off the
        human split to the pooled bucket when the split is too thin."""
        try:
            share_bucket = _bucket_005(float(share_to_responder), 0.0, 0.95)
        except (TypeError, ValueError):
            return None
        rl = _rounds_left_bucket(rounds_left)
        exact = self.barg_accept.get(_dumps(
            {"share_bucket": share_bucket, "rounds_left_bucket": rl, "human": bool(human)}
        ))
        ok = self._counts_ok(exact)
        if ok is not None:
            return ok[1] / ok[0]
        # Pool both human flags for this share/round bucket.
        total_n = total_k = 0.0
        for flag in (True, False):
            counts = self.barg_accept.get(_dumps(
                {"share_bucket": share_bucket, "rounds_left_bucket": rl, "human": flag}
            ))
            try:
                total_n += float(counts[0])
                total_k += float(counts[1])
            except (TypeError, ValueError, IndexError):
                continue
        if total_n >= MIN_BUCKET_N:
            return total_k / total_n
        return None

    def neg_accept_prob(
        self,
        rel_price: float | None,
        responder_role: str,
        rounds_left: int | None,
        human: bool,
    ) -> float | None:
        """P(field responder accepts | price / responder's value).

        rel_price=None: pooled over rel buckets (weak prior for hidden-value
        opponents; constant in price, so it can only break ties)."""
        rl = _rounds_left_bucket(rounds_left)
        if rel_price is None:
            ok = self._counts_ok(self._neg_marginal.get((responder_role, rl, bool(human))))
            if ok is not None:
                return ok[1] / ok[0]
            pooled = [0.0, 0.0]
            for flag in (True, False):
                c = self._neg_marginal.get((responder_role, rl, flag))
                if c:
                    pooled[0] += c[0]
                    pooled[1] += c[1]
            if pooled[0] >= MIN_BUCKET_N:
                return pooled[1] / pooled[0]
            return None
        try:
            rel_bucket = _bucket_005(float(rel_price), 0.0, 1.95)
        except (TypeError, ValueError):
            return None

        def _key(flag: bool) -> str:
            return _dumps({
                "rel_bucket": rel_bucket,
                "role": responder_role,
                "rounds_left_bucket": rl,
                "human": flag,
            })

        ok = self._counts_ok(self.neg_accept.get(_key(bool(human))))
        if ok is not None:
            return ok[1] / ok[0]
        total_n = total_k = 0.0
        for flag in (True, False):
            counts = self.neg_accept.get(_key(flag))
            try:
                total_n += float(counts[0])
                total_k += float(counts[1])
            except (TypeError, ValueError, IndexError):
                continue
        if total_n >= MIN_BUCKET_N:
            return total_k / total_n
        return None


# ------------------------------------------------------------ singleton

LIVE_PATH = Path(__file__).resolve().parents[3] / "data" / "live_targets.json"
LIVE_POOL_N = 40        # payoffs needed before a live pool overrides the prior
LIVE_BUCKET_N = 200     # observations needed before a live accept bucket does


def _merge_live(data: dict, live_path: Path) -> dict:
    """Overlay pools and accept curves measured on the LIVE field.

    The dataset key was generated by 2024-vintage models and is badly
    optimistic about this field: it prices a bargaining give of 0.50 at 68.6%
    acceptance where we realize 2.8%, and a negotiation price at the
    responder's own value at 20-31% where we realize 0.2-2.4%. An optimizer
    fed those numbers systematically over-prices.

    Live cells only override where they are well sampled; everywhere else the
    dataset still provides the shape (it covers offers we never make, so
    replacing it wholesale would let our own policy define its own evidence).
    Never raises: any problem leaves the dataset key untouched.
    """
    try:
        with open(live_path, encoding="utf-8") as fh:
            live = json.load(fh)
    except Exception:  # noqa: BLE001 — live overlay is strictly optional
        return data

    for fam in ("bargaining", "negotiation", "persuasion"):
        cells = live.get(fam) or {}
        base = data.setdefault(fam, {})
        for key, rec in cells.items():
            if not isinstance(rec, dict):
                continue
            slot = base.setdefault(key, {})
            pq = slot.setdefault("payoff_q", {})
            for role, grid in (rec.get("payoff_q") or {}).items():
                if isinstance(grid, list) and grid:
                    pq[role] = grid

    for curve in ("barg_accept", "neg_accept"):
        cells = live.get(curve) or {}
        base = data.setdefault(curve, {})
        for key, counts in cells.items():
            if (isinstance(counts, list) and len(counts) == 2
                    and counts[0] >= LIVE_BUCKET_N):
                base[key] = counts
    return data


def load_targets(path: str | Path | None = None, live: bool = True) -> Targets:
    """Load targets.json; a missing or corrupt file yields the Null object."""
    p = Path(path) if path is not None else DEFAULT_PATH
    try:
        with open(p, encoding="utf-8") as fh:
            data = json.load(fh)
        if live:
            data = _merge_live(data, LIVE_PATH)
        t = Targets(data)
        if t.is_null:
            logger.warning("targets file %s parsed but holds no pools", p)
        return t
    except FileNotFoundError:
        logger.info("targets file %s not found; using pure-theory fallback", p)
        return Targets.null()
    except Exception as e:  # noqa: BLE001 — never raise into a live turn
        logger.warning("targets file %s unreadable (%s); pure-theory fallback", p, e)
        return Targets.null()


_TARGETS: Targets | None = None


def get_targets() -> Targets:
    """Cached module-level singleton. Never raises."""
    global _TARGETS
    if _TARGETS is None:
        _TARGETS = load_targets()
    return _TARGETS


def set_targets(targets: Targets | None) -> None:
    """Override (or reset with None) the singleton — tests and reloads."""
    global _TARGETS
    _TARGETS = targets
