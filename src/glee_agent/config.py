"""Runtime configuration: env vars + per-agent strategy knobs.

Every tunable that an A/B arm might vary lives in Knobs, so two agents can run
the same code with different .env-selected knob sets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Knobs:
    """Strategy tunables. Defaults are the champion settings; A/B arms
    override individual fields via GLEE_KNOB_* env vars."""

    # Bargaining
    barg_anchor_agent: float = 0.80      # opening share vs agent/hidden opponents
    barg_anchor_human: float = 0.65      # opening share vs disclosed humans
    barg_floor_agent: float = 0.55       # lowest share we concede to (agents)
    barg_floor_human: float = 0.58       # lowest share we concede to (humans)
    barg_beta: float = 2.5               # Boulware exponent (higher = concede later)
    barg_final_round_give: float = 0.30  # share offered to responder in final round
    barg_accept_great: float = 0.65      # grab any offer at/above this share immediately
    barg_cont_realism: float = 0.85      # P(opponent accepts my next aggressive offer)
    barg_accept_pct: float = 0.70        # accept when payoff percentile vs pool >= this
    barg_patience_edge: float = 0.04     # |my_delta - opp_delta| >= this splits the regimes
    barg_adv_hold: float = 0.75          # advantage-regime acceptance threshold cap
    barg_dis_accept: float = 0.48        # disadvantage regime: accept any share >= this
    barg_book_gain: float = 0.0          # per-opponent anchor shift = (0.5 - our measured share
                                         # vs this name) x this. 0 = off. Our head-to-head share
                                         # spans 0.407-0.581 by opponent, ~5x any global knob
    barg_book_accept_gain: float = 0.0   # per-opponent ACCEPT-threshold shift = (0.5 - our
                                         # measured share vs this name) x this. The anchor
                                         # version moved nothing: 61% of bargaining games end by
                                         # round 2, so the binding decision is what we ACCEPT,
                                         # not what we open with. Per-opponent skill gaps are
                                         # real and large -- 0.16 spread even after adjusting
                                         # for config mix -- so the lever exists; it just has to
                                         # be applied where the outcome is actually decided
    barg_book_min_n: int = 60            # games vs a name before its record steers real money
    barg_dis_anchor: float = 0.58        # disadvantage regime opening share. Our pot melts
                                         # faster than theirs, so the whole point is to CLOSE:
                                         # measured live, our 0.575 offers here are accepted
                                         # 10.3% of the time and the regime realizes only
                                         # 0.325 of pot. Pricing to close should beat pricing
                                         # to win -- under test on an arm.
    barg_drip: float = 0.01              # max unreciprocated concession per offer (0 in advantage)

    # Negotiation
    neg_anchor_markup: float = 0.9       # how far past own value we anchor (fraction of value)
    neg_beta: float = 2.5                # Boulware exponent
    neg_max_planned_rounds: int = 12     # concession schedule length for long/unlimited games
    neg_min_margin_frac: float = 0.02    # final margin above reservation (fraction of value)
    neg_accept_pct: float = 0.70         # accept when payoff percentile vs pool >= this
    neg_accept_factor: float = 0.9       # accept when payoff >= this x my planned counter's payoff
    neg_anchor_markup_buyer: float | None = None  # buyer-side markup override (None = symmetric)
    neg_ci_floor_frac: float = 0.0       # complete info only: clamp the anchor feasible (opponent
                                         # can profitably accept) and floor concession at
                                         # value +/- frac*surplus instead of own reservation.
                                         # 0 = off; A/B arm runs 0.4
    neg_ci_anchor_frac: float = 0.80     # complete info: opening leaves the opponent at least
                                         # (1-this) of the surplus. 0.95 was functionally
                                         # infeasible -- live acceptance at that level is ~0%,
                                         # while leaving them 10-20% is accepted ~22%
    neg_reciprocal: bool = True          # concede at most what the opponent just conceded
                                         # (plus a drip). Their acceptance probability AND
                                         # their own concession both fall in our concession
                                         # speed, so a time-based schedule teaches them to wait
    neg_drip: float = 0.01               # unreciprocated concession per offer, as a fraction
                                         # of the anchor-to-floor range
    neg_never_walk: bool = False         # unlimited-horizon: walking pays exactly 0, the same
                                         # as running out of road, but forecloses any chance the
                                         # opponent caves. Measured: WE initiate 64% of walkaways,
                                         # all in unlimited games, and 73% of negotiation is
                                         # incomplete-info where the median payoff is already 0
    neg_stall_accept: int = 0            # unlimited-horizon: accept ANY positive profit from this
                                         # round on (0 = use neg_max_planned_rounds+4). Under
                                         # percentile scoring a small positive beats a zero by a
                                         # wide margin in no-deal-heavy pools
    neg_traj_pct_gate: float = 0.0       # trajectory-accept additionally requires pool pct >= this
                                         # (0 = off; A/B arm 0.45) — blocks the round-12 Boulware
                                         # floor collapse from accepting near-reservation crumbs

    # Persuasion
    pers_buyer_surplus: float = 0.10     # seller: leave the buyer at least this fraction of
                                         # price in expected surplus per purchase. KG's x* is
                                         # the buyer's INDIFFERENCE point -- at it they earn
                                         # exactly zero and a real buyer stops buying. Measured
                                         # live: in configs where p*v == price our seller ran a
                                         # lie rate of 1.000 and earned 0.39-0.56 of max, while
                                         # the field ran 0.28-0.43 and earned 0.70-0.77
    pers_blind_lie: float = 1.0          # blind seller (v hidden -- HALF our seller games):
                                         # rate of recommending a low-quality unit. 1.0 = pool,
                                         # the historical behaviour. The KG rate is uncomputable
                                         # without v, but pooling walks into the same
                                         # indifference trap that cost us 0.2-0.3 of max revenue
                                         # in the visible thin configs
    pers_kg_haircut: float = 0.8         # multiply KG x* by this (seller lying rate safety)
    pers_kg_haircut_human: float = 0.4   # vs disclosed human buyers
    pers_explore_frac: float = 0.33      # fraction of rounds with UCB exploration (buyer)
    pers_buy_margin: float = 0.02        # required edge on THIN margins: buy iff P(high)*v > price*(1+margin)
    pers_buy_margin_wide: float = -0.05  # tolerance on WIDE margins (headroom >= 0.5): field
                                         # buyers profit at the knife-edge; a passive 0 sits
                                         # below the pool median, so accept slight prior slack
    pers_prior_lie: float = 0.25         # prior P(recommend | low) before we observe anything.
                                         # MEASURED from ~900k live rounds of field sellers:
                                         # implied lie rate 0.26 (p=1/3), 0.29 (p=0.5), 0.19
                                         # (p=0.8) -- the field lies well BELOW the KG-rational
                                         # x*, so anchoring the prior on x* starts us far too
                                         # cynical and we refuse profitable recommendations.
    pers_forget: float = 0.9             # per-round decay on old observations (seller behaviour
                                         # shifts: trust repair early, pooling at the horizon)
    pers_probe_budget: float = 0.15      # max share of max-attainable surplus spent on
                                         # information-gathering (losing) purchases

    # LLM layer
    llm_enabled: bool = True
    llm_timeout_s: float = 8.0           # llm/client.py enforces this budget
    llm_breaker_failures: int = 3        # consecutive failures to trip the circuit breaker
    llm_breaker_cooldown_s: float = 600.0


def _knobs_from_env() -> Knobs:
    overrides = {}
    for f in Knobs.__dataclass_fields__.values():
        env_name = f"GLEE_KNOB_{f.name.upper()}"
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        if f.type == "bool" or isinstance(f.default, bool):
            overrides[f.name] = raw.strip().lower() in ("1", "true", "yes", "on")
        elif isinstance(f.default, int):
            overrides[f.name] = int(raw)
        else:
            overrides[f.name] = float(raw)
    knobs = Knobs(**overrides)
    if not _env_bool("GLEE_AGENT_LLM_ENABLED", True):
        knobs = Knobs(**{**overrides, "llm_enabled": False})
    return knobs


@dataclass(frozen=True)
class Settings:
    glee_api_key: str
    agent_label: str                     # "main" | "test_a" | "test_b" — selects key + log file names
    concurrency: int
    llm_api_base: str
    llm_api_key: str
    llm_model: str
    knobs: Knobs = field(default_factory=Knobs)


def load_settings(agent_label: str = "main") -> Settings:
    key_env = {
        "main": "GLEE_API_KEY_MAIN",
        "test_a": "GLEE_API_KEY_TEST_A",
        "test_b": "GLEE_API_KEY_TEST_B",
        "test_c": "GLEE_API_KEY_TEST_C",
        "test_d": "GLEE_API_KEY_TEST_D",
    }.get(agent_label)
    if key_env is None:
        raise ValueError(f"Unknown agent label: {agent_label!r}")
    api_key = os.environ.get(key_env, "").strip()
    if not api_key:
        raise ValueError(f"{key_env} is not set in .env")

    return Settings(
        glee_api_key=api_key,
        agent_label=agent_label,
        concurrency=int(_env_float("GLEE_AGENT_CONCURRENCY", 8)),
        llm_api_base=os.environ.get("LLM_API_BASE", "https://yunwu.ai/v1"),
        llm_api_key=os.environ.get("LLM_API_KEY", ""),
        llm_model=os.environ.get("LLM_MODEL", "deepseek-chat"),
        knobs=_knobs_from_env(),
    )
