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

    # Negotiation
    neg_anchor_markup: float = 0.9       # how far past own value we anchor (fraction of value)
    neg_beta: float = 2.5                # Boulware exponent
    neg_max_planned_rounds: int = 12     # concession schedule length for long/unlimited games
    neg_min_margin_frac: float = 0.02    # final margin above reservation (fraction of value)
    neg_accept_pct: float = 0.70         # accept when payoff percentile vs pool >= this

    # Persuasion
    pers_kg_haircut: float = 0.8         # multiply KG x* by this (seller lying rate safety)
    pers_kg_haircut_human: float = 0.4   # vs disclosed human buyers
    pers_explore_frac: float = 0.33      # fraction of rounds with UCB exploration (buyer)
    pers_buy_margin: float = 0.02        # required edge: buy iff P(high)*v > price*(1+margin)

    # LLM layer
    llm_enabled: bool = True
    llm_timeout_s: float = 15.0
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
