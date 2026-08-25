"""Read-only reports for the live strategy canaries.

The report deliberately reads JSONL rather than ``agent.db``.  That preserves
turn-level assignment evidence, includes terminal results written by the
reaper, and avoids making telemetry ingestion part of an experiment readout.

Enrollment is intentionally strict: a game must have no turn before the
experiment cut, and its first turn at/after the cut must be round 1 with an
empty embedded history.  Repeated polls are collapsed by
``(agent, game_id, round, phase)`` and the latest game-over result wins.

Usage::

    .venv/bin/python scripts/canary_report.py
    .venv/bin/python scripts/canary_report.py --json
    .venv/bin/python scripts/canary_report.py --experiment neg_terminal_close
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glee_agent.config import Knobs  # noqa: E402
from glee_agent.dispatcher import FAMILIES  # noqa: E402
from glee_agent.guard import guard  # noqa: E402
from glee_agent.schema import parse_game  # noqa: E402
from glee_agent.theory.targets import (  # noqa: E402
    DEFAULT_PATH as TARGETS_PATH,
    config_key_negotiation,
    load_targets,
)


# These are the champion knobs shared by all four live processes at the
# experiment cuts below.  Only the named canary knob differs within an
# experiment.  Keeping this explicit makes an old cohort reproducible after
# dataclass defaults move on.
LIVE_BASELINE = {
    "neg_anchor_markup": 0.50,
    "neg_beta": 1.80,
    "barg_beta": 1.50,
    "barg_accept_great": 0.60,
    "neg_traj_pct_gate": 0.45,
    "neg_max_planned_rounds": 22,
    "llm_enabled": False,
    "barg_anchor_agent": 0.58,
    "barg_accept_pct": 0.60,
    "neg_ci_floor_frac": 0.55,
    "pers_buyer_surplus": 0.20,
}


@dataclass(frozen=True)
class Experiment:
    name: str
    family: str
    cutoff: float
    treatment_agents: tuple[str, ...]
    control_agents: tuple[str, ...]
    knob: str
    treatment_value: Any
    control_value: Any

    @property
    def agents(self) -> tuple[str, ...]:
        return self.treatment_agents + self.control_agents

    def variant_for(self, agent: str) -> str:
        return "treatment" if agent in self.treatment_agents else "control"


EXPERIMENTS = (
    Experiment(
        "barg_dis_anchor",
        "bargaining",
        1787671708.0,
        ("test_b",),
        ("main", "test_c"),
        "barg_dis_anchor",
        0.50,
        0.58,
    ),
    Experiment(
        "neg_terminal_close",
        "negotiation",
        1787674193.0,
        ("test_a",),
        ("main", "test_b", "test_c"),
        "neg_terminal_close",
        True,
        False,
    ),
    Experiment(
        "pers_blind_lie",
        "persuasion",
        1787674193.0,
        ("test_a", "test_b"),
        ("main", "test_c"),
        "pers_blind_lie",
        0.40,
        1.0,
    ),
)


# Frozen before reading outcomes beyond the pre-gate pilot checkpoint.  The
# strategy trigger is hidden opponent *value* (complete_information=False),
# not hidden opponent identity.  A pre-cut reconstruction therefore freezes
# all eight identity x own-value cells.  P(role=buyer)=1 because the historical
# opportunity extraction was specifically buyer round-9 decision counters.
_NEG_TERMINAL_HISTORICAL_CELLS = (
    # opponent identity, own-value grid, resolved, direct, compatibility rate
    ("agent", "80", 237, 0, 0.0),
    ("agent", "100", 195, 18, 0.25467870697552675),
    ("agent", "120", 168, 20, 0.5082449941107184),
    ("agent", "150", 92, 10, 0.7539589059023688),
    ("hidden", "80", 244, 0, 0.0),
    ("hidden", "100", 207, 18, 0.25467870697552675),
    ("hidden", "120", 156, 19, 0.5082449941107184),
    ("hidden", "150", 83, 18, 0.7539589059023688),
)
_NEG_TERMINAL_REFERENCE_CELLS = tuple(
    {
        "role": "buyer",
        "own_value_grid": value,
        "phase": "decision",
        "horizon": "finite",
        "max_rounds": "10",
        "opponent_type": opponent_type,
        "complete_information": False,
        "weight": resolved / 1382,
        "compatibility_rate": compatibility,
        "historical_resolved": resolved,
        "historical_direct_converted": direct,
    }
    for opponent_type, value, resolved, direct, compatibility
    in _NEG_TERMINAL_HISTORICAL_CELLS
)

_NEG_TERMINAL_COMPATIBILITY_RATE = 0.28870841304554923
_Z_ONE_SIDED_95 = 1.6448536269514722
_Z_ONE_SIDED_90 = 1.2815515655446004
_NEG_TARGET_ARTIFACT = {
    "path": "data/targets.json",
    "sha256": "1d24a579ca2b611e3b30af4ddf7af5b84ad13e7198fa55b93a2f5e6617e65e25",
    "bytes": 642520,
    "git_commit": "cb6a7eefff4451baf44e91e293fbcff187846d86",
    "published_ts": 1787673703.0,
}
_NEG_STATIC_ASSIGNMENT = {
    "epoch_id": "static:neg-terminal-close:1787674193",
    "start": 1787674193.0,
    "treatment_agents": ("test_a",),
    "control_agents": ("main", "test_b", "test_c"),
}

NEG_TERMINAL_GATE_DESIGN = {
    "version": "hidden-value-terminal-close-v2-frozen-2026-08-25",
    "frozen_before_subsequent_outcomes": True,
    "pilot_checkpoint": {
        "treatment": {"direct_converted": 0, "direct_resolved": 2},
        "control": {"direct_converted": 1, "direct_resolved": 6},
        "used_to_tune_thresholds": False,
        "note": (
            "Pre-gate pilot was T 0/2 versus C 1/6; later outcomes were not "
            "used to set gates."
        ),
        "analysis_window": (
            "The report retains all strictly enrolled games from the experiment "
            "cutoff, including the disclosed pilot."
        ),
    },
    "estimand": {
        "unit": "strictly enrolled game at first exact action-level divergence",
        "joint_strata": [
            "role",
            "own_value_grid",
            "phase",
            "horizon",
            "max_rounds",
            "opponent_type",
            "complete_information",
        ],
        "role_weight": {"buyer": 1.0},
        "missing_support_policy": "report and block; never renormalize fixed weights",
        "agent_epoch_policy": (
            "Use the frozen timestamped initial assignment, then runtime manifests; "
            "a later treatment assignment never reclassifies earlier control rows."
        ),
        "causal_status_policy": (
            "Fixed-label evidence can reach screen_pass only. Causal promote "
            "requires two labels observed with meaningful treatment and control "
            "support across a manifest-backed switchback."
        ),
    },
    "reference_cells": [dict(cell) for cell in _NEG_TERMINAL_REFERENCE_CELLS],
    "static_assignment": {
        **_NEG_STATIC_ASSIGNMENT,
        "treatment_agents": list(_NEG_STATIC_ASSIGNMENT["treatment_agents"]),
        "control_agents": list(_NEG_STATIC_ASSIGNMENT["control_agents"]),
    },
    "payoff_target_artifact": dict(_NEG_TARGET_ARTIFACT),
    "historical_reference": {
        "window": "24h before cutoff 1787672701",
        "window_start": 1787586301.0,
        "window_end_exclusive": 1787672701.0,
        "extraction": (
            "latest raw buyer round-9 RejectOffer counter per agent/game; "
            "max_rounds=10; complete_information=false; terminal rollup resolved"
        ),
        "identity_counts_source": (
            "pre-cut raw turn logs plus immutable agent.db terminal lookup; "
            "1418 eligible, 1382 resolved, 36 censored"
        ),
        "resolved": 1382,
        "direct_converted": 103,
        "direct_conversion_rate": 103 / 1382,
        "recent_6h_direct_conversion_rate": 31 / 341,
        "compatible_opportunity_rate": _NEG_TERMINAL_COMPATIBILITY_RATE,
        "normalized_payoff_mean": 0.00606393,
        "normalized_payoff_sd": 0.033679,
        "payoff_percentile_mean": 0.722626,
        "payoff_percentile_sd": 0.22869,
    },
    "thresholds": {
        "promotion_sample": {"treatment": 340, "control": 1020},
        "promotion_per_joint_cell": {"treatment": 15, "control": 45},
        "direct_uplift_min": 0.060,
        "direct_uplift_one_sided_95_lower_min": 0.0,
        "normalized_payoff_noninferiority_margin": -0.005,
        "payoff_percentile_noninferiority_margin": -0.050,
        "treatment_agent_blocks_required": 2,
        "treatment_epoch_block_min": {
            "total": 60,
            "per_own_value_across_identity": 15,
            "per_opponent_identity": 20,
        },
        "switchback_labels_required": 2,
        "unsupported_slice": {
            "per_cell_sample": {"treatment": 15, "control": 45},
            "direct_noninferiority_margin": -0.050,
            "normalized_payoff_noninferiority_margin": -0.005,
            "payoff_percentile_noninferiority_margin": -0.050,
        },
        "compatibility_efficiency_diagnostic_target": 0.45,
        "interim_1": {"treatment": 50, "control": 150},
        "interim_2": {"treatment": 100, "control": 300},
        "interim_zero_conversion_rollback": True,
        "interim_normalized_payoff_rollback": -0.010,
        "interim_2_treatment_direct_futility": 0.050,
        "interim_2_conditional_power_futility": None,
        "treatment_error_rate_excess_max": 0.010,
        "affected_censor_rate_excess_max": 0.030,
    },
    "inference": {
        "confidence": "one-sided 95% promotion bounds; one-sided 90% futility upper bound",
        "binary_se": "unpooled fixed-weight normal contrast with Jeffreys cell variance",
        "continuous_se": "unpooled fixed-weight normal contrast with sample cell variance",
        "conditional_power": (
            "legacy approximation retained as diagnostic-only; it is not a "
            "rollback or promotion criterion"
        ),
        "payoff_percentile_pool": (
            "pinned data/targets.json loaded with live=False; mutable live targets "
            "and the get_targets singleton are excluded"
        ),
    },
}


def _knobs(experiment: Experiment, variant: str) -> Knobs:
    base = replace(Knobs(), **LIVE_BASELINE)
    value = (
        experiment.treatment_value
        if variant == "treatment"
        else experiment.control_value
    )
    return replace(base, **{experiment.knob: value})


def replay_action(game: dict, knobs: Knobs) -> dict:
    """Replay the current family strategy and validity guard without logging."""
    view = parse_game(game)
    decide = FAMILIES.get(view.family)
    if decide is None:
        return {}
    proposed = decide(view, knobs)
    return guard(proposed, view)[0]


@dataclass
class Turn:
    agent: str
    gid: str
    family: str
    ts: float
    round: int
    phase: str
    game: dict
    action: dict
    corrections: list
    error: Any


@dataclass
class ResultEvent:
    agent: str
    gid: str
    ts: float
    valid: Any
    game_over: bool
    error: Any
    result: dict


@dataclass
class RuntimeEvent:
    agent: str
    ts: float
    pid: int | None
    knobs: dict
    git_head: str | None
    strategy_sha256: str | None


@dataclass
class ParsedRecords:
    turns: dict[tuple[str, str, int, str], Turn]
    first_turns: dict[tuple[str, str], Turn]
    results: list[ResultEvent]
    terminals: dict[tuple[str, str], ResultEvent]
    duplicate_events: list[tuple[str, str, float]]
    gid_family: dict[tuple[str, str], str]
    runtimes: list[RuntimeEvent]


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _record_agent(record: dict) -> str:
    value = record.get("_agent", record.get("agent", ""))
    return value if isinstance(value, str) else ""


def parse_records(records: Iterable[dict]) -> ParsedRecords:
    turns: dict[tuple[str, str, int, str], Turn] = {}
    first_turns: dict[tuple[str, str], Turn] = {}
    results: list[ResultEvent] = []
    terminals: dict[tuple[str, str], ResultEvent] = {}
    runtimes: list[RuntimeEvent] = []
    turn_occurrences: dict[tuple[str, str, int, str], list[float]] = defaultdict(list)
    gid_family: dict[tuple[str, str], str] = {}

    for record in records:
        if not isinstance(record, dict):
            continue
        kind = record.get("type")
        agent = _record_agent(record)
        ts = _as_float(record.get("ts"))
        if not agent or ts is None:
            continue
        if kind == "runtime":
            hashes = record.get("content_hashes")
            hashes = hashes if isinstance(hashes, dict) else {}
            strategy = hashes.get("strategy_python")
            strategy = strategy if isinstance(strategy, dict) else {}
            pid_value = _as_float(record.get("pid"))
            runtimes.append(
                RuntimeEvent(
                    agent=agent,
                    ts=ts,
                    pid=int(pid_value) if pid_value is not None else None,
                    knobs=(
                        record.get("knobs")
                        if isinstance(record.get("knobs"), dict)
                        else {}
                    ),
                    git_head=(
                        record.get("git_head")
                        if isinstance(record.get("git_head"), str)
                        else None
                    ),
                    strategy_sha256=(
                        strategy.get("aggregate_sha256")
                        if isinstance(strategy.get("aggregate_sha256"), str)
                        else None
                    ),
                )
            )
        elif kind == "turn":
            game = record.get("game")
            if not isinstance(game, dict):
                continue
            gid = game.get("game_id")
            family = game.get("game_family")
            state = game.get("game_state")
            if not isinstance(gid, str) or not isinstance(family, str):
                continue
            state = state if isinstance(state, dict) else {}
            rnd_value = _as_float(state.get("round"))
            rnd = int(rnd_value) if rnd_value is not None else 1
            phase = game.get("phase", state.get("phase", ""))
            phase = phase if isinstance(phase, str) else ""
            turn = Turn(
                agent=agent,
                gid=gid,
                family=family,
                ts=ts,
                round=rnd,
                phase=phase,
                game=game,
                action=record.get("action") if isinstance(record.get("action"), dict) else {},
                corrections=(
                    record.get("corrections")
                    if isinstance(record.get("corrections"), list)
                    else []
                ),
                error=record.get("error"),
            )
            game_key = (agent, gid)
            old_first = first_turns.get(game_key)
            if old_first is None or turn.ts < old_first.ts:
                first_turns[game_key] = turn
            gid_family[game_key] = family
            key = (agent, gid, rnd, phase)
            turn_occurrences[key].append(ts)
            old = turns.get(key)
            if old is None or turn.ts >= old.ts:
                turns[key] = turn
        elif kind == "result":
            gid = record.get("game_id")
            if not isinstance(gid, str):
                continue
            event = ResultEvent(
                agent=agent,
                gid=gid,
                ts=ts,
                valid=record.get("valid"),
                game_over=bool(record.get("game_over")),
                error=record.get("error"),
                result=record.get("result") if isinstance(record.get("result"), dict) else {},
            )
            results.append(event)
            if event.game_over:
                key = (agent, gid)
                old = terminals.get(key)
                if old is None or event.ts >= old.ts:
                    terminals[key] = event

    duplicate_events: list[tuple[str, str, float]] = []
    for key, timestamps in turn_occurrences.items():
        agent, gid, _, _ = key
        family = gid_family.get((agent, gid), "")
        duplicate_events.extend(
            (agent, family, ts) for ts in sorted(timestamps)[1:]
        )

    return ParsedRecords(
        turns=turns,
        first_turns=first_turns,
        results=results,
        terminals=terminals,
        duplicate_events=duplicate_events,
        gid_family=gid_family,
        runtimes=sorted(runtimes, key=lambda event: (event.ts, event.agent)),
    )


def _history_empty(turn: Turn) -> bool:
    state = turn.game.get("game_state")
    return isinstance(state, dict) and state.get("history") == []


def _arm_empty() -> dict:
    return {
        "enrollment": {
            "enrolled": 0,
            "resolved": 0,
            "censored": 0,
            "terminal_reaped": 0,
            "excluded_pre_cut": 0,
            "excluded_partial": 0,
        },
        "health": {
            "turns": 0,
            "duplicate_turns": 0,
            "turn_errors": 0,
            "corrections": 0,
            "turns_with_corrections": 0,
            "result_events": 0,
            "invalid_results": 0,
            "result_errors": 0,
            "http_503": 0,
        },
        "routing": {
            "checked": 0,
            "assigned_matches": 0,
            "replay_errors": 0,
            "affected": 0,
            "affected_assigned_matches": 0,
            "affected_wrong_variant": 0,
            "affected_unknown": 0,
            "direction_violations": 0,
        },
    }


def _variant_summary() -> dict:
    return {
        "games": 0,
        "resolved": 0,
        "censored": 0,
        "affected_games": 0,
        "affected_turns": 0,
        "direction_violations": 0,
    }


def _action_equal(left: dict, right: dict) -> bool:
    return left == right


def _my_gain(action: dict, game: dict) -> float | None:
    player = game.get("your_player", "player_1")
    key = "alice_gain" if player == "player_1" else "bob_gain"
    value = _as_float(action.get(key))
    if value is not None:
        return value
    # The platform state uses player_* names while guarded actions use the
    # Alice/Bob aliases.  Accept both for synthetic fixtures and schema drift.
    return _as_float(action.get(f"{player}_gain"))


def _direction_violation(
    experiment: Experiment, turn: Turn, control: dict, treatment: dict
) -> bool:
    state = turn.game.get("game_state")
    state = state if isinstance(state, dict) else {}
    if experiment.name == "barg_dis_anchor":
        c_decision = control.get("decision")
        t_decision = treatment.get("decision")
        if c_decision is not None or t_decision is not None:
            # A lower disadvantage anchor is a close-fast policy.  Accepting
            # where the old anchor would reject is therefore in-direction;
            # rejecting where the old anchor would accept is the violation.
            return not (t_decision == "accept" and c_decision != "accept")
        c = _my_gain(control, turn.game)
        t = _my_gain(treatment, turn.game)
        return c is None or t is None or not t < c
    if experiment.name == "neg_terminal_close":
        c = _as_float(control.get("product_price"))
        t = _as_float(treatment.get("product_price"))
        if c is None or t is None:
            return True
        player = turn.game.get("your_player", "player_1")
        role = state.get(f"{player}_role", "seller" if player == "player_1" else "buyer")
        return not (t < c if role == "seller" else t > c)
    if experiment.name == "pers_blind_lie":
        return not (_recommendation(control) is True and _recommendation(treatment) is False)
    return False


def _recommendation(action: dict) -> bool | None:
    decision = action.get("decision")
    if isinstance(decision, str) and decision.lower() in ("yes", "no"):
        return decision.lower() == "yes"
    message = action.get("message")
    if not isinstance(message, str):
        return None
    text = message.lower().strip()
    negatives = ("don't", "do not", "skip", "pass", "avoid", "not this", "no ")
    return not any(word in text for word in negatives)


def _effective_offer_round(turn: Turn, action: dict) -> int | None:
    if turn.family == "bargaining":
        # Offer divergences act on this round; decision divergences act on
        # the incoming offer from this round (e.g. .50 accepts where .58
        # rejects).  Both are direct close opportunities.
        if turn.phase in ("offer", "decision"):
            return turn.round
    if turn.family == "negotiation":
        if turn.phase == "offer" and "product_price" in action:
            return turn.round
        if (
            turn.phase == "decision"
            and action.get("decision") == "RejectOffer"
            and "product_price" in action
        ):
            return turn.round + 1
    return None


def _horizon(turn: Turn) -> str:
    state = turn.game.get("game_state")
    state = state if isinstance(state, dict) else {}
    maximum = state.get("max_rounds")
    known = state.get("horizon_known", maximum is not None)
    return "finite" if known and maximum is not None else "unlimited"


def _max_rounds(turn: Turn) -> str:
    state = turn.game.get("game_state")
    state = state if isinstance(state, dict) else {}
    maximum = state.get("max_rounds")
    return str(maximum) if maximum is not None else "unlimited"


def _terminal_payoff(turn: Turn, terminal: ResultEvent) -> float | None:
    player = turn.game.get("your_player", "player_1")
    return _as_float(terminal.result.get(f"{player}_payoff"))


def _terminal_agreement(terminal: ResultEvent) -> bool:
    return terminal.result.get("outcome") in ("agreement", "completed")


def _round_number(value: Any) -> int | None:
    number = _as_float(value)
    if number is None or number < 0 or not number.is_integer():
        return None
    return int(number)


def _cell_key(cell: dict) -> str:
    return json.dumps(cell, sort_keys=True, separators=(",", ":"))


def _offer_cell(experiment: Experiment, turn: Turn, affected: dict) -> dict:
    state = turn.game.get("game_state")
    state = state if isinstance(state, dict) else {}
    player = turn.game.get("your_player", "player_1")
    opponent = turn.game.get("opponent")
    opponent = opponent if isinstance(opponent, dict) else {}
    common = {
        "horizon": _horizon(turn),
        "max_rounds": _max_rounds(turn),
        "phase": affected.get("phase", "unknown"),
        "opponent_type": opponent.get("type", "unknown"),
    }
    if experiment.family == "negotiation":
        return {
            **common,
            "complete_information": bool(state.get("complete_information", False)),
            "role": state.get(
                f"{player}_role", "seller" if player == "player_1" else "buyer"
            ),
        }
    my_index = 2 if player == "player_2" else 1
    return {
        **common,
        "role": player,
        "my_delta": _as_float(state.get(f"delta_{my_index}")),
        "opponent_delta": _as_float(state.get(f"delta_{3 - my_index}")),
    }


def _round_metric_empty() -> dict:
    return {"offers": 0, "resolved": 0, "converted": 0, "conversion_rate": None}


def _outcome_metric_empty() -> dict:
    return {
        "games": 0,
        "resolved": 0,
        "censored": 0,
        "agreements": 0,
        "no_deal": 0,
        "normalized_payoff_sum": 0.0,
        "mean_normalized_payoff": None,
        "direct_offers": 0,
        "direct_resolved": 0,
        "direct_converted": 0,
        "direct_conversion_rate": None,
    }


def _finish_outcome_metrics(metric: dict) -> None:
    resolved = metric.get("resolved", 0)
    metric["mean_normalized_payoff"] = (
        metric.pop("normalized_payoff_sum") / resolved if resolved else None
    )
    trials = metric.get("direct_resolved", 0)
    if "direct_resolved" in metric:
        metric["direct_conversion_rate"] = (
            metric.get("direct_converted", 0) / trials if trials else None
        )


def _offer_outcomes(
    affected: list[dict],
    enrolled: set[tuple[str, str]],
    parsed: ParsedRecords,
    experiment: Experiment,
) -> dict:
    by_variant = {
        "treatment": {
            **_outcome_metric_empty(),
            "affected_games": 0,
            "affected_turns": 0,
            "direction_violations": 0,
            "effective_offer_rounds": {},
            "horizon_strata": {},
            "max_rounds_strata": {},
            "cells": {},
        },
        "control": {
            **_outcome_metric_empty(),
            "affected_games": 0,
            "affected_turns": 0,
            "direction_violations": 0,
            "effective_offer_rounds": {},
            "horizon_strata": {},
            "max_rounds_strata": {},
            "cells": {},
        },
    }
    affected_by_game: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in affected:
        affected_by_game[(item["agent"], item["game_id"])].append(item)

    first_by_game: dict[tuple[str, str], Turn] = {}
    for turn in parsed.turns.values():
        key = (turn.agent, turn.gid)
        if key not in enrolled or turn.family != experiment.family:
            continue
        old = first_by_game.get(key)
        if old is None or turn.ts < old.ts:
            first_by_game[key] = turn

    for key, items in affected_by_game.items():
        turn = first_by_game.get(key)
        if turn is None:
            continue
        # Outcome extraction is one observation per game, anchored on the
        # first exact strategy divergence.  Later divergent turns remain in
        # ``affected_turns`` for routing diagnostics but cannot create extra
        # conversion trials or reweight the terminal payoff.
        first_affected = min(items, key=lambda item: (item["ts"], item["round"]))
        variant = first_affected["variant"]
        metric = by_variant[variant]
        metric["affected_games"] += 1
        metric["affected_turns"] += 1
        metric["direction_violations"] += int(
            bool(first_affected["direction_violation"])
        )
        terminal = parsed.terminals.get(key)
        horizon = _horizon(turn)
        maximum = _max_rounds(turn)
        hmetric = metric["horizon_strata"].setdefault(horizon, _outcome_metric_empty())
        mmetric = metric["max_rounds_strata"].setdefault(maximum, _outcome_metric_empty())
        cell = _offer_cell(experiment, turn, first_affected)
        cell_key = _cell_key(cell)
        cmetric = metric["cells"].setdefault(
            cell_key, {"cell": cell, **_outcome_metric_empty()}
        )
        targets = (metric, hmetric, mmetric, cmetric)
        for target in targets:
            target["games"] += 1
        if terminal is None:
            for target in targets:
                target["censored"] += 1
        else:
            payoff = _terminal_payoff(turn, terminal)
            state = turn.game.get("game_state")
            state = state if isinstance(state, dict) else {}
            if experiment.family == "bargaining":
                denominator = _as_float(state.get("money_to_divide"))
            else:
                player = turn.game.get("your_player", "player_1")
                denominator = _as_float(state.get(f"{player}_value"))
            normalized = (
                payoff / denominator
                if payoff is not None and denominator is not None and denominator > 0
                else 0.0
            )
            for target in targets:
                target["resolved"] += 1
                target["normalized_payoff_sum"] += normalized
                if _terminal_agreement(terminal):
                    target["agreements"] += 1
                else:
                    target["no_deal"] += 1

        offer_round = _round_number(first_affected.get("effective_offer_round"))
        if offer_round is None:
            continue
        for target in targets:
            target["direct_offers"] += 1
        rmetric = metric["effective_offer_rounds"].setdefault(
            str(offer_round), _round_metric_empty()
        )
        rmetric["offers"] += 1
        if terminal is None:
            continue
        for target in targets:
            target["direct_resolved"] += 1
        rmetric["resolved"] += 1
        agreed_round = _round_number(terminal.result.get("agreed_round"))
        converted = _terminal_agreement(terminal) and agreed_round == offer_round
        if converted:
            for target in targets:
                target["direct_converted"] += 1
            rmetric["converted"] += 1

    for metric in by_variant.values():
        _finish_outcome_metrics(metric)
        for strata_name in ("horizon_strata", "max_rounds_strata", "cells"):
            for submetric in metric[strata_name].values():
                _finish_outcome_metrics(submetric)
        for round_metric in metric["effective_offer_rounds"].values():
            n = round_metric["resolved"]
            round_metric["conversion_rate"] = (
                round_metric["converted"] / n if n else None
            )
    return by_variant


_NEG_GATE_CELL_FIELDS = (
    "role",
    "own_value_grid",
    "phase",
    "horizon",
    "max_rounds",
    "opponent_type",
    "complete_information",
)


def _neg_gate_cell(spec: dict) -> dict:
    return {field: spec.get(field) for field in _NEG_GATE_CELL_FIELDS}


def _neg_gate_cell_id(cell: dict) -> str:
    return _cell_key(_neg_gate_cell(cell))


def _neg_value_grid(value: Any) -> str:
    """Map scale-equivalent live values onto the public 80/100/120/150 grid."""
    number = _as_float(value)
    if number is None or number <= 0:
        return "unknown"
    for base in (80, 100, 120, 150):
        ratio = number / base
        exponent = round(math.log10(ratio))
        if math.isclose(ratio, 10**exponent, rel_tol=1e-9, abs_tol=1e-9):
            return str(base)
    return "other"


def _neg_target_artifact_identity(path: Path = TARGETS_PATH) -> dict:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        return {
            "path": "data/targets.json",
            "sha256": None,
            "bytes": None,
            "available": False,
        }
    return {
        "path": "data/targets.json",
        "sha256": digest.hexdigest(),
        "bytes": size,
        "available": True,
    }


def _neg_target_integrity(actual: dict | None = None) -> dict:
    observed = dict(actual) if actual is not None else _neg_target_artifact_identity()
    matches = bool(
        observed.get("available") is not False
        and observed.get("sha256") == _NEG_TARGET_ARTIFACT["sha256"]
        and observed.get("bytes") == _NEG_TARGET_ARTIFACT["bytes"]
    )
    return {
        "expected": dict(_NEG_TARGET_ARTIFACT),
        "observed": observed,
        "pool": "public-only (load_targets live=False)",
        "pass": matches,
    }


def _neg_gate_assignment(
    parsed: ParsedRecords,
    experiment: Experiment,
    agent: str,
    ts: float,
) -> tuple[str, str, str]:
    eligible = [
        event
        for event in parsed.runtimes
        if event.agent == agent
        and experiment.cutoff <= event.ts <= ts
        and experiment.knob in event.knobs
    ]
    if eligible:
        event = max(eligible, key=lambda item: item.ts)
        value = event.knobs.get(experiment.knob)
        if value == experiment.treatment_value:
            variant = "treatment"
        elif value == experiment.control_value:
            variant = "control"
        else:
            variant = "unknown"
        identity = event.strategy_sha256 or event.git_head or "unknown"
        epoch_id = f"runtime:{agent}:{event.ts:.6f}:{event.pid}:{identity[:12]}"
        return variant, epoch_id, "runtime_manifest"

    if agent in _NEG_STATIC_ASSIGNMENT["treatment_agents"]:
        variant = "treatment"
    elif agent in _NEG_STATIC_ASSIGNMENT["control_agents"]:
        variant = "control"
    else:
        variant = experiment.variant_for(agent)
    epoch_id = f"{_NEG_STATIC_ASSIGNMENT['epoch_id']}:{agent}:{variant}"
    return variant, epoch_id, "frozen_static_assignment"


def _neg_gate_payoff_percentile(
    turn: Turn,
    role: str,
    own_value: float | None,
    payoff: float | None,
    targets: Any,
) -> float | None:
    if own_value is None or payoff is None or role not in ("seller", "buyer"):
        return None
    try:
        full_key, role_key = config_key_negotiation(turn.game["game_state"], role, own_value)
        percentile = None
        if full_key is not None:
            percentile = targets.payoff_percentile(
                "negotiation", full_key, turn.game.get("your_player", "player_1"), payoff
            )
        if percentile is None and role_key is not None:
            percentile = targets.payoff_percentile(
                "negotiation", role_key, turn.game.get("your_player", "player_1"), payoff
            )
        return _as_float(percentile)
    except Exception:  # noqa: BLE001 - target drift must not kill a health report
        return None


def _neg_gate_unsupported_reason(cell: dict) -> str:
    if cell.get("complete_information") is not False:
        return "complete_information"
    if cell.get("role") != "buyer":
        return f"role={cell.get('role', 'unknown')}"
    if cell.get("own_value_grid") not in {"80", "100", "120", "150"}:
        return f"own_value_grid={cell.get('own_value_grid', 'unknown')}"
    if cell.get("phase") != "decision":
        return f"phase={cell.get('phase', 'unknown')}"
    if cell.get("horizon") != "finite" or cell.get("max_rounds") != "10":
        return (
            f"horizon={cell.get('horizon', 'unknown')},"
            f"max_rounds={cell.get('max_rounds', 'unknown')}"
        )
    if cell.get("opponent_type") not in ("agent", "hidden"):
        return f"opponent_type={cell.get('opponent_type', 'unknown')}"
    return "not_in_frozen_reference"


def _neg_terminal_gate_rows(
    affected: list[dict],
    enrolled: set[tuple[str, str]],
    parsed: ParsedRecords,
    experiment: Experiment,
) -> list[dict]:
    """Extract exactly one immutable gate observation per affected game."""
    affected_by_game: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in affected:
        affected_by_game[(item["agent"], item["game_id"])].append(item)
    turn_index = {
        (turn.agent, turn.gid, turn.round, turn.phase): turn
        for turn in parsed.turns.values()
        if (turn.agent, turn.gid) in enrolled and turn.family == "negotiation"
    }
    reference_ids = {
        _neg_gate_cell_id(spec) for spec in _NEG_TERMINAL_REFERENCE_CELLS
    }
    target_integrity = _neg_target_integrity()
    targets = (
        load_targets(TARGETS_PATH, live=False) if target_integrity["pass"] else None
    )
    rows: list[dict] = []
    for key, items in sorted(affected_by_game.items()):
        first = min(items, key=lambda item: (item["ts"], item["round"], item["phase"]))
        turn = turn_index.get((key[0], key[1], first["round"], first["phase"]))
        if turn is None:
            continue
        state = turn.game.get("game_state")
        state = state if isinstance(state, dict) else {}
        player = turn.game.get("your_player", "player_1")
        role = state.get(
            f"{player}_role", "seller" if player == "player_1" else "buyer"
        )
        own_value = _as_float(state.get(f"{player}_value"))
        opponent = turn.game.get("opponent")
        opponent = opponent if isinstance(opponent, dict) else {}
        cell = {
            "role": role,
            "own_value_grid": _neg_value_grid(own_value),
            "phase": first.get("phase", "unknown"),
            "horizon": _horizon(turn),
            "max_rounds": _max_rounds(turn),
            "opponent_type": opponent.get("type", "unknown"),
            "complete_information": bool(state.get("complete_information", False)),
        }
        cell_id = _neg_gate_cell_id(cell)
        terminal = parsed.terminals.get(key)
        payoff = _terminal_payoff(turn, terminal) if terminal is not None else None
        normalized = (
            payoff / own_value
            if payoff is not None and own_value is not None and own_value > 0
            else None
        )
        offer_round = _round_number(first.get("effective_offer_round"))
        direct = None
        if terminal is not None and offer_round is not None:
            direct = bool(
                _terminal_agreement(terminal)
                and _round_number(terminal.result.get("agreed_round")) == offer_round
            )
        compatibility = next(
            (
                spec["compatibility_rate"]
                for spec in _NEG_TERMINAL_REFERENCE_CELLS
                if _neg_gate_cell_id(spec) == cell_id
            ),
            None,
        )
        variant, epoch_id, assignment_source = _neg_gate_assignment(
            parsed, experiment, key[0], first["ts"]
        )
        rows.append(
            {
                "agent": key[0],
                "variant": variant,
                "assignment_epoch_id": epoch_id,
                "assignment_source": assignment_source,
                "game_id": key[1],
                "cell": cell,
                "cell_id": cell_id,
                "supported": cell_id in reference_ids,
                "unsupported_reason": (
                    None if cell_id in reference_ids else _neg_gate_unsupported_reason(cell)
                ),
                "resolved": terminal is not None,
                "censored": terminal is None,
                "terminal_reaped": terminal is not None and terminal.valid is None,
                "direct": direct,
                "effective_offer_round": offer_round,
                "normalized_payoff": normalized,
                "payoff_percentile": _neg_gate_payoff_percentile(
                    turn, str(role), own_value, payoff, targets
                ),
                "compatibility_rate": compatibility,
                "direction_violation": bool(first.get("direction_violation")),
                "assigned_match": bool(first.get("assigned_match")),
            }
        )
    return rows


def _neg_gate_sample_empty() -> dict:
    return {
        "affected": 0,
        "resolved": 0,
        "censored": 0,
        "terminal_reaped": 0,
        "direct_trials": 0,
        "direct_converted": 0,
        "normalized_payoff_n": 0,
        "payoff_percentile_n": 0,
    }


def _neg_gate_add_sample(sample: dict, row: dict) -> None:
    sample["affected"] += 1
    sample["resolved"] += int(bool(row.get("resolved")))
    sample["censored"] += int(bool(row.get("censored")))
    sample["terminal_reaped"] += int(bool(row.get("terminal_reaped")))
    if isinstance(row.get("direct"), bool):
        sample["direct_trials"] += 1
        sample["direct_converted"] += int(row["direct"])
    sample["normalized_payoff_n"] += int(row.get("normalized_payoff") is not None)
    sample["payoff_percentile_n"] += int(row.get("payoff_percentile") is not None)


def _neg_gate_counts(rows: list[dict]) -> dict:
    variants = {
        variant: {"all": _neg_gate_sample_empty(), "primary": _neg_gate_sample_empty()}
        for variant in ("treatment", "control")
    }
    cells: dict[str, dict] = {}
    for spec in _NEG_TERMINAL_REFERENCE_CELLS:
        cell = _neg_gate_cell(spec)
        cell_id = _neg_gate_cell_id(cell)
        cells[cell_id] = {
            "cell": cell,
            "weight": spec["weight"],
            "compatibility_rate": spec["compatibility_rate"],
            "treatment": _neg_gate_sample_empty(),
            "control": _neg_gate_sample_empty(),
        }
    agents: dict[str, dict] = {}
    unsupported_cells: dict[str, dict] = {}
    unsupported_reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        variant = row["variant"]
        _neg_gate_add_sample(variants[variant]["all"], row)
        agent = agents.setdefault(
            row["agent"],
            {
                "variant": variant,
                "all": _neg_gate_sample_empty(),
                "primary": _neg_gate_sample_empty(),
            },
        )
        _neg_gate_add_sample(agent["all"], row)
        if row.get("supported"):
            _neg_gate_add_sample(variants[variant]["primary"], row)
            _neg_gate_add_sample(agent["primary"], row)
            _neg_gate_add_sample(cells[row["cell_id"]][variant], row)
            continue
        reason = str(row.get("unsupported_reason") or "unknown")
        unsupported_reasons[reason] += 1
        entry = unsupported_cells.setdefault(
            row["cell_id"],
            {
                "cell": row.get("cell", {}),
                "reason": reason,
                "treatment": _neg_gate_sample_empty(),
                "control": _neg_gate_sample_empty(),
            },
        )
        _neg_gate_add_sample(entry[variant], row)
    return {
        "variants": variants,
        "cells": cells,
        "agents": agents,
        "unsupported": {
            "total": sum(unsupported_reasons.values()),
            "reasons": dict(sorted(unsupported_reasons.items())),
            "cells": dict(sorted(unsupported_cells.items())),
        },
    }


def _sample_variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((value - mean) ** 2 for value in values) / (len(values) - 1)


def _neg_gate_standardized(
    rows: list[dict], field: str, *, binary: bool
) -> dict:
    reference = {
        _neg_gate_cell_id(spec): spec for spec in _NEG_TERMINAL_REFERENCE_CELLS
    }
    values: dict[str, dict[str, list[float]]] = {
        cell_id: {"treatment": [], "control": []} for cell_id in reference
    }
    for row in rows:
        if not row.get("supported") or row.get(field) is None:
            continue
        value = float(bool(row[field])) if binary else _as_float(row[field])
        if value is not None:
            values[row["cell_id"]][row["variant"]].append(value)

    cell_output: dict[str, dict] = {}
    coverage = 0.0
    treatment_mean = 0.0
    control_mean = 0.0
    se_squared = 0.0
    variance_available = True
    for cell_id, spec in reference.items():
        weight = float(spec["weight"])
        arm_output: dict[str, dict] = {}
        both_present = True
        cell_variances: dict[str, float | None] = {}
        for variant in ("treatment", "control"):
            observed = values[cell_id][variant]
            mean = sum(observed) / len(observed) if observed else None
            if binary and observed:
                # Jeffreys smoothing is used only for the variance, preserving
                # the observed cell rate in the standardized point estimate.
                probability = (sum(observed) + 0.5) / (len(observed) + 1)
                variance = probability * (1 - probability)
            else:
                variance = _sample_variance(observed)
            cell_variances[variant] = variance
            arm_output[variant] = {
                "n": len(observed),
                "mean": mean,
                "variance": variance,
            }
            both_present = both_present and bool(observed)
        if both_present:
            coverage += weight
            treatment_mean += weight * arm_output["treatment"]["mean"]
            control_mean += weight * arm_output["control"]["mean"]
            for variant in ("treatment", "control"):
                variance = cell_variances[variant]
                n = arm_output[variant]["n"]
                if variance is None:
                    variance_available = False
                else:
                    se_squared += weight * weight * variance / n
        cell_output[cell_id] = {
            "cell": _neg_gate_cell(spec),
            "weight": weight,
            **arm_output,
        }

    complete = math.isclose(coverage, 1.0, rel_tol=0, abs_tol=1e-12)
    estimate = treatment_mean - control_mean if complete else None
    standard_error = (
        math.sqrt(se_squared) if complete and variance_available else None
    )
    return {
        "field": field,
        "method": (
            "fixed joint-stratum contrast; raw means; Jeffreys binary cell variance"
            if binary
            else "fixed joint-stratum contrast; unpooled sample cell variance"
        ),
        "reference_weight_coverage": coverage,
        "complete_fixed_support": complete,
        "treatment": treatment_mean if complete else None,
        "control": control_mean if complete else None,
        "uplift": estimate,
        "standard_error": standard_error,
        "one_sided_95_lower": (
            estimate - _Z_ONE_SIDED_95 * standard_error
            if estimate is not None and standard_error is not None
            else None
        ),
        "one_sided_90_upper": (
            estimate + _Z_ONE_SIDED_90 * standard_error
            if estimate is not None and standard_error is not None
            else None
        ),
        "cells": cell_output,
    }


def _neg_gate_two_arm_contrast(
    treatment: list[float], control: list[float], *, binary: bool
) -> dict:
    means = {
        "treatment": sum(treatment) / len(treatment) if treatment else None,
        "control": sum(control) / len(control) if control else None,
    }
    variances: dict[str, float | None] = {}
    for variant, values in (("treatment", treatment), ("control", control)):
        if binary and values:
            probability = (sum(values) + 0.5) / (len(values) + 1)
            variances[variant] = probability * (1 - probability)
        else:
            variances[variant] = _sample_variance(values)
    estimate = (
        means["treatment"] - means["control"]
        if means["treatment"] is not None and means["control"] is not None
        else None
    )
    standard_error = None
    if estimate is not None and all(value is not None for value in variances.values()):
        standard_error = math.sqrt(
            variances["treatment"] / len(treatment)
            + variances["control"] / len(control)
        )
    return {
        "treatment_n": len(treatment),
        "control_n": len(control),
        "treatment": means["treatment"],
        "control": means["control"],
        "uplift": estimate,
        "standard_error": standard_error,
        "one_sided_95_lower": (
            estimate - _Z_ONE_SIDED_95 * standard_error
            if estimate is not None and standard_error is not None
            else None
        ),
    }


def _neg_gate_unsupported_safety(rows: list[dict]) -> dict:
    unsupported = [
        row
        for row in rows
        if not row.get("supported") and row.get("variant") in ("treatment", "control")
    ]
    if not unsupported:
        return {
            "present": False,
            "pass": True,
            "harm_fail": False,
            "reason": "no unsupported policy-affected rows",
            "cells": {},
        }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in unsupported:
        grouped[row["cell_id"]].append(row)
    output: dict[str, dict] = {}
    all_pass = True
    harm_fail = False
    for cell_id, cell_rows in sorted(grouped.items()):
        metrics = {}
        for field, binary in (
            ("direct", True),
            ("normalized_payoff", False),
            ("payoff_percentile", False),
        ):
            values = {"treatment": [], "control": []}
            for row in cell_rows:
                value = row.get(field)
                if value is None:
                    continue
                parsed = float(bool(value)) if binary else _as_float(value)
                if parsed is not None:
                    values[row["variant"]].append(parsed)
            metrics[field] = _neg_gate_two_arm_contrast(
                values["treatment"], values["control"], binary=binary
            )
        sample_pass = all(
            metric["treatment_n"] >= 15 and metric["control_n"] >= 45
            for metric in metrics.values()
        )
        margins = {
            "direct": -0.050,
            "normalized_payoff": -0.005,
            "payoff_percentile": -0.050,
        }
        noninferior = {
            field: bool(
                metric["one_sided_95_lower"] is not None
                and metric["one_sided_95_lower"] > margins[field]
            )
            for field, metric in metrics.items()
        }
        cell_pass = sample_pass and all(noninferior.values())
        harm_fail = harm_fail or (sample_pass and not all(noninferior.values()))
        all_pass = all_pass and cell_pass
        output[cell_id] = {
            "cell": cell_rows[0].get("cell", {}),
            "sample_pass": sample_pass,
            "noninferiority": noninferior,
            "pass": cell_pass,
            "metrics": metrics,
        }
    return {
        "present": True,
        "pass": all_pass,
        "harm_fail": harm_fail,
        "reason": (
            "all unsupported cells meet frozen sample and noninferiority gates"
            if all_pass
            else "unsupported cells lack support or fail noninferiority"
        ),
        "cells": output,
    }


def _normal_cdf(value: float) -> float:
    return 0.5 * (1 + math.erf(value / math.sqrt(2)))


def _neg_gate_conditional_power(
    direct: dict, treatment_n: int, control_n: int
) -> float | None:
    estimate = direct.get("uplift")
    standard_error = direct.get("standard_error")
    if estimate is None or standard_error is None or standard_error <= 0:
        return None
    info = min(treatment_n / 340, control_n / 1020, 1.0)
    if info <= 0:
        return None
    if info >= 1:
        return float(estimate - _Z_ONE_SIDED_95 * standard_error > 0)
    historical_control = 31 / 341
    planned_treatment = historical_control + 0.060
    planned_se = math.sqrt(
        planned_treatment * (1 - planned_treatment) / 340
        + historical_control * (1 - historical_control) / 1020
    )
    planned_noncentrality = 0.060 / planned_se
    current_z = estimate / standard_error
    numerator = (
        math.sqrt(info) * current_z
        + planned_noncentrality * (1 - info)
        - _Z_ONE_SIDED_95
    )
    return _normal_cdf(numerator / math.sqrt(1 - info))


def _neg_gate_epoch_health(
    parsed: ParsedRecords,
    experiment: Experiment,
    affected: list[dict],
    agent_data: dict[str, dict],
) -> dict[str, dict]:
    by_variant = {
        variant: {
            "traffic_events": 0,
            "errors": 0,
            "invalid_results": 0,
            "corrections": 0,
            "replay_errors": 0,
            "affected_wrong_variant": 0,
            "affected_unknown": 0,
            "direction_violations": 0,
            "affected": 0,
            "censored": 0,
        }
        for variant in ("treatment", "control")
    }
    for turn in parsed.turns.values():
        if (
            turn.agent not in experiment.agents
            or turn.family != experiment.family
            or turn.ts < experiment.cutoff
        ):
            continue
        variant = _neg_gate_assignment(parsed, experiment, turn.agent, turn.ts)[0]
        if variant not in by_variant:
            continue
        target = by_variant[variant]
        target["traffic_events"] += 1
        target["errors"] += int(bool(turn.error))
        target["corrections"] += len(turn.corrections)
    for event in parsed.results:
        if (
            event.agent not in experiment.agents
            or event.ts < experiment.cutoff
            or parsed.gid_family.get((event.agent, event.gid)) != experiment.family
        ):
            continue
        variant = _neg_gate_assignment(parsed, experiment, event.agent, event.ts)[0]
        if variant not in by_variant:
            continue
        target = by_variant[variant]
        target["traffic_events"] += 1
        target["errors"] += int(bool(event.error))
        target["invalid_results"] += int(event.valid is False)
    for item in affected:
        variant = item.get("variant")
        if variant not in by_variant:
            continue
        target = by_variant[variant]
        target["affected_wrong_variant"] += int(
            not item.get("assigned_match") and item.get("other_variant_match")
        )
        target["affected_unknown"] += int(
            not item.get("assigned_match") and not item.get("other_variant_match")
        )
        target["direction_violations"] += int(bool(item.get("direction_violation")))
    # Replay exceptions are global report-integrity failures. They cannot
    # always be assigned to an epoch because no divergent row is then emitted.
    by_variant["treatment"]["replay_errors"] = sum(
        data.get("routing", {}).get("replay_errors", 0)
        for data in agent_data.values()
    )
    return by_variant


def _neg_gate_health(
    rows: list[dict],
    agent_data: dict[str, dict] | None,
    agent_variants: dict[str, str] | None,
    epoch_health: dict[str, dict] | None = None,
) -> dict:
    by_variant = {
        variant: {
            "traffic_events": 0,
            "errors": 0,
            "invalid_results": 0,
            "corrections": 0,
            "replay_errors": 0,
            "affected_wrong_variant": 0,
            "affected_unknown": 0,
            "direction_violations": 0,
            "affected": 0,
            "censored": 0,
        }
        for variant in ("treatment", "control")
    }
    observed_variants = {row["agent"]: row["variant"] for row in rows}
    if epoch_health is not None:
        for variant in ("treatment", "control"):
            for key, value in epoch_health.get(variant, {}).items():
                if key in by_variant[variant]:
                    by_variant[variant][key] = value
    elif agent_data:
        for agent, data in agent_data.items():
            variant = (agent_variants or {}).get(agent, observed_variants.get(agent))
            if variant not in by_variant:
                continue
            health = data.get("health", {})
            routing = data.get("routing", {})
            target = by_variant[variant]
            target["traffic_events"] += health.get("turns", 0) + health.get(
                "result_events", 0
            )
            target["errors"] += health.get("turn_errors", 0) + health.get(
                "result_errors", 0
            )
            target["invalid_results"] += health.get("invalid_results", 0)
            target["corrections"] += health.get("corrections", 0)
            for key in (
                "replay_errors",
                "affected_wrong_variant",
                "affected_unknown",
                "direction_violations",
            ):
                target[key] += routing.get(key, 0)
    else:
        # Pure evaluator callers still receive deterministic healthy rates.
        for row in rows:
            target = by_variant[row["variant"]]
            target["traffic_events"] += 1
            target["direction_violations"] += int(
                bool(row.get("direction_violation"))
            )
    for row in rows:
        target = by_variant[row["variant"]]
        target["affected"] += 1
        target["censored"] += int(bool(row.get("censored")))
    for target in by_variant.values():
        traffic = target["traffic_events"]
        affected = target["affected"]
        target["error_rate"] = target["errors"] / traffic if traffic else None
        target["affected_censor_rate"] = (
            target["censored"] / affected if affected else None
        )

    treatment = by_variant["treatment"]
    control = by_variant["control"]
    error_excess = (
        treatment["error_rate"] - control["error_rate"]
        if treatment["error_rate"] is not None and control["error_rate"] is not None
        else None
    )
    censor_excess = (
        treatment["affected_censor_rate"] - control["affected_censor_rate"]
        if treatment["affected_censor_rate"] is not None
        and control["affected_censor_rate"] is not None
        else None
    )
    structural_faults = sum(
        target[key]
        for target in by_variant.values()
        for key in (
            "replay_errors",
            "affected_wrong_variant",
            "affected_unknown",
            "direction_violations",
        )
    )
    treatment_validity_faults = treatment["invalid_results"] + treatment["corrections"]
    checks = {
        "routing_and_direction_clean": structural_faults == 0,
        "treatment_invalid_and_corrections_clean": treatment_validity_faults == 0,
        "treatment_error_rate_excess_within_0.01": (
            error_excess <= 0.010 if error_excess is not None else None
        ),
        "affected_censor_rate_excess_within_0.03": (
            censor_excess <= 0.030 if censor_excess is not None else None
        ),
    }
    return {
        "by_variant": by_variant,
        "treatment_error_rate_excess": error_excess,
        "affected_censor_rate_excess": censor_excess,
        "structural_faults": structural_faults,
        "treatment_validity_faults": treatment_validity_faults,
        "checks": checks,
        "pass": all(value is True for value in checks.values()),
        "hard_fail": any(value is False for value in checks.values()),
    }


def _neg_gate_agent_confirmation(rows: list[dict]) -> dict:
    treatment_epochs = sorted(
        {
            row.get("assignment_epoch_id", f"legacy:{row['agent']}")
            for row in rows
            if row["variant"] == "treatment"
        }
    )
    blocks = []
    for epoch_id in treatment_epochs:
        treatment_rows = [
            row
            for row in rows
            if row["variant"] == "treatment"
            and row.get("assignment_epoch_id", f"legacy:{row['agent']}") == epoch_id
            and row.get("supported")
            and isinstance(row.get("direct"), bool)
        ]
        block_rows = [
            row
            for row in rows
            if row["variant"] == "control" or row in treatment_rows
        ]
        metric = _neg_gate_standardized(block_rows, "direct", binary=True)
        by_value = Counter(row["cell"]["own_value_grid"] for row in treatment_rows)
        by_identity = Counter(row["cell"]["opponent_type"] for row in treatment_rows)
        sample_pass = bool(
            len(treatment_rows) >= 60
            and all(by_value[value] >= 15 for value in ("80", "100", "120", "150"))
            and all(by_identity[kind] >= 20 for kind in ("agent", "hidden"))
        )
        passes = bool(
            sample_pass
            and metric["complete_fixed_support"]
            and metric["uplift"] is not None
            and metric["uplift"] >= 0
        )
        agents = sorted({row["agent"] for row in treatment_rows})
        blocks.append(
            {
                "assignment_epoch_id": epoch_id,
                "agents": agents,
                "assignment_sources": sorted(
                    {
                        str(row.get("assignment_source", "legacy"))
                        for row in treatment_rows
                    }
                ),
                "direct_trials": len(treatment_rows),
                "direct_trials_by_value": dict(sorted(by_value.items())),
                "direct_trials_by_opponent_identity": dict(sorted(by_identity.items())),
                "sample_pass": sample_pass,
                "complete_fixed_support": metric["complete_fixed_support"],
                "standardized_direct_uplift": metric["uplift"],
                "nonnegative": passes,
            }
        )
    confirmed = sum(block["nonnegative"] for block in blocks)
    return {
        "unit": "timestamped treatment assignment epoch",
        "minimum_per_block": {
            "total": 60,
            "per_own_value_across_identity": 15,
            "per_opponent_identity": 20,
        },
        "required": 2,
        "confirmed": confirmed,
        "pass": confirmed >= 2,
        "blocks": blocks,
    }


def _neg_gate_block_sample(rows: list[dict]) -> dict:
    by_value = Counter(row["cell"]["own_value_grid"] for row in rows)
    by_identity = Counter(row["cell"]["opponent_type"] for row in rows)
    passes = bool(
        len(rows) >= 60
        and all(by_value[value] >= 15 for value in ("80", "100", "120", "150"))
        and all(by_identity[kind] >= 20 for kind in ("agent", "hidden"))
    )
    return {
        "direct_trials": len(rows),
        "direct_trials_by_value": dict(sorted(by_value.items())),
        "direct_trials_by_opponent_identity": dict(sorted(by_identity.items())),
        "pass": passes,
    }


def _neg_gate_switchback_confirmation(rows: list[dict]) -> dict:
    supported = [
        row
        for row in rows
        if row.get("supported")
        and row.get("variant") in ("treatment", "control")
        and isinstance(row.get("direct"), bool)
    ]
    labels = []
    for agent in sorted({row["agent"] for row in supported}):
        agent_rows = [row for row in supported if row["agent"] == agent]
        arm_rows = {
            variant: [row for row in agent_rows if row["variant"] == variant]
            for variant in ("treatment", "control")
        }
        samples = {
            variant: _neg_gate_block_sample(arm_rows[variant])
            for variant in ("treatment", "control")
        }
        runtime_arms = sorted(
            {
                row["variant"]
                for row in agent_rows
                if str(row.get("assignment_epoch_id", "")).startswith("runtime:")
            }
        )
        metric = _neg_gate_standardized(agent_rows, "direct", binary=True)
        passes = bool(
            samples["treatment"]["pass"]
            and samples["control"]["pass"]
            and runtime_arms
            and metric["complete_fixed_support"]
            and metric["uplift"] is not None
            and metric["uplift"] >= 0
        )
        labels.append(
            {
                "agent": agent,
                "runtime_manifest_arms": runtime_arms,
                "samples": samples,
                "complete_fixed_support": metric["complete_fixed_support"],
                "within_label_standardized_direct_uplift": metric["uplift"],
                "pass": passes,
            }
        )
    confirmed = sum(label["pass"] for label in labels)
    return {
        "unit": "agent label crossed over timestamped assignment epochs",
        "required_labels": 2,
        "confirmed_labels": confirmed,
        "pass": confirmed >= 2,
        "labels": labels,
        "status_cap_without_pass": "screen_pass",
    }


def _neg_terminal_gate_from_rows(
    rows: list[dict],
    agent_data: dict[str, dict] | None = None,
    agent_variants: dict[str, str] | None = None,
    target_artifact_identity: dict | None = None,
    epoch_health: dict[str, dict] | None = None,
) -> dict:
    """Pure deterministic evaluator for the frozen hidden terminal-close gate."""
    counts = _neg_gate_counts(rows)
    direct = _neg_gate_standardized(rows, "direct", binary=True)
    normalized = _neg_gate_standardized(rows, "normalized_payoff", binary=False)
    percentile = _neg_gate_standardized(rows, "payoff_percentile", binary=False)
    standardized = {
        "direct": direct,
        "normalized_payoff": normalized,
        "payoff_percentile": percentile,
    }
    health = _neg_gate_health(rows, agent_data, agent_variants, epoch_health)
    agent_confirmation = _neg_gate_agent_confirmation(rows)
    switchback_confirmation = _neg_gate_switchback_confirmation(rows)
    unsupported_safety = _neg_gate_unsupported_safety(rows)
    target_integrity = _neg_target_integrity(target_artifact_identity)
    primary = {
        variant: counts["variants"][variant]["primary"]
        for variant in ("treatment", "control")
    }
    treatment_n = primary["treatment"]["direct_trials"]
    control_n = primary["control"]["direct_trials"]
    conditional_power = _neg_gate_conditional_power(direct, treatment_n, control_n)

    interim_reasons: list[str] = []
    if treatment_n >= 50 and control_n >= 150:
        if primary["treatment"]["direct_converted"] == 0:
            interim_reasons.append("T>=50/C>=150 with zero treatment conversions")
        if (
            normalized["uplift"] is not None
            and normalized["uplift"] <= -0.010
        ):
            interim_reasons.append("standardized normalized-payoff uplift <= -0.010")
    if treatment_n >= 100 and control_n >= 300:
        if direct["treatment"] is not None and direct["treatment"] <= 0.050:
            interim_reasons.append("standardized treatment direct conversion <= 0.050")
        if (
            direct["one_sided_90_upper"] is not None
            and direct["one_sided_90_upper"] <= 0
        ):
            interim_reasons.append("one-sided 90% upper direct uplift <= 0")
    if health["hard_fail"]:
        interim_reasons.append("health or direction hard-fail")
    if unsupported_safety["harm_fail"]:
        interim_reasons.append("unsupported policy-affected slice failed noninferiority")

    final_sample = treatment_n >= 340 and control_n >= 1020
    per_cell_sample = all(
        cell["treatment"]["direct_trials"] >= 15
        and cell["control"]["direct_trials"] >= 45
        for cell in counts["cells"].values()
    )
    if final_sample and per_cell_sample:
        stage = "promotion_ready"
    elif treatment_n >= 100 and control_n >= 300:
        stage = "interim_2"
    elif treatment_n >= 50 and control_n >= 150:
        stage = "interim_1"
    else:
        stage = "collecting"
    interim = {
        "stage": stage,
        "treatment_direct_trials": treatment_n,
        "control_direct_trials": control_n,
        "information_fraction": min(treatment_n / 340, control_n / 1020, 1.0),
        "conditional_power": conditional_power,
        "conditional_power_binding": False,
        "conditional_power_warning": (
            "Legacy count-fraction approximation does not match the fixed "
            "eight-cell estimator; diagnostic only."
        ),
        "rollback": bool(interim_reasons),
        "reasons": interim_reasons,
    }

    passes = {
        "overall_sample": final_sample,
        "joint_cell_sample": per_cell_sample,
        "complete_fixed_support": direct["complete_fixed_support"],
        "direct_uplift_point_at_least_0.060": bool(
            direct["uplift"] is not None and direct["uplift"] >= 0.060
        ),
        "direct_uplift_one_sided_95_lower_above_0": bool(
            direct["one_sided_95_lower"] is not None
            and direct["one_sided_95_lower"] > 0
        ),
        "normalized_payoff_one_sided_95_lower_above_minus_0.005": bool(
            normalized["one_sided_95_lower"] is not None
            and normalized["one_sided_95_lower"] > -0.005
        ),
        "payoff_percentile_one_sided_95_lower_above_minus_0.050": bool(
            percentile["one_sided_95_lower"] is not None
            and percentile["one_sided_95_lower"] > -0.050
        ),
        "two_supported_nonnegative_treatment_epochs": agent_confirmation["pass"],
        "balanced_manifest_switchback": switchback_confirmation["pass"],
        "payoff_target_artifact_matches_cutoff": target_integrity["pass"],
        "unsupported_policy_slices_noninferior": unsupported_safety["pass"],
        "health": health["pass"],
        "no_interim_rollback": not interim["rollback"],
    }
    failed = [name for name, passed in passes.items() if not passed]
    if interim["rollback"]:
        status = "rollback"
    elif not failed:
        status = "promote"
    elif failed == ["balanced_manifest_switchback"]:
        status = "screen_pass"
    else:
        status = "continue"

    treatment_direct = direct["treatment"]
    control_direct = direct["control"]
    compatibility = {
        "fixed_compatible_opportunity_rate": _NEG_TERMINAL_COMPATIBILITY_RATE,
        "historical_control_efficiency_24h": (103 / 1382)
        / _NEG_TERMINAL_COMPATIBILITY_RATE,
        "historical_control_efficiency_6h": (31 / 341)
        / _NEG_TERMINAL_COMPATIBILITY_RATE,
        "diagnostic_target": 0.45,
        "standardized_treatment_efficiency": (
            treatment_direct / _NEG_TERMINAL_COMPATIBILITY_RATE
            if treatment_direct is not None
            else None
        ),
        "standardized_control_efficiency": (
            control_direct / _NEG_TERMINAL_COMPATIBILITY_RATE
            if control_direct is not None
            else None
        ),
        "standardized_uplift_per_compatible_opportunity": (
            direct["uplift"] / _NEG_TERMINAL_COMPATIBILITY_RATE
            if direct["uplift"] is not None
            else None
        ),
        "formal_promotion_gate": False,
        "note": (
            "Compatibility is an inferred rational ceiling diagnostic, not an "
            "observed-outcome denominator."
        ),
    }
    return {
        "design": json.loads(json.dumps(NEG_TERMINAL_GATE_DESIGN)),
        "counts": counts,
        "standardized": standardized,
        "compatibility_diagnostic": compatibility,
        "payoff_target_integrity": target_integrity,
        "unsupported_safety": unsupported_safety,
        "health": health,
        "interim": interim,
        "agent_confirmation": agent_confirmation,
        "switchback_confirmation": switchback_confirmation,
        "promotion": {
            "status": status,
            "passes": passes,
            "failed_checks": failed,
            "reasons": (
                interim_reasons
                if status == "rollback"
                else ["causal promotion capped pending balanced manifest switchback"]
                if status == "screen_pass"
                else [f"waiting for {name}" for name in failed]
            ),
        },
    }


def _p_key(value: Any) -> str:
    number = _as_float(value)
    return f"{number:.6g}" if number is not None else "unknown"


def _pers_leaf_empty() -> dict:
    return {
        "blind_seller_games": 0,
        "resolved": 0,
        "censored": 0,
        "revenue_share_sum": 0.0,
        "mean_revenue_share": None,
        "zero_sales": 0,
        "zero_sales_rate": None,
        "affected_games": 0,
        "affected_turns": 0,
        "direction_violations": 0,
        "deterministic_route_checked": 0,
        "deterministic_route_matches": 0,
    }


def _pers_metric_empty() -> dict:
    return {
        **_pers_leaf_empty(),
        "p_strata": {},
        "cells": {},
    }


def _finish_pers_metric(metric: dict) -> None:
    resolved = metric["resolved"]
    metric["mean_revenue_share"] = (
        metric.pop("revenue_share_sum") / resolved if resolved else None
    )
    metric["zero_sales_rate"] = metric["zero_sales"] / resolved if resolved else None


def _pers_cell(turn: Turn, experiment: Experiment) -> dict:
    state = turn.game.get("game_state")
    state = state if isinstance(state, dict) else {}
    opponent = turn.game.get("opponent")
    opponent = opponent if isinstance(opponent, dict) else {}
    total_rounds = _round_number(state.get("total_rounds"))
    return {
        "p": _as_float(state.get("p")),
        "message_type": state.get("seller_message_type", "unknown"),
        "price": _as_float(state.get("product_price")),
        "total_rounds": total_rounds if total_rounds is not None else "unknown",
        "opponent_type": opponent.get("type", "unknown"),
        "start_block_15m": int(max(turn.ts - experiment.cutoff, 0) // 900),
    }


def _persuasion_outcomes(
    affected: list[dict],
    enrolled: set[tuple[str, str]],
    parsed: ParsedRecords,
    experiment: Experiment,
) -> dict:
    metrics = {"treatment": _pers_metric_empty(), "control": _pers_metric_empty()}
    affected_by_game: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in affected:
        affected_by_game[(item["agent"], item["game_id"])].append(item)

    first_by_game: dict[tuple[str, str], Turn] = {}
    for turn in parsed.turns.values():
        key = (turn.agent, turn.gid)
        if key not in enrolled or turn.family != "persuasion":
            continue
        old = first_by_game.get(key)
        if old is None or turn.ts < old.ts:
            first_by_game[key] = turn

    for key, turn in first_by_game.items():
        state = turn.game.get("game_state")
        state = state if isinstance(state, dict) else {}
        player = turn.game.get("your_player", "player_1")
        seller = player == "player_1" or state.get(f"{player}_role") == "seller"
        if "is_seller_know_cv" in state:
            # Current payloads carry the authoritative visibility contract.
            # Only an explicit false belongs in the blind-seller cohort.
            blind = state.get("is_seller_know_cv") is False
        else:
            # Old archived payloads predate that flag; absent v is the only
            # available backward-compatible indication of seller blindness.
            blind = state.get("v") is None
        if not seller or not blind:
            continue
        variant = experiment.variant_for(turn.agent)
        metric = metrics[variant]
        pmetric = metric["p_strata"].setdefault(
            _p_key(state.get("p")), _pers_leaf_empty()
        )
        cell = _pers_cell(turn, experiment)
        cmetric = metric["cells"].setdefault(
            _cell_key(cell), {"cell": cell, **_pers_leaf_empty()}
        )
        targets = (metric, pmetric, cmetric)
        for target in targets:
            target["blind_seller_games"] += 1
        items = affected_by_game.get(key, [])
        if items:
            for target in targets:
                target["affected_games"] += 1
                target["affected_turns"] += len(items)
                target["direction_violations"] += sum(
                    bool(item["direction_violation"]) for item in items
                )
                target["deterministic_route_checked"] += len(items)
                target["deterministic_route_matches"] += sum(
                    bool(item["assigned_match"]) for item in items
                )
        terminal = parsed.terminals.get(key)
        if terminal is None:
            for target in targets:
                target["censored"] += 1
            continue
        payoff = _terminal_payoff(turn, terminal)
        price = _as_float(state.get("product_price"))
        rounds_value = _as_float(state.get("total_rounds"))
        denominator = (
            price * rounds_value
            if price is not None and price > 0 and rounds_value is not None and rounds_value > 0
            else None
        )
        share = payoff / denominator if payoff is not None and denominator else 0.0
        for target in targets:
            target["resolved"] += 1
            target["revenue_share_sum"] += share
            if payoff is None or payoff <= 0:
                target["zero_sales"] += 1

    for metric in metrics.values():
        for pmetric in metric["p_strata"].values():
            _finish_pers_metric(pmetric)
        for cmetric in metric["cells"].values():
            _finish_pers_metric(cmetric)
        _finish_pers_metric(metric)
    return metrics


ReplayFn = Callable[[dict, Knobs], dict]


def analyze_experiment(
    parsed: ParsedRecords,
    preexisting: set[tuple[str, str]],
    experiment: Experiment,
    replay: ReplayFn = replay_action,
) -> dict:
    agent_data = {agent: _arm_empty() for agent in experiment.agents}
    candidates: dict[tuple[str, str], Turn] = {}
    for turn in parsed.turns.values():
        if (
            turn.agent in experiment.agents
            and turn.family == experiment.family
            and turn.ts >= experiment.cutoff
        ):
            key = (turn.agent, turn.gid)
            old = candidates.get(key)
            if old is None or turn.ts < old.ts:
                candidates[key] = turn

    enrolled: set[tuple[str, str]] = set()
    for key, candidate in candidates.items():
        agent = key[0]
        first = parsed.first_turns.get(key, candidate)
        enrollment = agent_data[agent]["enrollment"]
        if key in preexisting or first.ts < experiment.cutoff:
            enrollment["excluded_pre_cut"] += 1
        elif first.round != 1 or not _history_empty(first):
            enrollment["excluded_partial"] += 1
        else:
            enrolled.add(key)
            enrollment["enrolled"] += 1
            terminal = parsed.terminals.get(key)
            if terminal is None:
                enrollment["censored"] += 1
            else:
                enrollment["resolved"] += 1
                # Reaper terminals have valid=None; direct SDK terminals have
                # a boolean.  Older logs did not persist the explicit reaped
                # marker, so this is the lossless inference available there.
                if terminal.valid is None:
                    enrollment["terminal_reaped"] += 1

    # Health is intentionally all matching arm traffic after the cut, not
    # only enrolled games: a partial game can still expose invalids or 503s.
    for turn in parsed.turns.values():
        if (
            turn.agent not in experiment.agents
            or turn.family != experiment.family
            or turn.ts < experiment.cutoff
        ):
            continue
        health = agent_data[turn.agent]["health"]
        health["turns"] += 1
        health["turn_errors"] += int(bool(turn.error))
        health["corrections"] += len(turn.corrections)
        health["turns_with_corrections"] += int(bool(turn.corrections))
        if "503" in str(turn.error or ""):
            health["http_503"] += 1
    for agent, family, ts in parsed.duplicate_events:
        if (
            agent in experiment.agents
            and family == experiment.family
            and ts >= experiment.cutoff
        ):
            agent_data[agent]["health"]["duplicate_turns"] += 1
    for event in parsed.results:
        key = (event.agent, event.gid)
        if (
            event.agent not in experiment.agents
            or event.ts < experiment.cutoff
            or parsed.gid_family.get(key) != experiment.family
        ):
            continue
        health = agent_data[event.agent]["health"]
        health["result_events"] += 1
        health["invalid_results"] += int(event.valid is False)
        health["result_errors"] += int(bool(event.error))
        if "503" in str(event.error or ""):
            health["http_503"] += 1

    control_knobs = _knobs(experiment, "control")
    treatment_knobs = _knobs(experiment, "treatment")
    affected: list[dict] = []
    games_with_divergence: set[tuple[str, str]] = set()
    for turn in sorted(parsed.turns.values(), key=lambda item: item.ts):
        key = (turn.agent, turn.gid)
        if key not in enrolled or turn.ts < experiment.cutoff:
            continue
        routing = agent_data[turn.agent]["routing"]
        try:
            control_action = replay(turn.game, control_knobs)
            treatment_action = replay(turn.game, treatment_knobs)
        except Exception as exc:  # report a replay bug as a routing miss
            control_action = {"_replay_error": f"{type(exc).__name__}: {exc}"}
            treatment_action = control_action
            routing["replay_errors"] += 1
        if experiment.name == "neg_terminal_close":
            assigned_variant = _neg_gate_assignment(
                parsed, experiment, turn.agent, turn.ts
            )[0]
        else:
            assigned_variant = experiment.variant_for(turn.agent)
        assigned_action = (
            treatment_action if assigned_variant == "treatment" else control_action
        )
        other_action = (
            control_action if assigned_variant == "treatment" else treatment_action
        )
        assigned_match = _action_equal(turn.action, assigned_action)
        routing["checked"] += 1
        routing["assigned_matches"] += int(assigned_match)
        if _action_equal(control_action, treatment_action):
            continue
        violation = _direction_violation(
            experiment, turn, control_action, treatment_action
        )
        other_match = _action_equal(turn.action, other_action)
        routing["affected"] += 1
        routing["affected_assigned_matches"] += int(assigned_match)
        routing["affected_wrong_variant"] += int(not assigned_match and other_match)
        routing["affected_unknown"] += int(not assigned_match and not other_match)
        routing["direction_violations"] += int(violation)
        first_for_game = key not in games_with_divergence
        games_with_divergence.add(key)
        affected.append(
            {
                "agent": turn.agent,
                "variant": assigned_variant,
                "game_id": turn.gid,
                "ts": turn.ts,
                "round": turn.round,
                "phase": turn.phase,
                "horizon": _horizon(turn),
                "max_rounds": _max_rounds(turn),
                "effective_offer_round": _effective_offer_round(turn, assigned_action),
                "first_for_game": first_for_game,
                "logged_action": turn.action,
                "control_action": control_action,
                "treatment_action": treatment_action,
                "assigned_match": assigned_match,
                "other_variant_match": other_match,
                "direction_violation": violation,
            }
        )

    variants = {"treatment": _variant_summary(), "control": _variant_summary()}
    for agent, data in agent_data.items():
        variant = experiment.variant_for(agent)
        enrollment = data["enrollment"]
        routing = data["routing"]
        variants[variant]["games"] += enrollment["enrolled"]
        variants[variant]["resolved"] += enrollment["resolved"]
        variants[variant]["censored"] += enrollment["censored"]
        variants[variant]["affected_turns"] += routing["affected"]
        variants[variant]["direction_violations"] += routing["direction_violations"]
    for variant in variants:
        variants[variant]["affected_games"] = len(
            {
                (item["agent"], item["game_id"])
                for item in affected
                if item["variant"] == variant
            }
        )

    if experiment.family == "persuasion":
        metrics = _persuasion_outcomes(affected, enrolled, parsed, experiment)
    else:
        metrics = _offer_outcomes(affected, enrolled, parsed, experiment)

    report = {
        "name": experiment.name,
        "family": experiment.family,
        "cutoff": experiment.cutoff,
        "cutoff_utc": datetime.fromtimestamp(experiment.cutoff, timezone.utc).isoformat(),
        "assignment": {
            "knob": experiment.knob,
            "treatment_value": experiment.treatment_value,
            "control_value": experiment.control_value,
            "treatment_agents": list(experiment.treatment_agents),
            "control_agents": list(experiment.control_agents),
        },
        "agents": agent_data,
        "variants": variants,
        "metrics": metrics,
        "affected_turns": affected,
    }
    if experiment.name == "neg_terminal_close":
        gate_rows = _neg_terminal_gate_rows(affected, enrolled, parsed, experiment)
        report["gate"] = _neg_terminal_gate_from_rows(
            gate_rows,
            agent_data,
            {agent: experiment.variant_for(agent) for agent in experiment.agents},
            epoch_health=_neg_gate_epoch_health(
                parsed, experiment, affected, agent_data
            ),
        )
    return report


def build_report(
    records: Iterable[dict],
    preexisting: set[tuple[str, str]] | None = None,
    experiments: Iterable[Experiment] = EXPERIMENTS,
    replay: ReplayFn = replay_action,
) -> dict:
    parsed = parse_records(records)
    prior = preexisting or set()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiments": [
            analyze_experiment(parsed, prior, experiment, replay)
            for experiment in experiments
        ],
    }


_TS_RE = re.compile(rb'"ts"\s*:\s*([0-9]+(?:\.[0-9]+)?)')
_GID_RE = re.compile(rb'"game_id"\s*:\s*"([^"]+)"')


def _line_ts(line: bytes) -> float | None:
    match = _TS_RE.search(line[:512])
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def seek_timestamp(path: Path, cutoff: float) -> int:
    """Return a line boundary at the first record whose timestamp >= cutoff."""
    size = path.stat().st_size
    if size == 0:
        return 0
    with path.open("rb") as handle:
        low = 0  # known line boundary; all consumed records are below cutoff
        high = size
        while high - low > 2 * 1024 * 1024:
            mid = (low + high) // 2
            handle.seek(mid)
            if mid:
                handle.readline()  # discard the partial line
            pos = handle.tell()
            if pos >= high:
                high = mid
                continue
            line = handle.readline()
            ts = _line_ts(line)
            if ts is None:
                low = handle.tell()
            elif ts < cutoff:
                low = handle.tell()
            else:
                high = pos
        handle.seek(low)
        while True:
            pos = handle.tell()
            line = handle.readline()
            if not line:
                return handle.tell()
            ts = _line_ts(line)
            if ts is not None and ts >= cutoff:
                return pos


@dataclass(frozen=True)
class LogSlice:
    agent: str
    path: Path
    offset: int


def discover_log_slices(
    log_dir: Path, experiments: Iterable[Experiment]
) -> tuple[list[LogSlice], set[tuple[str, str]]]:
    experiments = tuple(experiments)
    earliest = min(experiment.cutoff for experiment in experiments)
    earliest_day = datetime.fromtimestamp(earliest, timezone.utc).strftime("%Y%m%d")
    relevant_agents = sorted({agent for exp in experiments for agent in exp.agents})
    slices: list[LogSlice] = []
    preexisting: set[tuple[str, str]] = set()

    for agent in relevant_agents:
        for path in sorted(log_dir.glob(f"{agent}-*.jsonl")):
            suffix = path.stem.rsplit("-", 1)[-1]
            if len(suffix) != 8 or not suffix.isdigit() or suffix < earliest_day:
                continue
            offset = seek_timestamp(path, earliest) if suffix == earliest_day else 0
            slices.append(LogSlice(agent, path, offset))
            if offset <= 0:
                continue
            # Cheap full-prefix check: only extract turn IDs; JSON decoding the
            # large embedded histories before the cut would dominate runtime.
            with path.open("rb") as handle:
                while handle.tell() < offset:
                    line = handle.readline()
                    if not line or handle.tell() > offset:
                        break
                    if b'"type": "turn"' not in line:
                        continue
                    match = _GID_RE.search(line)
                    if match is not None:
                        preexisting.add((agent, match.group(1).decode("utf-8", "replace")))
    return slices, preexisting


def iter_log_records(slices: Iterable[LogSlice]) -> Iterator[dict]:
    for item in slices:
        with item.path.open("rb") as handle:
            handle.seek(item.offset)
            for raw in handle:
                try:
                    record = json.loads(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(record, dict):
                    record["_agent"] = item.agent
                    yield record


def _fmt_rate(value: Any) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _fmt_delta(value: Any) -> str:
    return "-" if value is None else f"{value:+.4f}"


def render_text(report: dict, detail_limit: int = 12) -> str:
    lines: list[str] = []
    for experiment in report["experiments"]:
        lines.append(
            f"{experiment['name']} ({experiment['family']}) since "
            f"{experiment['cutoff_utc']}"
        )
        for variant in ("treatment", "control"):
            cohort = experiment["variants"][variant]
            metric = experiment["metrics"][variant]
            if experiment["family"] == "persuasion":
                summary = (
                    f"blind={metric['blind_seller_games']} resolved={metric['resolved']} "
                    f"rev/max={_fmt_rate(metric['mean_revenue_share'])} "
                    f"zero={_fmt_rate(metric['zero_sales_rate'])}"
                )
            else:
                denominator_name = (
                    "pay/pot"
                    if experiment["family"] == "bargaining"
                    else "pay/value"
                )
                normalized = metric["mean_normalized_payoff"]
                normalized_text = "-" if normalized is None else f"{normalized:.4f}"
                summary = (
                    f"eligible={metric['affected_games']} resolved={metric['resolved']} "
                    f"{denominator_name}={normalized_text} "
                    f"direct={metric['direct_converted']}/{metric['direct_resolved']} "
                    f"({_fmt_rate(metric['direct_conversion_rate'])})"
                )
            lines.append(
                f"  {variant:<9} cohort={cohort['games']} "
                f"resolved={cohort['resolved']} censored={cohort['censored']} | {summary}"
            )
        gate = experiment.get("gate")
        if gate is not None:
            direct = gate["standardized"]["direct"]
            primary = gate["counts"]["variants"]
            lines.append(
                f"  frozen gate: {gate['promotion']['status']} "
                f"stage={gate['interim']['stage']} "
                f"primary_n=T{primary['treatment']['primary']['direct_trials']}/"
                f"C{primary['control']['primary']['direct_trials']} "
                f"direct_uplift={_fmt_delta(direct['uplift'])} "
                f"lower95={_fmt_delta(direct['one_sided_95_lower'])} "
                f"unsupported={gate['counts']['unsupported']['total']}"
            )
        lines.append("  health/routing by agent:")
        for agent, data in experiment["agents"].items():
            enrollment = data["enrollment"]
            health = data["health"]
            routing = data["routing"]
            lines.append(
                f"    {agent:<7} enrolled={enrollment['enrolled']} "
                f"censored={enrollment['censored']} turns={health['turns']} "
                f"invalid={health['invalid_results']} "
                f"errors={health['result_errors'] + health['turn_errors']} "
                f"503={health['http_503']} corrections={health['corrections']} "
                f"route={routing['assigned_matches']}/{routing['checked']} "
                f"replay_err={routing['replay_errors']} "
                f"affected={routing['affected']} wrong={routing['affected_wrong_variant']} "
                f"unknown={routing['affected_unknown']} dir_bad={routing['direction_violations']}"
            )
        affected = experiment["affected_turns"]
        if affected:
            if detail_limit <= 0:
                lines.append(
                    f"  exact affected: {len(affected)} turns (use --json for IDs)"
                )
                lines.append("")
                continue
            tokens = [
                f"{item['agent']}:{item['game_id']}:{item['round']}/{item['phase']}"
                for item in affected[:detail_limit]
            ]
            suffix = (
                ""
                if len(affected) <= detail_limit
                else f" ... +{len(affected) - detail_limit} (use --json)"
            )
            lines.append("  exact affected: " + ", ".join(tokens) + suffix)
        else:
            lines.append("  exact affected: none yet")
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", type=Path, default=ROOT / "logs")
    parser.add_argument("--json", action="store_true", help="emit complete JSON")
    parser.add_argument(
        "--experiment",
        choices=[experiment.name for experiment in EXPERIMENTS],
        action="append",
        help="limit to one or more experiments",
    )
    parser.add_argument("--detail-limit", type=int, default=12)
    args = parser.parse_args()

    selected = tuple(
        experiment
        for experiment in EXPERIMENTS
        if not args.experiment or experiment.name in args.experiment
    )
    slices, preexisting = discover_log_slices(args.logs, selected)
    report = build_report(iter_log_records(slices), preexisting, selected)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        print(render_text(report, max(args.detail_limit, 0)))


if __name__ == "__main__":
    main()
