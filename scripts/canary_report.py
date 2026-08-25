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
import json
import math
import re
import sys
from collections import defaultdict
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
class ParsedRecords:
    turns: dict[tuple[str, str, int, str], Turn]
    first_turns: dict[tuple[str, str], Turn]
    results: list[ResultEvent]
    terminals: dict[tuple[str, str], ResultEvent]
    duplicate_events: list[tuple[str, str, float]]
    gid_family: dict[tuple[str, str], str]


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
        if kind == "turn":
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
        variant = experiment.variant_for(turn.agent)
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

    return {
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
