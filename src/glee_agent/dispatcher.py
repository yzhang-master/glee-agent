"""The strategy entrypoint handed to GleeClient.run().

Pipeline per turn: parse -> assignment -> family strategy -> guard -> log ->
return.  Ordinary strategy failures are wrapped and guarded.  An assigned
canary move is intentionally withheld (by raising) only when its write-ahead
receipt is indeterminate or its turn audit cannot be written; returning the
base policy in either case could switch experiment arms mid-game.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from .canary_assignment import CanaryAssigner, LoadedAssignmentPlan
from .config import Settings
from .families import bargaining, negotiation, persuasion
from .guard import fallback_action, guard
from .logging_ import log_turn
from .schema import parse_game

logger = logging.getLogger("glee_agent")


class CanaryTelemetryUnavailable(RuntimeError):
    """An assigned canary move is withheld when its turn audit cannot land."""


FAMILIES = {
    "bargaining": bargaining.decide,
    "negotiation": negotiation.decide,
    "persuasion": persuasion.decide,
}


def build_strategy(
    settings: Settings,
    canary_assignment: LoadedAssignmentPlan | None = None,
) -> Callable[[dict], dict]:
    loaded = canary_assignment or LoadedAssignmentPlan(
        status="missing",
        artifact_path="data/canary_assignment.json",
        artifact_sha256=None,
        artifact_bytes=None,
    )
    assigner = CanaryAssigner(loaded, settings.agent_label)

    def strategy(game: dict) -> dict:
        start = time.monotonic()
        corrections: list[str] = []
        error: str | None = None
        view = parse_game(game)
        assignment = assigner.assignment_for(view)
        effective_knobs = assignment.apply(settings.knobs)
        try:
            decide = FAMILIES.get(view.family)
            if decide is None:
                corrections.append(f"no strategy for family {view.family!r}")
                proposed = fallback_action(view)
            else:
                proposed = decide(view, effective_knobs)
        except Exception as e:  # noqa: BLE001 — strategy bugs must not lose games
            error = f"{type(e).__name__}: {e}"
            logger.exception("Strategy error in %s game %s", view.family, view.game_id)
            proposed = fallback_action(view)

        action, guard_notes = guard(proposed, view)
        corrections.extend(guard_notes)
        corrections.extend(view.warnings)

        if corrections:
            logger.warning(
                "Turn corrections for %s (%s): %s", view.game_id, view.family, corrections
            )

        elapsed = time.monotonic() - start
        logged = log_turn(
            settings.agent_label,
            game,
            action,
            corrections,
            elapsed,
            error=error,
            canary_assignment=assignment.log_metadata(),
        )
        if assignment.assigned and not logged:
            raise CanaryTelemetryUnavailable(
                f"canary assignment telemetry unavailable for game {view.game_id!r}"
            )
        return action

    return strategy
