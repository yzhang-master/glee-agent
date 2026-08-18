"""The strategy entrypoint handed to GleeClient.run().

Pipeline per turn: parse -> family strategy -> guard -> log -> return.
The whole pipeline is wrapped so it can never raise and never exceed the
turn budget by much (family strategies are pure computation, microseconds).
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from .config import Settings
from .families import bargaining, negotiation, persuasion
from .guard import fallback_action, guard
from .logging_ import log_turn
from .schema import parse_game

logger = logging.getLogger("glee_agent")

FAMILIES = {
    "bargaining": bargaining.decide,
    "negotiation": negotiation.decide,
    "persuasion": persuasion.decide,
}


def build_strategy(settings: Settings) -> Callable[[dict], dict]:
    def strategy(game: dict) -> dict:
        start = time.monotonic()
        corrections: list[str] = []
        error: str | None = None
        view = parse_game(game)
        try:
            decide = FAMILIES.get(view.family)
            if decide is None:
                corrections.append(f"no strategy for family {view.family!r}")
                proposed = fallback_action(view)
            else:
                proposed = decide(view, settings.knobs)
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
        log_turn(settings.agent_label, game, action, corrections, elapsed, error=error)
        return action

    return strategy
