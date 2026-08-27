"""The exploration arm's analysis plan is pre-registered and pinned.

Written before any exploration data exists.  These tests exist so the plan
cannot be quietly loosened after the curve is seen -- which is the failure
mode the plan is designed to prevent.
"""

import json
from pathlib import Path

from glee_agent.families import negotiation

PLAN = json.loads(
    (Path(__file__).resolve().parents[1] / "data/price_explore_analysis_plan.json")
    .read_text()
)


def test_plan_matches_the_shipped_ladder_and_salt():
    assert tuple(PLAN["arm"]["ladder"]) == negotiation._II_EXPLORE_LADDER
    assert PLAN["arm"]["salt"].encode() == negotiation._II_EXPLORE_SALT
    assert PLAN["arm"]["default"] == 0.0


def test_plan_is_declared_before_data_exists():
    assert PLAN["status"] == "pre_registered_before_any_exploration_data_exists"


def test_unit_of_analysis_is_the_game():
    assert PLAN["prespecified_analysis"]["unit_of_analysis"].startswith("one game")


def test_selection_and_estimation_are_split():
    split = PLAN["prespecified_analysis"]["selection_and_estimation_must_be_split"]
    assert "first half" in split["rule"]
    assert "second half" in split["rule"]


def test_per_cell_argmax_is_forbidden():
    forbidden = PLAN["forbidden"]
    assert "per-cell argmax over the ladder" in forbidden
    assert "reporting an in-sample selected effect size" in forbidden
    assert any("turn or by rejection" in f for f in forbidden)


def test_a_null_result_is_prespecified_as_acceptable():
    assert "NO EFFECT" in PLAN["prespecified_null_result"]


def test_held_out_replication_is_required():
    req = PLAN["prespecified_analysis"]["replication_requirement"]
    assert "3 of 4" in req and "held-out" in req


def test_rejected_candidates_are_recorded_with_their_cause_of_death():
    killed = PLAN["overfitting_provenance"]["candidates_examined_and_rejected_2026_08_26"]
    assert len(killed) == 5
    assert all(c["killed_by"] for c in killed)
