"""Fail-closed, deterministic per-game live-canary assignment."""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest

import glee_agent.dispatcher as dispatcher
import glee_agent.canary_assignment as canary_assignment_module
from fixtures import bargaining_game, negotiation_game, persuasion_game
from glee_agent.canary_assignment import (
    MAX_RECEIPT_BYTES,
    CanaryAssigner,
    LoadedAssignmentPlan,
    load_assignment_plan,
    receipt_relative_path,
)
from glee_agent.config import Knobs
from glee_agent.runtime_manifest import build_runtime_manifest
from glee_agent.schema import parse_game


def _document(*, probability=0.5, activated_at=0, expires_at=4_000_000_000):
    return {
        "schema_version": 1,
        "plan_id": "rotation-2026-08-25",
        "assignment_salt": "public-seed-01",
        "activated_at": activated_at,
        "expires_at": expires_at,
        "agents": ["test_a", "test_b"],
        "rules": {
            "bargaining": {
                "rule_id": "barg-anchor-v1",
                "knob": "barg_dis_anchor",
                "control": 0.58,
                "treatment": 0.50,
                "treatment_probability": probability,
            },
            "negotiation": {
                "rule_id": "neg-close-v1",
                "knob": "neg_terminal_close",
                "control": False,
                "treatment": True,
                "treatment_probability": probability,
            },
            "persuasion": {
                "rule_id": "pers-lie-v1",
                "knob": "pers_blind_lie",
                "control": 1.0,
                "treatment": 0.4,
                "treatment_probability": probability,
            },
        },
    }


def _write_plan(tmp_path, document=None):
    path = tmp_path / "data/canary_assignment.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document or _document(), separators=(",", ":")))
    return load_assignment_plan(project_root=tmp_path), path


def _turn(game, *, game_id=None, round_=None, history=None):
    result = copy.deepcopy(game)
    if game_id is not None:
        result["game_id"] = game_id
    if round_ is not None:
        result["game_state"]["round"] = round_
    if history is not None:
        result["game_state"]["history"] = history
    return parse_game(result)


def test_missing_plan_is_disabled_and_has_explicit_artifact_identity(tmp_path):
    loaded = load_assignment_plan(project_root=tmp_path)

    assert loaded.status == "missing"
    assert loaded.plan is None
    assert loaded.manifest() == {
        "loader_status": "missing",
        "error_code": None,
        "artifact": {
            "path": "data/canary_assignment.json",
            "available": False,
            "sha256": None,
            "bytes": None,
        },
        "contract": None,
    }
    assignment = CanaryAssigner(loaded, "test_a", clock=lambda: 150).assignment_for(
        _turn(bargaining_game())
    )
    assert not assignment.assigned
    assert assignment.reason == "plan_missing"


@pytest.mark.parametrize(
    ("mutate", "error_code"),
    [
        (lambda d: d.update(schema_version="1"), "unsupported_schema_version"),
        (lambda d: d.update(extra=True), "invalid_plan_fields"),
        (
            lambda d: d["rules"]["bargaining"].update(knob="barg_anchor_agent"),
            "unknown_rule_knob",
        ),
        (
            lambda d: d["rules"]["bargaining"].update(treatment="0.5"),
            "invalid_rule_value",
        ),
        (
            lambda d: d["rules"]["negotiation"].update(treatment=1),
            "invalid_rule_value",
        ),
        (
            lambda d: d["rules"].update(
                unknown={
                    "rule_id": "bad",
                    "knob": "bad",
                    "control": 0,
                    "treatment": 1,
                    "treatment_probability": 0.5,
                }
            ),
            "unknown_family",
        ),
        (
            lambda d: d["rules"]["persuasion"].update(
                treatment_probability=True
            ),
            "invalid_treatment_probability",
        ),
        (lambda d: d.update(expires_at=d["activated_at"]), "invalid_enrollment_window"),
        (lambda d: d.update(agents=["test_typo"]), "unknown_agent"),
    ],
)
def test_malformed_unknown_knob_value_and_type_fail_closed(
    tmp_path, mutate, error_code
):
    document = _document()
    mutate(document)
    loaded, _ = _write_plan(tmp_path, document)

    assert loaded.status == "invalid"
    assert loaded.error_code == error_code
    assignment = CanaryAssigner(loaded, "test_a", clock=lambda: 150).assignment_for(
        _turn(bargaining_game())
    )
    assert not assignment.assigned
    assert assignment.reason == "plan_invalid"


def test_invalid_json_duplicate_key_and_oversize_are_fail_closed(tmp_path):
    path = tmp_path / "data/canary_assignment.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version":1,"schema_version":1}')
    duplicate = load_assignment_plan(project_root=tmp_path)
    assert duplicate.status == "invalid"
    assert duplicate.error_code == "duplicate_json_key"

    path.write_bytes(b" " * (64 * 1024 + 1))
    oversized = load_assignment_plan(project_root=tmp_path)
    assert oversized.status == "invalid"
    assert oversized.error_code == "artifact_too_large"
    assert oversized.artifact_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_absolute_parent_paths_and_noninteger_timestamps_are_rejected(tmp_path):
    absolute = load_assignment_plan(
        project_root=tmp_path, relative_path=tmp_path / "plan.json"
    )
    parent = load_assignment_plan(
        project_root=tmp_path, relative_path="../canary_assignment.json"
    )
    assert absolute.error_code == "invalid_artifact_path"
    assert parent.error_code == "invalid_artifact_path"

    document = _document(activated_at=1.0)
    loaded, _ = _write_plan(tmp_path, document)
    assert loaded.status == "invalid"
    assert loaded.error_code == "invalid_activated_at"


def test_assignment_is_stable_across_turns_and_after_expiry(tmp_path):
    loaded, _ = _write_plan(tmp_path)
    now = [150.0]
    assigner = CanaryAssigner(loaded, "test_a", clock=lambda: now[0])
    first = assigner.assignment_for(_turn(bargaining_game(), game_id="stable"))

    now[0] = 999.0
    later = assigner.assignment_for(
        _turn(
            bargaining_game(),
            game_id="stable",
            round_=7,
            history=[{"round": 1}, {"round": 2}],
        )
    )

    assert first.assigned
    assert later is first
    assert first.assignment_sha256 == later.assignment_sha256
    assert first.apply(Knobs()).barg_dis_anchor == first.value


def test_hash_domain_includes_agent_family_and_game(tmp_path):
    loaded, _ = _write_plan(tmp_path)
    bargain_a = CanaryAssigner(loaded, "test_a", clock=lambda: 150).assignment_for(
        _turn(bargaining_game(), game_id="same")
    )
    bargain_b = CanaryAssigner(loaded, "test_b", clock=lambda: 150).assignment_for(
        _turn(bargaining_game(), game_id="same")
    )
    persuasion_a = CanaryAssigner(loaded, "test_a", clock=lambda: 150).assignment_for(
        _turn(persuasion_game(actor="seller"), game_id="same")
    )
    other_game = CanaryAssigner(loaded, "test_a", clock=lambda: 150).assignment_for(
        _turn(bargaining_game(), game_id="different")
    )

    digests = {
        bargain_a.assignment_sha256,
        bargain_b.assignment_sha256,
        persuasion_a.assignment_sha256,
        other_game.assignment_sha256,
    }
    assert len(digests) == 4


def test_treatment_probability_uses_replayable_integer_threshold(tmp_path):
    loaded, _ = _write_plan(tmp_path, _document(probability=0.25))
    assigner = CanaryAssigner(loaded, "test_a", clock=lambda: 150)
    # Durability has dedicated tests below; avoid 2,000 fsyncs in a math test.
    assigner._persist_receipt = lambda _game_id, _assignment: True
    assignments = [
        assigner.assignment_for(_turn(bargaining_game(), game_id=f"g-{index}"))
        for index in range(2000)
    ]
    treatment = 0
    for assignment in assignments:
        expected = assignment.assignment_u64 < assignment.treatment_threshold_u64
        assert (assignment.arm == "treatment") is expected
        assert assignment.treatment_threshold_u64 == int(0.25 * (1 << 64))
        treatment += expected
    assert 0.22 < treatment / len(assignments) < 0.28


def test_activation_expiry_partial_and_agent_scope_are_strict(tmp_path):
    loaded, _ = _write_plan(
        tmp_path, _document(activated_at=100, expires_at=200)
    )

    before_time = [99.0]
    before = CanaryAssigner(loaded, "test_a", clock=lambda: before_time[0])
    assert before.assignment_for(_turn(bargaining_game(), game_id="before")).reason == (
        "before_activation"
    )
    before_time[0] = 150.0
    assert before.assignment_for(_turn(bargaining_game(), game_id="before")).reason == (
        "before_activation"
    )

    expired = CanaryAssigner(loaded, "test_a", clock=lambda: 200)
    assert expired.assignment_for(_turn(bargaining_game(), game_id="expired")).reason == (
        "after_expiry"
    )

    partial_round = CanaryAssigner(loaded, "test_a", clock=lambda: 150)
    assert partial_round.assignment_for(
        _turn(bargaining_game(), game_id="partial-round", round_=2)
    ).reason == "partial_game"
    assert partial_round.assignment_for(
        _turn(bargaining_game(), game_id="partial-round", round_=1, history=[])
    ).reason == "partial_game"

    partial_history = CanaryAssigner(loaded, "test_a", clock=lambda: 150)
    assert partial_history.assignment_for(
        _turn(bargaining_game(), game_id="partial-history", history=[{"round": 1}])
    ).reason == "partial_game"

    outsider = CanaryAssigner(loaded, "main", clock=lambda: 150)
    assert outsider.assignment_for(_turn(bargaining_game())).reason == (
        "agent_not_enrolled"
    )


def test_cache_is_thread_safe(tmp_path):
    loaded, _ = _write_plan(tmp_path)
    assigner = CanaryAssigner(loaded, "test_a", clock=lambda: 150)
    view = _turn(bargaining_game(), game_id="concurrent")

    with ThreadPoolExecutor(max_workers=32) as pool:
        results = list(pool.map(lambda _index: assigner.assignment_for(view), range(1000)))

    assert all(result is results[0] for result in results)
    assert results[0].assigned
    receipt = tmp_path / receipt_relative_path("test_a", loaded.artifact_sha256)
    assert len(receipt.read_text().splitlines()) == 1


def test_receipt_capacity_covers_twice_projected_72_hour_volume(tmp_path):
    loaded, _ = _write_plan(tmp_path)
    assigner = CanaryAssigner(loaded, "test_a", clock=lambda: 150)
    assignment = assigner.assignment_for(
        _turn(bargaining_game(), game_id="capacity-sample")
    )
    encoded = json.dumps(
        assigner._receipt_record("capacity-sample", assignment),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8") + b"\n"

    assert MAX_RECEIPT_BYTES // len(encoded) >= 60_000


def test_write_ahead_receipt_recovers_arm_after_process_restart(tmp_path):
    loaded, _ = _write_plan(tmp_path)
    first_process = CanaryAssigner(loaded, "test_a", clock=lambda: 150)
    enrolled = first_process.assignment_for(
        _turn(bargaining_game(), game_id="survives-crash")
    )
    receipt = tmp_path / receipt_relative_path("test_a", loaded.artifact_sha256)

    assert enrolled.assigned
    assert enrolled.enrollment == "new"
    assert receipt.exists()
    assert receipt.read_bytes().endswith(b"\n")

    # Existing games keep their arm even when recovery happens after expiry;
    # the window gates new enrollment, never mid-game behavior.
    restarted = CanaryAssigner(loaded, "test_a", clock=lambda: 5_000_000_000)
    recovered = restarted.assignment_for(
        _turn(
            bargaining_game(),
            game_id="survives-crash",
            round_=8,
            history=[{"round": 1}],
        )
    )

    assert recovered.assigned
    assert recovered.enrollment == "recovered"
    assert recovered.reason == "eligible"
    assert recovered.arm == enrolled.arm
    assert recovered.value == enrolled.value
    assert recovered.assignment_sha256 == enrolled.assignment_sha256
    assert recovered.artifact_sha256 == enrolled.artifact_sha256
    # Recovery is a cache hit and never creates a second enrollment receipt.
    assert len(receipt.read_text().splitlines()) == 1


def test_receipt_failure_or_corruption_disables_assignment(tmp_path):
    loaded, _ = _write_plan(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    unwritable = CanaryAssigner(loaded, "test_a", clock=lambda: 150)
    unwritable._receipt_path = blocker / "receipt.jsonl"

    failed = unwritable.assignment_for(
        _turn(bargaining_game(), game_id="cannot-persist")
    )
    assert not failed.assigned
    assert failed.reason == "receipt_persistence_failed"

    receipt = tmp_path / receipt_relative_path("test_a", loaded.artifact_sha256)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text('{"corrupt":true}\n')
    corrupt = CanaryAssigner(loaded, "test_a", clock=lambda: 150)
    disabled = corrupt.assignment_for(
        _turn(bargaining_game(), game_id="after-corruption")
    )
    assert not disabled.assigned
    assert disabled.reason == "receipt_store_invalid"


def test_truncated_receipt_tail_is_ignored_but_conflicts_fail_closed(tmp_path):
    loaded, _ = _write_plan(tmp_path)
    initial = CanaryAssigner(loaded, "test_a", clock=lambda: 150)
    enrolled = initial.assignment_for(
        _turn(bargaining_game(), game_id="receipt-integrity")
    )
    receipt = tmp_path / receipt_relative_path("test_a", loaded.artifact_sha256)
    valid_line = receipt.read_text().strip()

    with receipt.open("ab") as handle:
        handle.write(b'{"interrupted":')
    recovered = CanaryAssigner(loaded, "test_a", clock=lambda: 150).assignment_for(
        _turn(
            bargaining_game(),
            game_id="receipt-integrity",
            round_=2,
            history=[{"round": 1}],
        )
    )
    assert recovered.assigned
    assert recovered.arm == enrolled.arm
    assert recovered.enrollment == "recovered"

    conflict = json.loads(valid_line)
    conflict["enrolled_at"] += 1
    receipt.write_text(valid_line + "\n" + json.dumps(conflict) + "\n")
    conflicted = CanaryAssigner(loaded, "test_a", clock=lambda: 150)
    disabled = conflicted.assignment_for(
        _turn(bargaining_game(), game_id="new-after-conflict")
    )
    assert not disabled.assigned
    assert disabled.reason == "receipt_store_invalid"


def test_oversized_receipt_store_and_paths_fail_closed(
    tmp_path, monkeypatch
):
    loaded, _ = _write_plan(tmp_path)
    receipt = tmp_path / receipt_relative_path("test_a", loaded.artifact_sha256)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_bytes(b"x" * 101)
    monkeypatch.setattr(canary_assignment_module, "MAX_RECEIPT_BYTES", 100)

    assigner = CanaryAssigner(loaded, "test_a", clock=lambda: 150)
    assignment = assigner.assignment_for(
        _turn(bargaining_game(), game_id="oversized-store")
    )
    assert not assignment.assigned
    assert assignment.reason == "receipt_store_invalid"

    bounded = receipt_relative_path("../../escape", "../../also-escape")
    assert bounded.parts[:2] == ("logs", "canary-assignments")
    assert ".." not in bounded.parts


@pytest.mark.parametrize(
    ("family", "game", "knob"),
    [
        ("bargaining", bargaining_game(), "barg_dis_anchor"),
        ("negotiation", negotiation_game(role="seller"), "neg_terminal_close"),
        ("persuasion", persuasion_game(actor="seller"), "pers_blind_lie"),
    ],
)
def test_dispatcher_applies_family_override_and_logs_assignment(
    tmp_path, monkeypatch, family, game, knob
):
    loaded, _ = _write_plan(tmp_path)
    captured_knobs = []
    logged = []

    def decide(view, knobs):
        captured_knobs.append(knobs)
        if family == "bargaining":
            return {"alice_gain": 500, "bob_gain": 500}
        if family == "negotiation":
            return {"product_price": 100}
        return {"message": "recommend"}

    monkeypatch.setitem(dispatcher.FAMILIES, family, decide)
    monkeypatch.setattr(
        dispatcher, "log_turn", lambda *args, **kwargs: logged.append((args, kwargs))
    )
    base = replace(
        Knobs(llm_enabled=False),
        barg_dis_anchor=0.77,
        neg_terminal_close=False,
        pers_blind_lie=0.88,
    )
    strategy = dispatcher.build_strategy(
        SimpleNamespace(knobs=base, agent_label="test_a"),
        canary_assignment=loaded,
    )

    strategy(game)
    strategy(game)

    metadata = logged[0][1]["canary_assignment"]
    assert metadata["status"] == "assigned"
    assert metadata["knob"] == knob
    assert metadata == logged[1][1]["canary_assignment"]
    assert getattr(captured_knobs[0], knob) == metadata["value"]
    assert captured_knobs[0] == captured_knobs[1]
    for other in {"barg_dis_anchor", "neg_terminal_close", "pers_blind_lie"} - {knob}:
        assert getattr(captured_knobs[0], other) == getattr(base, other)


def test_runtime_manifest_pins_artifact_and_parsed_contract(tmp_path, monkeypatch):
    loaded, path = _write_plan(tmp_path)
    (tmp_path / "scripts").mkdir(exist_ok=True)
    (tmp_path / "scripts/run_agent.py").write_text("# runner\n")
    (tmp_path / "src/glee_agent").mkdir(parents=True)
    (tmp_path / "src/glee_agent/policy.py").write_text("# policy\n")
    monkeypatch.setattr("glee_agent.runtime_manifest._git_head", lambda _root: None)

    manifest = build_runtime_manifest(
        "test_a",
        Knobs(),
        project_root=tmp_path,
        pid=123,
        canary_assignment=loaded,
    )

    identity = manifest["content_hashes"]["canary_assignment"]
    assert identity == {
        "path": "data/canary_assignment.json",
        "available": True,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": len(path.read_bytes()),
    }
    canary = manifest["canary_assignment"]
    assert canary["loader_status"] == "valid"
    assert canary["contract"]["plan_id"] == "rotation-2026-08-25"
    assert canary["contract"]["assignment_algorithm"] == "sha256-u64-v1"
    assert {rule["knob"] for rule in canary["contract"]["rules"]} == {
        "barg_dis_anchor",
        "neg_terminal_close",
        "pers_blind_lie",
    }
    serialized = json.dumps(manifest, sort_keys=True).lower()
    assert "api_key" not in serialized
    assert "llm_api" not in serialized
