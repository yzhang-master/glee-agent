"""Security and identity tests for the frozen confirmation declaration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.confirmation_plan import (
    ASSIGNMENT_ARTIFACT_BYTES,
    ASSIGNMENT_ARTIFACT_SHA256,
    ASSIGNMENT_ENROLLMENT_CUTOFF_EXCLUSIVE,
    DECLARATION_BYTES,
    DECLARATION_CANONICAL_SHA256,
    DECLARATION_SHA256,
    DECLARED_AT,
    FINAL_ANALYSIS_AS_OF,
    MAX_DECLARATION_BYTES,
    load_confirmation_plan,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DECLARATION_SOURCE = REPOSITORY_ROOT / "data/canary_analysis_plan.json"
ASSIGNMENT_SOURCE = REPOSITORY_ROOT / "data/canary_assignment.json"


def _copy_valid_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data"
    data.mkdir()
    declaration = data / "canary_analysis_plan.json"
    assignment = data / "canary_assignment.json"
    declaration.write_bytes(DECLARATION_SOURCE.read_bytes())
    assignment.write_bytes(ASSIGNMENT_SOURCE.read_bytes())
    return declaration, assignment


def _rewrite_document(path: Path, mutate) -> None:
    document = json.loads(path.read_text())
    mutate(document)
    path.write_text(json.dumps(document, indent=2) + "\n")


def test_repository_default_path_resolves_outside_the_strategy_package():
    loaded = load_confirmation_plan()

    assert loaded.valid
    assert loaded.artifact_path == "data/canary_analysis_plan.json"


def test_frozen_declaration_and_linked_assignment_load_with_exact_identities(
    tmp_path,
):
    _copy_valid_artifacts(tmp_path)

    loaded = load_confirmation_plan(project_root=tmp_path)

    assert loaded.valid
    assert loaded.error_code is None
    assert loaded.linked_assignment_verified is True
    assert loaded.artifact_sha256 == DECLARATION_SHA256
    assert loaded.artifact_canonical_sha256 == DECLARATION_CANONICAL_SHA256
    assert loaded.artifact_bytes == DECLARATION_BYTES
    assert loaded.plan is not None
    assert loaded.plan.declared_at == DECLARED_AT
    assert loaded.plan.declared_at < loaded.plan.assignment_activated_at
    assert (
        loaded.plan.assignment_enrollment_cutoff_exclusive
        == ASSIGNMENT_ENROLLMENT_CUTOFF_EXCLUSIVE
    )
    assert loaded.plan.analysis_as_of == FINAL_ANALYSIS_AS_OF
    assert loaded.plan.analysis_as_of == (
        loaded.plan.assignment_enrollment_cutoff_exclusive + 1800
    )
    assert loaded.artifact_manifest() == {
        "path": "data/canary_analysis_plan.json",
        "available": True,
        "sha256": DECLARATION_SHA256,
        "canonical_sha256": DECLARATION_CANONICAL_SHA256,
        "bytes": DECLARATION_BYTES,
        "linked_assignment_verified": True,
    }
    assignment_raw = ASSIGNMENT_SOURCE.read_bytes()
    assert len(assignment_raw) == ASSIGNMENT_ARTIFACT_BYTES
    assert hashlib.sha256(assignment_raw).hexdigest() == ASSIGNMENT_ARTIFACT_SHA256


def test_missing_declaration_is_explicitly_unavailable(tmp_path):
    loaded = load_confirmation_plan(project_root=tmp_path)

    assert loaded.status == "missing"
    assert loaded.valid is False
    assert loaded.error_code is None
    assert loaded.artifact_sha256 is None


def test_semantically_equivalent_byte_tampering_fails_exact_identity(tmp_path):
    declaration, _assignment = _copy_valid_artifacts(tmp_path)
    declaration.write_bytes(declaration.read_bytes() + b" ")

    loaded = load_confirmation_plan(project_root=tmp_path)

    assert loaded.status == "invalid"
    assert loaded.error_code == "declaration_identity_mismatch"
    assert loaded.artifact_canonical_sha256 == DECLARATION_CANONICAL_SHA256
    assert loaded.artifact_sha256 != DECLARATION_SHA256


@pytest.mark.parametrize(
    ("mutate", "error_code"),
    [
        (lambda document: document.update(declared_at=float(DECLARED_AT)), "invalid_declared_at"),
        (lambda document: document.update(extra=True), "invalid_top_level_fields"),
        (
            lambda document: document["scheduled_look"].update(
                efficacy_binding=1
            ),
            "invalid_efficacy_binding",
        ),
        (
            lambda document: document["assignment"].update(artifact_path=[]),
            "invalid_assignment_artifact_path",
        ),
        (
            lambda document: document["families"]["persuasion"].update(
                maturity_lag_seconds="1800"
            ),
            "invalid_persuasion_maturity_lag",
        ),
    ],
)
def test_wrong_types_and_extra_fields_fail_closed(tmp_path, mutate, error_code):
    declaration, _assignment = _copy_valid_artifacts(tmp_path)
    _rewrite_document(declaration, mutate)

    loaded = load_confirmation_plan(project_root=tmp_path)

    assert loaded.status == "invalid"
    assert loaded.error_code == error_code
    assert loaded.plan is None


def test_duplicate_keys_and_oversize_fail_before_becoming_a_plan(tmp_path):
    declaration, _assignment = _copy_valid_artifacts(tmp_path)
    declaration.write_text('{"schema_version":1,"schema_version":1}')

    duplicate = load_confirmation_plan(project_root=tmp_path)

    assert duplicate.status == "invalid"
    assert duplicate.error_code == "duplicate_json_key"

    declaration.write_bytes(b" " * (MAX_DECLARATION_BYTES + 1))
    oversized = load_confirmation_plan(project_root=tmp_path)
    assert oversized.status == "invalid"
    assert oversized.error_code == "artifact_too_large"
    assert oversized.plan is None


def test_huge_json_integer_fails_closed_as_invalid_json(tmp_path):
    declaration, _assignment = _copy_valid_artifacts(tmp_path)
    declaration.write_bytes(b'{"declared_at":' + b"9" * 10_000 + b"}")

    loaded = load_confirmation_plan(project_root=tmp_path)

    assert loaded.status == "invalid"
    assert loaded.error_code == "invalid_json"
    assert loaded.plan is None


@pytest.mark.parametrize(
    "relative_path",
    (Path("../canary_analysis_plan.json"), Path("/tmp/canary_analysis_plan.json")),
)
def test_parent_and_absolute_declaration_paths_are_rejected(tmp_path, relative_path):
    loaded = load_confirmation_plan(
        project_root=tmp_path,
        relative_path=relative_path,
    )

    assert loaded.status == "invalid"
    assert loaded.error_code == "invalid_artifact_path"


def test_declaration_file_symlink_is_never_followed(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    outside = tmp_path / "outside-declaration.json"
    outside.write_bytes(DECLARATION_SOURCE.read_bytes())
    (data / "canary_analysis_plan.json").symlink_to(outside)
    (data / "canary_assignment.json").write_bytes(ASSIGNMENT_SOURCE.read_bytes())

    loaded = load_confirmation_plan(project_root=tmp_path)

    assert loaded.status == "invalid"
    assert loaded.error_code == "artifact_symlink"
    assert loaded.artifact_sha256 is None


def test_symlinked_parent_component_is_never_followed(tmp_path):
    actual = tmp_path / "actual-data"
    actual.mkdir()
    (actual / "canary_analysis_plan.json").write_bytes(DECLARATION_SOURCE.read_bytes())
    (actual / "canary_assignment.json").write_bytes(ASSIGNMENT_SOURCE.read_bytes())
    (tmp_path / "data").symlink_to(actual, target_is_directory=True)

    loaded = load_confirmation_plan(project_root=tmp_path)

    assert loaded.status == "invalid"
    assert loaded.error_code == "artifact_symlink"


def test_linked_assignment_symlink_and_content_tampering_fail_closed(tmp_path):
    _declaration, assignment = _copy_valid_artifacts(tmp_path)
    outside = tmp_path / "outside-assignment.json"
    outside.write_bytes(ASSIGNMENT_SOURCE.read_bytes())
    assignment.unlink()
    assignment.symlink_to(outside)

    symlinked = load_confirmation_plan(project_root=tmp_path)

    assert symlinked.status == "invalid"
    assert symlinked.error_code == "assignment_artifact_symlink"
    assert symlinked.linked_assignment_verified is False

    assignment.unlink()
    assignment.write_bytes(ASSIGNMENT_SOURCE.read_bytes() + b" ")
    tampered = load_confirmation_plan(project_root=tmp_path)
    assert tampered.status == "invalid"
    assert tampered.error_code == "assignment_artifact_identity_mismatch"
    assert tampered.linked_assignment_verified is False


def test_linked_assignment_missing_and_oversize_fail_closed(tmp_path):
    declaration, assignment = _copy_valid_artifacts(tmp_path)
    assignment.unlink()

    missing = load_confirmation_plan(project_root=tmp_path)

    assert declaration.exists()
    assert missing.status == "invalid"
    assert missing.error_code == "assignment_artifact_missing"

    assignment.write_bytes(b"x" * (64 * 1024 + 1))
    oversized = load_confirmation_plan(project_root=tmp_path)
    assert oversized.status == "invalid"
    assert oversized.error_code == "assignment_artifact_too_large"
