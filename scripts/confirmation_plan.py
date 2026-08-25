"""Strict loader for the frozen single-look canary-analysis declaration.

The declaration is intentionally independent of the live assignment loader.
It cannot affect play.  Its only job is to make the one confirmatory look,
continuous-monitoring boundary, and future physical-log-prefix procedure an
exact fail-closed artifact that reporting code can cite and verify.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECLARATION_PATH = Path("data/canary_analysis_plan.json")
SCHEMA_VERSION = 1
MAX_DECLARATION_BYTES = 64 * 1024
MAX_LINKED_ARTIFACT_BYTES = 64 * 1024

DECLARATION_ID = "confirmation-v2-final-look-20260828-2130z-r2"
DECLARATION_SHA256 = (
    "39943a3877adafae71f6bdacfab13a02f0065dc1b955ef6184fbb14dfe20e260"
)
DECLARATION_CANONICAL_SHA256 = (
    "72aefcf5c9477b0a51eda85da7da07e23f7dc0ee6ae7b5e815fc163a8f7d51b8"
)
DECLARATION_BYTES = 5950
DECLARED_AT = 1787689408

ASSIGNMENT_ARTIFACT_PATH = "data/canary_assignment.json"
ASSIGNMENT_ARTIFACT_SHA256 = (
    "b002b688d02df3233b7dd4f21a5595cf149b4cc8dd501a0bfc2ee5bccd11d745"
)
ASSIGNMENT_ARTIFACT_BYTES = 837
ASSIGNMENT_PLAN_ID = "confirmation-v2-20260825-2100z"
ASSIGNMENT_ACTIVATED_AT = 1787691600
ASSIGNMENT_ENROLLMENT_CUTOFF_EXCLUSIVE = 1787950800
ASSIGNMENT_ALGORITHM = "sha256-u64-v1"
AGENTS = ("main", "test_a", "test_b", "test_c")
STRATEGY_AGGREGATE_SHA256 = (
    "631ef69862d572644ba855174a411f80a220b11ed5c20e30b43ffc31f1303388"
)
TARGET_SHA256 = {
    "data/targets.json": (
        "1d24a579ca2b611e3b30af4ddf7af5b84ad13e7198fa55b93a2f5e6617e65e25"
    ),
    "data/live_targets.json": (
        "3dcaff69f17175648e4b46499859bf183bba03b1321364de329d01bed0e618a3"
    ),
}

FINAL_LOOK_ID = "final-confirmatory-expiry-plus-persuasion-maturity-v1"
FINAL_ANALYSIS_AS_OF = 1787952600
PREFIX_PROCEDURE_VERSION = "stable-filtered-jsonl-snapshot-v2"
PREFIX_CAPTURE_NOT_BEFORE = 1787952900
PREFIX_SETTLEMENT_DELAY_SECONDS = 300
PREFIX_OUTPUT_PATH_TEMPLATE = (
    "data/canary-confirmation-prefix/{declaration_sha256}.json"
)
FAMILY_RULES = {
    "bargaining": "barg-anchor-confirm-v2",
    "negotiation": "neg-terminal-confirm-v2",
    "persuasion": "pers-blind-confirm-v2",
}
FAMILY_POPULATIONS = {
    "bargaining": "all-strictly-enrolled-bargaining-games",
    "negotiation": (
        "supported-first-divergence-hidden-value-terminal-close-games"
    ),
    "persuasion": "all-strictly-enrolled-explicit-blind-seller-games",
}
FAMILY_MATURITY_LAGS = {
    "bargaining": 1200,
    "negotiation": 600,
    "persuasion": 1800,
}
UTC_DATES = ("20260825", "20260826", "20260827", "20260828")

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PATH_COMPONENT = re.compile(r"^[A-Za-z0-9._-]+$")

_TOP_FIELDS = {
    "schema_version",
    "declaration_id",
    "declared_at",
    "assignment",
    "scheduled_look",
    "continuous_monitoring",
    "families",
    "prefix_manifest",
}
_ASSIGNMENT_FIELDS = {
    "artifact_path",
    "artifact_sha256",
    "artifact_bytes",
    "plan_id",
    "activated_at",
    "enrollment_cutoff_exclusive",
    "assignment_algorithm",
    "agents",
    "strategy_aggregate_sha256",
    "target_sha256",
}
_LOOK_FIELDS = {
    "look_id",
    "analysis_as_of",
    "assignment_enrollment_cutoff_exclusive",
    "kind",
    "efficacy_binding",
    "promotion_binding",
    "only_efficacy_look",
}
_MONITORING_FIELDS = {
    "scope",
    "efficacy_binding",
    "promotion_binding",
    "optional_stopping_for_efficacy",
    "permitted_decisions",
}
_FAMILY_FIELDS = {
    "rule_id",
    "population_id",
    "population",
    "maturity_lag_seconds",
    "outcome_deadline_policy",
}
_PREFIX_FIELDS = {
    "procedure_version",
    "capture_not_before",
    "settlement_delay_seconds",
    "output_path_template",
    "path_order",
    "event_log_paths",
    "assignment_receipt_paths",
    "physical_read",
    "entry_schema",
    "aggregate_identity",
}
_SOURCE_FIELDS = {
    "path_template",
    "agents",
    "all_expanded_paths_required",
    "maximum_bytes_per_file",
    "prefix_rule",
}
_EVENT_SOURCE_FIELDS = _SOURCE_FIELDS | {"utc_dates"}
_PHYSICAL_READ_FIELDS = {
    "root",
    "follow_symlinks",
    "require_regular_files",
    "open_mode",
    "snapshot",
}
_ENTRY_SCHEMA = (
    "path",
    "kind",
    "device",
    "inode",
    "snapshot_bytes",
    "snapshot_sha256",
    "selected_bytes",
    "selected_records",
    "minimum_selected_timestamp",
    "maximum_selected_timestamp",
    "selection_sha256",
)


class _DeclarationError(ValueError):
    """Internal failure carrying a bounded public error code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ConfirmationPlan:
    schema_version: int
    declaration_id: str
    declared_at: int
    assignment_plan_id: str
    assignment_activated_at: int
    assignment_enrollment_cutoff_exclusive: int
    agents: tuple[str, ...]
    scheduled_look_id: str
    analysis_as_of: int
    family_maturity_lags: tuple[tuple[str, int], ...]
    prefix_procedure_version: str
    prefix_capture_not_before: int
    prefix_output_path_template: str

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "declaration_id": self.declaration_id,
            "declared_at": self.declared_at,
            "assignment_plan_id": self.assignment_plan_id,
            "assignment_activated_at": self.assignment_activated_at,
            "assignment_enrollment_cutoff_exclusive": (
                self.assignment_enrollment_cutoff_exclusive
            ),
            "agents": list(self.agents),
            "scheduled_look_id": self.scheduled_look_id,
            "analysis_as_of": self.analysis_as_of,
            "family_maturity_lags": dict(self.family_maturity_lags),
            "prefix_procedure_version": self.prefix_procedure_version,
            "prefix_capture_not_before": self.prefix_capture_not_before,
            "prefix_output_path_template": self.prefix_output_path_template,
        }


@dataclass(frozen=True)
class LoadedConfirmationPlan:
    status: Literal["valid", "missing", "invalid"]
    artifact_path: str
    artifact_sha256: str | None
    artifact_canonical_sha256: str | None
    artifact_bytes: int | None
    plan: ConfirmationPlan | None = None
    linked_assignment_verified: bool = False
    error_code: str | None = None

    @property
    def valid(self) -> bool:
        return self.status == "valid" and self.plan is not None

    def artifact_manifest(self) -> dict[str, Any]:
        return {
            "path": self.artifact_path,
            "available": self.artifact_sha256 is not None,
            "sha256": self.artifact_sha256,
            "canonical_sha256": self.artifact_canonical_sha256,
            "bytes": self.artifact_bytes,
            "linked_assignment_verified": self.linked_assignment_verified,
        }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DeclarationError("duplicate_json_key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise _DeclarationError("non_finite_json_number")


def _object(value: Any, fields: set[str], code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise _DeclarationError(code)
    return value


def _integer(value: Any, code: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _DeclarationError(code)
    return value


def _boolean(value: Any, expected: bool, code: str) -> bool:
    if type(value) is not bool or value is not expected:
        raise _DeclarationError(code)
    return value


def _string(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value:
        raise _DeclarationError(code)
    return value


def _sha256(value: Any, code: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise _DeclarationError(code)
    return value


def _exact_string_list(value: Any, expected: tuple[str, ...], code: str) -> None:
    if (
        not isinstance(value, list)
        or any(not isinstance(item, str) for item in value)
        or tuple(value) != expected
    ):
        raise _DeclarationError(code)


def _canonical_json(document: dict[str, Any]) -> bytes:
    try:
        return json.dumps(
            document,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError):
        raise _DeclarationError("invalid_canonical_json") from None


def _validate_relative_path(relative_path: Path, code: str) -> tuple[str, ...]:
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts:
        raise _DeclarationError(code)
    parts = tuple(relative.parts)
    if any(
        part in ("", ".", "..") or _PATH_COMPONENT.fullmatch(part) is None
        for part in parts
    ):
        raise _DeclarationError(code)
    return parts


def _safe_read(
    project_root: Path,
    relative_path: Path,
    *,
    maximum_bytes: int,
    prefix: str,
) -> bytes:
    """Read one regular file through a component-wise no-follow descriptor walk."""
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise _DeclarationError(f"{prefix}_nofollow_unavailable")
    parts = _validate_relative_path(relative_path, f"invalid_{prefix}_path")
    try:
        root = Path(project_root).resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        raise _DeclarationError(f"{prefix}_root_unreadable") from None

    descriptors: list[int] = []
    base_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        root_fd = os.open(root, base_flags | os.O_DIRECTORY)
        descriptors.append(root_fd)
        parent_fd = root_fd
        for component in parts[:-1]:
            try:
                metadata = os.stat(
                    component, dir_fd=parent_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                raise _DeclarationError(f"{prefix}_missing") from None
            except OSError:
                raise _DeclarationError(f"{prefix}_unreadable") from None
            if stat.S_ISLNK(metadata.st_mode):
                raise _DeclarationError(f"{prefix}_symlink")
            if not stat.S_ISDIR(metadata.st_mode):
                raise _DeclarationError(f"{prefix}_parent_not_directory")
            try:
                child_fd = os.open(
                    component,
                    base_flags | os.O_DIRECTORY,
                    dir_fd=parent_fd,
                )
            except OSError as error:
                if error.errno in (errno.ELOOP, errno.ENOTDIR):
                    raise _DeclarationError(f"{prefix}_symlink") from None
                raise _DeclarationError(f"{prefix}_unreadable") from None
            descriptors.append(child_fd)
            parent_fd = child_fd

        filename = parts[-1]
        try:
            metadata = os.stat(filename, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            raise _DeclarationError(f"{prefix}_missing") from None
        except OSError:
            raise _DeclarationError(f"{prefix}_unreadable") from None
        if stat.S_ISLNK(metadata.st_mode):
            raise _DeclarationError(f"{prefix}_symlink")

        final_flags = base_flags | getattr(os, "O_NONBLOCK", 0)
        try:
            file_fd = os.open(filename, final_flags, dir_fd=parent_fd)
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise _DeclarationError(f"{prefix}_symlink") from None
            raise _DeclarationError(f"{prefix}_unreadable") from None
        descriptors.append(file_fd)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise _DeclarationError(f"{prefix}_not_regular_file")
        if before.st_size > maximum_bytes:
            raise _DeclarationError(f"{prefix}_too_large")

        remaining = before.st_size
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(file_fd, min(remaining, 64 * 1024))
            if not chunk:
                raise _DeclarationError(f"{prefix}_changed_during_read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(file_fd)
        identity_before = (before.st_dev, before.st_ino, before.st_mode, before.st_size)
        identity_after = (after.st_dev, after.st_ino, after.st_mode, after.st_size)
        if identity_before != identity_after:
            raise _DeclarationError(f"{prefix}_changed_during_read")
        return b"".join(chunks)
    except _DeclarationError:
        raise
    except (OSError, OverflowError, ValueError):
        raise _DeclarationError(f"{prefix}_unreadable") from None
    finally:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass


def _parse_document(raw: bytes) -> tuple[dict[str, Any], ConfirmationPlan, str]:
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except _DeclarationError:
        raise
    # Python's integer-string digit limit raises a plain ``ValueError`` for
    # otherwise syntactically valid JSON containing an enormous integer.
    # Treat every decoder ValueError as invalid input instead of letting a
    # hostile declaration escape the fail-closed loader.
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise _DeclarationError("invalid_json") from None

    body = _object(document, _TOP_FIELDS, "invalid_top_level_fields")
    if _integer(body["schema_version"], "unsupported_schema_version") != SCHEMA_VERSION:
        raise _DeclarationError("unsupported_schema_version")
    if _string(body["declaration_id"], "invalid_declaration_id") != DECLARATION_ID:
        raise _DeclarationError("unexpected_declaration_id")
    declared_at = _integer(body["declared_at"], "invalid_declared_at")

    assignment = _object(
        body["assignment"], _ASSIGNMENT_FIELDS, "invalid_assignment_fields"
    )
    assignment_artifact_path = _string(
        assignment["artifact_path"], "invalid_assignment_artifact_path"
    )
    if assignment_artifact_path != ASSIGNMENT_ARTIFACT_PATH:
        raise _DeclarationError("unexpected_assignment_artifact_path")
    _validate_relative_path(
        Path(assignment_artifact_path), "invalid_assignment_artifact_path"
    )
    if (
        _sha256(assignment["artifact_sha256"], "invalid_assignment_artifact_hash")
        != ASSIGNMENT_ARTIFACT_SHA256
    ):
        raise _DeclarationError("unexpected_assignment_artifact_hash")
    if (
        _integer(assignment["artifact_bytes"], "invalid_assignment_artifact_bytes")
        != ASSIGNMENT_ARTIFACT_BYTES
    ):
        raise _DeclarationError("unexpected_assignment_artifact_bytes")
    if assignment["plan_id"] != ASSIGNMENT_PLAN_ID:
        raise _DeclarationError("unexpected_assignment_plan_id")
    activated_at = _integer(assignment["activated_at"], "invalid_activated_at")
    enrollment_cutoff = _integer(
        assignment["enrollment_cutoff_exclusive"],
        "invalid_enrollment_cutoff",
    )
    if activated_at != ASSIGNMENT_ACTIVATED_AT:
        raise _DeclarationError("unexpected_activated_at")
    if enrollment_cutoff != ASSIGNMENT_ENROLLMENT_CUTOFF_EXCLUSIVE:
        raise _DeclarationError("unexpected_enrollment_cutoff")
    if not declared_at < activated_at < enrollment_cutoff:
        raise _DeclarationError("invalid_declaration_timeline")
    if assignment["assignment_algorithm"] != ASSIGNMENT_ALGORITHM:
        raise _DeclarationError("unexpected_assignment_algorithm")
    _exact_string_list(assignment["agents"], AGENTS, "invalid_assignment_agents")
    if (
        _sha256(assignment["strategy_aggregate_sha256"], "invalid_strategy_hash")
        != STRATEGY_AGGREGATE_SHA256
    ):
        raise _DeclarationError("unexpected_strategy_hash")
    targets = _object(
        assignment["target_sha256"], set(TARGET_SHA256), "invalid_target_hashes"
    )
    for path, expected in TARGET_SHA256.items():
        if _sha256(targets[path], "invalid_target_hash") != expected:
            raise _DeclarationError("unexpected_target_hash")

    look = _object(body["scheduled_look"], _LOOK_FIELDS, "invalid_look_fields")
    if look["look_id"] != FINAL_LOOK_ID:
        raise _DeclarationError("unexpected_look_id")
    analysis_as_of = _integer(look["analysis_as_of"], "invalid_analysis_as_of")
    if analysis_as_of != FINAL_ANALYSIS_AS_OF:
        raise _DeclarationError("unexpected_analysis_as_of")
    if (
        _integer(
            look["assignment_enrollment_cutoff_exclusive"],
            "invalid_look_enrollment_cutoff",
        )
        != enrollment_cutoff
    ):
        raise _DeclarationError("look_enrollment_cutoff_mismatch")
    if look["kind"] != "single_final_confirmatory":
        raise _DeclarationError("invalid_look_kind")
    for field in ("efficacy_binding", "promotion_binding", "only_efficacy_look"):
        _boolean(look[field], True, f"invalid_{field}")

    monitoring = _object(
        body["continuous_monitoring"],
        _MONITORING_FIELDS,
        "invalid_monitoring_fields",
    )
    if monitoring["scope"] != "safety_only":
        raise _DeclarationError("invalid_monitoring_scope")
    for field in (
        "efficacy_binding",
        "promotion_binding",
        "optional_stopping_for_efficacy",
    ):
        _boolean(monitoring[field], False, f"invalid_monitoring_{field}")
    _exact_string_list(
        monitoring["permitted_decisions"],
        ("continue", "rollback_for_predeclared_safety_harm"),
        "invalid_monitoring_decisions",
    )

    families = _object(body["families"], set(FAMILY_RULES), "invalid_families")
    lags: list[tuple[str, int]] = []
    for family in sorted(FAMILY_RULES):
        entry = _object(
            families[family], _FAMILY_FIELDS, f"invalid_{family}_fields"
        )
        if entry["rule_id"] != FAMILY_RULES[family]:
            raise _DeclarationError(f"unexpected_{family}_rule")
        if entry["population_id"] != FAMILY_POPULATIONS[family]:
            raise _DeclarationError(f"unexpected_{family}_population")
        _string(entry["population"], f"invalid_{family}_population_text")
        lag = _integer(
            entry["maturity_lag_seconds"], f"invalid_{family}_maturity_lag"
        )
        if lag != FAMILY_MATURITY_LAGS[family]:
            raise _DeclarationError(f"unexpected_{family}_maturity_lag")
        _string(
            entry["outcome_deadline_policy"], f"invalid_{family}_deadline_policy"
        )
        lags.append((family, lag))
    if analysis_as_of != enrollment_cutoff + max(FAMILY_MATURITY_LAGS.values()):
        raise _DeclarationError("final_look_does_not_clear_maturity")

    prefix_manifest = _object(
        body["prefix_manifest"], _PREFIX_FIELDS, "invalid_prefix_fields"
    )
    if prefix_manifest["procedure_version"] != PREFIX_PROCEDURE_VERSION:
        raise _DeclarationError("unexpected_prefix_procedure")
    capture_not_before = _integer(
        prefix_manifest["capture_not_before"], "invalid_capture_not_before"
    )
    settlement_delay = _integer(
        prefix_manifest["settlement_delay_seconds"],
        "invalid_settlement_delay",
    )
    if (
        capture_not_before != PREFIX_CAPTURE_NOT_BEFORE
        or settlement_delay != PREFIX_SETTLEMENT_DELAY_SECONDS
        or capture_not_before != analysis_as_of + settlement_delay
    ):
        raise _DeclarationError("prefix_capture_timeline_mismatch")
    if (
        _string(
            prefix_manifest["output_path_template"],
            "invalid_prefix_output_path_template",
        )
        != PREFIX_OUTPUT_PATH_TEMPLATE
    ):
        raise _DeclarationError("unexpected_prefix_output_path_template")
    _string(prefix_manifest["path_order"], "invalid_prefix_path_order")

    event_source = _object(
        prefix_manifest["event_log_paths"],
        _EVENT_SOURCE_FIELDS,
        "invalid_event_source_fields",
    )
    if event_source["path_template"] != "logs/{agent}-{utc_date}.jsonl":
        raise _DeclarationError("unexpected_event_path_template")
    _exact_string_list(event_source["agents"], AGENTS, "invalid_event_agents")
    _exact_string_list(event_source["utc_dates"], UTC_DATES, "invalid_utc_dates")
    _boolean(
        event_source["all_expanded_paths_required"],
        True,
        "invalid_event_paths_required",
    )
    if (
        _integer(event_source["maximum_bytes_per_file"], "invalid_event_size")
        != 2 * 1024 * 1024 * 1024
    ):
        raise _DeclarationError("unexpected_event_size")
    _string(event_source["prefix_rule"], "invalid_event_prefix_rule")

    receipt_source = _object(
        prefix_manifest["assignment_receipt_paths"],
        _SOURCE_FIELDS,
        "invalid_receipt_source_fields",
    )
    expected_receipt_template = (
        "logs/canary-assignments/{agent}/"
        f"{ASSIGNMENT_ARTIFACT_SHA256}.jsonl"
    )
    if receipt_source["path_template"] != expected_receipt_template:
        raise _DeclarationError("unexpected_receipt_path_template")
    _exact_string_list(receipt_source["agents"], AGENTS, "invalid_receipt_agents")
    _boolean(
        receipt_source["all_expanded_paths_required"],
        True,
        "invalid_receipt_paths_required",
    )
    if (
        _integer(receipt_source["maximum_bytes_per_file"], "invalid_receipt_size")
        != 64 * 1024 * 1024
    ):
        raise _DeclarationError("unexpected_receipt_size")
    _string(receipt_source["prefix_rule"], "invalid_receipt_prefix_rule")

    physical = _object(
        prefix_manifest["physical_read"],
        _PHYSICAL_READ_FIELDS,
        "invalid_physical_read_fields",
    )
    if physical["root"] != ".":
        raise _DeclarationError("invalid_physical_root")
    _boolean(physical["follow_symlinks"], False, "invalid_follow_symlinks")
    _boolean(
        physical["require_regular_files"], True, "invalid_regular_file_policy"
    )
    _string(physical["open_mode"], "invalid_open_mode")
    _string(physical["snapshot"], "invalid_snapshot_policy")
    _exact_string_list(
        prefix_manifest["entry_schema"], _ENTRY_SCHEMA, "invalid_entry_schema"
    )
    _string(prefix_manifest["aggregate_identity"], "invalid_aggregate_identity")

    canonical_sha256 = hashlib.sha256(_canonical_json(body)).hexdigest()
    plan = ConfirmationPlan(
        schema_version=SCHEMA_VERSION,
        declaration_id=DECLARATION_ID,
        declared_at=declared_at,
        assignment_plan_id=ASSIGNMENT_PLAN_ID,
        assignment_activated_at=activated_at,
        assignment_enrollment_cutoff_exclusive=enrollment_cutoff,
        agents=AGENTS,
        scheduled_look_id=FINAL_LOOK_ID,
        analysis_as_of=analysis_as_of,
        family_maturity_lags=tuple(lags),
        prefix_procedure_version=PREFIX_PROCEDURE_VERSION,
        prefix_capture_not_before=capture_not_before,
        prefix_output_path_template=PREFIX_OUTPUT_PATH_TEMPLATE,
    )
    return body, plan, canonical_sha256


def load_confirmation_plan(
    *,
    project_root: Path = PROJECT_ROOT,
    relative_path: Path = DEFAULT_DECLARATION_PATH,
) -> LoadedConfirmationPlan:
    """Load the exact declaration and verify its linked assignment artifact."""
    artifact_path = Path(relative_path).as_posix()
    try:
        raw = _safe_read(
            project_root,
            Path(relative_path),
            maximum_bytes=MAX_DECLARATION_BYTES,
            prefix="artifact",
        )
    except _DeclarationError as error:
        status: Literal["missing", "invalid"] = (
            "missing" if error.code == "artifact_missing" else "invalid"
        )
        return LoadedConfirmationPlan(
            status,
            artifact_path,
            None,
            None,
            None,
            error_code=None if status == "missing" else error.code,
        )

    digest = hashlib.sha256(raw).hexdigest()
    size = len(raw)
    try:
        _document, plan, canonical_digest = _parse_document(raw)
    except _DeclarationError as error:
        return LoadedConfirmationPlan(
            "invalid",
            artifact_path,
            digest,
            None,
            size,
            error_code=error.code,
        )
    if (
        size != DECLARATION_BYTES
        or digest != DECLARATION_SHA256
        or canonical_digest != DECLARATION_CANONICAL_SHA256
    ):
        return LoadedConfirmationPlan(
            "invalid",
            artifact_path,
            digest,
            canonical_digest,
            size,
            error_code="declaration_identity_mismatch",
        )

    try:
        assignment_raw = _safe_read(
            project_root,
            Path(ASSIGNMENT_ARTIFACT_PATH),
            maximum_bytes=MAX_LINKED_ARTIFACT_BYTES,
            prefix="assignment_artifact",
        )
    except _DeclarationError as error:
        return LoadedConfirmationPlan(
            "invalid",
            artifact_path,
            digest,
            canonical_digest,
            size,
            error_code=error.code,
        )
    if (
        len(assignment_raw) != ASSIGNMENT_ARTIFACT_BYTES
        or hashlib.sha256(assignment_raw).hexdigest()
        != ASSIGNMENT_ARTIFACT_SHA256
    ):
        return LoadedConfirmationPlan(
            "invalid",
            artifact_path,
            digest,
            canonical_digest,
            size,
            error_code="assignment_artifact_identity_mismatch",
        )

    return LoadedConfirmationPlan(
        "valid",
        artifact_path,
        digest,
        canonical_digest,
        size,
        plan=plan,
        linked_assignment_verified=True,
    )
