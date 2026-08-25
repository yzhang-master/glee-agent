"""Deterministic, fail-closed assignment for live per-game canaries.

The optional ``data/canary_assignment.json`` artifact is deliberately small
and rigid.  A valid plan assigns an eligible game once, using a domain-
separated SHA-256 digest of the plan, rule, agent, family, and game identity.
The result is cached for the lifetime of the process so every turn of a game
uses the same arm, including turns that finish after the enrollment window.

Missing or invalid artifacts never affect play.  Games first observed after
round one (or with any embedded history) are also permanently left unassigned
within the process; silently enrolling partial games would bias a canary.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

from .config import Knobs
from .schema import GameView


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN_PATH = Path("data/canary_assignment.json")
SCHEMA_VERSION = 1
ASSIGNMENT_ALGORITHM = "sha256-u64-v1"
MAX_PLAN_BYTES = 64 * 1024
# More than two projected 72-hour fleets at the current 25-30k games/label,
# while still bounding startup reads and corruption exposure.
MAX_RECEIPT_BYTES = 64 * 1024 * 1024
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_FORMAT = "append-only-jsonl-v1"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_KNOWN_AGENTS = {"main", "test_a", "test_b", "test_c", "test_d"}
_FAMILY_KNOBS: dict[str, tuple[str, type, float | None, float | None]] = {
    "bargaining": ("barg_dis_anchor", float, 0.0, 1.0),
    "persuasion": ("pers_blind_lie", float, 0.0, 1.0),
    "negotiation": ("neg_terminal_close", bool, None, None),
}
_TOP_LEVEL_FIELDS = {
    "schema_version",
    "plan_id",
    "assignment_salt",
    "activated_at",
    "expires_at",
    "agents",
    "rules",
}
_RULE_FIELDS = {
    "rule_id",
    "knob",
    "control",
    "treatment",
    "treatment_probability",
}


class _PlanError(ValueError):
    """Internal parse failure carrying only a bounded, non-secret code."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class ReceiptPersistenceUncertain(RuntimeError):
    """Raised before play when a failed append cannot be durably rolled back."""


@dataclass(frozen=True)
class AssignmentRule:
    family: str
    rule_id: str
    knob: str
    control: float | bool
    treatment: float | bool
    treatment_probability: float

    def manifest(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "rule_id": self.rule_id,
            "knob": self.knob,
            "control": self.control,
            "treatment": self.treatment,
            "treatment_probability": self.treatment_probability,
        }


@dataclass(frozen=True)
class AssignmentPlan:
    schema_version: int
    plan_id: str
    assignment_salt: str
    activated_at: int
    expires_at: int
    agents: tuple[str, ...]
    rules: tuple[AssignmentRule, ...]

    def rule_for(self, family: str) -> AssignmentRule | None:
        return next((rule for rule in self.rules if rule.family == family), None)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "assignment_salt": self.assignment_salt,
            "assignment_salt_visibility": "public_replay_seed",
            "activated_at": self.activated_at,
            "expires_at": self.expires_at,
            "agents": list(self.agents),
            "assignment_algorithm": ASSIGNMENT_ALGORITHM,
            "enrollment": {
                "first_seen_round": 1,
                "requires_empty_history": True,
                "assigned_games_remain_stable_after_expiry": True,
            },
            "rules": [rule.manifest() for rule in self.rules],
        }


@dataclass(frozen=True)
class LoadedAssignmentPlan:
    """The complete result of one startup artifact read."""

    status: Literal["valid", "missing", "invalid"]
    artifact_path: str
    artifact_sha256: str | None
    artifact_bytes: int | None
    plan: AssignmentPlan | None = None
    error_code: str | None = None
    project_root: Path = PROJECT_ROOT

    @property
    def available(self) -> bool:
        return self.artifact_sha256 is not None

    def artifact_manifest(self) -> dict[str, Any]:
        return {
            "path": self.artifact_path,
            "available": self.available,
            "sha256": self.artifact_sha256,
            "bytes": self.artifact_bytes,
        }

    def manifest(self, *, agent_label: str | None = None) -> dict[str, Any]:
        result = {
            "loader_status": self.status,
            "error_code": self.error_code,
            "artifact": self.artifact_manifest(),
            "contract": self.plan.manifest() if self.plan is not None else None,
        }
        if agent_label is not None:
            result["receipt_store"] = {
                "format": RECEIPT_FORMAT,
                "path": receipt_relative_path(
                    str(agent_label), self.artifact_sha256
                ).as_posix(),
                "write_ahead_fsync": True,
                "max_bytes": MAX_RECEIPT_BYTES,
            }
        return result


@dataclass(frozen=True)
class GameAssignment:
    status: Literal["assigned", "unassigned"]
    reason: str
    artifact_sha256: str | None
    plan_id: str | None = None
    rule_id: str | None = None
    family: str | None = None
    knob: str | None = None
    arm: Literal["control", "treatment"] | None = None
    value: float | bool | None = None
    treatment_probability: float | None = None
    assignment_sha256: str | None = None
    assignment_u64: int | None = None
    treatment_threshold_u64: int | None = None
    enrollment: Literal["new", "recovered"] | None = None
    enrolled_at: float | None = None

    @property
    def assigned(self) -> bool:
        return self.status == "assigned"

    def apply(self, knobs: Knobs) -> Knobs:
        if not self.assigned or self.knob is None:
            return knobs
        return replace(knobs, **{self.knob: self.value})

    def log_metadata(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "status": self.status,
            "reason": self.reason,
            "artifact_sha256": self.artifact_sha256,
        }
        if self.assigned:
            record.update(
                {
                    "plan_id": self.plan_id,
                    "rule_id": self.rule_id,
                    "family": self.family,
                    "knob": self.knob,
                    "arm": self.arm,
                    "value": self.value,
                    "treatment_probability": self.treatment_probability,
                    "assignment_algorithm": ASSIGNMENT_ALGORITHM,
                    "assignment_sha256": self.assignment_sha256,
                    "assignment_u64": self.assignment_u64,
                    "treatment_threshold_u64": self.treatment_threshold_u64,
                    "enrollment": self.enrollment,
                    "enrolled_at": self.enrolled_at,
                }
            )
        return record


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _PlanError("duplicate_json_key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise _PlanError("non_finite_json_number")


def _require_exact_fields(value: Any, expected: set[str], code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise _PlanError(code)
    return value


def _identifier(value: Any, code: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise _PlanError(code)
    return value


def _timestamp(value: Any, code: str) -> int:
    if type(value) is not int:
        raise _PlanError(code)
    if value < 0:
        raise _PlanError(code)
    return value


def _probability(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _PlanError("invalid_treatment_probability")
    try:
        probability = float(value)
    except (OverflowError, TypeError, ValueError):
        raise _PlanError("invalid_treatment_probability") from None
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise _PlanError("invalid_treatment_probability")
    return probability


def _rule_value(
    value: Any, expected_type: type, lower: float | None, upper: float | None
) -> float | bool:
    if expected_type is bool:
        if not isinstance(value, bool):
            raise _PlanError("invalid_rule_value")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _PlanError("invalid_rule_value")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        raise _PlanError("invalid_rule_value") from None
    if not math.isfinite(result):
        raise _PlanError("invalid_rule_value")
    if lower is not None and result < lower:
        raise _PlanError("invalid_rule_value")
    if upper is not None and result > upper:
        raise _PlanError("invalid_rule_value")
    return result


def _parse_plan(raw: bytes) -> AssignmentPlan:
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except _PlanError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise _PlanError("invalid_json") from None

    body = _require_exact_fields(document, _TOP_LEVEL_FIELDS, "invalid_plan_fields")
    schema_version = body["schema_version"]
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        raise _PlanError("unsupported_schema_version")
    plan_id = _identifier(body["plan_id"], "invalid_plan_id")
    salt = _identifier(body["assignment_salt"], "invalid_assignment_salt")
    activated_at = _timestamp(body["activated_at"], "invalid_activated_at")
    expires_at = _timestamp(body["expires_at"], "invalid_expires_at")
    if activated_at >= expires_at:
        raise _PlanError("invalid_enrollment_window")

    agents_raw = body["agents"]
    if not isinstance(agents_raw, list) or not agents_raw:
        raise _PlanError("invalid_agents")
    agents = tuple(_identifier(value, "invalid_agent") for value in agents_raw)
    if len(set(agents)) != len(agents):
        raise _PlanError("duplicate_agent")
    if not set(agents).issubset(_KNOWN_AGENTS):
        raise _PlanError("unknown_agent")

    rules_raw = body["rules"]
    if not isinstance(rules_raw, dict) or not rules_raw:
        raise _PlanError("invalid_rules")
    if not set(rules_raw).issubset(_FAMILY_KNOBS):
        raise _PlanError("unknown_family")

    rules: list[AssignmentRule] = []
    rule_ids: set[str] = set()
    for family in sorted(rules_raw):
        rule_body = _require_exact_fields(
            rules_raw[family], _RULE_FIELDS, "invalid_rule_fields"
        )
        expected_knob, expected_type, lower, upper = _FAMILY_KNOBS[family]
        knob = rule_body["knob"]
        if knob != expected_knob:
            raise _PlanError("unknown_rule_knob")
        rule_id = _identifier(rule_body["rule_id"], "invalid_rule_id")
        if rule_id in rule_ids:
            raise _PlanError("duplicate_rule_id")
        rule_ids.add(rule_id)
        control = _rule_value(rule_body["control"], expected_type, lower, upper)
        treatment = _rule_value(rule_body["treatment"], expected_type, lower, upper)
        if control == treatment:
            raise _PlanError("identical_rule_arms")
        rules.append(
            AssignmentRule(
                family=family,
                rule_id=rule_id,
                knob=knob,
                control=control,
                treatment=treatment,
                treatment_probability=_probability(rule_body["treatment_probability"]),
            )
        )
    return AssignmentPlan(
        schema_version=SCHEMA_VERSION,
        plan_id=plan_id,
        assignment_salt=salt,
        activated_at=activated_at,
        expires_at=expires_at,
        agents=agents,
        rules=tuple(rules),
    )


def load_assignment_plan(
    *,
    project_root: Path = PROJECT_ROOT,
    relative_path: Path = DEFAULT_PLAN_PATH,
) -> LoadedAssignmentPlan:
    """Read and validate the optional plan once; all failures disable it."""
    relative = Path(relative_path)
    artifact_path = relative.as_posix()
    if relative.is_absolute() or ".." in relative.parts:
        return LoadedAssignmentPlan(
            "invalid",
            artifact_path,
            None,
            None,
            error_code="invalid_artifact_path",
            project_root=project_root.resolve(strict=False),
        )
    root = project_root.resolve(strict=False)
    path = root / relative
    try:
        resolved_path = path.resolve(strict=True)
    except FileNotFoundError:
        return LoadedAssignmentPlan(
            "missing", artifact_path, None, None, project_root=root
        )
    except (OSError, ValueError):
        return LoadedAssignmentPlan(
            "invalid",
            artifact_path,
            None,
            None,
            error_code="artifact_unreadable",
            project_root=root,
        )
    try:
        resolved_path.relative_to(root)
    except ValueError:
        return LoadedAssignmentPlan(
            "invalid",
            artifact_path,
            None,
            None,
            error_code="artifact_path_escape",
            project_root=root,
        )
    if not resolved_path.is_file():
        return LoadedAssignmentPlan(
            "invalid",
            artifact_path,
            None,
            None,
            error_code="artifact_not_regular_file",
            project_root=root,
        )
    try:
        raw = resolved_path.read_bytes()
    except (OSError, ValueError):
        return LoadedAssignmentPlan(
            "invalid",
            artifact_path,
            None,
            None,
            error_code="artifact_unreadable",
            project_root=root,
        )

    digest = hashlib.sha256(raw).hexdigest()
    size = len(raw)
    if size > MAX_PLAN_BYTES:
        return LoadedAssignmentPlan(
            "invalid",
            artifact_path,
            digest,
            size,
            error_code="artifact_too_large",
            project_root=root,
        )
    try:
        plan = _parse_plan(raw)
    except _PlanError as error:
        return LoadedAssignmentPlan(
            "invalid",
            artifact_path,
            digest,
            size,
            error_code=error.code,
            project_root=root,
        )
    return LoadedAssignmentPlan(
        "valid", artifact_path, digest, size, plan=plan, project_root=root
    )


def receipt_relative_path(agent_label: str, artifact_sha256: str | None = None) -> Path:
    """Return the non-top-level JSONL path used for one agent's receipts."""
    safe_label = agent_label if _IDENTIFIER.fullmatch(agent_label) else "invalid-agent"
    safe_artifact = (
        artifact_sha256
        if isinstance(artifact_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", artifact_sha256)
        else "unavailable"
    )
    return Path("logs/canary-assignments") / safe_label / f"{safe_artifact}.jsonl"


def disabled_assignment(reason: str, loaded: LoadedAssignmentPlan) -> GameAssignment:
    return GameAssignment(
        status="unassigned",
        reason=reason,
        artifact_sha256=loaded.artifact_sha256,
    )


def require_runtime_manifest(
    loaded: LoadedAssignmentPlan, manifest_persisted: bool
) -> LoadedAssignmentPlan:
    """Disable a valid plan unless its startup provenance was durable."""
    if loaded.status != "valid" or manifest_persisted:
        return loaded
    return LoadedAssignmentPlan(
        status="invalid",
        artifact_path=loaded.artifact_path,
        artifact_sha256=loaded.artifact_sha256,
        artifact_bytes=loaded.artifact_bytes,
        error_code="runtime_manifest_unavailable",
        project_root=loaded.project_root,
    )


class CanaryAssigner:
    """Thread-safe, process-local game assignment and decision cache."""

    def __init__(
        self,
        loaded: LoadedAssignmentPlan,
        agent_label: str,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.loaded = loaded
        self.agent_label = str(agent_label)
        self._clock = clock
        self._cache: dict[tuple[str, str], GameAssignment] = {}
        self._lock = threading.Lock()
        relative_receipt = receipt_relative_path(
            self.agent_label, loaded.artifact_sha256
        )
        self._receipt_parent_parts = relative_receipt.parts[:-1]
        self._receipt_filename = relative_receipt.name
        self._receipt_store_healthy = True
        self._persistence_uncertain = False
        self._load_receipts()

    @staticmethod
    def _directory_flags() -> int:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        return flags

    def _open_receipt_parent(self, *, create: bool) -> int | None:
        """Walk from the resolved project root without following symlinks.

        Every newly created directory entry is fsynced in its already-open
        parent before traversal continues.  Returning the final directory fd
        lets callers use ``openat`` semantics and closes the containment TOCTOU
        left by resolving an absolute path and opening it later.
        """
        descriptor: int | None = None
        flags = self._directory_flags()
        try:
            descriptor = os.open(
                self.loaded.project_root.resolve(strict=False), flags
            )
            for component in self._receipt_parent_parts:
                try:
                    child = os.open(component, flags, dir_fd=descriptor)
                except FileNotFoundError:
                    if not create:
                        closing = descriptor
                        descriptor = None
                        os.close(closing)
                        return None
                    try:
                        os.mkdir(component, 0o700, dir_fd=descriptor)
                    except FileExistsError:
                        # A concurrent creator is accepted only if the no-follow
                        # directory open below validates what appeared.
                        pass
                    else:
                        os.fsync(descriptor)
                    child = os.open(component, flags, dir_fd=descriptor)
                closing = descriptor
                descriptor = child
                os.close(closing)
            result = descriptor
            descriptor = None
            return result
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def assignment_for(self, view: GameView) -> GameAssignment:
        game_id = view.game_id
        family = view.family
        if not isinstance(game_id, str) or not game_id:
            return disabled_assignment("invalid_game_id", self.loaded)
        key = (family, game_id)
        with self._lock:
            if self._persistence_uncertain:
                raise ReceiptPersistenceUncertain(
                    "canary receipt durability remains indeterminate"
                )
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            try:
                assignment = self._first_assignment(view)
                if assignment.assigned and not self._persist_receipt(view.game_id, assignment):
                    assignment = disabled_assignment(
                        "receipt_persistence_failed", self.loaded
                    )
            except ReceiptPersistenceUncertain:
                # Returning the base policy here could cross arms after a
                # restart if the uncertain receipt survived.  Stop this move.
                raise
            except Exception:  # noqa: BLE001 - assignment must be fail-closed
                assignment = disabled_assignment("assignment_error", self.loaded)
            self._cache[key] = assignment
            return assignment

    def _first_assignment(self, view: GameView) -> GameAssignment:
        plan = self.loaded.plan
        if self.loaded.status != "valid" or plan is None:
            return disabled_assignment(f"plan_{self.loaded.status}", self.loaded)
        if self.agent_label not in plan.agents:
            return disabled_assignment("agent_not_enrolled", self.loaded)
        rule = plan.rule_for(view.family)
        if rule is None:
            return disabled_assignment("family_not_enrolled", self.loaded)
        if not self._is_complete_first_sighting(view):
            return disabled_assignment("partial_game", self.loaded)

        try:
            observed_at = float(self._clock())
        except (TypeError, ValueError, OverflowError):
            return disabled_assignment("invalid_clock", self.loaded)
        if not math.isfinite(observed_at):
            return disabled_assignment("invalid_clock", self.loaded)
        if observed_at < plan.activated_at:
            return disabled_assignment("before_activation", self.loaded)
        if observed_at >= plan.expires_at:
            return disabled_assignment("after_expiry", self.loaded)

        if not self._receipt_store_healthy:
            return disabled_assignment("receipt_store_invalid", self.loaded)
        return self._make_assignment(plan, rule, view.game_id, observed_at, "new")

    def _make_assignment(
        self,
        plan: AssignmentPlan,
        rule: AssignmentRule,
        game_id: str,
        enrolled_at: float,
        enrollment: Literal["new", "recovered"],
    ) -> GameAssignment:
        digest = self._assignment_digest(plan, rule, game_id)
        bucket = int.from_bytes(digest[:8], "big", signed=False)
        threshold = int(rule.treatment_probability * (1 << 64))
        arm: Literal["control", "treatment"] = (
            "treatment" if bucket < threshold else "control"
        )
        value = rule.treatment if arm == "treatment" else rule.control
        return GameAssignment(
            status="assigned",
            reason="eligible",
            artifact_sha256=self.loaded.artifact_sha256,
            plan_id=plan.plan_id,
            rule_id=rule.rule_id,
            family=rule.family,
            knob=rule.knob,
            arm=arm,
            value=value,
            treatment_probability=rule.treatment_probability,
            assignment_sha256=digest.hex(),
            assignment_u64=bucket,
            treatment_threshold_u64=threshold,
            enrollment=enrollment,
            enrolled_at=enrolled_at,
        )

    def _receipt_record(self, game_id: str, assignment: GameAssignment) -> dict[str, Any]:
        return {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "record_type": "canary_assignment_receipt",
            "agent": self.agent_label,
            "game_id": game_id,
            "family": assignment.family,
            "artifact_sha256": assignment.artifact_sha256,
            "plan_id": assignment.plan_id,
            "rule_id": assignment.rule_id,
            "assignment_algorithm": ASSIGNMENT_ALGORITHM,
            "assignment_sha256": assignment.assignment_sha256,
            "assignment_u64": assignment.assignment_u64,
            "treatment_threshold_u64": assignment.treatment_threshold_u64,
            "arm": assignment.arm,
            "knob": assignment.knob,
            "value": assignment.value,
            "treatment_probability": assignment.treatment_probability,
            "enrolled_at": assignment.enrolled_at,
        }

    def _persist_receipt(self, game_id: str, assignment: GameAssignment) -> bool:
        if not self._receipt_store_healthy:
            return False
        descriptor: int | None = None
        parent_descriptor: int | None = None
        starting_size: int | None = None
        created = False
        try:
            line = json.dumps(
                self._receipt_record(game_id, assignment),
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            payload = (line + "\n").encode("utf-8")
            parent_descriptor = self._open_receipt_parent(create=True)
            if parent_descriptor is None:
                raise OSError("receipt parent unavailable")
            flags = os.O_WRONLY | os.O_APPEND
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(
                    self._receipt_filename,
                    flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=parent_descriptor,
                )
                created = True
            except FileExistsError:
                descriptor = os.open(
                    self._receipt_filename, flags, dir_fd=parent_descriptor
                )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            starting_size = os.fstat(descriptor).st_size
            if starting_size + len(payload) > MAX_RECEIPT_BYTES:
                self._receipt_store_healthy = False
                return False
            written = 0
            while written < len(payload):
                count = os.write(descriptor, payload[written:])
                if count <= 0:
                    raise OSError("receipt append made no progress")
                written += count
            os.fsync(descriptor)
            if created:
                os.fsync(parent_descriptor)
            return True
        except (OSError, TypeError, ValueError, OverflowError):
            self._receipt_store_healthy = False
            if descriptor is not None and starting_size is not None:
                try:
                    os.ftruncate(descriptor, starting_size)
                    os.fsync(descriptor)
                    if created:
                        if parent_descriptor is None:
                            raise OSError("receipt parent unavailable for rollback")
                        os.fsync(parent_descriptor)
                except OSError as rollback_error:
                    self._persistence_uncertain = True
                    raise ReceiptPersistenceUncertain(
                        "canary receipt append and rollback were both uncertain"
                    ) from rollback_error
            return False
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if parent_descriptor is not None:
                try:
                    os.close(parent_descriptor)
                except OSError:
                    pass

    def _load_receipts(self) -> None:
        plan = self.loaded.plan
        if self.loaded.status != "valid" or plan is None:
            return
        if self.agent_label not in plan.agents:
            # A receipt is never authority to expand a plan's agent scope.
            return
        if not self._receipt_store_healthy:
            return
        descriptor: int | None = None
        parent_descriptor: int | None = None
        try:
            parent_descriptor = self._open_receipt_parent(create=False)
            if parent_descriptor is None:
                return
            flags = os.O_RDWR
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                descriptor = os.open(
                    self._receipt_filename, flags, dir_fd=parent_descriptor
                )
            except FileNotFoundError:
                return
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            size = os.fstat(descriptor).st_size
            if size > MAX_RECEIPT_BYTES:
                self._receipt_store_healthy = False
                return
            chunks: list[bytes] = []
            remaining = size
            while remaining:
                chunk = os.read(descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    raise OSError("receipt journal ended before its stated size")
                chunks.append(chunk)
                remaining -= len(chunk)
            raw = b"".join(chunks)
            # An interrupted final append is safe to discard: the writer only
            # returns an assigned arm after the full line is file-fsynced.
            if raw and not raw.endswith(b"\n"):
                cutoff = raw.rfind(b"\n") + 1
                os.ftruncate(descriptor, cutoff)
                os.fsync(descriptor)
                raw = raw[:cutoff]
        except OSError:
            self._receipt_store_healthy = False
            return
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if parent_descriptor is not None:
                try:
                    os.close(parent_descriptor)
                except OSError:
                    pass
        lines = raw.splitlines()
        recovered: dict[tuple[str, str], GameAssignment] = {}
        try:
            for line in lines:
                if not line:
                    raise _PlanError("invalid_receipt")
                record = json.loads(
                    line.decode("utf-8"),
                    object_pairs_hook=_object_without_duplicates,
                    parse_constant=_reject_json_constant,
                )
                assignment = self._validate_receipt(record, plan)
                key = (assignment.family or "", str(record["game_id"]))
                previous = recovered.get(key)
                if previous is not None and previous != assignment:
                    raise _PlanError("conflicting_receipt")
                recovered[key] = assignment
        except (
            _PlanError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            OverflowError,
            RecursionError,
        ):
            self._receipt_store_healthy = False
            self._cache.clear()
            return
        self._cache.update(recovered)

    def _validate_receipt(
        self, record: Any, plan: AssignmentPlan
    ) -> GameAssignment:
        if self.agent_label not in plan.agents:
            raise _PlanError("receipt_agent_not_enrolled")
        expected_fields = {
            "schema_version",
            "record_type",
            "agent",
            "game_id",
            "family",
            "artifact_sha256",
            "plan_id",
            "rule_id",
            "assignment_algorithm",
            "assignment_sha256",
            "assignment_u64",
            "treatment_threshold_u64",
            "arm",
            "knob",
            "value",
            "treatment_probability",
            "enrolled_at",
        }
        body = _require_exact_fields(record, expected_fields, "invalid_receipt")
        game_id = body["game_id"]
        family = body["family"]
        enrolled_at = body["enrolled_at"]
        if not isinstance(game_id, str) or not game_id:
            raise _PlanError("invalid_receipt")
        if not isinstance(family, str):
            raise _PlanError("invalid_receipt")
        if (
            isinstance(enrolled_at, bool)
            or not isinstance(enrolled_at, (int, float))
            or not math.isfinite(float(enrolled_at))
            or not plan.activated_at <= float(enrolled_at) < plan.expires_at
        ):
            raise _PlanError("invalid_receipt")
        rule = plan.rule_for(family)
        if rule is None:
            raise _PlanError("invalid_receipt")
        expected = self._make_assignment(
            plan, rule, game_id, float(enrolled_at), "recovered"
        )
        expected_record = self._receipt_record(game_id, expected)
        # Enrollment is a runtime classification, not part of the receipt.
        if body != expected_record:
            raise _PlanError("invalid_receipt")
        if body["schema_version"] != RECEIPT_SCHEMA_VERSION:
            raise _PlanError("invalid_receipt")
        if body["record_type"] != "canary_assignment_receipt":
            raise _PlanError("invalid_receipt")
        return expected

    def _assignment_digest(
        self, plan: AssignmentPlan, rule: AssignmentRule, game_id: str
    ) -> bytes:
        components = (
            "glee-canary-assignment-v1",
            plan.assignment_salt,
            plan.plan_id,
            rule.rule_id,
            self.agent_label,
            rule.family,
            game_id,
        )
        payload = b"\0".join(component.encode("utf-8") for component in components)
        return hashlib.sha256(payload).digest()

    @staticmethod
    def _is_complete_first_sighting(view: GameView) -> bool:
        state = view.raw.get("game_state") if isinstance(view.raw, dict) else None
        if not isinstance(state, dict):
            return False
        round_value = state.get("round")
        if isinstance(round_value, bool) or not isinstance(round_value, (int, float)):
            return False
        try:
            numeric_round = float(round_value)
        except (OverflowError, TypeError, ValueError):
            return False
        if not math.isfinite(numeric_round) or numeric_round != 1.0:
            return False
        history = state.get("history")
        return isinstance(history, list) and len(history) == 0
