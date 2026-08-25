"""Auditable, non-secret runtime policy manifests.

One manifest is appended when ``scripts/run_agent.py`` acquires its instance
lock.  It records the complete :class:`~glee_agent.config.Knobs` dataclass and
content identities for every Python file shipped in the agent package, the
runner entry point, the payoff-target artifacts, and the optional validated
canary-assignment artifact and parsed contract.

This module intentionally accepts ``Knobs`` rather than ``Settings``.  API
keys, LLM credentials, endpoint configuration, and the ambient environment
therefore cannot accidentally enter the record.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .canary_assignment import LoadedAssignmentPlan, load_assignment_plan
from .config import Knobs
from .logging_ import log_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HASH_ALGORITHM = "sha256"
TARGET_PATHS = (Path("data/targets.json"), Path("data/live_targets.json"))
_GIT_OID = re.compile(r"^[0-9a-fA-F]{40,64}$")


def _hash_file(path: Path) -> dict[str, Any]:
    """Return a stable file identity, including an explicit missing marker."""
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except (OSError, ValueError):
        return {"sha256": None, "bytes": None, "available": False}
    return {"sha256": digest.hexdigest(), "bytes": size, "available": True}


def _git_head(project_root: Path) -> str | None:
    """Resolve HEAD without letting a missing/broken Git binary delay startup."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=1.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = completed.stdout.strip()
    return head.lower() if completed.returncode == 0 and _GIT_OID.fullmatch(head) else None


def _strategy_source_paths(project_root: Path) -> list[Path]:
    """Return the deterministic, de-duplicated strategy source inventory."""
    candidates = [project_root / "scripts/run_agent.py"]
    package = project_root / "src/glee_agent"
    try:
        candidates.extend(package.rglob("*.py"))
    except OSError:
        # A partial manifest is still more useful than no startup evidence.
        pass
    unique = {path.resolve(strict=False): path for path in candidates if path.is_file()}
    return sorted(unique.values(), key=lambda path: path.relative_to(project_root).as_posix())


def _source_hashes(project_root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    aggregate = hashlib.sha256()
    for path in _strategy_source_paths(project_root):
        relative = path.relative_to(project_root).as_posix()
        identity = _hash_file(path)
        files[relative] = identity
        # Length-prefixing is unnecessary because both separators are outside
        # the hex/path alphabets, but keeping explicit delimiters documents the
        # aggregate contract for independent verification.
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        digest = identity["sha256"] or "unavailable"
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")
    return {
        "aggregate_sha256": aggregate.hexdigest(),
        "count": len(files),
        "files": files,
    }


def build_runtime_manifest(
    agent_label: str,
    knobs: Knobs,
    *,
    project_root: Path = PROJECT_ROOT,
    pid: int | None = None,
    canary_assignment: LoadedAssignmentPlan | None = None,
) -> dict[str, Any]:
    """Build a JSON-safe manifest containing policy state and no credentials."""
    root = project_root.resolve(strict=False)
    targets = {
        relative.as_posix(): _hash_file(root / relative) for relative in TARGET_PATHS
    }
    assignment = canary_assignment or load_assignment_plan(project_root=root)
    return {
        "agent": str(agent_label),
        "pid": os.getpid() if pid is None else int(pid),
        "knobs": asdict(knobs),
        "git_head": _git_head(root),
        "canary_assignment": assignment.manifest(agent_label=agent_label),
        "content_hashes": {
            "algorithm": HASH_ALGORITHM,
            "strategy_python": _source_hashes(root),
            "targets": targets,
            "canary_assignment": assignment.artifact_manifest(),
        },
    }


def append_runtime_manifest(
    agent_label: str,
    knobs: Knobs,
    *,
    project_root: Path = PROJECT_ROOT,
    canary_assignment: LoadedAssignmentPlan | None = None,
) -> bool:
    """Build and append one startup manifest; never interfere with play."""
    try:
        return log_runtime(
            agent_label,
            build_runtime_manifest(
                agent_label,
                knobs,
                project_root=project_root,
                canary_assignment=canary_assignment,
            ),
        )
    except Exception:  # noqa: BLE001 - runtime provenance is strictly fail-open
        return False
