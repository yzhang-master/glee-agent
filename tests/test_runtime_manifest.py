"""Runtime policy manifests are complete, deterministic, and non-secret."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

from glee_agent import logging_, runtime_manifest
from glee_agent.config import Knobs


def _project(tmp_path):
    files = {
        "scripts/run_agent.py": b"print('runner')\n",
        "src/glee_agent/config.py": b"POLICY = 1\n",
        "src/glee_agent/families/bargaining.py": b"def decide(): return 1\n",
        "data/targets.json": b'{"public": 1}\n',
        "data/live_targets.json": b'{"live": 2}\n',
    }
    for relative, content in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return files


def _records(path, prefix):
    records = []
    for log in sorted(path.glob(f"{prefix}-*.jsonl")):
        records.extend(json.loads(line) for line in log.read_text().splitlines())
    return records


def test_manifest_has_full_knobs_pid_git_and_deterministic_content_hashes(
    tmp_path, monkeypatch
):
    files = _project(tmp_path)
    knobs = replace(
        Knobs(),
        barg_dis_anchor=0.50,
        neg_terminal_close=True,
        pers_blind_lie=0.40,
        llm_enabled=False,
    )
    head = "a" * 40
    monkeypatch.setattr(runtime_manifest, "_git_head", lambda _root: head)

    first = runtime_manifest.build_runtime_manifest(
        "test_a", knobs, project_root=tmp_path, pid=4321
    )
    second = runtime_manifest.build_runtime_manifest(
        "test_a", knobs, project_root=tmp_path, pid=4321
    )

    assert first == second
    assert first["agent"] == "test_a"
    assert first["pid"] == 4321
    assert first["git_head"] == head
    assert first["knobs"] == asdict(knobs)
    assert set(first["knobs"]) == set(Knobs.__dataclass_fields__)
    sources = first["content_hashes"]["strategy_python"]
    assert set(sources["files"]) == {
        "scripts/run_agent.py",
        "src/glee_agent/config.py",
        "src/glee_agent/families/bargaining.py",
    }
    for relative in sources["files"]:
        assert sources["files"][relative] == {
            "sha256": hashlib.sha256(files[relative]).hexdigest(),
            "bytes": len(files[relative]),
            "available": True,
        }
    targets = first["content_hashes"]["targets"]
    assert targets["data/targets.json"]["sha256"] == hashlib.sha256(
        files["data/targets.json"]
    ).hexdigest()
    assert targets["data/live_targets.json"]["sha256"] == hashlib.sha256(
        files["data/live_targets.json"]
    ).hexdigest()


def test_source_aggregate_changes_with_content_not_filesystem_metadata(
    tmp_path, monkeypatch
):
    _project(tmp_path)
    monkeypatch.setattr(runtime_manifest, "_git_head", lambda _root: None)
    knobs = Knobs()
    before = runtime_manifest.build_runtime_manifest(
        "main", knobs, project_root=tmp_path, pid=1
    )
    source = tmp_path / "src/glee_agent/config.py"
    source.touch()
    touched = runtime_manifest.build_runtime_manifest(
        "main", knobs, project_root=tmp_path, pid=1
    )
    assert (
        before["content_hashes"]["strategy_python"]["aggregate_sha256"]
        == touched["content_hashes"]["strategy_python"]["aggregate_sha256"]
    )

    source.write_text("POLICY = 2\n")
    changed = runtime_manifest.build_runtime_manifest(
        "main", knobs, project_root=tmp_path, pid=1
    )
    assert (
        before["content_hashes"]["strategy_python"]["aggregate_sha256"]
        != changed["content_hashes"]["strategy_python"]["aggregate_sha256"]
    )


def test_missing_target_is_explicit_and_manifest_contains_no_secret_fields(
    tmp_path, monkeypatch
):
    _project(tmp_path)
    (tmp_path / "data/live_targets.json").unlink()
    monkeypatch.setattr(runtime_manifest, "_git_head", lambda _root: None)

    manifest = runtime_manifest.build_runtime_manifest(
        "main", Knobs(), project_root=tmp_path, pid=7
    )

    assert manifest["content_hashes"]["targets"]["data/live_targets.json"] == {
        "sha256": None,
        "bytes": None,
        "available": False,
    }
    serialized = json.dumps(manifest, sort_keys=True).lower()
    assert "api_key" not in serialized
    assert "llm_api" not in serialized
    assert "endpoint" not in serialized


def test_append_writes_one_runtime_record_and_is_fail_open(tmp_path, monkeypatch):
    _project(tmp_path)
    monkeypatch.setattr(logging_, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(runtime_manifest, "_git_head", lambda _root: "b" * 40)

    runtime_manifest.append_runtime_manifest(
        "test_b", Knobs(llm_enabled=False), project_root=tmp_path
    )

    records = _records(tmp_path / "logs", "test_b")
    assert len(records) == 1
    assert records[0]["type"] == "runtime"
    assert records[0]["agent"] == "test_b"
    assert records[0]["pid"] > 0
    assert isinstance(records[0]["ts"], float)

    monkeypatch.setattr(
        runtime_manifest,
        "build_runtime_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("hash failed")),
    )
    runtime_manifest.append_runtime_manifest("test_b", Knobs(), project_root=tmp_path)
    assert len(_records(tmp_path / "logs", "test_b")) == 1
