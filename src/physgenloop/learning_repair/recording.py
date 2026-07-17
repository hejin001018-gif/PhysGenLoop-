"""Append-only Trial recording and immutable, versioned Memory publication."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
from typing import Iterable

from .compatibility import sha256_file
from .contracts import LearningTargetV1, RepairTrialV1, utc_now


class JsonlTrialRecorder:
    """Single-process append-only recorder; it never rewrites prior trials."""

    def __init__(self, path: str | Path, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self._lock = threading.Lock()

    def append(self, trial: RepairTrialV1) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            trial.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        with self._lock:
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(payload + "\n")
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())


def read_trials(path: str | Path) -> tuple[RepairTrialV1, ...]:
    records = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            trial = RepairTrialV1.from_dict(json.loads(line))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid trial at line {line_number}: {exc}") from exc
        if trial.trial_id in seen:
            raise ValueError(f"duplicate trial_id at line {line_number}: {trial.trial_id}")
        seen.add(trial.trial_id)
        records.append(trial)
    if not records:
        raise ValueError("trial manifest contains no records")
    return tuple(records)


def read_targets(path: str | Path) -> tuple[LearningTargetV1, ...]:
    records = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            target = LearningTargetV1.from_dict(json.loads(line))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid target at line {line_number}: {exc}") from exc
        if target.sample_id in seen:
            raise ValueError(f"duplicate sample_id at line {line_number}: {target.sample_id}")
        seen.add(target.sample_id)
        records.append(target)
    if not records:
        raise ValueError("target manifest contains no records")
    return tuple(records)


def write_targets(
    targets: Iterable[LearningTargetV1],
    path: str | Path,
    *,
    allow_existing: bool = False,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if allow_existing else "x"
    with destination.open(mode, encoding="utf-8", newline="\n") as handle:
        for target in targets:
            handle.write(
                json.dumps(
                    target.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    return destination


class VersionedMemoryWriter:
    """Publish a new Memory version while keeping all older versions immutable."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def publish(
        self,
        trials: Iterable[RepairTrialV1],
        *,
        version: str,
        critic_id: str,
        executor_manifest: dict,
    ) -> tuple[Path, Path]:
        if not version.strip() or any(char in version for char in "/\\"):
            raise ValueError("memory version must be a non-empty path-safe string")
        records = tuple(trials)
        if not records:
            raise ValueError("cannot publish an empty Memory")
        self.root.mkdir(parents=True, exist_ok=True)
        memory_path = self.root / f"repair_memory_{version}.jsonl"
        manifest_path = self.root / f"repair_memory_{version}.manifest.json"
        if memory_path.exists() or manifest_path.exists():
            raise FileExistsError(f"Memory version {version!r} already exists")
        with memory_path.open("x", encoding="utf-8", newline="\n") as handle:
            for trial in records:
                handle.write(
                    json.dumps(
                        trial.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        manifest = {
            "schema_version": "repair-memory-manifest/1.0",
            "version": version,
            "created_at": utc_now(),
            "record_count": len(records),
            "successful_count": sum(item.successful for item in records),
            "failure_count": sum(not item.successful for item in records),
            "domains": sorted({item.domain for item in records}),
            "critic_id": critic_id,
            "compatibility_ids": sorted(
                {
                    item.decision.compatibility_id
                    for item in records
                    if item.decision.compatibility_id
                }
            ),
            "executor_manifest": executor_manifest,
            "memory_file": memory_path.name,
            "memory_sha256": sha256_file(memory_path),
            "source_trial_set_sha256": hashlib.sha256(
                "\n".join(sorted(item.trial_id for item in records)).encode("utf-8")
            ).hexdigest(),
        }
        with manifest_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        return memory_path, manifest_path
