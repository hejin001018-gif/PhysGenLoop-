"""Run a resumable, quality-gated Blender Repair data campaign.

Each shard is finalized by the frozen Critic before it is admitted to the merged
manifest.  The script is intentionally generation-only: training and destructive
cleanup are separate stages with their own completion gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


VARIANTS_PER_GROUP = 13
RECORDS_PER_GROUP = {"standard": 13, "hard-v1": 35, "hard-v1.1": 35}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument("--total-groups", type=int, default=600)
    parser.add_argument("--groups-per-shard", type=int, default=25)
    parser.add_argument("--start-group", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument(
        "--difficulty-profile",
        choices=tuple(RECORDS_PER_GROUP),
        default="standard",
    )
    parser.add_argument("--runner", type=Path)
    args = parser.parse_args()
    if args.total_groups < 1 or args.groups_per_shard < 1:
        parser.error("total-groups and groups-per-shard must be positive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def source_fingerprint(repo: Path) -> dict[str, str]:
    relative_paths = (
        "Blender_video/scripts/generate_repair_shard.py",
        "Blender_video/scripts/finalize_repair_shard.py",
        "Blender_video/run_cloud_shard.sh",
        "configs/default.yaml",
    )
    return {relative: sha256(repo / relative) for relative in relative_paths}


def rewrite_manifest(source: Path, destination: Path, shard_name: str) -> dict[str, Any]:
    records = []
    sample_ids: set[str] = set()
    group_ids: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record["sample_id"])
        if sample_id in sample_ids:
            raise ValueError(f"duplicate sample_id {sample_id!r} in {source}:{line_number}")
        sample_ids.add(sample_id)
        group_ids.add(str(record["group_id"]))
        artifacts = dict(record.get("artifacts") or {})
        rewritten = {}
        for key, raw_path in artifacts.items():
            path = Path(str(raw_path))
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe artifact path {raw_path!r} in {source}")
            rewritten[str(key)] = (Path("shards") / shard_name / path).as_posix()
        record["artifacts"] = rewritten
        record.setdefault("metadata", {})["campaign_shard"] = shard_name
        records.append(record)
    text = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records)
    atomic_text(destination, text)
    return {
        "record_count": len(records),
        "group_count": len(group_ids),
        "sha256": sha256(destination),
    }


def merge_snapshots(manifest_dir: Path, output: Path, expected_records: int) -> dict[str, Any]:
    lines = []
    sample_ids: set[str] = set()
    group_ids: set[str] = set()
    for snapshot in sorted(manifest_dir.glob("shard_*.jsonl")):
        for line_number, line in enumerate(snapshot.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(
                    f"duplicate campaign sample_id {sample_id!r} at {snapshot}:{line_number}"
                )
            sample_ids.add(sample_id)
            group_ids.add(str(record["group_id"]))
            lines.append(json.dumps(record, ensure_ascii=False) + "\n")
    if len(lines) != expected_records:
        raise ValueError(
            f"merged record count {len(lines)} does not match expected {expected_records}"
        )
    atomic_text(output, "".join(lines))
    return {
        "record_count": len(lines),
        "group_count": len(group_ids),
        "sha256": sha256(output),
    }


def main() -> int:
    args = parse_args()
    repo = Path(__file__).resolve().parents[2]
    runner = (args.runner or repo / "Blender_video/run_cloud_shard.sh").resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"cloud shard runner not found: {runner}")
    campaign = args.campaign_root.resolve()
    shards_root = campaign / "shards"
    manifest_dir = campaign / "manifests"
    logs_dir = campaign / "logs"
    for path in (shards_root, manifest_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    config = {
        "total_groups": args.total_groups,
        "groups_per_shard": args.groups_per_shard,
        "start_group": args.start_group,
        "seed": args.seed,
        "frames": args.frames,
        "width": args.width,
        "height": args.height,
        "samples": args.samples,
        "difficulty_profile": args.difficulty_profile,
        "variants_per_group": VARIANTS_PER_GROUP,
        "records_per_group": RECORDS_PER_GROUP[args.difficulty_profile],
        "source_fingerprint": source_fingerprint(repo),
    }
    state_path = campaign / "campaign_state.json"
    if state_path.is_file():
        state = load_json(state_path)
        if state.get("config") != config:
            raise ValueError(
                "existing campaign configuration/source fingerprint differs; "
                "use a new campaign root"
            )
    else:
        state = {
            "schema_version": "1.0",
            "status": "generating",
            "created_at_unix": time.time(),
            "config": config,
            "shards": {},
        }
        atomic_json(state_path, state)

    shard_index = 0
    generated_groups = 0
    while generated_groups < args.total_groups:
        group_count = min(args.groups_per_shard, args.total_groups - generated_groups)
        start_group = args.start_group + generated_groups
        shard_name = f"shard_{shard_index:04d}"
        shard_root = shards_root / shard_name
        snapshot = manifest_dir / f"{shard_name}.jsonl"
        prior = state["shards"].get(shard_name, {})
        if prior.get("status") == "complete":
            if not snapshot.is_file() or sha256(snapshot) != prior.get("manifest_sha256"):
                raise ValueError(f"completed shard snapshot failed checksum: {shard_name}")
            print(f"SKIP verified {shard_name}", flush=True)
        else:
            state["shards"][shard_name] = {
                "status": "running",
                "start_group": start_group,
                "group_count": group_count,
                "started_at_unix": time.time(),
            }
            atomic_json(state_path, state)
            env = os.environ.copy()
            env.update(
                {
                    "OUTPUT_ROOT": str(shard_root),
                    "SHARD_ID": shard_name,
                    "START_GROUP": str(start_group),
                    "GROUP_COUNT": str(group_count),
                    "SEED": str(args.seed),
                    "FRAMES": str(args.frames),
                    "WIDTH": str(args.width),
                    "HEIGHT": str(args.height),
                    "SAMPLES": str(args.samples),
                    "DIFFICULTY_PROFILE": args.difficulty_profile,
                }
            )
            log_path = logs_dir / f"{shard_name}.log"
            print(
                f"RUN {shard_name} groups={start_group}..{start_group + group_count - 1}",
                flush=True,
            )
            started = time.monotonic()
            with log_path.open("a", encoding="utf-8") as log:
                result = subprocess.run(
                    [str(runner)],
                    cwd=repo,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode:
                state["shards"][shard_name].update(
                    {"status": "failed", "returncode": result.returncode}
                )
                atomic_json(state_path, state)
                raise RuntimeError(f"{shard_name} failed; inspect {log_path}")
            final_report = load_json(shard_root / "finalize_report.json")
            if not final_report["audit"]["valid"] or not final_report["semantic_gate"]["valid"]:
                raise ValueError(f"{shard_name} did not pass artifact/semantic gates")
            snapshot_result = rewrite_manifest(
                shard_root / "repair_manifest.jsonl", snapshot, shard_name
            )
            expected = group_count * RECORDS_PER_GROUP[args.difficulty_profile]
            if snapshot_result["record_count"] != expected:
                raise ValueError(
                    f"{shard_name} contains {snapshot_result['record_count']} records, "
                    f"expected {expected}"
                )
            state["shards"][shard_name].update(
                {
                    "status": "complete",
                    "elapsed_sec": round(time.monotonic() - started, 3),
                    "manifest_sha256": snapshot_result["sha256"],
                    "record_count": snapshot_result["record_count"],
                    "completed_at_unix": time.time(),
                }
            )
            atomic_json(state_path, state)
            print(
                f"PASS {shard_name} records={snapshot_result['record_count']} "
                f"elapsed={state['shards'][shard_name]['elapsed_sec']}s",
                flush=True,
            )
        generated_groups += group_count
        shard_index += 1

    merged = merge_snapshots(
        manifest_dir,
        campaign / "campaign_manifest.jsonl",
        args.total_groups * RECORDS_PER_GROUP[args.difficulty_profile],
    )
    state.update(
        {
            "status": "generated",
            "merged_manifest": "campaign_manifest.jsonl",
            "merged_manifest_sha256": merged["sha256"],
            "record_count": merged["record_count"],
            "group_count": merged["group_count"],
            "generation_completed_at_unix": time.time(),
        }
    )
    atomic_json(state_path, state)
    print(json.dumps({"status": "generated", **merged}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
