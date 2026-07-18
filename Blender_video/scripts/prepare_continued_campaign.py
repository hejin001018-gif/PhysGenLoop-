"""Combine the completed campaign manifest with a new continuation campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-campaign", required=True, type=Path)
    parser.add_argument("--second-campaign", required=True, type=Path)
    parser.add_argument("--output-campaign", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_manifest(source: Path, output, sample_ids: set[str], group_ids: set[str]) -> tuple[int, set[str]]:
    count = 0
    source_groups: set[str] = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = str(record["sample_id"])
        group_id = str(record["group_id"])
        if sample_id in sample_ids:
            raise ValueError(f"duplicate continuation sample_id {sample_id!r} at {source}:{line_number}")
        sample_ids.add(sample_id)
        group_ids.add(group_id)
        source_groups.add(group_id)
        record.setdefault("metadata", {})["continuation_sources"] = [source.parent.name]
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        count += 1
    return count, source_groups


def main() -> int:
    args = parse_args()
    first = args.first_campaign.resolve()
    second = args.second_campaign.resolve()
    output = args.output_campaign.resolve()
    first_manifest = first / "campaign_manifest.jsonl"
    second_manifest = second / "campaign_manifest.jsonl"
    if not first_manifest.is_file() or not second_manifest.is_file():
        raise FileNotFoundError("both source campaigns must have a merged manifest")
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "campaign_manifest.jsonl"
    sample_ids: set[str] = set()
    group_ids: set[str] = set()
    with destination.open("w", encoding="utf-8") as handle:
        first_count, first_groups = append_manifest(
            first_manifest, handle, sample_ids, group_ids
        )
        second_count, second_groups = append_manifest(
            second_manifest, handle, sample_ids, group_ids
        )
    state = {
        "schema_version": "1.0",
        "status": "generated",
        "created_at_unix": time.time(),
        "config": {
            "total_groups": len(group_ids),
            "groups_per_shard": 10,
            "records_per_group": None,
            "continuation": True,
            "source_campaigns": [str(first), str(second)],
        },
        "source_campaigns": {
            "first_manifest_sha256": sha256(first_manifest),
            "second_manifest_sha256": sha256(second_manifest),
            "first_group_count": len(first_groups),
            "second_group_count": len(second_groups),
        },
        "merged_manifest": destination.name,
        "merged_manifest_sha256": sha256(destination),
        "record_count": first_count + second_count,
        "group_count": len(group_ids),
        "generation_completed_at_unix": time.time(),
        "shards": {},
    }
    atomic_json(output / "campaign_state.json", state)
    print(json.dumps({"status": "generated", "groups": len(group_ids), "records": first_count + second_count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
