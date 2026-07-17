"""Safely remove continuation Blender shards after cumulative release validation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined-campaign", required=True, type=Path)
    parser.add_argument("--data-campaign", required=True, type=Path)
    parser.add_argument("--confirm", required=True, choices=("delete-verified-continuation-shards",))
    return parser.parse_args()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_in(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_release(release: Path, expected_manifest_sha: str) -> None:
    manifest_path = release / "release_manifest.json"
    if not manifest_path.is_file() or sha256(manifest_path) != expected_manifest_sha:
        raise ValueError("cumulative release manifest checksum mismatch")
    manifest = load(manifest_path)
    for relative, expected in manifest["files"].items():
        path = (release / relative).resolve()
        if release not in path.parents or not path.is_file():
            raise ValueError(f"missing/unsafe release file: {relative}")
        if sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"release file verification failed: {relative}")


def main() -> int:
    args = parse_args()
    combined = args.combined_campaign.resolve()
    data = args.data_campaign.resolve()
    state_path = combined / "campaign_state.json"
    state = load(state_path)
    if state.get("status") != "trained":
        raise ValueError("combined campaign is not successfully trained")
    selection_path = combined / state["selection_report"]
    if sha256(selection_path) != state["selection_report_sha256"]:
        raise ValueError("combined selection report checksum mismatch")
    selection = load(selection_path)
    if not selection.get("release_smoke", {}).get("valid"):
        raise ValueError("combined release smoke gate failed")
    release = (combined / state["release"]).resolve()
    verify_release(release, state["release_manifest_sha256"])

    shards = (data / "shards").resolve()
    if shards.parent != data or shards.name != "shards":
        raise ValueError(f"refusing unsafe continuation shard path: {shards}")
    deleted_bytes = bytes_in(shards) if shards.is_dir() else 0
    if shards.is_dir():
        shutil.rmtree(shards)

    receipt = {
        "status": "cleaned",
        "deleted_path": str(shards),
        "deleted_bytes": deleted_bytes,
        "combined_release_manifest_sha256": state["release_manifest_sha256"],
        "cleaned_at_unix": time.time(),
    }
    receipt_path = combined / "cleanup_receipt.json"
    atomic_json(receipt_path, receipt)
    state.update(
        {
            "status": "cleaned",
            "cleanup_receipt": receipt_path.name,
            "cleanup_receipt_sha256": sha256(receipt_path),
        }
    )
    atomic_json(state_path, state)
    data_state_path = data / "campaign_state.json"
    data_state = load(data_state_path)
    data_state.update({"status": "cleaned", "cleanup_receipt": str(receipt_path)})
    atomic_json(data_state_path, data_state)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
