"""Delete only verified Repair-v3 Blender shards and write an audit receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--smoke", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True, type=Path)
    parser.add_argument("--campaign", action="append", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--confirm",
        required=True,
        choices=("delete-verified-repair-v3-shards",),
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    release = args.release.resolve()
    manifest_path = release / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative, expected in manifest["files"].items():
        path = (release / relative).resolve()
        if release not in path.parents or not path.is_file():
            raise ValueError(f"missing or unsafe release file: {relative}")
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"release verification failed: {relative}")
    smoke = json.loads(args.smoke.read_text(encoding="utf-8"))
    if not smoke.get("valid") or smoke.get("checked") != 4:
        raise ValueError("release smoke is not a valid four-action gate")
    archive = args.archive.resolve()
    expected_archive = args.archive_sha256.read_text(encoding="utf-8").split()[0]
    actual_archive = sha256(archive)
    if actual_archive != expected_archive:
        raise ValueError("release archive SHA256 mismatch")

    allowed_root = Path("/workspace/pavg/campaigns").resolve()
    rows = []
    for raw_campaign in args.campaign:
        campaign = raw_campaign.resolve()
        if campaign.parent != allowed_root:
            raise ValueError(f"campaign is outside the allowed root: {campaign}")
        shards = (campaign / "shards").resolve()
        if shards.parent != campaign or shards.name != "shards":
            raise ValueError(f"unsafe shards path: {shards}")
        deleted_bytes = (
            sum(item.stat().st_size for item in shards.rglob("*") if item.is_file())
            if shards.is_dir()
            else 0
        )
        if shards.is_dir():
            shutil.rmtree(shards)
        state_path = campaign / "campaign_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        previous_status = state.get("status")
        state["pre_v3_cleanup_status"] = previous_status
        state["status"] = "cleaned_after_repair_v3"
        state["repair_v3_release_manifest_sha256"] = sha256(manifest_path)
        state["repair_v3_archive_sha256"] = actual_archive
        atomic_json(state_path, state)
        rows.append(
            {
                "campaign": str(campaign),
                "shards": str(shards),
                "deleted_bytes": deleted_bytes,
                "pre_cleanup_status": previous_status,
            }
        )
    receipt = {
        "schema_version": "repair-v3-cleanup-receipt/1.0",
        "status": "cleaned",
        "release_manifest_sha256": sha256(manifest_path),
        "archive_sha256": actual_archive,
        "release_smoke_sha256": sha256(args.smoke.resolve()),
        "campaigns": rows,
        "total_deleted_bytes": sum(item["deleted_bytes"] for item in rows),
        "cleaned_at_unix": time.time(),
    }
    atomic_json(args.receipt.resolve(), receipt)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
