"""Delete generated Blender shards only after the trained release is verified."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", required=True, type=Path)
    parser.add_argument(
        "--confirm",
        required=True,
        choices=("delete-verified-blender-shards",),
        help="Explicit guard for the irreversible shard deletion.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def verify_release(release: Path, expected_manifest_sha256: str) -> dict[str, Any]:
    manifest_path = release / "release_manifest.json"
    if not manifest_path.is_file() or sha256(manifest_path) != expected_manifest_sha256:
        raise ValueError("release manifest does not match the trained campaign state")
    manifest = load_json(manifest_path)
    for relative, expected in manifest["files"].items():
        path = (release / relative).resolve()
        if release not in path.parents or not path.is_file():
            raise ValueError(f"unsafe or missing release file: {relative}")
        if sha256(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"release file verification failed: {relative}")
    return {"model_id": manifest.get("model_id"), "file_count": len(manifest["files"])}


def main() -> int:
    args = parse_args()
    campaign = args.campaign_root.resolve()
    state_path = campaign / "campaign_state.json"
    state = load_json(state_path)
    if state.get("status") == "cleaned":
        print(json.dumps({"status": "already_cleaned"}))
        return 0
    if state.get("status") != "trained":
        raise ValueError("campaign shards can only be deleted after successful training")
    selection = campaign / str(state["selection_report"])
    if not selection.is_file() or sha256(selection) != state["selection_report_sha256"]:
        raise ValueError("selection report failed checksum verification")
    selection_report = load_json(selection)
    if not selection_report.get("release_smoke", {}).get("valid"):
        raise ValueError("release smoke gate is not valid")
    release = (campaign / str(state["release"])).resolve()
    release_result = verify_release(release, state["release_manifest_sha256"])

    shards = (campaign / "shards").resolve()
    if shards.parent != campaign or shards.name != "shards":
        raise ValueError(f"refusing unsafe shard path: {shards}")
    deleted_bytes = directory_bytes(shards) if shards.is_dir() else 0
    if shards.is_dir():
        shutil.rmtree(shards)
    receipt = {
        "status": "cleaned",
        "deleted_path": "shards",
        "deleted_bytes": deleted_bytes,
        "release": release_result,
        "release_manifest_sha256": state["release_manifest_sha256"],
        "cleaned_at_unix": time.time(),
    }
    receipt_path = campaign / "cleanup_receipt.json"
    atomic_json(receipt_path, receipt)
    state.update(
        {
            "status": "cleaned",
            "cleanup_receipt": receipt_path.name,
            "cleanup_receipt_sha256": sha256(receipt_path),
        }
    )
    atomic_json(state_path, state)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
