from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from common import BASE, load_config, sha256_file, write_json


def git_commit(repository: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()


def file_record(path: Path, include_hash: bool = True) -> dict:
    record = {
        "path": str(path.relative_to(BASE)).replace("\\", "/"),
        "bytes": path.stat().st_size,
    }
    if include_hash:
        record["sha256"] = sha256_file(path)
    return record


def main() -> None:
    config = load_config()
    critical = [
        BASE / "config.json",
        BASE / "README.md",
        BASE / "BATCH_GENERATION_PROMPT.md",
        BASE / "run_pipeline.ps1",
        BASE / config["dataset"]["archive"],
        BASE / config["sam2"]["checkpoint"],
        BASE
        / "downloads"
        / "wheels"
        / "torch-2.7.1+cu128-cp312-cp312-win_amd64.whl",
        BASE / "external" / "ProPainter" / "weights" / "raft-things.pth",
        BASE / "external" / "ProPainter" / "weights" / "recurrent_flow_completion.pth",
        BASE / "external" / "ProPainter" / "weights" / "ProPainter.pth",
    ]
    output_videos = sorted((BASE / "outputs").rglob("*.mp4"))
    inventory_roots = [
        BASE / name for name in ("data", "work", "outputs", "reports", "scripts")
    ]
    manifest_target = BASE / "reports" / "reproducibility_manifest.json"
    inventory_files = sorted(
        path
        for root in inventory_roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and path != manifest_target
    )
    pip_freeze = subprocess.check_output(
        [str(BASE / "runtime" / "env" / "python.exe"), "-m", "pip", "freeze"],
        text=True,
    ).splitlines()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_version": config["project_version"],
        "source_license": "DAVIS 2017 archive declares CC BY-NC 4.0; see data/provenance/README.md and SOURCES.md.",
        "repositories": {
            "sam2": {
                "url": "https://github.com/facebookresearch/sam2",
                "commit": git_commit(BASE / "external" / "sam2"),
            },
            "ProPainter": {
                "url": "https://github.com/sczhou/ProPainter",
                "commit": git_commit(BASE / "external" / "ProPainter"),
            },
        },
        "environment": {
            "python": subprocess.check_output(
                [str(BASE / "runtime" / "env" / "python.exe"), "--version"],
                text=True,
                stderr=subprocess.STDOUT,
            ).strip(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pip_freeze": pip_freeze,
        },
        "critical_inputs": [file_record(path) for path in critical],
        "pipeline_code": [
            file_record(path) for path in sorted((BASE / "scripts").glob("*.py"))
        ],
        "deliverable_videos": [file_record(path) for path in output_videos],
        "inventory": {
            "scope": [str(root.relative_to(BASE)) for root in inventory_roots],
            "file_count": len(inventory_files),
            "total_bytes": sum(path.stat().st_size for path in inventory_files),
            "files": [file_record(path, include_hash=False) for path in inventory_files],
        },
    }
    write_json(manifest_target, manifest)
    print(
        f"Manifest contains {len(output_videos)} videos and "
        f"{len(inventory_files)} inventoried files"
    )


if __name__ == "__main__":
    main()
