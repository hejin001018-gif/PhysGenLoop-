"""Repackage a verified legacy v3 bundle for the canonical Learning Repair API."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import textwrap
from typing import Any, Mapping


RELEASE_SCHEMA_VERSION = "repair-agent-action-value-release/1.1"
RELEASE_VERSION = "3.1.0"
GENERATED_FILES = frozenset(
    {"README.md", "inference.py", "release_manifest.json", "requirements.txt"}
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(raw)


def safe_member(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise ValueError(f"invalid release member path: {relative!r}")
    destination = (root / relative).resolve()
    if root.resolve() not in destination.parents:
        raise ValueError(f"release member escapes its root: {relative!r}")
    return destination


def verify_source_release(source: Path) -> dict[str, Any]:
    manifest_path = source / "release_manifest.json"
    manifest = load_object(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("source release manifest has no file inventory")
    failures: list[str] = []
    for relative, metadata in files.items():
        if not isinstance(metadata, Mapping):
            failures.append(f"{relative}: invalid metadata")
            continue
        path = safe_member(source, str(relative))
        if not path.is_file():
            failures.append(f"{relative}: missing")
            continue
        expected_size = int(metadata.get("bytes", -1))
        expected_sha = str(metadata.get("sha256", ""))
        actual_size = path.stat().st_size
        actual_sha = sha256(path)
        if actual_size != expected_size or actual_sha != expected_sha:
            failures.append(
                f"{relative}: expected {expected_size}/{expected_sha}, "
                f"got {actual_size}/{actual_sha}"
            )
    if failures:
        raise ValueError("source release verification failed: " + "; ".join(failures))
    return manifest


INFERENCE = '''#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from physgenloop.learning_repair import (
    ActionValueDecisionPolicy,
    CompatibilityManifest,
    RepairContext,
    TorchActionValuePolicy,
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-report", required=True, type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--candidate-id", default="deployment-candidate")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    compatibility = CompatibilityManifest.load(
        root / "config" / "critic_compatibility_v1.json"
    )
    learned = TorchActionValuePolicy.load(
        root / "model" / "best_action_value_policy.pt",
        device=args.device,
        compatibility_manifest=compatibility,
    )
    policy = ActionValueDecisionPolicy(learned, minimum_confidence=0.0)
    context = RepairContext.from_dict(
        json.loads(args.context.read_text(encoding="utf-8"))
        if args.context
        else None
    )
    report = json.loads(args.critic_report.read_text(encoding="utf-8"))
    if isinstance(report, dict) and isinstance(report.get("report"), dict):
        report = report["report"]
    decision = policy.decide(
        critic_report=report,
        candidate=Candidate(candidate_id=args.candidate_id),
        prompt=args.prompt,
        context=context,
    ).to_dict()
    payload = {
        "decision": decision,
        "model_id": learned.model_id,
        "compatibility_id": compatibility.compatibility_id,
        "selection_mode": learned.selection_mode,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def readme(manifest: Mapping[str, Any]) -> str:
    return textwrap.dedent(
        f"""
        # Repair Agent v3.1 Canonical Pre-release

        This bundle contains the trained Repair Agent checkpoint, frozen Critic and
        feature compatibility inputs, proxy Memory, evaluation evidence, and a
        canonical inference entry point.

        - Model: `{manifest['model_id']}`
        - Release version: `{manifest['release_version']}`
        - Canonical namespace: `{manifest['canonical_namespace']}`
        - Source revision: `{manifest['release_source_revision']}` (`hj`)
        - Compatibility: `{manifest['compatibility_id']}`
        - Selection mode: `{manifest['selection_mode']}`
        - Proxy labels: `{manifest['proxy_label_count']}`
        - Actual Executor trials: `{manifest['actual_trial_count']}`
        - Deployment ready: `{str(manifest['deployment_ready']).lower()}`

        ## Install the canonical runtime

        ```bash
        git clone --branch hj https://github.com/hejin001018-gif/PhysGenLoop-.git
        cd PhysGenLoop-
        python -m pip install -e ".[train]"
        ```

        ## Inference

        ```bash
        python inference.py \\
          --critic-report critic_report.json \\
          --context context.json \\
          --device cuda
        ```

        The checkpoint is trained from Blender selected-action proxy labels.  It is
        a research pre-release and must not be reported as HunyuanVideo repair
        success until real Prompt/Global/Local Executor trials and source-disjoint
        Hunyuan calibration/test evaluation are completed.
        """
    ).lstrip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--target-branch", default="hj")
    return parser.parse_args()


def build_deterministic_archive(source: Path, archive: Path) -> tuple[Path, str]:
    archive.parent.mkdir(parents=True, exist_ok=True)
    members = sorted(item for item in source.rglob("*") if item.is_file())
    with archive.open("xb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as bundle:
                for path in members:
                    relative = path.relative_to(source).as_posix()
                    info = bundle.gettarinfo(str(path), arcname=relative)
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    info.mtime = 0
                    info.pax_headers = {}
                    with path.open("rb") as handle:
                        bundle.addfile(info, handle)
    with tarfile.open(archive, mode="r:gz") as bundle:
        archived = tuple(item.name for item in bundle.getmembers() if item.isfile())
    expected = tuple(path.relative_to(source).as_posix() for path in members)
    if archived != expected:
        raise RuntimeError("archive inventory does not match the release directory")
    digest = sha256(archive)
    checksum = Path(f"{archive}.sha256")
    checksum.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    return checksum, digest


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    archive = args.archive.resolve()
    checksum = Path(f"{archive}.sha256")
    if not source.is_dir():
        raise FileNotFoundError(f"source release directory is missing: {source}")
    if output.exists():
        raise FileExistsError(f"release output already exists: {output}")
    if archive.exists() or checksum.exists():
        raise FileExistsError(f"release archive or checksum already exists: {archive}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_revision):
        raise ValueError("source-revision must be a full lowercase Git SHA")

    source_manifest = verify_source_release(source)
    output.mkdir(parents=True)
    for relative in sorted(source_manifest["files"]):
        relative = str(relative)
        if relative in GENERATED_FILES:
            continue
        source_path = safe_member(source, relative)
        destination = safe_member(output, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    (output / "inference.py").write_text(INFERENCE, encoding="utf-8")
    (output / "inference.py").chmod(0o755)
    (output / "requirements.txt").write_text(
        "torch>=2.7.1\nPyYAML>=6.0\n", encoding="utf-8"
    )

    training_report = load_object(output / "reports" / "training_report.json")
    compatibility = load_object(output / "config" / "critic_compatibility_v1.json")
    provenance = training_report.get("label_provenance", {})
    if not isinstance(provenance, Mapping):
        provenance = {}
    compatibility_source = str(compatibility.get("source_revision", "unknown"))
    manifest: dict[str, Any] = {
        "schema_version": RELEASE_SCHEMA_VERSION,
        "release_version": RELEASE_VERSION,
        "release_channel": "prerelease",
        "canonical_namespace": "physgenloop.learning_repair",
        "target_branch": args.target_branch,
        "release_source_revision": args.source_revision,
        "model_id": source_manifest.get("model_id"),
        "compatibility_id": source_manifest.get("compatibility_id"),
        "compatibility_source_revision": compatibility_source,
        "selection_mode": provenance.get("selection_mode", "classification_proxy"),
        "label_type": source_manifest.get("label_type"),
        "actual_trial_count": int(source_manifest.get("actual_trial_count", 0)),
        "proxy_label_count": int(source_manifest.get("proxy_label_count", 0)),
        "deployment_ready": compatibility_source not in {"", "unknown"},
        "source_targets_sha256": source_manifest.get("source_targets_sha256"),
        "source_release_manifest_sha256": sha256(source / "release_manifest.json"),
        "files": {},
        "limitations": list(source_manifest.get("limitations", ())),
    }
    manifest["limitations"].append(
        "This canonical v3.1 package preserves the v3 checkpoint; it does not retrain or relabel data."
    )
    (output / "README.md").write_text(readme(manifest), encoding="utf-8")

    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        manifest["files"][relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    (output / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    verify_source_release(output)
    checksum, archive_sha256 = build_deterministic_archive(output, archive)
    print(
        json.dumps(
            {
                "release": str(output),
                "release_version": RELEASE_VERSION,
                "file_count": len(manifest["files"]),
                "model_id": manifest["model_id"],
                "source_revision": manifest["release_source_revision"],
                "archive": str(archive),
                "archive_sha256": archive_sha256,
                "checksum": str(checksum),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
