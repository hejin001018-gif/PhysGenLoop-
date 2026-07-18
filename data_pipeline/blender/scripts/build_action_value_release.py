"""Build and verify a self-contained Action-Value Repair Agent release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import textwrap


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--memory-targets", required=True, type=Path)
    parser.add_argument("--compatibility", required=True, type=Path)
    parser.add_argument("--critic-config", required=True, type=Path)
    parser.add_argument("--feature-schema", required=True, type=Path)
    parser.add_argument("--adaptation-report", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


INFERENCE = '''#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

from physgenloop.learning_repair.contracts import RepairContext
from physgenloop.learning_repair.compatibility import CompatibilityManifest
from physgenloop.learning_repair.value_policy import (
    ActionValueDecisionPolicy,
    TorchActionValuePolicy,
)

@dataclass(frozen=True)
class Candidate:
    candidate_id: str = "deployment-candidate"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-report", required=True, type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    compatibility = CompatibilityManifest.load(root / "config" / "critic_compatibility_v1.json")
    learned = TorchActionValuePolicy.load(
        root / "model" / "best_action_value_policy.pt",
        device=args.device,
        compatibility_manifest=compatibility,
    )
    policy = ActionValueDecisionPolicy(learned, minimum_confidence=0.0)
    context = RepairContext.from_dict(
        json.loads(args.context.read_text(encoding="utf-8")) if args.context else None
    )
    report = json.loads(args.critic_report.read_text(encoding="utf-8"))
    decision = policy.decide(
        critic_report=report,
        candidate=Candidate(),
        prompt=args.prompt,
        context=context,
    ).to_dict()
    payload = {"decision": decision, "model_id": learned.model_id, "compatibility_id": compatibility.compatibility_id}
    if args.output:
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
'''


def main() -> int:
    args = parse_args()
    training = args.training_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"release output already exists: {output}")
    required = {
        "model/best_action_value_policy.pt": training / "best_action_value_policy.pt",
        "reports/training_report.json": training / "training_report.json",
        "memory/proxy_memory_train.jsonl": args.memory_targets.resolve(),
        "config/critic_compatibility_v1.json": args.compatibility.resolve(),
        "config/critic_config.yaml": args.critic_config.resolve(),
        "config/feature_schema.json": args.feature_schema.resolve(),
        "reports/proxy_adaptation.json": args.adaptation_report.resolve(),
        "reports/evaluation_test.json": args.evaluation.resolve(),
    }
    missing = [str(source) for source in required.values() if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"missing release inputs: {missing}")
    output.mkdir(parents=True)
    for relative, source in required.items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    (output / "inference.py").write_text(INFERENCE, encoding="utf-8")
    (output / "inference.py").chmod(0o755)

    report = json.loads(args.adaptation_report.read_text(encoding="utf-8"))
    training_report = json.loads((training / "training_report.json").read_text(encoding="utf-8"))
    from physgenloop.learning_repair.compatibility import CompatibilityManifest

    compatibility = CompatibilityManifest.load(args.compatibility).to_dict()
    manifest = {
        "schema_version": "repair-agent-action-value-release/1.0",
        "model_id": training_report.get("model_id"),
        "compatibility_id": compatibility.get("compatibility_id"),
        "label_type": report.get("label_type"),
        "actual_trial_count": report.get("actual_trial_count", 0),
        "proxy_label_count": report.get("sample_count", 0),
        "source_targets_sha256": sha256(args.targets.resolve()),
        "files": {},
        "limitations": report.get("limitations", []),
    }
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "release_manifest.json"):
        relative = path.relative_to(output).as_posix()
        manifest["files"][relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (output / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    readme = textwrap.dedent(
        f"""
        # Repair Agent v3 Action-Value Release

        This release uses the canonical Learning Repair package with four action
        values, capability masking, and the frozen Critic compatibility contract.

        - Model: `{manifest['model_id']}`
        - Compatibility: `{manifest['compatibility_id']}`
        - Proxy labels: `{manifest['proxy_label_count']}`
        - Actual Executor trials: `{manifest['actual_trial_count']}`

        The current checkpoint is Blender proxy-trained. It must not be reported as
        HunyuanVideo repair success until real Executor trials are collected.

        ```bash
        python inference.py --critic-report critic_report.json --context context.json --device cuda
        ```
        """
    ).lstrip()
    (output / "README.md").write_text(readme, encoding="utf-8")
    # README is part of the release manifest, so append it after writing it.
    manifest["files"]["README.md"] = {
        "bytes": (output / "README.md").stat().st_size,
        "sha256": sha256(output / "README.md"),
    }
    (output / "release_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"release": str(output), "file_count": len(manifest["files"]), "model_id": manifest["model_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
