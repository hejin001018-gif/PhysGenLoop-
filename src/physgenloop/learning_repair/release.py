"""将训练产物导出为不依赖 Blender 的可迁移 Repair Agent bundle。"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any

import yaml

from .agent import AgentConfig
from .dataset import load_repair_manifest
from .policy import require_torch


_INFERENCE_SCRIPT = '''"""Run the exported PhysGenLoop Learning Repair Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from physgenloop.learning_repair import (
    AgentConfig,
    LearningRepairAgent,
    RepairContext,
    RepairMemory,
    TorchMLPRepairPolicy,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--critic-report", required=True, type=Path)
    parser.add_argument("--context", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8")) or {}
    policy = TorchMLPRepairPolicy.load(root / "model.pt", device=args.device)
    memory_path = root / "repair_memory.jsonl"
    memory = (
        RepairMemory.from_manifest(memory_path, encoder=policy.encoder)
        if memory_path.is_file()
        else None
    )
    agent = LearningRepairAgent(
        policy,
        memory=memory,
        config=AgentConfig(**dict(config.get("agent", {}))),
    )
    report = json.loads(args.critic_report.read_text(encoding="utf-8"))
    if isinstance(report, dict) and isinstance(report.get("report"), dict):
        report = report["report"]
    context = (
        RepairContext.from_dict(json.loads(args.context.read_text(encoding="utf-8")))
        if args.context
        else RepairContext()
    )
    payload = agent.decide(
        critic_report=report,
        prompt=args.prompt,
        context=context,
    ).to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


_README = """# Exported Learning Repair Agent

This bundle is the deployable output of the Blender training stage. Blender and the
training videos are not needed at inference time.

Install the matching PhysGenLoop source revision and runtime dependencies, then run:

```bash
python inference.py --critic-report critic_report.json --device cuda
```

`critic_snapshot.json` identifies the frozen Critic contract used for training.
`feature_schema.json` replaces a tokenizer: this policy consumes deterministic
structured CriticReport features and does not tokenize natural-language text.

The returned action is a strategy decision. Prompt repair, global regeneration, local
editing, and candidate rejection still require deployment-specific execution adapters.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _load_checkpoint(path: Path) -> dict[str, Any]:
    torch = require_torch()
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6 compatibility
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != "1.0":
        raise ValueError("unsupported Repair Policy checkpoint")
    return checkpoint


def export_release(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path | None = None,
    memory_path: str | Path | None = None,
    critic_config_path: str | Path | None = None,
    critic_model_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """构造完整 bundle；成功前不会把半成品暴露为最终目录。"""

    checkpoint_source = Path(checkpoint_path).resolve()
    if not checkpoint_source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_source}")
    checkpoint = _load_checkpoint(checkpoint_source)
    destination = Path(output_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"release directory already exists: {destination}")
        if not destination.is_dir():
            raise ValueError(f"release output is not a directory: {destination}")
        shutil.rmtree(destination)

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        shutil.copy2(checkpoint_source, temporary / "model.pt")
        if config_path is not None:
            shutil.copy2(Path(config_path).resolve(), temporary / "config.yaml")
        else:
            (temporary / "config.yaml").write_text(
                yaml.safe_dump(
                    {"agent": asdict(AgentConfig())},
                    sort_keys=False,
                    allow_unicode=True,
                ),
                encoding="utf-8",
            )
        if memory_path is not None:
            memory_source = Path(memory_path).resolve()
            # Validate before copying so a corrupt memory cannot enter a release.
            load_repair_manifest(memory_source)
            shutil.copy2(memory_source, temporary / "repair_memory.jsonl")

        critic_snapshot: dict[str, Any] = {
            "source_revision": _git_revision(),
            "critic_model_id": critic_model_id,
        }
        if critic_config_path is not None:
            critic_source = Path(critic_config_path).resolve()
            shutil.copy2(critic_source, temporary / "critic_config.yaml")
            critic_snapshot.update(
                {
                    "config_file": "critic_config.yaml",
                    "config_sha256": _sha256(critic_source),
                }
            )
        (temporary / "critic_snapshot.json").write_text(
            json.dumps(critic_snapshot, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (temporary / "feature_schema.json").write_text(
            json.dumps(
                {
                    "feature_config": checkpoint["feature_config"],
                    "feature_names": checkpoint["feature_names"],
                    "action_order": checkpoint["action_order"],
                    "model_id": checkpoint.get("model_id"),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (temporary / "inference.py").write_text(_INFERENCE_SCRIPT, encoding="utf-8")
        (temporary / "README.md").write_text(_README, encoding="utf-8")
        (temporary / "requirements.txt").write_text(
            "torch>=2.7.1\nPyYAML>=6.0\njsonschema>=4.20,<5\n",
            encoding="utf-8",
        )
        files = sorted(path for path in temporary.rglob("*") if path.is_file())
        manifest = {
            "format_version": "1.0",
            "model_id": checkpoint.get("model_id"),
            "source_revision": _git_revision(),
            "files": {
                path.relative_to(temporary).as_posix(): {
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            },
        }
        (temporary / "release_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination
