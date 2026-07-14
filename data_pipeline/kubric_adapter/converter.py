"""Kubric 输出 → PAVG sample.schema.json 适配。

Kubric 官方输出包含 metadata.json / data_ranges.json / segmentation / depth / flow / instances
本模块只做结构映射与命名归一，不做任何物理判定。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "1.0"


@dataclass
class KubricSample:
    root: Path
    metadata_path: Path
    video_path: Path


def _load_metadata(sample_root: Path) -> dict[str, Any]:
    meta = sample_root / "metadata.json"
    if not meta.exists():
        raise FileNotFoundError(f"Kubric metadata missing: {meta}")
    return json.loads(meta.read_text(encoding="utf-8"))


def _extract_objects(meta: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for inst in meta.get("instances", []):
        out.append({
            "id": str(inst.get("asset_id") or inst.get("id")),
            "category": inst.get("category", "unknown"),
            "role": inst.get("role", "actor"),
        })
    return out


def convert_kubric_output(
    sample_root: str | Path,
    sample_id: str,
    is_physical: bool,
    violations: list[dict[str, Any]] | None = None,
    scene_template: str | None = None,
) -> dict[str, Any]:
    """把 Kubric 单样本目录归一为符合 sample.schema.json 的 dict。

    Args:
        sample_root: Kubric 输出根目录（包含 metadata.json / rgba/ 等）
        sample_id: 分配的样本 id
        is_physical: 是否为正常物理样本
        violations: 若为异常样本，列出违规条目
        scene_template: 场景模板名（用于按模板分组切分）
    """
    root = Path(sample_root)
    meta = _load_metadata(root)

    resolution = meta.get("resolution", [0, 0])
    frame_rate = meta.get("frame_rate") or meta.get("fps") or 24

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "source": "kubric_synthetic",
        "is_physical": is_physical,
        "video_path": str((root / "video.mp4").as_posix()),
        "scene_template": scene_template or meta.get("scene", "unknown"),
        "seed": int(meta.get("seed", 0)),
        "fps": float(frame_rate),
        "num_frames": int(meta.get("num_frames", 0)),
        "resolution": {"width": int(resolution[0]), "height": int(resolution[1])},
        "objects": _extract_objects(meta),
        "violations": violations or [],
        "provenance": {
            "engine": "kubric",
            "engine_version": meta.get("kubric_version", "unknown"),
            "renderer": meta.get("renderer", "blender"),
            "created_at": meta.get("created_at", ""),
            "pipeline_version": "0.1.0",
        },
    }
    return payload


if __name__ == "__main__":
    # smoke test 占位。真实调用需先跑 Kubric worker。
    import sys
    if len(sys.argv) < 3:
        print("usage: python -m data_pipeline.kubric_adapter.converter <sample_root> <sample_id>")
        sys.exit(1)
    out = convert_kubric_output(sys.argv[1], sys.argv[2], is_physical=True)
    print(json.dumps(out, indent=2, ensure_ascii=False))
