"""一次性生成步骤：加载 Wan2.2，生成单个候选，写出 JSON 后立即退出。

进程退出即释放 Wan2.2 占用的全部显存，供后续 vLLM 评估阶段使用。
直接使用 WanGenerator（底层推理），不经过 WanPhysicsGenerator（常驻封装），
避免在一次性子进程里做无意义的"常驻"语义。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

# 双卡角色分工：允许把 Wan2.2 生成固定到某张卡（默认 GPU0），与 vLLM(GPU1) 分离。
# 必须在 import torch / WanGenerator 之前设置 CUDA_VISIBLE_DEVICES 才生效。
_GPU_ID_ENV = os.environ.get("WAN_GPU_ID")
if _GPU_ID_ENV is not None and _GPU_ID_ENV != "":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(_GPU_ID_ENV)

from generators.wanphysics.wan_generator import WanGenerator  # noqa: E402


def _candidate_id(prompt: str, seed: int) -> str:
    digest = hashlib.sha256(f"{prompt}\0{seed}".encode("utf-8")).hexdigest()[:6]
    return f"wan-{seed:04d}-{digest}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", default=str(_ROOT / "outputs"))
    parser.add_argument("--model-path", default=str(_ROOT / "models" / "wan2.2_ti2v_5b"))
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--gpu-id", default=None, help="固定生成使用的 GPU（如 0）；也可用 WAN_GPU_ID 环境变量")
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    # --gpu-id 优先于已在模块顶层读取的 WAN_GPU_ID；若给出且尚未设置则应用。
    if args.gpu_id is not None and os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    candidate_id = _candidate_id(args.prompt, args.seed)
    candidate_dir = Path(args.output_root) / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    video_path = candidate_dir / f"{candidate_id}-v01.mp4"

    generator = WanGenerator(model_path=args.model_path, device="cuda")
    generator.generate_video(
        prompt=args.prompt,
        output_path=str(video_path),
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
    )

    metadata = {
        "backend": "wan2.2-ti2v-5b",
        "is_real_video": True,
        "seed": args.seed,
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "fps": args.fps,
    }
    (candidate_dir / "prompt.txt").write_text(args.prompt, encoding="utf-8")
    (candidate_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # critic.json 初始占位，status=waiting；Critic 完成后回填
    critic_init = {
        "video": f"{candidate_id}-v01.mp4",
        "status": "waiting",
        "physics_violation": None,
        "reason": None,
        "confidence": None,
    }
    (candidate_dir / "critic.json").write_text(
        json.dumps(critic_init, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    payload = {
        "candidate_id": candidate_id,
        "video_path": str(video_path),
        "prompt": args.prompt,
        "seed": args.seed,
        "metadata": metadata,
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("GEN_OK", candidate_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
