"""纯生成入口，不经过 Critic 和闭环控制器。

用于调试单个视频效果，或提前批量生成供离线评估。
生成完成后写出 critic.json（status: waiting），等待 Critic 填写。

用法：
  python agents/wanphysics/generate_only.py --prompt "a red ball rolling on a flat table"
  python agents/wanphysics/generate_only.py --prompt "..." --task-id 0015 --seed 42
  python agents/wanphysics/generate_only.py --config configs/loop.yaml --task-id 0015
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "/root/PhysGenLoop-")
sys.path.insert(0, "/root/PhysGenLoop-/src")

from dotenv import load_dotenv

load_dotenv("/root/PhysGenLoop-/.env")

from generators.wanphysics.adapter import WanSubprocessGenerator

_DEFAULT_CONFIG = Path("/root/PhysGenLoop-/configs/loop.yaml")


def _load_cfg(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    parser = argparse.ArgumentParser(description="纯生成入口（不含 Critic/闭环）")
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG))
    parser.add_argument("--prompt", default=None, help="覆盖 yaml 中的 prompt")
    parser.add_argument("--task-id", default=None,
                        help="输出目录名，默认使用 prompt 前 20 字符的 slug")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子，默认使用 yaml 中的 base_seed")
    parser.add_argument("--output-root", default=None, help="覆盖输出根目录")
    args = parser.parse_args()

    cfg = _load_cfg(Path(args.config))
    paths = cfg["paths"]
    gen = cfg["generator"]
    loop = cfg["loop"]

    prompt = args.prompt or loop["prompt"]
    seed = args.seed if args.seed is not None else loop["base_seed"]
    output_root = Path(args.output_root or paths["outputs"])

    # task_id 优先用 --task-id，否则用 prompt 前 20 字符做 slug
    if args.task_id:
        task_id = args.task_id
    else:
        slug = prompt[:20].strip().replace(" ", "_").replace("/", "-")
        task_id = slug if slug else "task"

    task_dir = output_root / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    generator = WanSubprocessGenerator(
        python=paths["envs"]["main"],
        model_path=paths["models"]["wan"],
        output_root=str(task_dir),
        num_frames=gen["num_frames"],
        height=gen["height"],
        width=gen["width"],
        fps=gen["fps"],
        negative_prompt=gen.get("negative_prompt"),
    )

    from pavg_critic.schemas import PhysicsPlan
    candidate = generator.generate(prompt=prompt, physics_plan=PhysicsPlan(), seed=seed)

    # 把视频和附属文件移到 task_dir 顶层（去掉 candidate_id 子目录层级）
    import shutil
    src_dir = Path(candidate.video_path).parent
    if src_dir != task_dir:
        for f in src_dir.iterdir():
            dest = task_dir / f.name
            if dest.exists():
                dest.unlink()
            shutil.move(str(f), str(dest))
        src_dir.rmdir()

    video_name = Path(candidate.video_path).name
    video_path = task_dir / video_name

    # 写 critic.json 占位（status: waiting）
    critic_init = {
        "video": video_name,
        "status": "waiting",
        "physics_violation": None,
        "reason": None,
        "confidence": None,
    }
    (task_dir / "critic.json").write_text(
        json.dumps(critic_init, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    result = {
        "task_id": task_id,
        "candidate_id": candidate.candidate_id,
        "video_path": str(video_path),
        "prompt": prompt,
        "seed": seed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[generate_only] 完成  task_id={task_id}  video={video_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
