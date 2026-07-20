"""
Wan2.2-TI2V-5B 推理封装器
模型权重需自行从 Hugging Face 下载到 ./models/wan2.2_ti2v_5b/
使用方式： python -m generators.wan_generator --prompt "your prompt"
"""

import torch
import numpy as np
from diffusers import WanPipeline
from PIL import Image
import os
import argparse

class WanGenerator:
    def __init__(self, model_path="./models/wan2.2_ti2v_5b", device="cuda"):
        """
        初始化 Wan2.2 模型
        :param model_path: 本地模型权重路径
        :param device: cuda 或 cpu
        """
        self.device = device
        self.pipe = WanPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        ).to(device)
        print("✅ Wan2.2-TI2V-5B 模型加载完成")

    def generate_video(
        self,
        prompt,
        output_path="./outputs/wan_output.mp4",
        num_frames=81,
        height=704,
        width=1280,
        fps=24,
        seed=None,
        negative_prompt=None,
    ):
        """
        根据文本生成视频
        :param prompt: 提示词
        :param output_path: 输出视频路径
        :param num_frames: 视频帧数，需满足官方推荐的 4n+1 格式（默认 81）
        :param height: 输出高度，官方 720P 推荐值为 704
        :param width: 输出宽度，官方 720P 推荐值为 1280
        :param fps: 导出视频的帧率，官方推荐 24fps
        :param seed: 可选随机种子，用于复现或制造候选之间的差异
        :param negative_prompt: 可选负向提示词
        """
        print(f"📝 生成提示词: {prompt}")
        generator = (
            torch.Generator(device=self.device).manual_seed(seed)
            if seed is not None
            else None
        )
        with torch.no_grad():
            video_frames = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                height=height,
                width=width,
                generator=generator,
            ).frames[0]  # 返回 PIL Image 列表

        # 保存为视频（依赖 imageio 库）
        import imageio
        writer = imageio.get_writer(output_path, fps=fps)
        for frame in video_frames:
            writer.append_data(np.array(frame))
        writer.close()
        print(f"✅ 视频已保存: {output_path}")
        return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True, help="生成视频的提示词")
    parser.add_argument("--output", type=str, default="./outputs/wan_output.mp4", help="输出路径")
    parser.add_argument("--model_path", type=str, default="./models/wan2.2_ti2v_5b", help="模型路径")
    args = parser.parse_args()

    generator = WanGenerator(model_path=args.model_path)
    generator.generate_video(args.prompt, args.output)