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
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device)
        print("✅ Wan2.2-TI2V-5B 模型加载完成")

    def generate_video(self, prompt, output_path="./outputs/wan_output.mp4", num_frames=16):
        """
        根据文本生成视频
        :param prompt: 提示词
        :param output_path: 输出视频路径
        :param num_frames: 视频帧数
        """
        print(f"📝 生成提示词: {prompt}")
        with torch.no_grad():
            video_frames = self.pipe(
                prompt=prompt,
                num_frames=num_frames,
                height=480,
                width=720,
            ).frames[0]  # 返回 PIL Image 列表

        # 保存为视频（依赖 imageio 库）
        import imageio
        writer = imageio.get_writer(output_path, fps=8)
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