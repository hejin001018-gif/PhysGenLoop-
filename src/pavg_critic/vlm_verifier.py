"""默认 VLM 复核边界。

真实 Video-VLM 适配器应实现 ``interfaces.VLMVerifier``，根据 request 中的视频和
``critical_frames`` 读取证据帧，并返回结构化 ``VLMReview``。核心包不会把未调用
VLM 的情况伪装成一个模型分数。
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .interfaces import MultimodalStructuredModel
from .schemas import CriticRequest, ViolationCandidate, VLMReview


class CriticalFrameLoader(Protocol):
    """从视频读取关键帧并编码为图像 data URL。"""

    def load(self, video_path: str, frame_indices: tuple[int, ...]) -> tuple[str, ...]: ...


class NoOpVLMVerifier:
    """显式返回“无 VLM 证据”，用于纯规则基线和消融实验。"""

    def verify(
        self,
        request: CriticRequest,
        candidate: ViolationCandidate,
        critical_frames: Sequence[int],
    ) -> VLMReview | None:
        """返回 ``None``，让融合器仅以 detector/rule 分数作出判断。"""

        return None


class EvidenceGroundedVLMVerifier:
    """只用定位后的关键帧复核一个规则候选，避免整段视频无证据泛评。"""

    _OUTPUT_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["violation_score", "reason", "repair_instruction"],
        "properties": {
            "violation_score": {"type": "number", "minimum": 0, "maximum": 1},
            "reason": {"type": "string"},
            "repair_instruction": {"type": "string"},
        },
    }

    def __init__(
        self,
        model: MultimodalStructuredModel,
        *,
        frame_loader: CriticalFrameLoader | None = None,
        model_name: str = "multimodal_model",
    ) -> None:
        self.model = model
        self.frame_loader = frame_loader or OpenCVFrameDataUrlLoader()
        self.model_name = model_name

    def verify(
        self,
        request: CriticRequest,
        candidate: ViolationCandidate,
        critical_frames: Sequence[int],
    ) -> VLMReview | None:
        frames = tuple(dict.fromkeys(int(frame) for frame in critical_frames))
        if not frames:
            return None
        images = self.frame_loader.load(request.video_path, frames)
        if not images:
            return None
        payload = self.model.generate_json_with_images(
            system_prompt=(
                "You are a physics-video verifier. Judge only the claimed violation using "
                "the supplied chronological keyframes. Do not infer unseen frames."
            ),
            user_prompt=json.dumps(
                {
                    "video_prompt": request.prompt,
                    "candidate": {
                        "object": candidate.object,
                        "category": candidate.category,
                        "reason": candidate.reason,
                        "rules": candidate.rules,
                        "critical_frames": frames,
                    },
                },
                ensure_ascii=False,
            ),
            image_data_urls=images,
            schema=self._OUTPUT_SCHEMA,
        )
        return VLMReview(
            score=float(payload["violation_score"]),
            reason=str(payload["reason"]),
            repair_instruction=str(payload["repair_instruction"]),
            model=self.model_name,
        )


class CategoryGroupedVLMVerifier(EvidenceGroundedVLMVerifier):
    """Review all candidates with at most one multimodal call per category."""

    def verify_many(
        self,
        request: CriticRequest,
        candidates: Sequence[ViolationCandidate],
        keyframes: Mapping[int, Sequence[int]],
    ) -> dict[int, VLMReview | None]:
        groups: dict[str, list[int]] = {}
        for index, candidate in enumerate(candidates):
            groups.setdefault(candidate.category, []).append(index)
        reviews: dict[int, VLMReview | None] = {}
        for category in sorted(groups):
            indices = groups[category]
            representative = max(
                indices,
                key=lambda index: (candidates[index].detector_score, -index),
            )
            frames = tuple(
                dict.fromkeys(int(frame) for frame in keyframes.get(representative, ()))
            )
            if not frames:
                for index in indices:
                    reviews[index] = None
                continue
            images = self.frame_loader.load(request.video_path, frames)
            if not images:
                for index in indices:
                    reviews[index] = None
                continue
            payload = self.model.generate_json_with_images(
                system_prompt=(
                    "You are verifying one category of a proposed physics-video "
                    "violation. Use only the chronological keyframes and the video "
                    "prompt. Reject detector/tracking artifacts and events explicitly "
                    "expected by the prompt. Return one score for whether this category "
                    "is a real physical violation in the video."
                ),
                user_prompt=json.dumps(
                    {
                        "video_prompt": request.prompt,
                        "category": category,
                        "representative_critical_frames": frames,
                        "candidates": [
                            {
                                "object": candidates[index].object,
                                "reason": candidates[index].reason,
                                "detector_score": candidates[index].detector_score,
                                "start_frame": candidates[index].start_frame,
                                "end_frame": candidates[index].end_frame,
                            }
                            for index in sorted(
                                indices,
                                key=lambda item: -candidates[item].detector_score,
                            )[:5]
                        ],
                    },
                    ensure_ascii=False,
                ),
                image_data_urls=images,
                schema=self._OUTPUT_SCHEMA,
            )
            review = VLMReview(
                score=float(payload["violation_score"]),
                reason=str(payload["reason"]),
                repair_instruction=str(payload["repair_instruction"]),
                model=self.model_name,
            )
            for index in indices:
                reviews[index] = review
        return reviews


class OpenCVFrameDataUrlLoader:
    """按帧号随机读取视频并编码 JPEG；OpenCV 保持可选依赖。"""

    def __init__(self, *, jpeg_quality: int = 85) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        self.jpeg_quality = jpeg_quality

    def load(self, video_path: str, frame_indices: tuple[int, ...]) -> tuple[str, ...]:
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Critical-frame loading requires pavg-critic[video]") from exc
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        encoded: list[str] = []
        try:
            for frame_index in frame_indices:
                if frame_index < 0:
                    raise ValueError("critical frame indices must be non-negative")
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, image = capture.read()
                if not ok:
                    continue
                ok, buffer = cv2.imencode(
                    ".jpg",
                    image,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
                )
                if ok:
                    data = base64.b64encode(buffer.tobytes()).decode("ascii")
                    encoded.append(f"data:image/jpeg;base64,{data}")
        finally:
            capture.release()
        return tuple(encoded)
