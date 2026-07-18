"""默认 VLM 复核边界。

真实 Video-VLM 适配器应实现 ``interfaces.VLMVerifier``，根据 request 中的视频和
``critical_frames`` 读取证据帧，并返回结构化 ``VLMReview``。核心包不会把未调用
VLM 的情况伪装成一个模型分数。
"""

from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .interfaces import MultimodalStructuredModel
from .schemas import CriticRequest, TrackSequence, ViolationCandidate, VLMReview


class CriticalFrameLoader(Protocol):
    """从视频读取关键帧并编码为图像 data URL。"""

    def load(self, video_path: str, frame_indices: tuple[int, ...]) -> tuple[str, ...]: ...


def with_track_evidence(
    candidate: ViolationCandidate,
    tracks: Sequence[TrackSequence],
    *,
    max_states: int = 24,
) -> ViolationCandidate:
    """Attach a bounded chronological SAM2 track snapshot to a candidate."""

    if max_states < 2:
        raise ValueError("max_states must be at least 2")
    track = next(
        (
            item
            for item in tracks
            if item.track_id == candidate.track_id
            or (item.object == candidate.object and item.track_id == candidate.track_id)
        ),
        None,
    )
    if track is None:
        return candidate
    states = tuple(sorted(track.states, key=lambda state: state.frame))
    if len(states) <= max_states:
        selected = states
    else:
        indices = {
            round(index * (len(states) - 1) / (max_states - 1))
            for index in range(max_states)
        }
        selected = tuple(states[index] for index in sorted(indices))
    serialized = [state.to_dict() for state in selected]
    evidence = dict(candidate.evidence)
    evidence["sam2_track"] = {
        "track_id": track.track_id,
        "object": track.object,
        "state_count": len(states),
        "visible_count": sum(1 for state in states if state.visible),
        "frame_range": [states[0].frame, states[-1].frame] if states else [],
        "states": serialized,
    }
    return replace(candidate, evidence=evidence)


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
            "claim_status": {
                "type": "string",
                "enum": ["confirmed", "rejected", "uncertain"],
            },
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
                "the supplied chronological keyframes and SAM2 track evidence. Do not infer "
                "unseen frames. Distinguish confirmed, rejected, and uncertain claims. "
                "A tracking or segmentation loss is not a physical disappearance: reject "
                "the claim when the object remains visible, and return uncertain when dust, "
                "occlusion, crop, or missing frames prevent a visual decision. Confirm only "
                "a prompt-relevant physical actor, not incidental background clutter. "
                "Do not reject events that the video prompt explicitly expects."
            ),
            user_prompt=json.dumps(
                {
                    "video_prompt": request.prompt,
                    "expected_event_policy": "do_not_reject_prompt_expected_events",
                    "candidate": {
                        "object": candidate.object,
                        "category": candidate.category,
                        "reason": candidate.reason,
                        "rules": candidate.rules,
                        "critical_frames": frames,
                        "evidence": candidate.evidence,
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
            claim_status=str(payload.get("claim_status", "uncertain")),
        )


class CategoryGroupedVLMVerifier(EvidenceGroundedVLMVerifier):
    """Review candidates grouped by object, category, and temporal segment."""

    def verify_many(
        self,
        request: CriticRequest,
        candidates: Sequence[ViolationCandidate],
        keyframes: Mapping[int, Sequence[int]],
    ) -> dict[int, VLMReview | None]:
        groups: dict[tuple[str, str, int, int], list[int]] = {}
        for index, candidate in enumerate(candidates):
            group_key = (
                candidate.object,
                candidate.category,
                candidate.start_frame,
                candidate.end_frame,
            )
            groups.setdefault(group_key, []).append(index)
        reviews: dict[int, VLMReview | None] = {}
        for object_name, category, start_frame, end_frame in sorted(groups):
            indices = groups[(object_name, category, start_frame, end_frame)]
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
                    "You are verifying one object/category/time-segment claim of a "
                    "proposed physics-video violation. Use only chronological keyframes, "
                    "SAM2 track evidence, and the video prompt. Reject detector/tracking "
                    "artifacts. A tracking or segmentation loss is not physical "
                    "disappearance; dust, occlusion, crop, or missing frames require an "
                    "uncertain result unless the violation is visually clear. Confirm only "
                    "a prompt-relevant physical actor, not incidental background clutter. "
                    "Do not reject events explicitly expected by the prompt. "
                    "Return whether the claim is confirmed, rejected, or uncertain and a "
                    "violation score for this segment."
                ),
                user_prompt=json.dumps(
                    {
                        "video_prompt": request.prompt,
                        "expected_event_policy": "do_not_reject_prompt_expected_events",
                        "object": object_name,
                        "category": category,
                        "time_segment": {
                            "start_frame": start_frame,
                            "end_frame": end_frame,
                        },
                        "representative_critical_frames": frames,
                        "candidates": [
                            {
                                "object": candidates[index].object,
                                "category": candidates[index].category,
                                "reason": candidates[index].reason,
                                "detector_score": candidates[index].detector_score,
                                "start_frame": candidates[index].start_frame,
                                "end_frame": candidates[index].end_frame,
                                "evidence": candidates[index].evidence,
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
                claim_status=str(payload.get("claim_status", "uncertain")),
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
