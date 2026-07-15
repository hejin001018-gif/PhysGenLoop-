"""Deterministic frame sampling shared by benchmark methods."""

from __future__ import annotations

import base64
from dataclasses import dataclass


def uniform_indices(total_frames: int, count: int) -> tuple[int, ...]:
    if total_frames <= 0 or count <= 0:
        raise ValueError("total_frames and count must be positive")
    if count >= total_frames:
        return tuple(range(total_frames))
    if count == 1:
        return (0,)
    return tuple(
        round(index * (total_frames - 1) / (count - 1)) for index in range(count)
    )


@dataclass(frozen=True)
class SampledFrames:
    indices: tuple[int, ...]
    data_urls: tuple[str, ...]
    total_frames: int
    fps: float


def sample_video_frames(
    video_path: str,
    *,
    count: int = 16,
    jpeg_quality: int = 85,
) -> SampledFrames:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        if total <= 0:
            raise ValueError(f"video reports no frames: {video_path}")
        requested = uniform_indices(total, count)
        actual: list[int] = []
        urls: list[str] = []
        for frame_index in requested:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
            )
            if ok:
                actual.append(frame_index)
                urls.append(
                    "data:image/jpeg;base64,"
                    + base64.b64encode(encoded.tobytes()).decode("ascii")
                )
    finally:
        cap.release()
    if not urls:
        raise ValueError(f"no requested frames could be decoded: {video_path}")
    return SampledFrames(tuple(actual), tuple(urls), total, fps)
