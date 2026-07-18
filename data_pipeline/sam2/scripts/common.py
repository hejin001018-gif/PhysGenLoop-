from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import imageio_ffmpeg
import numpy as np


BASE = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with (BASE / "config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def numbered_files(directory: Path, suffixes: tuple[str, ...]) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir() if path.suffix.lower() in suffixes
    )


def read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr):
        raise OSError(f"Unable to write image: {path}")


def write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8)):
        raise OSError(f"Unable to write mask: {path}")


def encode_rgb_frames(
    frames: Iterable[np.ndarray],
    output: Path,
    fps: int,
    width: int,
    height: int,
    config: dict,
) -> None:
    """Encode all deliverables through one deterministic H.264 path."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    output.parent.mkdir(parents=True, exist_ok=True)
    enc = config["encoding"]
    gop = int(fps * int(enc["gop_seconds"]))
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        enc["video_codec"],
        "-preset",
        enc["preset"],
        "-crf",
        str(enc["crf"]),
        "-g",
        str(gop),
        "-keyint_min",
        str(gop),
        "-sc_threshold",
        "0",
        "-pix_fmt",
        enc["pixel_format"],
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    try:
        for frame in frames:
            if frame.shape != (height, width, 3):
                raise ValueError(
                    f"Frame shape {frame.shape} does not match {(height, width, 3)}"
                )
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
    return_code = process.wait()
    if return_code:
        raise RuntimeError(f"FFmpeg failed with return code {return_code}: {output}")


def iter_rgb_files(files: Iterable[Path]) -> Iterator[np.ndarray]:
    for path in files:
        yield read_rgb(path)


def binary_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    if not len(xs):
        raise ValueError("Cannot compute a bounding box for an empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments((mask > 0).astype(np.uint8))
    if moments["m00"] == 0:
        raise ValueError("Cannot compute a centroid for an empty mask")
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def intersection_over_union(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a > 0
    b_bool = b > 0
    union = np.logical_or(a_bool, b_bool).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a_bool, b_bool).sum() / union)
