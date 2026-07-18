from __future__ import annotations

import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from common import (
    BASE,
    binary_bbox,
    encode_rgb_frames,
    iter_rgb_files,
    load_config,
    mask_centroid,
    numbered_files,
    read_rgb,
    write_json,
    write_mask,
    write_rgb,
)


BAR_HEIGHT = 44


def feather(mask: np.ndarray, sigma: float = 1.15) -> np.ndarray:
    alpha = (mask > 0).astype(np.float32)
    return np.clip(cv2.GaussianBlur(alpha, (0, 0), sigma), 0.0, 1.0)


def ground_shadow_alpha(mask: np.ndarray, opacity: float) -> np.ndarray:
    """Build a soft, geometry-derived contact shadow without copying source texture."""
    height, width = mask.shape
    x1, y1, x2, y2 = binary_bbox(mask)
    object_width = max(2, x2 - x1 + 1)
    object_height = max(2, y2 - y1 + 1)
    center = (
        int(round((x1 + x2) / 2)),
        min(height - 1, int(round(y2 + max(2, object_height * 0.035)))),
    )
    axes = (
        max(3, int(round(object_width * 0.43))),
        max(2, int(round(object_height * 0.085))),
    )
    shadow = np.zeros((height, width), dtype=np.float32)
    cv2.ellipse(shadow, center, axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    sigma = max(2.0, object_width * 0.022)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigma)
    maximum = float(shadow.max())
    if maximum > 0:
        shadow *= float(opacity) / maximum
    return np.clip(shadow, 0.0, float(opacity))


def warp(image: np.ndarray, dx: float, dy: float, interpolation: int) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = np.float32([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def clip_displacement(mask: np.ndarray, dx: float, dy: float, margin: int = 3) -> tuple[float, float]:
    x1, y1, x2, y2 = binary_bbox(mask)
    height, width = mask.shape
    dx = float(np.clip(dx, margin - x1, width - 1 - margin - x2))
    dy = float(np.clip(dy, margin - y1, height - 1 - margin - y2))
    return dx, dy


def calculate_transforms(
    sequence: dict, masks: list[np.ndarray], onset: int
) -> list[tuple[float, float]]:
    count = len(masks)
    height, width = masks[0].shape
    anomaly = sequence["anomaly"]
    transforms: list[tuple[float, float]] = []

    if anomaly == "midair_hover":
        anchor_x, anchor_y = mask_centroid(masks[onset])
        for index, mask in enumerate(masks):
            if index < onset:
                transforms.append((0.0, 0.0))
                continue
            center_x, center_y = mask_centroid(mask)
            transforms.append(
                clip_displacement(mask, anchor_x - center_x, anchor_y - center_y)
            )
    elif anomaly == "instant_teleport":
        onset_box = binary_bbox(masks[onset])
        room_right = width - 1 - onset_box[2]
        room_left = onset_box[0]
        magnitude = abs(float(sequence["dx_fraction"]) * width)
        direction = 1.0 if room_right >= room_left else -1.0
        wanted_dx = direction * magnitude
        wanted_dy = float(sequence["dy_fraction"]) * height
        for index, mask in enumerate(masks):
            transforms.append(
                (0.0, 0.0)
                if index < onset
                else clip_displacement(mask, wanted_dx, wanted_dy)
            )
    elif anomaly == "gravity_reversal":
        denominator = max(1, count - onset - 1)
        for index, mask in enumerate(masks):
            if index < onset:
                transforms.append((0.0, 0.0))
                continue
            progress = (index - onset) / denominator
            upward = -float(sequence["upward_acceleration_fraction"]) * height * progress**2
            transforms.append(clip_displacement(mask, 0.0, upward))
    else:
        raise ValueError(f"Unknown anomaly: {anomaly}")
    return transforms


def composite(background: np.ndarray, foreground: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha_3d = alpha[..., None].astype(np.float32)
    value = background.astype(np.float32) * (1.0 - alpha_3d)
    value += foreground.astype(np.float32) * alpha_3d
    return np.clip(np.rint(value), 0, 255).astype(np.uint8)


def comparison_frame(
    original: np.ndarray,
    sham: np.ndarray,
    anomaly: np.ndarray,
    frame_index: int,
    onset: int,
    anomaly_name: str,
) -> np.ndarray:
    height, width = original.shape[:2]
    canvas = np.zeros((height + BAR_HEIGHT, width * 3, 3), dtype=np.uint8)
    canvas[BAR_HEIGHT:, :width] = original
    canvas[BAR_HEIGHT:, width : width * 2] = sham
    canvas[BAR_HEIGHT:, width * 2 :] = anomaly
    canvas[:BAR_HEIGHT] = (22, 25, 32)
    labels = ["ORIGINAL", "SHAM-EDIT (NORMAL)", f"ANOMALY: {anomaly_name.upper()}"]
    for panel, label in enumerate(labels):
        cv2.putText(
            canvas,
            label,
            (panel * width + 14, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (235, 239, 245),
            1,
            cv2.LINE_AA,
        )
    status = "PRE-ONSET" if frame_index < onset else f"ACTIVE  f={frame_index}"
    status_color = (255, 210, 70) if frame_index < onset else (255, 74, 74)
    status_size = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)[0]
    cv2.putText(
        canvas,
        status,
        (width * 3 - status_size[0] - 14, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        status_color,
        1,
        cv2.LINE_AA,
    )
    if frame_index >= onset:
        cv2.rectangle(
            canvas,
            (width * 2 + 2, BAR_HEIGHT + 2),
            (width * 3 - 3, height + BAR_HEIGHT - 3),
            (255, 65, 65),
            3,
        )
    return canvas


def process_sequence(sequence: dict, config: dict) -> None:
    sequence_id = sequence["id"]
    source_dir = BASE / "data" / "sources" / sequence_id / "frames"
    sam_dir = BASE / "work" / sequence_id / "masks" / "sam2"
    repair_dir = BASE / "work" / sequence_id / "masks" / "repair"
    background_dir = BASE / "work" / sequence_id / "propainter" / source_dir.name / "frames"
    frame_files = numbered_files(source_dir, (".jpg", ".jpeg"))
    mask_files = numbered_files(sam_dir, (".png",))
    repair_files = numbered_files(repair_dir, (".png",))
    background_files = numbered_files(background_dir, (".png",))
    counts = {len(frame_files), len(mask_files), len(repair_files), len(background_files)}
    if len(counts) != 1:
        raise RuntimeError(
            f"Input count mismatch for {sequence_id}: frames={len(frame_files)}, "
            f"SAM2={len(mask_files)}, repair={len(repair_files)}, backgrounds={len(background_files)}"
        )

    masks = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in mask_files]
    if any(mask is None for mask in masks):
        raise FileNotFoundError(f"Unreadable SAM2 mask in {sam_dir}")
    onset = int(round((len(frame_files) - 1) * float(sequence["onset_fraction"])))
    transforms = calculate_transforms(sequence, masks, onset)

    work_root = BASE / "work" / sequence_id / "composites"
    directories = {
        "original": work_root / "original_frames",
        "sham": work_root / "sham_frames",
        "anomaly": work_root / "anomaly_frames",
        "comparison": work_root / "comparison_frames",
        "alpha": BASE / "work" / sequence_id / "masks" / "alpha",
        "anomaly_mask": BASE / "work" / sequence_id / "masks" / "anomaly",
        "source_shadow": BASE / "work" / sequence_id / "masks" / "shadow_source",
        "anomaly_shadow": BASE / "work" / sequence_id / "masks" / "shadow_anomaly",
        "sham_support": BASE / "work" / sequence_id / "masks" / "sham_support",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    transform_records = []
    for index, (frame_path, background_path, repair_path, mask, (dx, dy)) in enumerate(
        zip(frame_files, background_files, repair_files, masks, transforms, strict=True)
    ):
        original = read_rgb(frame_path)
        background = read_rgb(background_path)
        repair_mask = cv2.imread(str(repair_path), cv2.IMREAD_GRAYSCALE)
        if repair_mask is None:
            raise FileNotFoundError(repair_path)
        if original.shape != background.shape:
            background = cv2.resize(
                background, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_CUBIC
            )
        # ProPainter processes dimensions divisible by 8 (854 -> 848 -> 854 here),
        # which otherwise resamples the entire frame. Restrict its contribution to
        # the declared repair region and preserve all unrelated source pixels exactly.
        background[repair_mask == 0] = original[repair_mask == 0]
        alpha = feather(mask)
        use_shadow = sequence.get("shadow_mode") == "synthetic_ground"
        if use_shadow:
            # Restore the real source shadow in the Sham control. The wide soft support
            # also restores clean road pixels, so it does not invent a dark ellipse.
            source_shadow_support = ground_shadow_alpha(mask, 0.92)
            sham_alpha = np.maximum(alpha, source_shadow_support)
        else:
            source_shadow_support = np.zeros_like(alpha)
        # Retain a controlled fraction of source content across the inpainting support.
        # This improves a normal Sham reconstruction while leaving most fill artifacts
        # visible for shortcut diagnostics.
        repair_support = feather(repair_mask, sigma=3.0) * 0.30
        sham_alpha = np.maximum.reduce([alpha, source_shadow_support, repair_support])
        sham = composite(background, original, sham_alpha)

        if index < onset:
            abnormal = sham.copy()
            abnormal_alpha = alpha
            anomaly_shadow = source_shadow_support
        else:
            moved_foreground = warp(original, dx, dy, cv2.INTER_CUBIC)
            abnormal_alpha = warp(alpha, dx, dy, cv2.INTER_LINEAR)
            shadowed_background = background
            if use_shadow:
                opacity = float(sequence.get("shadow_opacity", 0.28))
                if sequence["anomaly"] == "gravity_reversal":
                    progress = (index - onset) / max(1, len(frame_files) - onset - 1)
                    opacity *= 1.0 - 0.70 * progress
                    anomaly_shadow = ground_shadow_alpha(mask, opacity)
                else:
                    moved_binary = warp(mask, dx, dy, cv2.INTER_NEAREST)
                    anomaly_shadow = ground_shadow_alpha(moved_binary, opacity)
                shadowed_background = np.clip(
                    background.astype(np.float32)
                    * (1.0 - anomaly_shadow[..., None]),
                    0,
                    255,
                ).astype(np.uint8)
            else:
                anomaly_shadow = np.zeros_like(alpha)
            abnormal = composite(shadowed_background, moved_foreground, abnormal_alpha)

        compare = comparison_frame(
            original, sham, abnormal, index, onset, sequence["anomaly"]
        )
        name = f"{index:05d}.png"
        write_rgb(directories["original"] / name, original)
        write_rgb(directories["sham"] / name, sham)
        write_rgb(directories["anomaly"] / name, abnormal)
        write_rgb(directories["comparison"] / name, compare)
        write_mask(directories["alpha"] / name, np.rint(alpha * 255))
        write_mask(
            directories["anomaly_mask"] / name,
            np.rint(np.clip(abnormal_alpha, 0.0, 1.0) * 255),
        )
        write_mask(
            directories["source_shadow"] / name,
            np.rint(np.clip(source_shadow_support, 0.0, 1.0) * 255),
        )
        write_mask(
            directories["anomaly_shadow"] / name,
            np.rint(np.clip(anomaly_shadow, 0.0, 1.0) * 255),
        )
        write_mask(
            directories["sham_support"] / name,
            np.rint(np.clip(sham_alpha, 0.0, 1.0) * 255),
        )
        transform_records.append(
            {
                "frame_index": index,
                "time_seconds": index / float(config["dataset"]["fps"]),
                "active": index >= onset,
                "dx_pixels": dx,
                "dy_pixels": dy,
            }
        )

    output_root = BASE / "outputs" / sequence_id
    output_root.mkdir(parents=True, exist_ok=True)
    height, width = read_rgb(frame_files[0]).shape[:2]
    fps = int(config["dataset"]["fps"])
    encoded = {}
    for kind in ("original", "sham", "anomaly"):
        files = numbered_files(directories[kind], (".png",))
        target = output_root / f"{kind}.mp4"
        encode_rgb_frames(iter_rgb_files(files), target, fps, width, height, config)
        encoded[kind] = str(target.relative_to(BASE))
    comparison_files = numbered_files(directories["comparison"], (".png",))
    comparison_target = output_root / "comparison.mp4"
    encode_rgb_frames(
        iter_rgb_files(comparison_files),
        comparison_target,
        fps,
        width * 3,
        height + BAR_HEIGHT,
        config,
    )
    encoded["comparison"] = str(comparison_target.relative_to(BASE))
    top_level = BASE / "outputs" / f"{sequence['order']:02d}_{sequence_id}_comparison.mp4"
    shutil.copy2(comparison_target, top_level)

    metadata = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "video_id": f"davis-{sequence_id}-{sequence['anomaly']}",
        "source_video_id": f"davis-2017-{sequence_id}",
        "split_group": f"davis-2017-{sequence_id}",
        "recommended_role": "synthetic training/development sample; not a standalone real-world test sample",
        "sequence": sequence,
        "fps": fps,
        "frame_count": len(frame_files),
        "width": width,
        "height": height,
        "anomaly_onset_frame": onset,
        "anomaly_onset_seconds": onset / fps,
        "negative_control": "Sham-edit uses the same SAM2, ProPainter, alpha-compositing and encoding path without anomalous displacement.",
        "pre_onset_invariant": "anomaly frames are pixel-identical to sham frames before onset",
        "transform_ground_truth": transform_records,
        "paths": {
            "source_frames": str(source_dir.relative_to(BASE)),
            "sam2_masks": str(sam_dir.relative_to(BASE)),
            "repair_masks": str(repair_dir.relative_to(BASE)),
            "background_frames": str(background_dir.relative_to(BASE)),
            "anomaly_masks": str(directories["anomaly_mask"].relative_to(BASE)),
            "source_shadow_masks": str(directories["source_shadow"].relative_to(BASE)),
            "anomaly_shadow_masks": str(directories["anomaly_shadow"].relative_to(BASE)),
            "sham_support_masks": str(directories["sham_support"].relative_to(BASE)),
            "videos": encoded,
            "comparison_copy": str(top_level.relative_to(BASE)),
        },
    }
    write_json(output_root / "metadata.json", metadata)
    print(
        f"{sequence_id}: onset={onset}/{len(frame_files)}, "
        f"comparison={comparison_target.relative_to(BASE)}"
    )


def main() -> None:
    config = load_config()
    for sequence in config["sequences"]:
        process_sequence(sequence, config)


if __name__ == "__main__":
    main()
