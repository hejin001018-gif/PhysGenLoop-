from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common import (
    BASE,
    load_config,
    numbered_files,
    read_rgb,
    sha256_file,
    write_json,
    write_rgb,
)


def probe_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {path}")
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    result = {
        "path": str(path.relative_to(BASE)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fourcc": "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)),
    }
    capture.release()
    return result


def make_contact_sheet(sequence_id: str, onset: int) -> Path:
    directory = BASE / "work" / sequence_id / "composites" / "comparison_frames"
    files = numbered_files(directory, (".png",))
    indices = sorted(set([max(0, onset - 1), onset, (onset + len(files) - 1) // 2, len(files) - 1]))
    thumbnails = []
    for index in indices:
        frame = read_rgb(files[index])
        target_width = 960
        scale = target_width / frame.shape[1]
        thumbnails.append(
            cv2.resize(frame, (target_width, round(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        )
    sheet = np.vstack(thumbnails)
    target = BASE / "reports" / sequence_id / "contact_sheet.png"
    write_rgb(target, sheet)
    return target


def audit_sequence(sequence: dict, config: dict) -> dict:
    sequence_id = sequence["id"]
    output = BASE / "outputs" / sequence_id
    with (output / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    onset = int(metadata["anomaly_onset_frame"])
    work = BASE / "work" / sequence_id / "composites"
    original_files = numbered_files(work / "original_frames", (".png",))
    sham_files = numbered_files(work / "sham_frames", (".png",))
    anomaly_files = numbered_files(work / "anomaly_frames", (".png",))
    mask_root = BASE / "work" / sequence_id / "masks"
    repair_files = numbered_files(mask_root / "repair", (".png",))
    anomaly_mask_files = numbered_files(mask_root / "anomaly", (".png",))
    anomaly_shadow_files = numbered_files(mask_root / "shadow_anomaly", (".png",))
    frame_count = len(original_files)

    psnr_values = []
    ssim_values = []
    prefix_max_error = 0
    post_difference = []
    outside_support_max_error = 0
    outside_support_mean_errors = []
    for index, (
        original_path,
        sham_path,
        anomaly_path,
        repair_path,
        anomaly_mask_path,
        anomaly_shadow_path,
    ) in enumerate(
        zip(
            original_files,
            sham_files,
            anomaly_files,
            repair_files,
            anomaly_mask_files,
            anomaly_shadow_files,
            strict=True,
        )
    ):
        original = read_rgb(original_path)
        sham = read_rgb(sham_path)
        anomaly = read_rgb(anomaly_path)
        psnr_values.append(peak_signal_noise_ratio(original, sham, data_range=255))
        ssim_values.append(
            structural_similarity(original, sham, channel_axis=2, data_range=255)
        )
        if index < onset:
            prefix_max_error = max(
                prefix_max_error,
                int(np.abs(sham.astype(np.int16) - anomaly.astype(np.int16)).max()),
            )
        else:
            post_difference.append(float(np.abs(sham.astype(np.float32) - anomaly).mean()))
        repair = cv2.imread(str(repair_path), cv2.IMREAD_GRAYSCALE)
        anomaly_mask = cv2.imread(str(anomaly_mask_path), cv2.IMREAD_GRAYSCALE)
        anomaly_shadow = cv2.imread(str(anomaly_shadow_path), cv2.IMREAD_GRAYSCALE)
        if repair is None or anomaly_mask is None or anomaly_shadow is None:
            raise FileNotFoundError(f"Missing edit-support mask for {sequence_id} frame {index}")
        support = (repair > 0) | (anomaly_mask > 0) | (anomaly_shadow > 0)
        outside = ~support
        outside_difference = np.abs(
            original.astype(np.int16) - anomaly.astype(np.int16)
        )[outside]
        if outside_difference.size:
            outside_support_max_error = max(
                outside_support_max_error, int(outside_difference.max())
            )
            outside_support_mean_errors.append(float(outside_difference.mean()))

    video_probes = {
        kind: probe_video(output / f"{kind}.mp4")
        for kind in ("original", "sham", "anomaly", "comparison")
    }
    individual_specs = {
        (
            video_probes[kind]["frame_count"],
            round(video_probes[kind]["fps"], 3),
            video_probes[kind]["width"],
            video_probes[kind]["height"],
            video_probes[kind]["fourcc"],
        )
        for kind in ("original", "sham", "anomaly")
    }

    tracking_path = BASE / "reports" / sequence_id / "sam2_tracking.json"
    with tracking_path.open("r", encoding="utf-8") as handle:
        tracking = json.load(handle)
    transforms = metadata["transform_ground_truth"]
    active_displacements = [
        math.hypot(item["dx_pixels"], item["dy_pixels"])
        for item in transforms
        if item["active"]
    ]
    diagonal = math.hypot(metadata["width"], metadata["height"])
    metrics = {
        "mean_sam2_iou_vs_davis_gt": tracking["mean_iou_vs_davis_gt"],
        "minimum_sam2_iou_vs_davis_gt": tracking["minimum_iou_vs_davis_gt"],
        "mean_sam2_iou_vs_davis_gt_non_prompt": tracking[
            "mean_iou_vs_davis_gt_non_prompt"
        ],
        "mean_sham_psnr_db": float(np.mean(psnr_values)),
        "minimum_sham_psnr_db": float(np.min(psnr_values)),
        "mean_sham_ssim": float(np.mean(ssim_values)),
        "minimum_sham_ssim": float(np.min(ssim_values)),
        "pre_onset_sham_anomaly_max_pixel_error": prefix_max_error,
        "mean_post_onset_sham_anomaly_abs_difference": float(np.mean(post_difference)),
        "maximum_active_displacement_pixels": float(np.max(active_displacements)),
        "maximum_active_displacement_fraction_of_diagonal": float(
            np.max(active_displacements) / diagonal
        ),
        "outside_declared_edit_support_max_pixel_error": outside_support_max_error,
        "outside_declared_edit_support_mean_abs_error": float(
            np.mean(outside_support_mean_errors)
        ),
    }
    checks = {
        "all_intermediate_frame_counts_match": len(
            {len(original_files), len(sham_files), len(anomaly_files)}
        )
        == 1,
        "individual_video_specs_match": len(individual_specs) == 1,
        "encoded_frame_count_matches_intermediate": all(
            video_probes[kind]["frame_count"] == frame_count
            for kind in ("original", "sham", "anomaly", "comparison")
        ),
        "pre_onset_pixel_identity": prefix_max_error == 0,
        "sam2_non_prompt_mean_iou_at_least_0_75": metrics[
            "mean_sam2_iou_vs_davis_gt_non_prompt"
        ]
        >= 0.75,
        "sham_mean_ssim_at_least_0_95": metrics["mean_sham_ssim"] >= 0.95,
        "sham_minimum_ssim_at_least_0_95": metrics["minimum_sham_ssim"] >= 0.95,
        "sham_mean_psnr_at_least_30_db": metrics["mean_sham_psnr_db"] >= 30.0,
        "post_onset_visual_change_nonzero": metrics[
            "mean_post_onset_sham_anomaly_abs_difference"
        ]
        > 0.25,
        "anomaly_displacement_at_least_3_percent_diagonal": metrics[
            "maximum_active_displacement_fraction_of_diagonal"
        ]
        >= 0.03,
        "pixels_outside_declared_edit_support_unchanged": metrics[
            "outside_declared_edit_support_max_pixel_error"
        ]
        <= 1,
    }
    sheet = make_contact_sheet(sequence_id, onset)
    return {
        "sequence_id": sequence_id,
        "anomaly": sequence["anomaly"],
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "video_probes": video_probes,
        "contact_sheet": str(sheet.relative_to(BASE)),
    }


def main() -> None:
    config = load_config()
    reports = [audit_sequence(sequence, config) for sequence in config["sequences"]]
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strict_audit_passed": all(report["passed"] for report in reports),
        "reports": reports,
        "methodological_note": (
            "A passing synthetic audit does not replace evaluation on an independently "
            "collected real-anomaly test set. Source-video grouping is mandatory for splits."
        ),
    }
    write_json(BASE / "reports" / "quality_audit.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["strict_audit_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
