from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

from common import (
    BASE,
    binary_bbox,
    intersection_over_union,
    load_config,
    numbered_files,
    write_json,
    write_mask,
)


def selected_gt_mask(path: Path, object_label: int | None = None) -> tuple[np.ndarray, int]:
    # DAVIS annotations are palette-index PNGs. Pillow preserves object IDs;
    # OpenCV expands the palette to BGR and would turn colors into false IDs.
    with Image.open(path) as image:
        raw = np.asarray(image)
    if raw.ndim != 2:
        raise ValueError(f"Expected a palette-index mask, got shape {raw.shape}: {path}")
    labels, counts = np.unique(raw[raw > 0], return_counts=True)
    if not len(labels):
        raise ValueError(f"No foreground DAVIS label in {path}")
    if object_label is None:
        object_label = int(labels[np.argmax(counts)])
    return (raw == object_label).astype(np.uint8) * 255, object_label


def main() -> None:
    config = load_config()
    sam_cfg = config["sam2"]
    checkpoint = BASE / sam_cfg["checkpoint"]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this SAM2 run")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    predictor = build_sam2_video_predictor(
        sam_cfg["model_config"],
        str(checkpoint),
        device=sam_cfg["device"],
        apply_postprocessing=True,
        vos_optimized=False,
    )

    reports = []
    for sequence in config["sequences"]:
        sequence_id = sequence["id"]
        source = BASE / "data" / "sources" / sequence_id
        frames_dir = source / "frames"
        gt_dir = source / "davis_gt_masks"
        output_dir = BASE / "work" / sequence_id / "masks" / "sam2"
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_files = numbered_files(frames_dir, (".jpg", ".jpeg"))
        gt_files = numbered_files(gt_dir, (".png",))

        first_gt, label = selected_gt_mask(gt_files[0])
        height, width = first_gt.shape
        padding = max(4, round(0.02 * max(width, height)))
        prompt_indices = sorted(set([0, len(frame_files) // 3, 2 * len(frame_files) // 3]))
        prompt_records = []

        start = time.perf_counter()
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
            async_loading_frames=True,
        )
        for prompt_index in prompt_indices:
            prompt_mask, _ = selected_gt_mask(gt_files[prompt_index], label)
            x1, y1, x2, y2 = binary_bbox(prompt_mask)
            box = np.array(
                [
                    max(0, x1 - padding),
                    max(0, y1 - padding),
                    min(width - 1, x2 + padding),
                    min(height - 1, y2 + padding),
                ],
                dtype=np.float32,
            )
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=prompt_index,
                obj_id=1,
                box=box,
            )
            prompt_records.append(
                {"frame_index": prompt_index, "box_xyxy": box.tolist()}
            )

        masks: dict[int, np.ndarray] = {}
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for frame_idx, object_ids, logits in predictor.propagate_in_video(state):
                object_index = list(map(int, object_ids)).index(1)
                mask = (logits[object_index] > float(sam_cfg["mask_threshold"]))
                mask = mask.detach().cpu().numpy().squeeze().astype(np.uint8) * 255
                masks[int(frame_idx)] = mask

        if len(masks) != len(frame_files):
            raise RuntimeError(
                f"SAM2 returned {len(masks)} masks for {len(frame_files)} frames in {sequence_id}"
            )

        ious = []
        areas = []
        for frame_idx, gt_path in enumerate(gt_files):
            predicted = masks[frame_idx]
            gt, _ = selected_gt_mask(gt_path, label)
            write_mask(output_dir / f"{frame_idx:05d}.png", predicted)
            ious.append(intersection_over_union(predicted, gt))
            areas.append(int((predicted > 0).sum()))

        report = {
            "sequence_id": sequence_id,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "davis_object_label": label,
            "prompt_type": "fixed_sparse_bounding_boxes",
            "prompt_schedule": "0%, 33%, 67%; fixed before inference and not selected from SAM2 errors",
            "prompts": prompt_records,
            "frame_count": len(frame_files),
            "elapsed_seconds": time.perf_counter() - start,
            "mean_iou_vs_davis_gt": float(np.mean(ious)),
            "minimum_iou_vs_davis_gt": float(np.min(ious)),
            "median_iou_vs_davis_gt": float(np.median(ious)),
            "mean_iou_vs_davis_gt_non_prompt": float(
                np.mean([value for index, value in enumerate(ious) if index not in prompt_indices])
            ),
            "minimum_iou_vs_davis_gt_non_prompt": float(
                np.min([value for index, value in enumerate(ious) if index not in prompt_indices])
            ),
            "per_frame_iou": ious,
            "per_frame_mask_area_pixels": areas,
            "sam2_checkpoint": sam_cfg["checkpoint"],
            "sam2_model_config": sam_cfg["model_config"],
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
        }
        write_json(BASE / "reports" / sequence_id / "sam2_tracking.json", report)
        reports.append(report)
        predictor.reset_state(state)
        print(
            f"{sequence_id}: {len(frame_files)} frames, "
            f"mean IoU={report['mean_iou_vs_davis_gt']:.4f}"
        )

    write_json(BASE / "reports" / "sam2_tracking_summary.json", reports)


if __name__ == "__main__":
    main()
