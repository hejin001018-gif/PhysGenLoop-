from __future__ import annotations

import os
import subprocess
import sys
import hashlib
import json
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from common import BASE, load_config, numbered_files, write_json, write_mask


def make_repair_masks(sequence: dict, config: dict) -> Path:
    sequence_id = sequence["id"]
    source = BASE / "work" / sequence_id / "masks" / "sam2"
    target = BASE / "work" / sequence_id / "masks" / "repair"
    target.mkdir(parents=True, exist_ok=True)
    radius = int(config["inpainting"]["mask_dilation_pixels"])
    vertical = int(config["inpainting"]["mask_vertical_shadow_extension_pixels"])
    vertical += int(sequence.get("shadow_repair_extension_pixels", 0))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    for index, path in enumerate(numbered_files(source, (".png",))):
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        repair = cv2.dilate((mask > 0).astype(np.uint8) * 255, kernel)
        if vertical > 0:
            shifted = np.zeros_like(repair)
            shifted[vertical:] = repair[:-vertical]
            repair = np.maximum(repair, shifted)
        write_mask(target / f"{index:05d}.png", repair)
    return target


def directory_signature(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in numbered_files(directory, (".png",)):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def run_sequence(sequence: dict, config: dict) -> None:
    sequence_id = sequence["id"]
    source_frames = BASE / "data" / "sources" / sequence_id / "frames"
    repair_masks = make_repair_masks(sequence, config)
    output_root = BASE / "work" / sequence_id / "propainter"
    expected = output_root / source_frames.name / "frames"
    stage_manifest = output_root / "stage_manifest.json"
    source_count = len(numbered_files(source_frames, (".jpg", ".jpeg")))
    signature = directory_signature(repair_masks)
    prior_signature = None
    if stage_manifest.is_file():
        with stage_manifest.open("r", encoding="utf-8") as handle:
            prior_signature = json.load(handle).get("repair_mask_sha256")
    if (
        len(numbered_files(expected, (".png",))) == source_count
        and prior_signature == signature
    ):
        print(f"{sequence_id}: reusing {source_count} existing ProPainter frames")
        return

    repository = BASE / config["inpainting"]["repository"]
    command = [
        sys.executable,
        "inference_propainter.py",
        "--video",
        str(source_frames),
        "--mask",
        str(repair_masks),
        "--output",
        str(output_root),
        "--mask_dilation",
        "0",
        "--ref_stride",
        "10",
        "--neighbor_length",
        "10",
        "--subvideo_length",
        "80",
        "--raft_iter",
        "20",
        "--save_fps",
        str(config["dataset"]["fps"]),
        "--save_frames",
        "--fp16",
    ]
    environment = os.environ.copy()
    cache = BASE / "runtime" / "cache"
    temp = BASE / "runtime" / "tmp"
    cache.mkdir(parents=True, exist_ok=True)
    temp.mkdir(parents=True, exist_ok=True)
    environment.update(
        {
            "HOME": str(BASE / "runtime" / "home"),
            "HF_HOME": str(cache / "huggingface"),
            "TORCH_HOME": str(cache / "torch"),
            "XDG_CACHE_HOME": str(cache),
            "TEMP": str(temp),
            "TMP": str(temp),
            "IMAGEIO_FFMPEG_EXE": imageio_ffmpeg.get_ffmpeg_exe(),
        }
    )
    log_dir = BASE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"propainter_{sequence_id}.stdout.log").open(
        "w", encoding="utf-8"
    ) as stdout, (log_dir / f"propainter_{sequence_id}.stderr.log").open(
        "w", encoding="utf-8"
    ) as stderr:
        subprocess.run(
            command,
            cwd=repository,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    result_count = len(numbered_files(expected, (".png",)))
    if result_count != source_count:
        raise RuntimeError(
            f"ProPainter output count differs for {sequence_id}: "
            f"expected {source_count}, got {result_count}"
        )
    write_json(
        stage_manifest,
        {
            "sequence_id": sequence_id,
            "frame_count": result_count,
            "repair_mask_sha256": signature,
            "mask_dilation_in_propainter": 0,
            "fp16": True,
        },
    )
    print(f"{sequence_id}: generated {result_count} ProPainter background frames")


def main() -> None:
    config = load_config()
    for sequence in config["sequences"]:
        run_sequence(sequence, config)


if __name__ == "__main__":
    main()
