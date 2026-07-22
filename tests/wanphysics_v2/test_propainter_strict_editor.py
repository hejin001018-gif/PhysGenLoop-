"""Strict ProPainter V2 editor mask handling."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from physgenloop.learning_repair.contracts import LocalEditTarget
from generators.wanphysics.v2.mask_manifest import MaskFrame, MaskManifest, MaskObject
from generators.wanphysics.v2.propainter_strict_editor import StrictProPainterLocalEditor


cv2 = pytest.importorskip("cv2")


def _write_video(path, *, width=64, height=48, frames=4):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 4, (width, height))
    if not writer.isOpened():
        pytest.skip("cv2 VideoWriter mp4v is unavailable")
    for idx in range(frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 1] = idx * 20
        writer.write(frame)
    writer.release()


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_strict_editor_uses_manifest_per_frame(tmp_path):
    video = tmp_path / "source.mp4"
    _write_video(video)

    mask1 = np.zeros((48, 64), dtype=np.uint8)
    mask1[10:16, 10:16] = 255
    mask3 = np.zeros((48, 64), dtype=np.uint8)
    mask3[20:26, 20:26] = 255
    mask1_path = tmp_path / "ball_00001.png"
    mask3_path = tmp_path / "ball_00003.png"
    cv2.imwrite(str(mask1_path), mask1)
    cv2.imwrite(str(mask3_path), mask3)

    manifest = MaskManifest(
        candidate_id="c1",
        video=str(video),
        video_width=64,
        video_height=48,
        video_frames=4,
        objects=(
            MaskObject(
                name="ball",
                normalized_name="ball",
                frames=(
                    MaskFrame(frame_index=1, path=str(mask1_path), sha256=_sha(mask1_path), valid=True, width=64, height=48, nonzero_ratio=float((mask1 > 0).sum()) / float(64 * 48)),
                    MaskFrame(frame_index=3, path=str(mask3_path), sha256=_sha(mask3_path), valid=True, width=64, height=48, nonzero_ratio=float((mask3 > 0).sum()) / float(64 * 48)),
                ),
            ),
        ),
    )
    manifest_path = tmp_path / "mask_manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict(), ensure_ascii=False), encoding="utf-8")

    out_masks = tmp_path / "out_masks"
    out_masks.mkdir()
    target = LocalEditTarget(
        parent_candidate_id="c1",
        objects=("ball",),
        critical_frames=(1, 3),
        mask_uri=str(manifest_path),
    )
    StrictProPainterLocalEditor(propainter_repo=str(tmp_path))._build_masks(out_masks, 4, target, video)

    assert cv2.imread(str(out_masks / "00000.png"), cv2.IMREAD_GRAYSCALE).sum() == 0
    assert cv2.imread(str(out_masks / "00001.png"), cv2.IMREAD_GRAYSCALE).sum() > 0
    assert cv2.imread(str(out_masks / "00002.png"), cv2.IMREAD_GRAYSCALE).sum() == 0
    assert cv2.imread(str(out_masks / "00003.png"), cv2.IMREAD_GRAYSCALE).sum() > 0


def test_strict_editor_rejects_single_png_mask_uri(tmp_path):
    video = tmp_path / "source.mp4"
    _write_video(video)
    mask = tmp_path / "ball_00001.png"
    cv2.imwrite(str(mask), np.ones((48, 64), dtype=np.uint8) * 255)
    out_masks = tmp_path / "out_masks"
    out_masks.mkdir()
    target = LocalEditTarget(parent_candidate_id="c1", objects=("ball",), critical_frames=(1,), mask_uri=str(mask))

    with pytest.raises(RuntimeError, match="mask_manifest.json"):
        StrictProPainterLocalEditor(propainter_repo=str(tmp_path))._build_masks(out_masks, 4, target, video)


def test_strict_encoder_accepts_propainter_mp4(tmp_path):
    result_dir = tmp_path / "result"
    nested = result_dir / "frames"
    nested.mkdir(parents=True)
    source_mp4 = nested / "inpaint_out.mp4"
    source_mp4.write_bytes(b"fake-mp4")
    out_video = tmp_path / "out.mp4"

    StrictProPainterLocalEditor(propainter_repo=str(tmp_path))._encode_video(result_dir, out_video)

    assert out_video.read_bytes() == b"fake-mp4"
