"""Strict V2 ProPainter local editor.

This adapter keeps the legacy :class:`ProPainterLocalEditor` intact and only
overrides the mask-building contract used by V2:

* ``LocalEditTarget.mask_uri`` must point to ``mask_manifest.json``.
* Masks are consumed per frame from the manifest; the first mask is never
  repeated over every critical frame.
* Missing, empty, almost full-frame, size-mismatched, SHA-mismatched, and
  out-of-range masks fail closed before ProPainter is called.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from physgenloop.contracts import GeneratedCandidate
from physgenloop.learning_repair.contracts import LocalEditTarget

from ..local_editor import ProPainterLocalEditor
from .mask_manifest import (
    MAX_COVERAGE_RATIO,
    MIN_NONZERO_RATIO,
    MaskFrame,
    MaskManifest,
    verify_manifest,
)


class StrictProPainterLocalEditor(ProPainterLocalEditor):
    """V2-only strict local editor backed by a mask manifest."""

    def edit(
        self,
        *,
        candidate: GeneratedCandidate,
        target: LocalEditTarget,
        instruction: str,
        critic_report: Any,
        seed: int,
    ) -> GeneratedCandidate:
        edited = super().edit(
            candidate=candidate,
            target=target,
            instruction=instruction,
            critic_report=critic_report,
            seed=seed,
        )
        output_validation = self._validate_output(
            Path(candidate.video_path),
            Path(edited.video_path),
        )
        metadata = {
            **dict(getattr(edited, "metadata", {}) or {}),
            "backend": "propainter-strict-local-edit",
            "mask_manifest_uri": target.mask_uri,
            "strict_mask_manifest": True,
            "editor": "StrictProPainterLocalEditor",
            "editor_backend": "ProPainter",
            "repair_mode": "strict-mask-video-inpainting",
            "propainter": {
                "repo": str(self._repo.resolve()),
                "script": str((self._repo / "inference_propainter.py").resolve()),
                "weights_dir": str((self._repo / "weights").resolve()),
                "python": str(self._python),
            },
            "output_validation": output_validation,
        }
        out_dir = Path(edited.video_path).resolve().parent
        try:
            (out_dir / "metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            # Metadata refresh is useful for audit, but the edited candidate
            # should still be returned if ProPainter itself succeeded.
            pass
        return GeneratedCandidate(
            candidate_id=edited.candidate_id,
            video_path=edited.video_path,
            prompt=edited.prompt,
            seed=edited.seed,
            metadata=metadata,
        )

    def _validate_output(self, source: Path, output: Path) -> dict[str, Any]:
        if not output.exists() or output.stat().st_size <= 0:
            raise RuntimeError(f"ProPainter output missing or empty: {output}")

        def _video_info(path: Path) -> tuple[int, int, int]:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise RuntimeError(f"cannot decode video: {path}")
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            capture.release()
            if frames <= 0 or width <= 0 or height <= 0:
                raise RuntimeError(f"invalid video geometry: {path}")
            return frames, width, height

        source_frames, source_width, source_height = _video_info(source)
        frames, width, height = _video_info(output)
        if frames != source_frames:
            raise RuntimeError(
                f"ProPainter frame count mismatch: {frames}!={source_frames}"
            )
        if (width, height) != (source_width, source_height):
            raise RuntimeError(
                "ProPainter frame size mismatch: "
                f"{width}x{height}!={source_width}x{source_height}"
            )
        return {
            "exists": True,
            "decode_ok": True,
            "frame_count": frames,
            "source_frame_count": source_frames,
            "frame_count_match": True,
            "width": width,
            "height": height,
            "source_width": source_width,
            "source_height": source_height,
            "size_match": True,
            "candidate_prefix_ok": output.parent.name.startswith("propainter-"),
        }

    def _build_masks(
        self,
        masks_dir: Path,
        frame_count: int,
        target: LocalEditTarget,
        video_path: Path,
    ) -> None:
        manifest_path = self._manifest_path(target)
        manifest = self._load_manifest(manifest_path)

        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"cannot read video first frame: {video_path}")
        height, width = frame.shape[:2]

        self._validate_video_shape(manifest, width=width, height=height, frame_count=frame_count)
        ok_manifest, problems = verify_manifest(manifest, check_sha=True)
        if not ok_manifest:
            raise RuntimeError(f"mask manifest invalid: {problems[:12]}")

        frame_masks = self._collect_frame_masks(
            manifest=manifest,
            target=target,
            frame_count=frame_count,
            width=width,
            height=height,
        )
        if not frame_masks:
            raise RuntimeError("strict local editing requires at least one valid per-frame mask")

        empty = np.zeros((height, width), dtype=np.uint8)
        for idx in range(frame_count):
            mask = frame_masks.get(idx, empty)
            cv2.imwrite(str(masks_dir / f"{idx:05d}.png"), mask)

    def _encode_video(self, result_dir: Path, out_video: Path) -> None:
        """Encode/copy ProPainter output robustly for V2 strict runs.

        Upstream ProPainter may save both frame folders and ``inpaint_out.mp4``
        under a nested directory named after the input folder, e.g.
        ``result/frames/inpaint_out.mp4`` and ``result/frames/frames/*.png``.
        The legacy editor only expects PNG frames; V2 accepts the official mp4
        first, then falls back to recursive frame encoding.
        """

        mp4s = sorted(result_dir.rglob("inpaint_out.mp4"))
        if not mp4s:
            mp4s = sorted(p for p in result_dir.rglob("*.mp4") if p.name != out_video.name)
        if mp4s:
            out_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(mp4s[0], out_video)
            return

        frames = sorted(result_dir.rglob("*.png")) + sorted(result_dir.rglob("*.jpg"))
        if not frames:
            raise RuntimeError(f"ProPainter produced neither inpaint_out.mp4 nor frames under: {result_dir}")
        first = cv2.imread(str(frames[0]))
        if first is None:
            raise RuntimeError(f"cannot read first ProPainter output frame: {frames[0]}")
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(
            str(out_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot open VideoWriter for: {out_video}")
        for frame_path in frames:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"cannot read ProPainter output frame: {frame_path}")
            writer.write(frame)
        writer.release()

    def _manifest_path(self, target: LocalEditTarget) -> Path:
        if not target.mask_uri:
            raise RuntimeError("strict local editing requires target.mask_uri=mask_manifest.json")
        path = Path(target.mask_uri).expanduser()
        if not path.exists():
            raise RuntimeError(f"mask manifest missing: {path}")
        if path.suffix.lower() != ".json":
            raise RuntimeError(f"strict local editing expects mask_manifest.json, got: {path}")
        return path

    def _load_manifest(self, path: Path) -> MaskManifest:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return MaskManifest.from_dict(raw)

    def _validate_video_shape(
        self,
        manifest: MaskManifest,
        *,
        width: int,
        height: int,
        frame_count: int,
    ) -> None:
        if manifest.video_width is not None and int(manifest.video_width) != int(width):
            raise RuntimeError(f"mask manifest width mismatch: {manifest.video_width}!={width}")
        if manifest.video_height is not None and int(manifest.video_height) != int(height):
            raise RuntimeError(f"mask manifest height mismatch: {manifest.video_height}!={height}")
        if manifest.video_frames is not None and int(manifest.video_frames) != int(frame_count):
            raise RuntimeError(f"mask manifest frame_count mismatch: {manifest.video_frames}!={frame_count}")

    def _collect_frame_masks(
        self,
        *,
        manifest: MaskManifest,
        target: LocalEditTarget,
        frame_count: int,
        width: int,
        height: int,
    ) -> dict[int, np.ndarray]:
        critical_frames = tuple(int(f) for f in (target.critical_frames or ()))
        if not critical_frames:
            raise RuntimeError("strict local editing requires explicit critical_frames")

        object_names = tuple(target.objects) if target.objects else tuple(obj.name for obj in manifest.objects)
        if not object_names:
            raise RuntimeError("strict local editing requires target objects or manifest objects")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        frame_masks: dict[int, np.ndarray] = {}
        for frame_index in critical_frames:
            if frame_index < 0 or frame_index >= frame_count:
                raise RuntimeError(f"critical frame out of range: {frame_index}/{frame_count}")

            combined = np.zeros((height, width), dtype=np.uint8)
            matched = False
            for object_name in object_names:
                frames = manifest.frames_for(str(object_name))
                mask_frame = frames.get(frame_index)
                if mask_frame is None:
                    continue
                matched = True
                combined = np.maximum(
                    combined,
                    self._read_mask_frame(mask_frame, width=width, height=height),
                )
            if not matched:
                raise RuntimeError(f"no manifest mask for frame {frame_index} and objects={object_names}")

            dilated = cv2.dilate(combined, kernel)
            ratio = float((dilated > 0).sum()) / float(width * height)
            if ratio < MIN_NONZERO_RATIO:
                raise RuntimeError(f"empty dilated local mask at frame {frame_index}")
            if ratio > MAX_COVERAGE_RATIO:
                raise RuntimeError(f"local mask too large after dilation at frame {frame_index}: {ratio:.4f}")
            frame_masks[frame_index] = dilated
        return frame_masks

    def _read_mask_frame(self, mask_frame: MaskFrame, *, width: int, height: int) -> np.ndarray:
        if not mask_frame.valid:
            raise RuntimeError(f"invalid mask frame {mask_frame.frame_index}: {mask_frame.reason}")
        path = Path(mask_frame.path)
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"cannot read mask frame: {path}")
        if mask.shape[:2] != (height, width):
            raise RuntimeError(
                f"mask size mismatch at frame {mask_frame.frame_index}: "
                f"{mask.shape[1]}x{mask.shape[0]} != {width}x{height}"
            )
        ratio = float((mask > 0).sum()) / float(width * height)
        if ratio < MIN_NONZERO_RATIO:
            raise RuntimeError(f"empty mask at frame {mask_frame.frame_index}")
        if ratio > MAX_COVERAGE_RATIO:
            raise RuntimeError(f"mask too large at frame {mask_frame.frame_index}: {ratio:.4f}")
        return mask
