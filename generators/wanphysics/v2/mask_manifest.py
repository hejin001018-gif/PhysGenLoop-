"""Mask manifest 构建与校验（V2）。

修复 P0-2 的落盘侧：把 SAM2 已经产出的 ``sam2_masks/{object}_{frame:05d}.png``
汇总成一个可校验、可追溯的 ``mask_manifest.json``，并据此构造 Local Editing 需要
的 :class:`LocalEditTarget`。

严格拒绝以下无效输入（对齐修复方案 §13），从源头杜绝 "全白 mask 伪装局部修复"：
  - mask 文件不存在 / 无法读取；
  - mask 尺寸与视频不一致；
  - critical frame 越界；
  - object name 无法匹配；
  - mask 覆盖超过整帧 95%（几乎全帧，等价于全局）；
  - mask 非零像素低于 0.01%（空 mask）；
  - manifest SHA 校验失败。

cv2 为可选依赖：缺失时仅跳过像素级校验（尺寸/nonzero），其余结构化校验照常，
以便在无 GPU / 无 OpenCV 的 CPU 环境跑逻辑测试。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from physgenloop.learning_repair.contracts import LocalEditTarget

MASK_MANIFEST_SCHEMA_VERSION = "mask-manifest/1.0"

# 与 sam2_detector.materialize_masks 的落盘命名保持一致：{object}_{frame:05d}.png
_MASK_FILENAME_RE = re.compile(r"^(?P<object>.+)_(?P<frame>\d{5})\.png$")

# 覆盖率边界：超过视为全局、过低视为空 mask。
MAX_COVERAGE_RATIO = 0.95
MIN_NONZERO_RATIO = 0.0001


def normalize_object_name(name: str) -> str:
    """object name 标准化：小写、去首尾空白、内部空白折叠为单空格。"""

    return re.sub(r"\s+", " ", str(name).strip().lower())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_cv2():
    try:
        import cv2  # noqa: PLC0415

        return cv2
    except Exception:  # noqa: BLE001
        return None


@dataclass(frozen=True)
class MaskFrame:
    frame_index: int
    path: str
    sha256: str | None = None
    nonzero_ratio: float | None = None
    width: int | None = None
    height: int | None = None
    valid: bool = True
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "path": self.path,
            "sha256": self.sha256,
            "nonzero_ratio": self.nonzero_ratio,
            "width": self.width,
            "height": self.height,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MaskObject:
    name: str
    normalized_name: str
    frames: tuple[MaskFrame, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "normalized_name": self.normalized_name,
            "frames": [frame.to_dict() for frame in self.frames],
        }


@dataclass(frozen=True)
class MaskManifest:
    candidate_id: str
    video: str
    objects: tuple[MaskObject, ...]
    video_width: int | None = None
    video_height: int | None = None
    video_frames: int | None = None
    source: str = "sam2"
    postprocess_enabled: bool = False
    status: str = "ok"  # "ok" | "empty" | "invalid"
    schema_version: str = MASK_MANIFEST_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "candidate_id": self.candidate_id,
            "video": self.video,
            "video_width": self.video_width,
            "video_height": self.video_height,
            "video_frames": self.video_frames,
            "source": self.source,
            "postprocess_enabled": self.postprocess_enabled,
            "status": self.status,
            "objects": [obj.to_dict() for obj in self.objects],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MaskManifest":
        objects = []
        for obj in raw.get("objects", ()):
            frames = tuple(
                MaskFrame(
                    frame_index=int(fr["frame_index"]),
                    path=str(fr["path"]),
                    sha256=fr.get("sha256"),
                    nonzero_ratio=fr.get("nonzero_ratio"),
                    width=fr.get("width"),
                    height=fr.get("height"),
                    valid=bool(fr.get("valid", True)),
                    reason=fr.get("reason"),
                )
                for fr in obj.get("frames", ())
            )
            objects.append(
                MaskObject(
                    name=str(obj["name"]),
                    normalized_name=str(obj.get("normalized_name", normalize_object_name(obj["name"]))),
                    frames=frames,
                )
            )
        return cls(
            schema_version=str(raw.get("schema_version", MASK_MANIFEST_SCHEMA_VERSION)),
            candidate_id=str(raw["candidate_id"]),
            video=str(raw["video"]),
            video_width=raw.get("video_width"),
            video_height=raw.get("video_height"),
            video_frames=raw.get("video_frames"),
            source=str(raw.get("source", "sam2")),
            postprocess_enabled=bool(raw.get("postprocess_enabled", False)),
            status=str(raw.get("status", "ok")),
            objects=tuple(objects),
        )

    def frames_for(self, object_name: str) -> dict[int, MaskFrame]:
        target = normalize_object_name(object_name)
        for obj in self.objects:
            if obj.normalized_name == target:
                return {fr.frame_index: fr for fr in obj.frames}
        return {}

    def valid_frame_paths(self, object_name: str) -> dict[int, str]:
        return {
            idx: fr.path for idx, fr in self.frames_for(object_name).items() if fr.valid
        }


def _measure_mask(cv2, path: Path) -> tuple[int | None, int | None, float | None, str | None]:
    """返回 (width, height, nonzero_ratio, reason)；cv2 缺失时返回结构占位。"""

    if cv2 is None:
        return None, None, None, None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None, None, None, "unreadable"
    h, w = mask.shape[:2]
    total = float(h * w) if h and w else 0.0
    nonzero = float((mask > 0).sum())
    ratio = (nonzero / total) if total > 0 else 0.0
    reason: str | None = None
    if ratio < MIN_NONZERO_RATIO:
        reason = "empty_mask"
    elif ratio > MAX_COVERAGE_RATIO:
        reason = "coverage_too_large"
    return w, h, ratio, reason


def build_manifest(
    *,
    candidate_id: str,
    video_path: str,
    mask_dir: str | Path,
    violations: Sequence[Any],
    video_width: int | None = None,
    video_height: int | None = None,
    video_frames: int | None = None,
    postprocess_enabled: bool = False,
    compute_sha: bool = True,
) -> MaskManifest:
    """扫描 ``mask_dir`` 下与 violation 对应的 mask，产出经校验的 manifest。

    只收录 violation.critical_frames 命中的 (object, frame)，与 SAM2 的
    "只为 critical_frames 落盘" 策略保持一致（不为全部帧生成，控制开销）。
    """

    cv2 = _load_cv2()
    mask_dir = Path(mask_dir)
    objects: list[MaskObject] = []
    any_valid = False
    any_invalid = False

    # 聚合 violation 期望的 (object -> frames)。
    wanted: dict[str, set[int]] = {}
    display_name: dict[str, str] = {}
    for violation in violations:
        obj_raw = str(getattr(violation, "object", "")).strip()
        norm = normalize_object_name(obj_raw)
        display_name.setdefault(norm, obj_raw)
        for frame in getattr(violation, "critical_frames", ()) or ():
            try:
                wanted.setdefault(norm, set()).add(int(frame))
            except (TypeError, ValueError):
                continue

    for norm, frames in wanted.items():
        disp = display_name.get(norm, norm)
        mask_frames: list[MaskFrame] = []
        for frame in sorted(frames):
            filename = f"{disp or 'object'}_{int(frame):05d}.png"
            path = mask_dir / filename
            if not path.exists():
                any_invalid = True
                mask_frames.append(
                    MaskFrame(frame_index=frame, path=str(path), valid=False, reason="missing")
                )
                continue
            if video_frames is not None and (frame < 0 or frame >= video_frames):
                any_invalid = True
                mask_frames.append(
                    MaskFrame(frame_index=frame, path=str(path), valid=False, reason="frame_out_of_range")
                )
                continue
            w, h, ratio, reason = _measure_mask(cv2, path)
            size_bad = (
                video_width is not None
                and video_height is not None
                and w is not None
                and h is not None
                and (w != video_width or h != video_height)
            )
            if size_bad:
                reason = "size_mismatch"
            valid = reason is None
            any_valid = any_valid or valid
            any_invalid = any_invalid or not valid
            mask_frames.append(
                MaskFrame(
                    frame_index=frame,
                    path=str(path),
                    sha256=_sha256_file(path) if compute_sha else None,
                    nonzero_ratio=ratio,
                    width=w,
                    height=h,
                    valid=valid,
                    reason=reason,
                )
            )
        objects.append(MaskObject(name=disp, normalized_name=norm, frames=tuple(mask_frames)))

    status = "invalid" if any_invalid else ("ok" if any_valid else ("empty" if not objects else "invalid"))
    return MaskManifest(
        candidate_id=candidate_id,
        video=Path(video_path).name,
        objects=tuple(objects),
        video_width=video_width,
        video_height=video_height,
        video_frames=video_frames,
        postprocess_enabled=postprocess_enabled,
        status=status,
    )


def verify_manifest(manifest: MaskManifest, *, check_sha: bool = True) -> tuple[bool, list[str]]:
    """复核 manifest：文件存在性 + SHA 一致性。返回 (all_ok, problems)。"""

    problems: list[str] = []
    for obj in manifest.objects:
        for fr in obj.frames:
            if not fr.valid:
                problems.append(f"{obj.normalized_name}#{fr.frame_index}: {fr.reason}")
                continue
            path = Path(fr.path)
            if not path.exists():
                problems.append(f"{obj.normalized_name}#{fr.frame_index}: missing")
                continue
            if check_sha and fr.sha256 is not None:
                if _sha256_file(path) != fr.sha256:
                    problems.append(f"{obj.normalized_name}#{fr.frame_index}: sha_mismatch")
    return (not problems), problems


def has_valid_masks(manifest: MaskManifest) -> bool:
    return any(fr.valid for obj in manifest.objects for fr in obj.frames)


def build_local_edit_target(
    *,
    parent_candidate_id: str,
    violation: Any,
    manifest: MaskManifest,
    manifest_uri: str | Path | None = None,
) -> LocalEditTarget | None:
    """从单个 violation + manifest 构造 LocalEditTarget；无有效 mask 时返回 None。

    Local Editor 需要逐帧 mask，因此 mask_uri 指向 manifest（editor 据此按帧取 mask），
    而不是把第一张 mask 复制到全部帧。
    """

    if manifest_uri is None:
        return None

    valid_paths = manifest.valid_frame_paths(getattr(violation, "object", ""))
    critical = tuple(
        f for f in (getattr(violation, "critical_frames", ()) or ()) if int(f) in valid_paths
    )
    if not critical:
        return None
    return LocalEditTarget(
        parent_candidate_id=parent_candidate_id,
        objects=(str(getattr(violation, "object", "")),),
        start_frame=int(getattr(violation, "start_frame", min(critical))),
        end_frame=int(getattr(violation, "end_frame", max(critical))),
        critical_frames=tuple(int(f) for f in critical),
        mask_uri=str(manifest_uri),
    )
