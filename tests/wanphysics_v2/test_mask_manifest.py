"""mask manifest 构建/校验/LocalEditTarget 构造。"""
from generators.wanphysics.v2.mask_manifest import (
    MaskManifest, MaskObject, MaskFrame, normalize_object_name,
    build_local_edit_target, build_manifest, has_valid_masks,
)


class _V:
    def __init__(self):
        self.object = "Baseball"
        self.category = "penetration"
        self.start_frame = 12
        self.end_frame = 15
        self.critical_frames = (12, 13, 15)


def test_normalize_object_name():
    assert normalize_object_name("  Red  Ball ") == "red ball"


def _manifest():
    frames = (
        MaskFrame(frame_index=12, path="/tmp/baseball_00012.png", valid=True),
        MaskFrame(frame_index=13, path="/tmp/baseball_00013.png", valid=False, reason="missing"),
        MaskFrame(frame_index=15, path="/tmp/baseball_00015.png", valid=True),
    )
    obj = MaskObject(name="Baseball", normalized_name="baseball", frames=frames)
    return MaskManifest(candidate_id="c1", video="c1.mp4", objects=(obj,), video_frames=81)


def test_valid_frame_paths_filters_invalid():
    m = _manifest()
    paths = m.valid_frame_paths("baseball")
    assert set(paths) == {12, 15}
    assert has_valid_masks(m)


def test_build_local_edit_target_only_valid_frames():
    target = build_local_edit_target(
        parent_candidate_id="c1",
        violation=_V(),
        manifest=_manifest(),
        manifest_uri="/tmp/c1/mask_manifest.json",
    )
    assert target is not None
    assert target.critical_frames == (12, 15)  # 13 无效被剔除
    assert target.mask_uri.endswith("mask_manifest.json")


def test_target_none_without_manifest_uri():
    assert build_local_edit_target(parent_candidate_id="c1", violation=_V(), manifest=_manifest()) is None


def test_target_none_when_no_valid_mask():
    empty = MaskManifest(candidate_id="c1", video="c1.mp4", objects=(), video_frames=81)
    assert build_local_edit_target(parent_candidate_id="c1", violation=_V(), manifest=empty) is None


def test_manifest_dict_roundtrip():
    m = _manifest()
    assert MaskManifest.from_dict(m.to_dict()).to_dict() == m.to_dict()


def test_manifest_status_invalid_when_any_requested_mask_missing(tmp_path):
    import numpy as np
    import pytest

    cv2 = pytest.importorskip("cv2")

    class _V2:
        object = "ball"
        critical_frames = (1, 2)

    mask_dir = tmp_path / "sam2_masks"
    mask_dir.mkdir()
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    cv2.imwrite(str(mask_dir / "ball_00001.png"), mask)
    manifest = build_manifest(
        candidate_id="c1",
        video_path="/tmp/c1.mp4",
        mask_dir=mask_dir,
        violations=(_V2(),),
        compute_sha=False,
    )
    assert manifest.status == "invalid"
    assert has_valid_masks(manifest) is True
