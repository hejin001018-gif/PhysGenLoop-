from pathlib import Path

import numpy as np
import pytest
import torch

from pavg_critic.sam2_detector import SAM2ObjectDetector


class ObjectSeedModel:
    def __init__(self):
        self.schema = None
        self.system_prompt = None
        self.user_prompt = None

    def generate_json_with_images(self, **kwargs):
        self.schema = kwargs["schema"]
        self.system_prompt = kwargs["system_prompt"]
        self.user_prompt = kwargs["user_prompt"]
        return {
            "objects": [
                {
                    "name": "ball",
                    "description": "a ball",
                    "x_pct": 101,
                    "y_pct": -1,
                }
            ]
        }


def _write_test_video(path: Path) -> None:
    import cv2

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 64),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    for value in range(3):
        writer.write(np.full((64, 64, 3), value * 30, dtype=np.uint8))
    writer.release()


def test_sam2_uses_jpeg_folder_and_squeezes_mask_channel(tmp_path, monkeypatch):
    video = tmp_path / "three.avi"
    _write_test_video(video)
    observed = {}

    class FakePredictor:
        def init_state(self, *, video_path, **kwargs):
            frame_dir = Path(video_path)
            observed["is_dir"] = frame_dir.is_dir()
            observed["frames"] = sorted(item.name for item in frame_dir.glob("*.jpg"))
            return {}

        def reset_state(self, state):
            return None

        def add_new_points_or_box(self, **kwargs):
            observed["seed_points"] = kwargs["points"].tolist()
            return 0, (0,), None

        def propagate_in_video(self, state):
            for frame_index in range(3):
                logits = torch.full((1, 1, 64, 64), -1.0)
                logits[:, :, 20:40, 10:30] = 1.0
                yield frame_index, (0,), logits

    def fake_build(config, checkpoint, *, device):
        observed["config"] = config
        observed["checkpoint"] = checkpoint
        observed["device"] = device
        return FakePredictor()

    monkeypatch.setattr("sam2.build_sam.build_sam2_video_predictor", fake_build)
    seed_model = ObjectSeedModel()
    detector = SAM2ObjectDetector(
        seed_model,
        str(video),
        model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
        model_ckpt="checkpoint.pt",
        prompt="A ball falls onto the floor.",
    )

    assert observed["is_dir"] is True
    assert observed["frames"] == ["00000.jpg", "00001.jpg", "00002.jpg"]
    assert observed["config"] == "configs/sam2.1/sam2.1_hiera_b+.yaml"
    assert observed["seed_points"] == [[63.0, 0.0]]
    assert seed_model.schema["properties"]["objects"]["items"]["properties"][
        "x_pct"
    ] == {"type": "number", "minimum": 0, "maximum": 100}
    assert "A ball falls onto the floor." in seed_model.user_prompt
    assert "prompt-relevant" in seed_model.system_prompt
    assert "background" in seed_model.system_prompt
    assert len(detector.detect(None, 0, 0.0)) == 1
    assert len(detector.detect(None, 2, 0.2)) == 1
    assert detector.detect(None, 0, 0.0)[0].track_id == "sam2:0"
    assert detector.detect(None, 0, 0.0)[0].bbox == (10.0, 20.0, 29.0, 39.0)


def test_sam2_rejects_non_finite_object_seed(tmp_path, monkeypatch):
    video = tmp_path / "three.avi"
    _write_test_video(video)

    class FakePredictor:
        def init_state(self, **kwargs):
            return {}

        def reset_state(self, state):
            return None

    monkeypatch.setattr(
        "sam2.build_sam.build_sam2_video_predictor",
        lambda *args, **kwargs: FakePredictor(),
    )
    detector = object.__new__(SAM2ObjectDetector)
    detector._cache = {}
    detector._object_names = {}
    detector._width = 64
    detector._height = 64

    with pytest.raises(ValueError, match="must be finite"):
        detector._track_with_sam2(
            str(video),
            [{"name": "ball", "x_pct": float("nan"), "y_pct": 50}],
            "config.yaml",
            "checkpoint.pt",
            10.0,
            85,
        )
