from pathlib import Path

import numpy as np
import pytest
import torch

from pavg_critic.sam2_detector import SAM2ObjectDetector


class ObjectSeedModel:
    def generate_json_with_images(self, **kwargs):
        return {
            "objects": [
                {
                    "name": "ball",
                    "description": "a ball",
                    "x_pct": 50,
                    "y_pct": 50,
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
    detector = SAM2ObjectDetector(
        ObjectSeedModel(),
        str(video),
        model_cfg="configs/sam2.1/sam2.1_hiera_b+.yaml",
        model_ckpt="checkpoint.pt",
    )

    assert observed["is_dir"] is True
    assert observed["frames"] == ["00000.jpg", "00001.jpg", "00002.jpg"]
    assert observed["config"] == "configs/sam2.1/sam2.1_hiera_b+.yaml"
    assert len(detector.detect(None, 0, 0.0)) == 1
    assert len(detector.detect(None, 2, 0.2)) == 1
    assert detector.detect(None, 0, 0.0)[0].bbox == (10.0, 20.0, 29.0, 39.0)
