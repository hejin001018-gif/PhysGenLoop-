"""preflight capability mask：ProPainter 缺失/不完整 → local 移除。"""
from generators.wanphysics.v2 import preflight
from generators.wanphysics.v2.preflight import run_preflight, check_propainter, port_in_use


def test_missing_propainter_masks_local(tmp_path):
    rep = run_preflight(propainter_repo=str(tmp_path / "nope"), vllm_port=59999)
    assert rep.capability_mask["local_editing"] is False
    assert rep.capability_mask["reject"] is True
    assert "global_regeneration" not in rep.capability_mask
    assert "propainter_repo" in rep.missing or "propainter_script" in rep.missing


def _stub_required_weights(root, monkeypatch):
    monkeypatch.setattr(
        preflight,
        "REQUIRED_PROPAINTER_WEIGHTS",
        {"raft-things.pth": 1, "recurrent_flow_completion.pth": 1, "ProPainter.pth": 1},
    )
    weights = root / "weights"
    weights.mkdir()
    for name in preflight.REQUIRED_PROPAINTER_WEIGHTS:
        (weights / name).write_bytes(b"x")


def test_propainter_present(tmp_path, monkeypatch):
    (tmp_path / "inference_propainter.py").write_text("# stub", encoding="utf-8")
    _stub_required_weights(tmp_path, monkeypatch)
    assert check_propainter(str(tmp_path)).ok


def test_propainter_missing_required_weight_fails(tmp_path, monkeypatch):
    (tmp_path / "inference_propainter.py").write_text("# stub", encoding="utf-8")
    _stub_required_weights(tmp_path, monkeypatch)
    (tmp_path / "weights" / "ProPainter.pth").unlink()
    result = check_propainter(str(tmp_path))
    assert result.ok is False
    assert result.name == "propainter_weights"
    assert "ProPainter.pth" in result.detail


def test_propainter_part_file_fails(tmp_path, monkeypatch):
    (tmp_path / "inference_propainter.py").write_text("# stub", encoding="utf-8")
    _stub_required_weights(tmp_path, monkeypatch)
    (tmp_path / "weights" / "ProPainter.pth.part").write_bytes(b"partial")
    result = check_propainter(str(tmp_path))
    assert result.ok is False
    assert "incomplete downloads" in result.detail


def test_port_check_free():
    assert port_in_use("127.0.0.1", 59998) is False
