import csv
from pathlib import Path

from pavg_critic.benchmarking.datasets import load_videophy_csv, select_smoke_samples


def test_videophy_csv_aliases_and_thresholds(tmp_path: Path):
    video = tmp_path / "a.mp4"
    video.write_bytes(b"fake")
    csv_path = tmp_path / "data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "caption", "videopath", "sa", "pc", "model", "action"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "1",
                "caption": "ball rolls",
                "videopath": str(video),
                "sa": "4",
                "pc": "3",
                "model": "gen-a",
                "action": "roll",
            }
        )
    sample = load_videophy_csv(csv_path, benchmark="videophy2", split="test")[0]
    assert sample.semantic_label == "adherent"
    assert sample.physics_label == "violation"
    assert sample.prompt_group_id == "roll"


def test_smoke_selection_is_stable_and_balanced(sample_factory):
    samples = tuple(
        sample_factory(index=i, physical=(i % 2 == 0), generator=f"g{i % 3}")
        for i in range(30)
    )
    first = select_smoke_samples(samples, count=20, seed=20260715)
    second = select_smoke_samples(samples, count=20, seed=20260715)
    assert [item.sample_id for item in first] == [item.sample_id for item in second]
    assert {item.physics_label for item in first} == {"physical", "violation"}
    assert len({item.generator for item in first}) == 3


def test_smoke_selection_rejects_impossible_count(sample_factory):
    samples = (sample_factory(index=0, physical=True, generator="g0"),)
    try:
        select_smoke_samples(samples, count=2, seed=20260715)
    except ValueError as exc:
        assert "between 1" in str(exc)
    else:
        raise AssertionError("expected an invalid count to fail")
