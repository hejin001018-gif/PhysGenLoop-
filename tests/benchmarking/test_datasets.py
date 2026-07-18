import csv
import io
import json
from pathlib import Path

from benchmarks.prepare_videophy_manifest import main as prepare_manifest_main
from pavg_critic.benchmarking import datasets
from pavg_critic.benchmarking.datasets import (
    load_manifest,
    load_videophy_csv,
    materialize_video_csv,
    select_smoke_samples,
    split_diagnostic_samples,
    write_manifest,
    write_source_smoke_csv,
)


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


def test_videophy_csv_uses_explicit_group_for_missing_action(tmp_path: Path):
    video = tmp_path / "missing-action.mp4"
    video.write_bytes(b"fake")
    csv_path = tmp_path / "data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["caption", "local_path", "sa", "pc", "model_name", "action"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "caption": "object moves",
                "local_path": str(video),
                "sa": "4",
                "pc": "3",
                "model_name": "gen-a",
                "action": "",
            }
        )

    sample = load_videophy_csv(csv_path, benchmark="videophy2", split="test")[0]

    assert sample.prompt_group_id == "__missing_action__"


def test_smoke_selection_is_stable_and_balanced(sample_factory):
    samples = tuple(
        sample_factory(index=i, physical=(i % 2 == 0), generator=f"g{i % 3}")
        for i in range(30)
    )
    first = select_smoke_samples(samples, count=20, seed=20260715)
    second = select_smoke_samples(samples, count=20, seed=20260715)
    assert [item.sample_id for item in first] == [item.sample_id for item in second]
    assert {item.physics_label for item in first} == {"physical", "violation"}
    assert sum(item.physics_label == "physical" for item in first) == 10
    assert len({item.generator for item in first}) == 3


def test_smoke_selection_rejects_impossible_count(sample_factory):
    samples = (sample_factory(index=0, physical=True, generator="g0"),)
    try:
        select_smoke_samples(samples, count=2, seed=20260715)
    except ValueError as exc:
        assert "between 1" in str(exc)
    else:
        raise AssertionError("expected an invalid count to fail")


def test_real_videophy2_schema_derives_stable_id_and_rules(tmp_path: Path):
    video = tmp_path / "real.mp4"
    video.write_bytes(b"fake")
    csv_path = tmp_path / "real-schema.csv"
    fields = [
        "caption",
        "video_url",
        "local_path",
        "sa",
        "pc",
        "action",
        "model_name",
        "physics_rules_followed",
        "physics_rules_unfollowed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "caption": "a ball falls",
                "video_url": "https://example.test/a.mp4",
                "local_path": str(video),
                "sa": "5",
                "pc": "2",
                "action": "falling",
                "model_name": "wan",
                "physics_rules_followed": "['The ball remains visible.']",
                "physics_rules_unfollowed": "['Gravity points downward.']",
            }
        )
    first = load_videophy_csv(csv_path, benchmark="videophy2", split="test")[0]
    second = load_videophy_csv(csv_path, benchmark="videophy2", split="test")[0]
    assert first.sample_id == second.sample_id
    assert first.sample_id.startswith("videophy2-")
    assert first.generator == "wan"
    assert first.source_url == "https://example.test/a.mp4"
    assert first.physical_rules == (
        "The ball remains visible.",
        "Gravity points downward.",
    )


def test_videophy_rule_nan_list_is_treated_as_missing(tmp_path: Path):
    video = tmp_path / "nan-rule.mp4"
    video.write_bytes(b"fake")
    csv_path = tmp_path / "data.csv"
    fields = [
        "caption",
        "local_path",
        "sa",
        "pc",
        "action",
        "model_name",
        "physics_rules_cannot_be_determined",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "caption": "object moves",
                "local_path": str(video),
                "sa": "5",
                "pc": "2",
                "action": "move",
                "model_name": "gen-a",
                "physics_rules_cannot_be_determined": "[nan]",
            }
        )

    sample = load_videophy_csv(csv_path, benchmark="videophy2", split="test")[0]

    assert sample.physical_rules == ()


def test_source_smoke_selection_happens_before_video_download(tmp_path: Path):
    source = tmp_path / "all.csv"
    output = tmp_path / "smoke.csv"
    fields = ["caption", "video_url", "sa", "pc", "action", "model_name"]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(30):
            writer.writerow(
                {
                    "caption": f"prompt {index}",
                    "video_url": f"https://example.test/{index}.mp4",
                    "sa": "5",
                    "pc": "5" if index % 2 == 0 else "2",
                    "action": f"action-{index // 2}",
                    "model_name": f"g{index % 3}",
                }
            )
    write_source_smoke_csv(source, output, count=20, seed=20260715)
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert {"2", "5"} <= {row["pc"] for row in rows}
    assert sum(float(row["pc"]) >= 4 for row in rows) == 10
    assert len({row["model_name"] for row in rows}) == 3


def test_source_pilot_covers_actions_and_balances_labels_and_generators(
    tmp_path: Path,
):
    source = tmp_path / "all.csv"
    first = tmp_path / "pilot-first.csv"
    second = tmp_path / "pilot-second.csv"
    fields = ["caption", "video_url", "sa", "pc", "action", "model_name"]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for action in range(8):
            for generator in range(3):
                for physical in (False, True):
                    index = action * 6 + generator * 2 + int(physical)
                    writer.writerow(
                        {
                            "caption": f"prompt {index}",
                            "video_url": f"https://example.test/{index}.mp4",
                            "sa": "5",
                            "pc": "5" if physical else "2",
                            "action": f"action-{action}",
                            "model_name": f"g{generator}",
                        }
                    )

    assert hasattr(datasets, "write_source_pilot_csv")
    datasets.write_source_pilot_csv(source, first, count=12, seed=20260716)
    datasets.write_source_pilot_csv(source, second, count=12, seed=20260716)

    with first.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert first.read_bytes() == second.read_bytes()
    assert len(rows) == 12
    assert len({row["action"] for row in rows}) == 8
    assert sum(float(row["pc"]) >= 4 for row in rows) == 6
    assert sorted(
        sum(row["model_name"] == generator for row in rows)
        for generator in ("g0", "g1", "g2")
    ) == [4, 4, 4]


def test_source_pilot_balances_labels_after_covering_actions(tmp_path: Path):
    source = tmp_path / "imbalanced-actions.csv"
    output = tmp_path / "pilot.csv"
    fields = ["id", "caption", "video_url", "sa", "pc", "action", "model_name"]
    rows = []
    for index, action in enumerate(("A", "A", "B", "B", "C", "C")):
        rows.append(
            {
                "id": str(index),
                "caption": f"physical {index}",
                "video_url": f"https://example.test/{index}.mp4",
                "sa": "5",
                "pc": "5",
                "action": action,
                "model_name": "g0",
            }
        )
    for index in range(6, 12):
        rows.append(
            {
                "id": str(index),
                "caption": f"violation {index}",
                "video_url": f"https://example.test/{index}.mp4",
                "sa": "5",
                "pc": "2",
                "action": "D",
                "model_name": "g1",
            }
        )
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    datasets.write_source_pilot_csv(source, output, count=8, seed=20260716)

    with output.open(newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    assert {row["action"] for row in selected} == {"A", "B", "C", "D"}
    assert sum(float(row["pc"]) >= 4 for row in selected) == 4


def test_source_pilot_keeps_rows_with_missing_action(tmp_path: Path):
    source = tmp_path / "all.csv"
    output = tmp_path / "pilot.csv"
    fields = ["caption", "video_url", "sa", "pc", "action", "model_name"]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "caption": "missing action",
                "video_url": "https://example.test/missing.mp4",
                "sa": "5",
                "pc": "5",
                "action": "",
                "model_name": "g0",
            }
        )
        writer.writerow(
            {
                "caption": "known action",
                "video_url": "https://example.test/known.mp4",
                "sa": "5",
                "pc": "2",
                "action": "known",
                "model_name": "g1",
            }
        )

    datasets.write_source_pilot_csv(source, output, count=2, seed=20260716)

    with output.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_prepare_manifest_cli_exposes_source_pilot(tmp_path: Path):
    source = tmp_path / "all.csv"
    output = tmp_path / "pilot.csv"
    fields = ["caption", "video_url", "sa", "pc", "action", "model_name"]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(4):
            writer.writerow(
                {
                    "caption": f"prompt {index}",
                    "video_url": f"https://example.test/{index}.mp4",
                    "sa": "5",
                    "pc": "5" if index % 2 else "2",
                    "action": f"action-{index}",
                    "model_name": f"g{index % 2}",
                }
            )

    assert prepare_manifest_main(
        [
            "source-pilot",
            "--csv",
            str(source),
            "--count",
            "4",
            "--seed",
            "20260716",
            "--output-csv",
            str(output),
        ]
    ) == 0
    with output.open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 4


def test_video_download_percent_encodes_unicode_url_path(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "source.csv"
    output = tmp_path / "local.csv"
    fields = ["caption", "video_url", "sa", "pc", "action", "model_name"]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "caption": "hands slip",
                "video_url": "https://example.test/gymnast’s_hands.mp4",
                "sa": "5",
                "pc": "2",
                "action": "gymnastics",
                "model_name": "cosmos",
            }
        )
    requested_urls = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        return io.BytesIO(b"video")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    failures = materialize_video_csv(
        source,
        video_dir=tmp_path / "videos",
        output_csv=output,
    )

    assert failures == ()
    assert requested_urls == [
        "https://example.test/gymnast%E2%80%99s_hands.mp4"
    ]


def test_video_download_preserves_existing_percent_encoded_path(
    tmp_path: Path,
    monkeypatch,
):
    source = tmp_path / "source.csv"
    output = tmp_path / "local.csv"
    fields = ["caption", "video_url", "sa", "pc", "action", "model_name"]
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "caption": "encoded path",
                "video_url": "https://example.test/gymnast%E2%80%99s_hands.mp4",
                "sa": "5",
                "pc": "2",
                "action": "gymnastics",
                "model_name": "cosmos",
            }
        )
    requested_urls = []

    def fake_urlopen(request, timeout):
        requested_urls.append(request.full_url)
        return io.BytesIO(b"video")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert materialize_video_csv(
        source,
        video_dir=tmp_path / "videos",
        output_csv=output,
    ) == ()
    assert requested_urls == [
        "https://example.test/gymnast%E2%80%99s_hands.mp4"
    ]


def test_manifest_stores_portable_video_paths(sample_factory, tmp_path: Path):
    sample = sample_factory(index=1, physical=True, generator="g")
    manifest = tmp_path / "manifests" / "smoke.json"
    write_manifest((sample,), manifest)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    assert not Path(raw["samples"][0]["video_path"]).is_absolute()
    assert "\\" not in raw["samples"][0]["video_path"]
    assert load_manifest(manifest) == (sample,)


def test_diagnostic_split_is_disjoint_balanced_and_group_preserving(
    sample_factory, tmp_path: Path
):
    samples = tuple(
        sample_factory(index=index, physical=index % 2 == 0, generator=f"g{index % 4}")
        for index in range(20)
    )
    dev, evaluation = split_diagnostic_samples(
        samples,
        dev_count=10,
        seed=20260716,
    )
    assert len(dev) == len(evaluation) == 10
    assert {item.sample_id for item in dev}.isdisjoint(
        item.sample_id for item in evaluation
    )
    assert {item.physics_label for item in dev} == {"physical", "violation"}
    assert {item.physics_label for item in evaluation} == {"physical", "violation"}
    dev_groups = {item.prompt_group_id for item in dev}
    eval_groups = {item.prompt_group_id for item in evaluation}
    assert dev_groups.isdisjoint(eval_groups)

    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_manifest(dev, first_path)
    repeated, _ = split_diagnostic_samples(
        samples,
        dev_count=10,
        seed=20260716,
    )
    write_manifest(repeated, second_path)
    assert first_path.read_bytes() == second_path.read_bytes()
