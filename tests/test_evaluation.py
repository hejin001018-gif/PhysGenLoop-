"""基线、消融配置和评估指标。"""

from __future__ import annotations

from pavg_critic.evaluation import (
    EvaluationRecord,
    build_ablation_config,
    compute_metrics,
    load_evaluation_samples,
    load_pqsg_evaluation_records,
    run_rule_evaluation,
)


def test_metrics_include_unknown_and_violation_f1():
    records = (
        EvaluationRecord("a", "violation", "violation", 0.2, 0.8),
        EvaluationRecord("b", "physical", "physical", 0.9, 0.7),
        EvaluationRecord("c", "violation", "unknown", 0.5, 0.1),
    )

    metrics = compute_metrics(records)

    assert metrics.accuracy == 2 / 3
    assert metrics.violation_precision == 1.0
    assert metrics.violation_recall == 0.5
    assert metrics.unknown_rate == 1 / 3
    assert metrics.mean_coverage == (0.8 + 0.7 + 0.1) / 3


def test_ablation_modes_enable_modules_incrementally():
    b1 = build_ablation_config("B1_RULE")
    m2 = build_ablation_config("M2_CHECKLIST")
    m3 = build_ablation_config("M3_MECHANICS")

    assert b1.question_graph.enabled is False
    assert b1.checklist.enabled is False
    assert b1.mechanics.enabled is False
    assert m2.checklist.enabled is True
    assert m2.mechanics.enabled is False
    assert m3.checklist.enabled is True
    assert m3.mechanics.enabled is True


def test_frozen_mini_fixture_runs_rule_baseline():
    samples = load_evaluation_samples("evaluation/fixtures/critic_mini.json")

    records, metrics = run_rule_evaluation(samples, mode="B1_RULE")

    assert len(records) == 6
    assert metrics.accuracy == 1.0
    assert metrics.violation_recall == 1.0


def test_official_pqsg_output_format_is_adapted_as_independent_b0(tmp_path):
    path = tmp_path / "pqsg_results.json"
    path.write_text(
        """[
          {
            "id": "normal",
            "label": "physical",
            "score": 0.9,
            "psg": {"nodes": {"object_existence": {"O1": "q"}, "action_verification": {}, "physics": {"P1": "q"}}},
            "answers": {"object_existence": {"O1": {"is_correct": true}}, "action_verification": {}, "physics": {"P1": {"is_correct": true}}}
          },
          {
            "id": "bad",
            "label": "violation",
            "score": 0.2,
            "psg": {"nodes": {"object_existence": {"O1": "q"}, "action_verification": {}, "physics": {"P1": "q"}}},
            "answers": {"object_existence": {"O1": {"is_correct": true}}, "action_verification": {}, "physics": {"P1": {"is_correct": false}}}
          }
        ]""",
        encoding="utf-8",
    )

    records = load_pqsg_evaluation_records(path, threshold=0.5)
    metrics = compute_metrics(records)

    assert [record.prediction for record in records] == ["physical", "violation"]
    assert all(record.coverage == 1.0 for record in records)
    assert metrics.accuracy == 1.0
