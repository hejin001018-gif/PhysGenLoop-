import pytest

from generators.wanphysics.v2.trials import WanRepairTrialV3
from physgenloop.learning_repair.base_contracts import RepairAction
from physgenloop.learning_repair.contracts import CandidateRecord, RepairDecision, ScoreBundle


def _decision():
    probabilities = {
        RepairAction.PROMPT_REPAIR: 0.8,
        RepairAction.LOCAL_EDITING: 0.1,
        RepairAction.REJECT: 0.1,
    }
    return RepairDecision(
        action=RepairAction.PROMPT_REPAIR,
        confidence=0.8,
        instruction="fix gravity",
        action_probabilities=probabilities,
        per_action_values=probabilities,
        source="three_action_policy",
    )


def _trial(**overrides):
    values = dict(
        trial_id="t1",
        group_id="g1",
        source_candidate=CandidateRecord("c1", "/tmp/c1.mp4", "p", 1),
        original_prompt="p",
        prompt="p",
        critic_before={"decision": "violation"},
        decision=_decision(),
        guard={"status": "allowed"},
        execution={"status": "succeeded", "backend_id": "prompt"},
        before_scores=ScoreBundle(0.2, 0.9, 0.9, 0.9),
        critic_after={"decision": "physical"},
        after_scores=ScoreBundle(0.9, 0.9, 0.9, 0.9),
        repair_improved=True,
        successful=True,
        failure_reason=None,
        metadata={"gates": {"after": {"status": "ACCEPTED"}}},
    )
    values.update(overrides)
    return WanRepairTrialV3(**values)


def test_v3_roundtrip_and_three_action_provenance():
    trial = _trial()
    payload = trial.to_dict()
    assert payload["schema_version"] == "wan-repair-trial/3.0"
    assert payload["decision_source"] == "three_action_policy"
    assert WanRepairTrialV3.from_dict(payload).to_dict() == payload


def test_partial_improvement_is_not_successful():
    trial = _trial(
        successful=False,
        failure_reason="after_gate_rejected",
        metadata={"gates": {"after": {"status": "REJECTED"}}},
    )
    assert trial.repair_improved and not trial.successful


def test_success_requires_strict_regate():
    with pytest.raises(ValueError, match="Strict Re-Gate ACCEPTED"):
        _trial(metadata={"gates": {"after": {"status": "REJECTED"}}})
