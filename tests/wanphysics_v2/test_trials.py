"""WanRepairTrialV2 + canonical adapter 需批准。"""
from physgenloop.learning_repair.contracts import CandidateRecord, RepairDecision, ScoreBundle
from physgenloop.learning_repair.base_contracts import RepairAction
from generators.wanphysics.v2.trials import WanRepairTrialV2, to_canonical_trial


def _dec():
    return RepairDecision(action=RepairAction.PROMPT_REPAIR, confidence=0.5, instruction="i",
                          action_probabilities={a.value: 0.25 for a in RepairAction},
                          per_action_values={a.value: 0.0 for a in RepairAction})


def _trial():
    return WanRepairTrialV2(
        trial_id="t1", group_id="g1", source_candidate=CandidateRecord(candidate_id="c", video_path="/tmp/c.mp4", prompt="p", seed=1),
        prompt="p", critic_before={"decision": "violation"}, decision=_dec(),
        execution={"status": "succeeded"}, before_scores=ScoreBundle(physics=0.2),
        critic_after={"decision": "physical"}, after_scores=ScoreBundle(physics=0.85), successful=True,
    )


def test_wan_trial_roundtrip_and_provenance():
    t = _trial()
    d = t.to_dict()
    assert d["generator"]["family"] == "wan" and d["research_only"] is True
    assert abs(d["physics_gain"] - 0.65) < 1e-6
    assert WanRepairTrialV2.from_dict(d).to_dict() == d


def test_canonical_requires_approval():
    try:
        to_canonical_trial(_trial(), domain="hunyuan", approved=False)
        assert False
    except ValueError:
        pass


def test_canonical_approved_maps():
    v1 = to_canonical_trial(_trial(), domain="fake", approved=True)
    assert v1.domain == "fake"
    assert v1.compatibility["mapped_from"] == "wan_repair_trial_v2"
