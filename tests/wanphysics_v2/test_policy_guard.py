"""scope 判定 + capability mask + action override。"""
from generators.wanphysics.v2.policy_guard import normalize_action, classify_scope, resolve_action
from physgenloop.learning_repair.base_contracts import RepairAction


class _V:
    def __init__(self, frames):
        self.critical_frames = frames
        self.evidence = {}


class _Report:
    def __init__(self, violations):
        self.violations = violations


def test_normalize_action():
    assert normalize_action(RepairAction.PROMPT_REPAIR) == "prompt_repair"
    assert normalize_action("RepairAction.LOCAL_EDITING") == "local_editing"
    assert normalize_action("REJECT") == "reject"


def test_scope_unknown_global_local():
    assert classify_scope(_Report(()), 10, 0.4, mask_valid=True) == "unknown"
    assert classify_scope(_Report((_V((1,)),)), 10, 0.4, mask_valid=True) == "local"
    assert classify_scope(_Report((_V(tuple(range(6))),)), 10, 0.4, mask_valid=True) == "global"
    # mask 无效 → 强制 global
    assert classify_scope(_Report((_V((1,)),)), 10, 0.4, mask_valid=False) == "global"


def _caps(local=True):
    return {"prompt_repair": True, "global_regeneration": True, "local_editing": local, "reject": True}


def test_local_scope_selects_local_editing():
    r = resolve_action(policy_action="prompt_repair", report=_Report((_V((1,)),)), total_frames=10, local_threshold=0.4, capability_available=_caps(True), mask_valid=True)
    assert r.final_action == "local_editing" and r.overridden


def test_local_unavailable_falls_back():
    r = resolve_action(policy_action="local_editing", report=_Report((_V((1,)),)), total_frames=10, local_threshold=0.4, capability_available=_caps(False), mask_valid=False)
    assert r.final_action != "local_editing"
    assert r.override_reason


def test_global_scope_overrides_policy_local():
    r = resolve_action(policy_action="local_editing", report=_Report((_V(tuple(range(6))),)), total_frames=10, local_threshold=0.4, capability_available=_caps(True), mask_valid=True)
    assert r.final_action == "global_regeneration"
