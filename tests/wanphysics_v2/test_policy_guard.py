from generators.wanphysics.v2.policy_guard import classify_scope, resolve_action


class _Violation:
    repair_instruction = "fix contact"
    critical_frames = (2, 3)


class _Report:
    violations = (_Violation(),)


def _caps(local=True):
    return {"prompt_repair": True, "local_editing": local, "reject": True}


def test_guard_allows_policy_action_without_rewriting():
    result = resolve_action(
        policy_action="local_editing",
        report=_Report(),
        total_frames=20,
        local_threshold=0.4,
        capability_available=_caps(),
        mask_valid=True,
    )
    assert result.allowed
    assert result.policy_action == result.final_action == "local_editing"


def test_guard_blocks_but_does_not_choose_prompt_fallback():
    result = resolve_action(
        policy_action="local_editing",
        report=_Report(),
        total_frames=20,
        local_threshold=0.4,
        capability_available=_caps(local=False),
        mask_valid=False,
    )
    assert not result.allowed
    assert result.policy_action == "local_editing"
    assert result.final_action == "reject"
    assert result.blocked_reason.startswith("capability_unavailable")


def test_scope_uses_broad_not_global():
    assert classify_scope(_Report(), 2, 0.4) == "broad"
