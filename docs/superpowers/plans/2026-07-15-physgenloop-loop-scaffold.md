# PhysGenLoop Loop Scaffold Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an importable, deterministic orchestration shell around the existing Physics Critic so future HunyuanVideo, Repair Agent, and Best-of-K work has stable contracts and a tested bounded loop.

**Architecture:** Keep `pavg_critic` as the physics-analysis package and add a separate `physgenloop` package for cross-component orchestration. All real model integrations sit behind Protocols; the initial generator is an explicit fake, selection and repair are deterministic, and the controller records every round while enforcing finite stopping conditions.

**Tech Stack:** Python 3.10+, frozen dataclasses, typing Protocols, pytest, existing `pavg_critic` schemas.

---

## Execution order and file map

First execute `docs/superpowers/plans/2026-07-15-prompt-physics-plan.md`; it owns the Prompt → PhysicsPlan implementation inside `pavg_critic`. Then execute this plan.

- Create `src/physgenloop/contracts.py`: loop configuration, generated candidate, evaluated candidate, round record, and final result.
- Create `src/physgenloop/interfaces.py`: generator, candidate critic, repairer, and selector Protocols.
- Create `src/physgenloop/generator.py`: deterministic fake generator for CPU tests and framework demonstrations.
- Create `src/physgenloop/critic_adapter.py`: adapter from generated candidates to the existing `PhysicsCritic` API.
- Create `src/physgenloop/repairer.py`: deterministic structured repair-instruction aggregation.
- Create `src/physgenloop/selector.py`: stable evidence-aware candidate ranking.
- Create `src/physgenloop/controller.py`: bounded Best-of-K feedback loop.
- Create `src/physgenloop/__init__.py`: stable public orchestration API.
- Create `tests/test_loop_contracts.py`: validation and fake-generation tests.
- Create `tests/test_loop_controller.py`: repair, selection, stop, history, and max-round tests.
- Modify `README.md`: current/ scaffold/future status, architecture, Planner, fake loop, structure, and roadmap.

### Task 1: Add frozen orchestration contracts

**Files:**
- Create: `src/physgenloop/contracts.py`
- Create: `tests/test_loop_contracts.py`

- [ ] **Step 1: Write failing contract validation tests**

```python
import pytest

from physgenloop.contracts import GeneratedCandidate, LoopConfig


def test_loop_config_rejects_unbounded_or_empty_runs():
    with pytest.raises(ValueError, match="max_rounds"):
        LoopConfig(max_rounds=0)
    with pytest.raises(ValueError, match="candidates_per_round"):
        LoopConfig(candidates_per_round=0)
    with pytest.raises(ValueError, match="acceptance_score"):
        LoopConfig(acceptance_score=1.1)


def test_generated_candidate_requires_identity_and_video_path():
    with pytest.raises(ValueError, match="candidate_id"):
        GeneratedCandidate(candidate_id="", video_path="fake://a", prompt="p", seed=1)
    with pytest.raises(ValueError, match="video_path"):
        GeneratedCandidate(candidate_id="a", video_path="", prompt="p", seed=1)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_loop_contracts.py -q`

Expected: collection fails because `physgenloop` does not exist.

- [ ] **Step 3: Implement the frozen contracts**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pavg_critic.schemas import CriticReport, PhysicsPlan


@dataclass(frozen=True)
class LoopConfig:
    max_rounds: int = 3
    candidates_per_round: int = 2
    acceptance_score: float = 0.8
    base_seed: int = 42

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        if self.candidates_per_round < 1:
            raise ValueError("candidates_per_round must be at least 1")
        if not 0.0 <= self.acceptance_score <= 1.0:
            raise ValueError("acceptance_score must be within [0, 1]")


@dataclass(frozen=True)
class GeneratedCandidate:
    candidate_id: str
    video_path: str
    prompt: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must not be empty")
        if not self.video_path.strip():
            raise ValueError("video_path must not be empty")


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: GeneratedCandidate
    report: CriticReport


@dataclass(frozen=True)
class LoopRound:
    round_index: int
    prompt: str
    evaluations: tuple[CandidateEvaluation, ...]
    selected_candidate_id: str


@dataclass(frozen=True)
class LoopResult:
    best: CandidateEvaluation
    history: tuple[LoopRound, ...]
    stop_reason: str
    resolved_plan: PhysicsPlan
```

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_loop_contracts.py -q`

Expected: both tests pass.

- [ ] **Step 5: Commit contracts**

```powershell
git add src/physgenloop/contracts.py tests/test_loop_contracts.py
git commit -m "feat: add physgenloop orchestration contracts"
```

### Task 2: Add component Protocols, fake generator, and Critic adapter

**Files:**
- Create: `src/physgenloop/interfaces.py`
- Create: `src/physgenloop/generator.py`
- Create: `src/physgenloop/critic_adapter.py`
- Modify: `tests/test_loop_contracts.py`

- [ ] **Step 1: Write a failing deterministic generator test**

```python
from pavg_critic.schemas import PhysicsPlan

from physgenloop.generator import DeterministicFakeGenerator


def test_fake_generator_is_deterministic_and_explicitly_fake():
    generator = DeterministicFakeGenerator()
    plan = PhysicsPlan(objects=("ball",), expected_events=("fall",))
    first = generator.generate(prompt="A ball falls.", physics_plan=plan, seed=42)
    second = generator.generate(prompt="A ball falls.", physics_plan=plan, seed=42)
    assert first == second
    assert first.video_path.startswith("fake://")
    assert first.metadata == {"backend": "deterministic_fake", "is_real_video": False}
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `python -m pytest tests/test_loop_contracts.py::test_fake_generator_is_deterministic_and_explicitly_fake -q`

Expected: import fails because `physgenloop.generator` does not exist.

- [ ] **Step 3: Define dependency-injection Protocols**

```python
from typing import Protocol

from pavg_critic.schemas import CriticReport, PhysicsPlan

from .contracts import CandidateEvaluation, GeneratedCandidate


class VideoGenerator(Protocol):
    def generate(
        self, *, prompt: str, physics_plan: PhysicsPlan, seed: int
    ) -> GeneratedCandidate: ...


class CandidateCritic(Protocol):
    def evaluate(
        self, candidate: GeneratedCandidate, *, prompt: str, physics_plan: PhysicsPlan
    ) -> CriticReport: ...


class PromptRepairer(Protocol):
    def repair(self, *, prompt: str, report: CriticReport) -> str: ...


class CandidateSelector(Protocol):
    def select(
        self, evaluations: tuple[CandidateEvaluation, ...]
    ) -> CandidateEvaluation: ...
```

- [ ] **Step 4: Implement the deterministic generator**

```python
import hashlib
import json

from pavg_critic.schemas import PhysicsPlan

from .contracts import GeneratedCandidate


class DeterministicFakeGenerator:
    def generate(
        self, *, prompt: str, physics_plan: PhysicsPlan, seed: int
    ) -> GeneratedCandidate:
        plan_json = json.dumps(
            physics_plan.to_dict(), ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(
            f"{prompt}\0{plan_json}\0{seed}".encode("utf-8")
        ).hexdigest()[:12]
        return GeneratedCandidate(
            candidate_id=f"fake-{digest}",
            video_path=f"fake://{digest}.mp4",
            prompt=prompt,
            seed=seed,
            metadata={"backend": "deterministic_fake", "is_real_video": False},
        )
```

- [ ] **Step 5: Implement the real Critic boundary adapter**

```python
from pavg_critic import CriticRequest, PhysicsCritic
from pavg_critic.schemas import CriticReport, PhysicsPlan

from .contracts import GeneratedCandidate


class PhysicsCriticAdapter:
    def __init__(self, critic: PhysicsCritic) -> None:
        self.critic = critic

    def evaluate(
        self, candidate: GeneratedCandidate, *, prompt: str, physics_plan: PhysicsPlan
    ) -> CriticReport:
        return self.critic.analyze(
            CriticRequest(
                video_path=candidate.video_path,
                prompt=prompt,
                physics_plan=physics_plan,
            )
        )
```

- [ ] **Step 6: Run focused tests**

Run: `python -m pytest tests/test_loop_contracts.py -q`

Expected: all contract and generator tests pass.

- [ ] **Step 7: Commit component boundaries**

```powershell
git add src/physgenloop/interfaces.py src/physgenloop/generator.py src/physgenloop/critic_adapter.py tests/test_loop_contracts.py
git commit -m "feat: add generator and critic boundaries"
```

### Task 3: Add deterministic repair and selection policies

**Files:**
- Create: `src/physgenloop/repairer.py`
- Create: `src/physgenloop/selector.py`
- Create: `tests/test_loop_controller.py`

- [ ] **Step 1: Write failing repair and stable-selection tests**

```python
from pavg_critic.schemas import CriticReport, Violation

from physgenloop.contracts import CandidateEvaluation, GeneratedCandidate
from physgenloop.repairer import InstructionPromptRepairer
from physgenloop.selector import EvidenceAwareSelector


def report(decision: str, score: float, confidence: float, instruction: str = ""):
    violations = ()
    if instruction:
        violations = (Violation(
            object="ball", category="gravity", start_frame=1, peak_frame=2,
            end_frame=3, critical_frames=(1, 2, 3), reason="bad",
            repair_instruction=instruction, evidence={},
        ),)
    return CriticReport(
        is_physical=decision == "physical", decision=decision,
        physics_score=score, confidence=confidence, violations=violations,
    )


def candidate(name: str, item_report: CriticReport):
    generated = GeneratedCandidate(name, f"fake://{name}", "p", 1)
    return CandidateEvaluation(generated, item_report)


def test_repairer_appends_unique_structured_instructions():
    item_report = report("violation", 0.2, 0.9, "Keep the ball falling until contact.")
    repaired = InstructionPromptRepairer().repair(prompt="A ball falls.", report=item_report)
    assert repaired == (
        "A ball falls.\nPhysics correction: Keep the ball falling until contact."
    )


def test_selector_prefers_decision_then_score_and_is_stable():
    first = candidate("first", report("physical", 0.8, 0.6))
    second = candidate("second", report("physical", 0.8, 0.6))
    violation = candidate("bad", report("violation", 0.99, 1.0))
    assert EvidenceAwareSelector().select((first, second, violation)) is first
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_loop_controller.py -q`

Expected: imports fail because repairer and selector modules do not exist.

- [ ] **Step 3: Implement repair-instruction aggregation**

```python
from pavg_critic.schemas import CriticReport


class InstructionPromptRepairer:
    def repair(self, *, prompt: str, report: CriticReport) -> str:
        instructions = tuple(dict.fromkeys(
            item.repair_instruction.strip()
            for item in report.violations
            if item.repair_instruction.strip()
        ))
        if not instructions:
            return prompt
        return f"{prompt}\nPhysics correction: {' '.join(instructions)}"
```

- [ ] **Step 4: Implement stable evidence-aware selection**

```python
from .contracts import CandidateEvaluation


_DECISION_RANK = {"violation": 0, "unknown": 1, "physical": 2}


class EvidenceAwareSelector:
    def select(
        self, evaluations: tuple[CandidateEvaluation, ...]
    ) -> CandidateEvaluation:
        if not evaluations:
            raise ValueError("evaluations must not be empty")
        return max(
            evaluations,
            key=lambda item: (
                _DECISION_RANK[item.report.decision],
                item.report.physics_score,
                item.report.confidence,
            ),
        )
```

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_loop_controller.py -q`

Expected: repair and selector tests pass.

- [ ] **Step 6: Commit policies**

```powershell
git add src/physgenloop/repairer.py src/physgenloop/selector.py tests/test_loop_controller.py
git commit -m "feat: add deterministic repair and selection"
```

### Task 4: Implement the bounded Best-of-K controller

**Files:**
- Create: `src/physgenloop/controller.py`
- Modify: `tests/test_loop_controller.py`

- [ ] **Step 1: Write failing success and max-round tests**

```python
from pavg_critic.planner import PhysicsPlanResolver, TemplatePhysicsPlanner
from pavg_critic.schemas import CriticRequest, PhysicsPlan

from physgenloop.contracts import LoopConfig
from physgenloop.controller import LoopController
from physgenloop.generator import DeterministicFakeGenerator
from physgenloop.repairer import InstructionPromptRepairer
from physgenloop.selector import EvidenceAwareSelector


class ScriptedCritic:
    def __init__(self, reports):
        self.reports = iter(reports)

    def evaluate(self, candidate, *, prompt, physics_plan):
        return next(self.reports)


def test_controller_stops_on_accepted_physical_candidate():
    controller = LoopController(
        generator=DeterministicFakeGenerator(),
        critic=ScriptedCritic((report("physical", 0.9, 0.8),)),
        repairer=InstructionPromptRepairer(),
        selector=EvidenceAwareSelector(),
        config=LoopConfig(max_rounds=3, candidates_per_round=1, acceptance_score=0.8),
    )
    result = controller.run(prompt="A ball falls.", physics_plan=PhysicsPlan())
    assert result.stop_reason == "accepted"
    assert len(result.history) == 1


def test_controller_keeps_global_best_at_max_rounds():
    controller = LoopController(
        generator=DeterministicFakeGenerator(),
        critic=ScriptedCritic((
            report("violation", 0.4, 0.8, "Fix gravity."),
            report("violation", 0.2, 0.9, "Fix contact."),
        )),
        repairer=InstructionPromptRepairer(),
        selector=EvidenceAwareSelector(),
        config=LoopConfig(max_rounds=2, candidates_per_round=1),
    )
    result = controller.run(prompt="A ball falls.", physics_plan=PhysicsPlan())
    assert result.stop_reason == "max_rounds"
    assert result.best.report.physics_score == 0.4
    assert len(result.history) == 2
```

- [ ] **Step 2: Run controller tests and verify RED**

Run: `python -m pytest tests/test_loop_controller.py -k controller -q`

Expected: import fails because `physgenloop.controller` does not exist.

- [ ] **Step 3: Implement the bounded controller**

```python
from pavg_critic.planner import PhysicsPlanResolver, TemplatePhysicsPlanner
from pavg_critic.schemas import CriticRequest, PhysicsPlan

from .contracts import CandidateEvaluation, LoopConfig, LoopResult, LoopRound
from .interfaces import (
    CandidateCritic, CandidateSelector, PlanResolver, PromptRepairer, VideoGenerator,
)


class LoopController:
    def __init__(
        self, *, generator: VideoGenerator, critic: CandidateCritic,
        repairer: PromptRepairer, selector: CandidateSelector,
        plan_resolver: PlanResolver | None = None,
        config: LoopConfig | None = None,
    ) -> None:
        self.generator = generator
        self.critic = critic
        self.repairer = repairer
        self.selector = selector
        self.plan_resolver = plan_resolver or PhysicsPlanResolver(
            TemplatePhysicsPlanner()
        )
        self.config = config or LoopConfig()

    def run(
        self, *, prompt: str, physics_plan: PhysicsPlan | None = None
    ) -> LoopResult:
        explicit_plan = physics_plan if physics_plan is not None else PhysicsPlan()
        resolved_plan = self.plan_resolver.resolve(
            CriticRequest(
                video_path="pending://generation",
                prompt=prompt,
                physics_plan=explicit_plan,
            )
        ).plan
        current_prompt = prompt
        history = []
        round_winners = []
        for round_index in range(self.config.max_rounds):
            evaluations = []
            for offset in range(self.config.candidates_per_round):
                seed = (
                    self.config.base_seed
                    + round_index * self.config.candidates_per_round
                    + offset
                )
                candidate = self.generator.generate(
                    prompt=current_prompt, physics_plan=resolved_plan, seed=seed
                )
                item_report = self.critic.evaluate(
                    candidate, prompt=current_prompt, physics_plan=resolved_plan
                )
                evaluations.append(CandidateEvaluation(candidate, item_report))
            frozen_evaluations = tuple(evaluations)
            selected = self.selector.select(frozen_evaluations)
            round_winners.append(selected)
            history.append(LoopRound(
                round_index=round_index,
                prompt=current_prompt,
                evaluations=frozen_evaluations,
                selected_candidate_id=selected.candidate.candidate_id,
            ))
            if (
                selected.report.decision == "physical"
                and selected.report.physics_score >= self.config.acceptance_score
            ):
                return LoopResult(
                    best=selected, history=tuple(history), stop_reason="accepted",
                    resolved_plan=resolved_plan,
                )
            current_prompt = self.repairer.repair(
                prompt=current_prompt, report=selected.report
            )
        best = self.selector.select(tuple(round_winners))
        return LoopResult(
            best=best, history=tuple(history), stop_reason="max_rounds",
            resolved_plan=resolved_plan,
        )
```

- [ ] **Step 4: Add and run a Best-of-K assertion**

```python
def test_controller_selects_best_of_k_within_one_round():
    controller = LoopController(
        generator=DeterministicFakeGenerator(),
        critic=ScriptedCritic((
            report("violation", 0.3, 0.9, "Fix gravity."),
            report("physical", 0.9, 0.8),
        )),
        repairer=InstructionPromptRepairer(),
        selector=EvidenceAwareSelector(),
        config=LoopConfig(max_rounds=1, candidates_per_round=2),
    )
    result = controller.run(prompt="A ball falls.", physics_plan=PhysicsPlan())
    assert len(result.history[0].evaluations) == 2
    assert result.best.report.decision == "physical"
    assert result.history[0].selected_candidate_id == result.best.candidate.candidate_id
```

Run: `python -m pytest tests/test_loop_controller.py -q`

Expected: all controller, repair, and selector tests pass.

- [ ] **Step 5: Commit controller**

```powershell
git add src/physgenloop/controller.py tests/test_loop_controller.py
git commit -m "feat: add bounded physics feedback loop"
```

### Task 5: Publish the package API and update project documentation

**Files:**
- Create: `src/physgenloop/__init__.py`
- Modify: `README.md`
- Test: `tests/test_package.py`

- [ ] **Step 1: Write a failing public-import test**

```python
def test_physgenloop_public_api_is_importable():
    from physgenloop import (
        DeterministicFakeGenerator,
        EvidenceAwareSelector,
        InstructionPromptRepairer,
        LoopConfig,
        LoopController,
        PhysicsCriticAdapter,
    )

    assert LoopConfig().max_rounds == 3
```

- [ ] **Step 2: Run the import test and verify RED**

Run: `python -m pytest tests/test_package.py::test_physgenloop_public_api_is_importable -q`

Expected: import fails because the public package initializer does not exist.

- [ ] **Step 3: Export the stable orchestration API**

```python
from .contracts import (
    CandidateEvaluation, GeneratedCandidate, LoopConfig, LoopResult, LoopRound,
)
from .controller import LoopController
from .critic_adapter import PhysicsCriticAdapter
from .generator import DeterministicFakeGenerator
from .repairer import InstructionPromptRepairer
from .selector import EvidenceAwareSelector

__all__ = [
    "CandidateEvaluation", "DeterministicFakeGenerator",
    "EvidenceAwareSelector", "GeneratedCandidate",
    "InstructionPromptRepairer", "LoopConfig", "LoopController",
    "LoopResult", "LoopRound", "PhysicsCriticAdapter",
]
```

- [ ] **Step 4: Rewrite README around the latest architecture**

Document the Mermaid flow `Prompt → PhysicsPlan Resolver → Generator → Critic → Repairer → Selector → LoopController`, clearly label current Critic/Planner as implemented, the deterministic loop as scaffolding, and HunyuanVideo/Blender/training as future work. Preserve installation, no-API Critic quick start, model adapter examples, evaluation commands, tests, known limitations, and links to both design documents.

- [ ] **Step 5: Run import and README link checks**

Run:

```powershell
python -m pytest tests/test_package.py tests/test_loop_contracts.py tests/test_loop_controller.py -q
python -c "from pathlib import Path; text=Path('README.md').read_text(encoding='utf-8'); assert 'PhysicsPlan Resolver' in text; assert 'HunyuanVideo' in text"
```

Expected: tests pass and the README assertion exits 0.

- [ ] **Step 6: Commit API and documentation**

```powershell
git add src/physgenloop/__init__.py README.md tests/test_package.py
git commit -m "docs: publish physgenloop architecture scaffold"
```

### Task 6: Full verification

**Files:**
- Verify all Planner and loop-scaffold files.

- [ ] **Step 1: Run the full automated suite**

Run: `python -m pytest -q`

Expected: all tests pass with no failures or errors.

- [ ] **Step 2: Run compile, dependency, JSON, and diff gates**

```powershell
python -m compileall -q src tests benchmarks
python -m pip check
python -m json.tool evaluation/external_manifest.json > $null
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 3: Run the no-API Critic example**

```powershell
python -m pavg_critic --request examples/critic_request.json --observations examples/observations.json --config configs/default.yaml --floor-y 100 --output outputs/planner_example.json
python -c "import json; r=json.load(open('outputs/planner_example.json', encoding='utf-8')); assert r['diagnostics']['planner']['resolved_plan']['expected_events']; print('planner example ok')"
```

Expected: `planner example ok`.

- [ ] **Step 4: Inspect final working tree without changing user-owned files**

Run: `git status --short`

Expected: implementation files are visible alongside any pre-existing user-owned changes; no unrelated file has been reverted or deleted.
