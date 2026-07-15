# Prompt PhysicsPlan Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pre-Critic Planner that resolves a prompt and optional partial plan into one validated, auditable PhysicsPlan before question-graph generation and all downstream evaluation.

**Architecture:** Extend the backward-compatible PhysicsPlan schema with relations, constraints, and planner metadata. Add deterministic and structured-model planners behind a resolver; explicit non-empty core fields win, optional provider failures fall back to the template planner, and Pipeline passes one resolved request to every downstream node.

**Tech Stack:** Python 3.10+, frozen dataclasses, Protocol-based dependency injection, JSON Schema 2020-12, pytest, existing OpenAI/DeepSeek `StructuredTextModel` adapters.

---

## File map

- Create `src/pavg_critic/planner.py`: template/model planners, merge logic, resolution result and provider fallback.
- Create `tests/test_planner.py`: schema, template, model, merge and fallback tests.
- Modify `src/pavg_critic/schemas.py`: `PhysicsRelation`, `PhysicsConstraint`, `PlannerMetadata`, extended `PhysicsPlan`, and `CriticArtifacts.resolved_request`.
- Modify `src/pavg_critic/interfaces.py`: `PhysicsPlanner` protocol.
- Modify `src/pavg_critic/pipeline.py`: planner selection, pre-graph resolution, one resolved request for all downstream stages, and diagnostics.
- Modify `src/pavg_critic/pqsg.py`: pass relations and constraints to model QG.
- Modify `src/pavg_critic/__init__.py`: public Planner exports.
- Modify `schemas/critic_output.schema.json`: allow planner diagnostics through the existing diagnostics object; no top-level breaking change.
- Modify `README.md`, `docs/operation-guide.md`, and `examples/critic_request.json`: prompt-only and API Planner usage.

### Task 1: Extend PhysicsPlan schema without breaking old requests

**Files:**
- Modify: `src/pavg_critic/schemas.py:91-108`
- Test: `tests/test_planner.py`

- [ ] **Step 1: Write failing backward-compatibility and extension tests**

```python
from pavg_critic.schemas import (
    PhysicsConstraint,
    PhysicsPlan,
    PhysicsRelation,
    PlannerMetadata,
    SchemaError,
)


def test_old_physics_plan_remains_valid():
    plan = PhysicsPlan.from_dict({
        "objects": ["red_ball"],
        "expected_events": ["fall"],
    })
    assert plan.objects == ("red_ball",)
    assert plan.expected_events == ("fall",)
    assert plan.relations == ()
    assert plan.physics_constraints == ()
    assert plan.planner_metadata.source == "empty"


def test_extended_physics_plan_parses_and_validates_references():
    plan = PhysicsPlan.from_dict({
        "objects": ["red_ball", "floor"],
        "expected_events": ["fall", "floor_contact"],
        "relations": [{
            "id": "R1",
            "subject": "red_ball",
            "relation": "expected_to_collide_with",
            "object": "floor",
        }],
        "physics_constraints": [{
            "id": "C1",
            "domain": "contact",
            "subjects": ["red_ball", "floor"],
            "condition": "during_contact",
            "expectation": "no_interpenetration",
        }],
    })
    plan.validate_references()
    assert plan.relations[0].id == "R1"
    assert plan.physics_constraints[0].domain == "contact"


def test_plan_rejects_unknown_constraint_subject():
    plan = PhysicsPlan.from_dict({
        "objects": ["red_ball"],
        "physics_constraints": [{
            "id": "C1",
            "domain": "contact",
            "subjects": ["red_ball", "floor"],
            "expectation": "no_interpenetration",
        }],
    })
    with pytest.raises(SchemaError, match="floor"):
        plan.validate_references()
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest tests/test_planner.py -q
```

Expected: collection fails because `PhysicsConstraint`, `PhysicsRelation`, and `PlannerMetadata` do not exist.

- [ ] **Step 3: Implement frozen extension dataclasses and parsing**

Add to `schemas.py` before `PhysicsPlan`:

```python
@dataclass(frozen=True)
class PhysicsRelation:
    id: str
    subject: str
    relation: str
    object: str

    def __post_init__(self) -> None:
        if not all(value.strip() for value in (self.id, self.subject, self.relation, self.object)):
            raise SchemaError("physics relation fields must not be empty")


@dataclass(frozen=True)
class PhysicsConstraint:
    id: str
    domain: str
    subjects: tuple[str, ...]
    expectation: str
    condition: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.domain.strip() or not self.expectation.strip():
            raise SchemaError("physics constraint id/domain/expectation must not be empty")
        if not self.subjects or any(not value.strip() for value in self.subjects):
            raise SchemaError("physics constraint requires non-empty subjects")


@dataclass(frozen=True)
class PlannerMetadata:
    source: str = "empty"
    confidence: float = 0.0
    fallback_used: bool = False
    model: str | None = None

    def __post_init__(self) -> None:
        if self.source not in {"explicit", "template", "model", "merged", "template_fallback", "empty"}:
            raise SchemaError(f"invalid planner source: {self.source!r}")
        _score(self.confidence, "planner confidence")
```

Extend `PhysicsPlan`, add `from_dict()` parsing, stable de-duplication, duplicate-ID checks, `validate_references()`, and `to_dict()` using `_jsonable`.

- [ ] **Step 4: Run focused and existing schema tests**

Run:

```powershell
python -m pytest tests/test_planner.py tests/test_foundation.py tests/test_schemas.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit schema extension**

```powershell
git add src/pavg_critic/schemas.py tests/test_planner.py
git commit -m "feat: extend physics plan schema"
```

### Task 2: Add deterministic TemplatePhysicsPlanner

**Files:**
- Create: `src/pavg_critic/planner.py`
- Modify: `src/pavg_critic/interfaces.py`
- Test: `tests/test_planner.py`

- [ ] **Step 1: Write failing English, Chinese and empty-prompt tests**

```python
from pavg_critic.planner import TemplatePhysicsPlanner


@pytest.mark.parametrize(
    "prompt",
    [
        "A red ball falls from a table, hits the floor, and bounces once.",
        "一个红球从桌子上掉落，接触地面后反弹一次。",
    ],
)
def test_template_planner_builds_fall_contact_rebound_plan(prompt):
    plan = TemplatePhysicsPlanner().generate(prompt)
    assert plan.objects == ("red_ball", "table", "floor")
    assert plan.expected_events == (
        "leave_support", "fall", "floor_contact", "rebound"
    )
    assert {item.domain for item in plan.physics_constraints} == {
        "gravity", "contact", "rebound"
    }
    assert plan.planner_metadata.source == "template"


def test_template_planner_empty_prompt_returns_empty_plan():
    plan = TemplatePhysicsPlanner().generate("")
    assert plan.objects == ()
    assert plan.expected_events == ()
    assert plan.planner_metadata.source == "empty"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
python -m pytest tests/test_planner.py -k "template" -q
```

Expected: import fails because `pavg_critic.planner` does not exist.

- [ ] **Step 3: Add protocol and minimal deterministic implementation**

Add to `interfaces.py`:

```python
class PhysicsPlanner(Protocol):
    def generate(self, prompt: str) -> PhysicsPlan: ...
```

Create `planner.py` with ordered bilingual object patterns, event predicates, canonical event order, primary-dynamic-object selection, derived relations, and derived constraints. The implementation must not emit numeric physical parameters.

- [ ] **Step 4: Add projectile and collision tests, then implement mappings**

```python
def test_template_planner_detects_projectile():
    plan = TemplatePhysicsPlanner().generate("A ball is thrown through the air.")
    assert "projectile" in plan.expected_events
    assert any(item.domain == "projectile" for item in plan.physics_constraints)


def test_template_planner_detects_collision():
    plan = TemplatePhysicsPlanner().generate("Two balls collide with each other.")
    assert "collision" in plan.expected_events
    assert any(item.domain == "collision" for item in plan.physics_constraints)
```

Run after implementation:

```powershell
python -m pytest tests/test_planner.py -k "template" -q
```

Expected: all template tests pass.

- [ ] **Step 5: Commit template planner**

```powershell
git add src/pavg_critic/planner.py src/pavg_critic/interfaces.py tests/test_planner.py
git commit -m "feat: add template physics planner"
```

### Task 3: Add ModelPhysicsPlanner and PhysicsPlanResolver

**Files:**
- Modify: `src/pavg_critic/planner.py`
- Test: `tests/test_planner.py`

- [ ] **Step 1: Write failing model parsing and fallback tests**

```python
class FakePlanModel:
    model = "fake-plan-model"

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def generate_json(self, **kwargs):
        self.calls += 1
        if self.error:
            raise self.error
        return self.payload


def test_model_planner_parses_valid_structured_plan():
    model = FakePlanModel(payload={
        "objects": ["red_ball", "floor"],
        "expected_events": ["fall", "floor_contact"],
        "relations": [],
        "physics_constraints": [],
    })
    plan = ModelPhysicsPlanner(model).generate("A red ball falls to the floor.")
    assert plan.objects == ("red_ball", "floor")
    assert plan.planner_metadata.source == "model"
    assert plan.planner_metadata.model == "fake-plan-model"


def test_resolver_falls_back_after_timeout():
    model = FakePlanModel(error=TimeoutError("planner timeout"))
    resolver = PhysicsPlanResolver(
        planner=ModelPhysicsPlanner(model),
        fallback=TemplatePhysicsPlanner(),
        fallback_on_provider_error=True,
    )
    resolution = resolver.resolve(
        CriticRequest(video_path="unused.mp4", prompt="A red ball falls.")
    )
    assert resolution.plan.planner_metadata.source == "template_fallback"
    assert resolution.provider_failure["stage"] == "physics_planner"
```

- [ ] **Step 2: Run focused tests and verify RED**

```powershell
python -m pytest tests/test_planner.py -k "model or resolver" -q
```

Expected: `ModelPhysicsPlanner` and `PhysicsPlanResolver` are undefined.

- [ ] **Step 3: Implement strict model schema and parse boundary**

`ModelPhysicsPlanner.generate()` must call `StructuredTextModel.generate_json()` with a strict object schema requiring `objects`, `expected_events`, `relations`, and `physics_constraints`, then use `PhysicsPlan.from_dict()`, replace metadata with system-assigned model metadata, and call `validate_references()`.

- [ ] **Step 4: Implement resolution, merge, full-plan skip and provider failure record**

```python
@dataclass(frozen=True)
class PhysicsPlanResolution:
    plan: PhysicsPlan
    provider_failure: dict[str, object] | None = None


class PhysicsPlanResolver:
    def resolve(self, request: CriticRequest) -> PhysicsPlanResolution:
        explicit = request.physics_plan
        if explicit.objects and explicit.expected_events:
            explicit.validate_references()
            return PhysicsPlanResolution(_with_metadata(explicit, "explicit", 1.0))
        try:
            generated = self.planner.generate(request.prompt)
        except PROVIDER_ERRORS as error:
            if not self.fallback_on_provider_error:
                raise
            generated = _as_fallback(self.fallback.generate(request.prompt))
            return PhysicsPlanResolution(
                _merge_plans(explicit, generated),
                _failure_record(error),
            )
        return PhysicsPlanResolution(_merge_plans(explicit, generated))
```

- [ ] **Step 5: Add merge-priority and malformed-output tests**

Cover explicit non-empty core fields, extension ID override, `null` arrays, invalid references and empty prompt. Run:

```powershell
python -m pytest tests/test_planner.py -q
```

Expected: all planner tests pass.

- [ ] **Step 6: Commit model planner and resolver**

```powershell
git add src/pavg_critic/planner.py tests/test_planner.py
git commit -m "feat: add model physics plan resolver"
```

### Task 4: Integrate Planner before question graph and expose resolved request

**Files:**
- Modify: `src/pavg_critic/pipeline.py:45-250`
- Modify: `src/pavg_critic/schemas.py:CriticArtifacts`
- Modify: `src/pavg_critic/__init__.py`
- Test: `tests/test_planner.py`
- Test: `tests/test_provider_resilience.py`

- [ ] **Step 1: Write failing model-selection and pipeline-order tests**

```python
def test_pipeline_reuses_question_model_for_planner_when_planner_model_missing():
    model = SequencedModel(plan_payload=PLAN_PAYLOAD, graph_payload={"nodes": []})
    artifacts = PhysicsCritic(question_model=model).analyze_detailed(
        CriticRequest(video_path="unused.mp4", prompt="A red ball falls."),
        observations=(),
        floor_y=100,
    )
    assert model.calls == ["physics_plan", "question_graph"]
    assert artifacts.resolved_request.physics_plan.objects == ("red_ball",)


def test_pipeline_prefers_dedicated_planner_model():
    planner_model = FakePlanModel(payload=PLAN_PAYLOAD)
    question_model = FakeGraphModel({"nodes": []})
    PhysicsCritic(
        planner_model=planner_model,
        question_model=question_model,
    ).analyze_detailed(
        CriticRequest(video_path="unused.mp4", prompt="A red ball falls."),
        observations=(),
        floor_y=100,
    )
    assert planner_model.calls == 1
    assert question_model.calls == 1
```

- [ ] **Step 2: Run integration tests and verify RED**

```powershell
python -m pytest tests/test_planner.py -k "pipeline" -q
```

Expected: `PhysicsCritic` rejects `planner_model` and artifacts lack `resolved_request`.

- [ ] **Step 3: Implement constructor selection and resolver setup**

Add `physics_planner` and `planner_model` keyword arguments. Reject simultaneous `physics_planner` and `planner_model`. Select dedicated model, shared question model, or template planner in that order. Only model-backed resolvers enable provider fallback.

- [ ] **Step 4: Resolve before QG and pass resolved request everywhere**

At the start of `analyze_detailed()`, resolve the plan, replace the request, append any planner provider failure, then use `resolved_request` for graph generation, rule context, checklist, mechanics, VLM and visual evidence extractors.

- [ ] **Step 5: Add Planner diagnostics and artifacts**

Add `resolved_request: CriticRequest | None = None` to `CriticArtifacts`. Attach:

```python
diagnostics["planner"] = {
    "source": plan.planner_metadata.source,
    "confidence": plan.planner_metadata.confidence,
    "fallback_used": plan.planner_metadata.fallback_used,
    "model": plan.planner_metadata.model,
    "resolved_plan": plan.to_dict(),
}
```

- [ ] **Step 6: Run Planner, provider and legacy pipeline tests**

```powershell
python -m pytest tests/test_planner.py tests/test_provider_resilience.py tests/test_rule_baseline.py tests/test_pqsg.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Pipeline integration**

```powershell
git add src/pavg_critic/pipeline.py src/pavg_critic/schemas.py src/pavg_critic/__init__.py tests/test_planner.py tests/test_provider_resilience.py
git commit -m "feat: resolve physics plans before critic"
```

### Task 5: Feed extended plan context into PQSG

**Files:**
- Modify: `src/pavg_critic/pqsg.py:61-99`
- Test: `tests/test_pqsg.py`

- [ ] **Step 1: Write a failing QG prompt-context test**

```python
def test_pqsg_generator_receives_relations_and_constraints():
    model = FakeStructuredModel({"nodes": []})
    request = CriticRequest(
        video_path="unused.mp4",
        prompt="A ball hits the floor.",
        physics_plan=EXTENDED_PLAN,
    )
    PQSGQuestionGraphGenerator(model).generate(request)
    user_payload = json.loads(model.calls[0][1])
    assert user_payload["relations"][0]["relation"] == "expected_to_collide_with"
    assert user_payload["physics_constraints"][0]["domain"] == "contact"
```

- [ ] **Step 2: Run and verify RED**

```powershell
python -m pytest tests/test_pqsg.py::test_pqsg_generator_receives_relations_and_constraints -q
```

Expected: `relations` is missing from the model input payload.

- [ ] **Step 3: Serialize extended context in PQSG user prompt**

Use `PhysicsPlan.to_dict()` and exclude planner metadata from semantic QG input, leaving objects, events, relations and constraints.

- [ ] **Step 4: Run PQSG tests and commit**

```powershell
python -m pytest tests/test_pqsg.py -q
git add src/pavg_critic/pqsg.py tests/test_pqsg.py
git commit -m "feat: ground PQSG in extended physics plan"
```

### Task 6: Update schema contract, examples and operations guide

**Files:**
- Modify: `examples/critic_request.json`
- Modify: `README.md`
- Modify: `docs/operation-guide.md`
- Modify: `schemas/README.md`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Add a prompt-only request fixture and CLI integration test**

Update the example so it can omit `physics_plan`, and add a test that loads the request, runs with observations, and asserts non-empty planner diagnostics and graph nodes.

- [ ] **Step 2: Document no-API and API Planner calls**

Document:

```python
PhysicsCritic()  # Template Planner
PhysicsCritic(question_model=model)  # shared Planner + PQSG model
PhysicsCritic(planner_model=planner_model, question_model=qg_model)
```

Explain that a shared model causes two API calls when the plan core is incomplete, while a complete explicit core skips the Planner call.

- [ ] **Step 3: Run schema and example tests**

```powershell
python -m pytest tests/test_schemas.py tests/test_foundation.py tests/test_planner.py -q
python -m pavg_critic --request examples/critic_request.json --observations examples/observations.json --config configs/default.yaml --floor-y 100 --output outputs/planner_example.json
```

Expected: tests pass and output contains `diagnostics.planner.resolved_plan`.

- [ ] **Step 4: Commit documentation and examples**

```powershell
git add README.md docs/operation-guide.md schemas/README.md examples/critic_request.json tests
git commit -m "docs: document prompt physics planner"
```

### Task 7: Full verification and review

**Files:**
- Verify all modified files

- [ ] **Step 1: Run full test and compile gates**

```powershell
python -m pytest -q
python -m compileall -q src tests benchmarks
python -m pip check
git diff --check
```

Expected: all tests pass, compile/pip/diff checks exit 0.

- [ ] **Step 2: Run no-API example and validate schema**

```powershell
python -m pavg_critic --request examples/critic_request.json --observations examples/observations.json --config configs/default.yaml --floor-y 100 --output outputs/planner_example.json
python -c "import json,jsonschema; s=json.load(open('schemas/critic_output.schema.json')); r=json.load(open('outputs/planner_example.json')); jsonschema.validate(r,s); assert r['diagnostics']['planner']['resolved_plan']['expected_events']; print('planner schema ok')"
```

Expected: `planner schema ok`.

- [ ] **Step 3: Request independent code review**

Review from the design commit through HEAD for explicit-plan precedence, model fallback isolation, one resolved request across downstream consumers, schema compatibility, API security and test completeness.

- [ ] **Step 4: Fix all Critical/Important review findings with red-green tests**

For every valid finding, add a focused failing test, run it to confirm RED, implement the minimum fix, then run the focused and full suites.

- [ ] **Step 5: Record final branch state**

```powershell
git status --short
git log --oneline -8
```

Expected: only pre-existing user-owned changes remain outside the implementation worktree; feature commits are present on the isolated branch.
