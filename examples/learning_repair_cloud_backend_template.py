"""Copy this NEW adapter file on the isolated cloud host and fill model hooks.

Do not point this template at the four-person team server.  It is intentionally
not imported by the package and cannot run until the placeholders are replaced.
The existing Critic, Generator, and model I/O contracts remain unchanged.
"""

from physgenloop.learning_repair.cloud_campaign import CloudBackendBundle
from physgenloop.learning_repair.executors import (
    ExecutorRegistry,
    GlobalRegenerationExecutor,
    LocalEditingExecutor,
    PromptRepairExecutor,
    RejectExecutor,
)


def build_backend(*, manifest, compatibility):
    """Return cloud objects; replace every placeholder with frozen adapters."""

    # Expected objects:
    # critic.evaluate(candidate, prompt=..., physics_plan=...) -> CriticReport
    # generator.generate(prompt=..., physics_plan=..., seed=...) -> GeneratedCandidate
    # prompt_rewriter.repair(prompt=..., report=...) -> str
    # local_editor.edit(candidate=..., target=..., instruction=..., ...) -> candidate
    # selector.select(tuple[CandidateEvaluation, ...]) -> CandidateEvaluation
    # source_loader(CampaignItem) -> GeneratedCandidate for item.source_video
    # plan_provider(CampaignItem) -> PhysicsPlan
    # semantic_scorer(before_candidate, after_candidate, prompt) -> float in [0, 1]
    # quality_scorer(before_candidate, after_candidate, prompt) -> float in [0, 1]
    raise RuntimeError(
        "Fill examples/learning_repair_cloud_backend_template.py on the isolated "
        "cloud host before running a Blender/Hunyuan campaign."
    )

    # Once the objects above exist, return:
    # return CloudBackendBundle(
    #     critic=critic,
    #     executors=ExecutorRegistry((
    #         PromptRepairExecutor(prompt_rewriter=prompt_rewriter, generator=generator),
    #         GlobalRegenerationExecutor(generator=generator),
    #         # Omit LocalEditingExecutor until a real editor is available; it is masked.
    #         LocalEditingExecutor(editor=local_editor),
    #         RejectExecutor(selector=selector),
    #     )),
    #     source_loader=source_loader,
    #     physics_plan_provider=plan_provider,
    #     semantic_scorer=semantic_scorer,
    #     quality_scorer=quality_scorer,
    # )
