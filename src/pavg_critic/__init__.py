"""Physics Critic 的稳定公开 API。

业务代码优先从此模块导入配置、请求 schema 和 ``PhysicsCritic``，避免依赖内部模块
布局。未列入 ``__all__`` 的类仍可用于高级扩展，但不承诺在后续版本中保持导入路径。
"""

from .api_models import DeepSeekChatModel, OpenAIChatModel, OpenAIResponsesModel
from .config import CriticConfig, load_config
from .execution_trace import (
    TraceRecorder,
    TraceValidationPolicy,
    TraceValidationReport,
    validate_trace,
)
from .pipeline import PhysicsCritic
from .planner import ModelPhysicsPlanner, PhysicsPlanResolver, TemplatePhysicsPlanner
from .pqsg import HybridQuestionGraphGenerator, PQSGQuestionGraphGenerator
from .schemas import (
    CriticReport,
    CriticRequest,
    FrameState,
    NodeResult,
    PhysicsConstraint,
    PhysicsPlan,
    PhysicsRelation,
    PlannerMetadata,
    QuestionGraph,
    QuestionNode,
)
from .sam2_detector import SAM2ObjectDetector
from .vlm_detector import VLMObjectDetector
from .vlm_verifier import EvidenceGroundedVLMVerifier

__all__ = [
    "CriticConfig",
    "CriticReport",
    "CriticRequest",
    "DeepSeekChatModel",
    "EvidenceGroundedVLMVerifier",
    "FrameState",
    "HybridQuestionGraphGenerator",
    "NodeResult",
    "OpenAIChatModel",
    "OpenAIResponsesModel",
    "ModelPhysicsPlanner",
    "PhysicsConstraint",
    "PhysicsCritic",
    "PhysicsPlan",
    "PhysicsPlanResolver",
    "PhysicsRelation",
    "PlannerMetadata",
    "PQSGQuestionGraphGenerator",
    "QuestionGraph",
    "SAM2ObjectDetector",
    "QuestionNode",
    "TemplatePhysicsPlanner",
    "TraceRecorder",
    "TraceValidationPolicy",
    "TraceValidationReport",
    "VLMObjectDetector",
    "load_config",
    "validate_trace",
]

# 包版本与 pyproject.toml 保持一致，便于实验报告记录确切实现版本。
__version__ = "0.3.0"
