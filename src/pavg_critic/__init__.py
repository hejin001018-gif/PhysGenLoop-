"""Physics Critic 的稳定公开 API。

业务代码优先从此模块导入配置、请求 schema 和 ``PhysicsCritic``，避免依赖内部模块
布局。未列入 ``__all__`` 的类仍可用于高级扩展，但不承诺在后续版本中保持导入路径。
"""

from .config import CriticConfig, load_config
from .pipeline import PhysicsCritic
from .schemas import (
    CriticReport,
    CriticRequest,
    FrameState,
    NodeResult,
    QuestionGraph,
    QuestionNode,
)

__all__ = [
    "CriticConfig",
    "CriticReport",
    "CriticRequest",
    "FrameState",
    "NodeResult",
    "PhysicsCritic",
    "QuestionGraph",
    "QuestionNode",
    "load_config",
]

# 0.2.0 对应问题图第一阶段；包版本与 pyproject.toml 保持一致，便于报告记录代码版本。
__version__ = "0.3.0"
