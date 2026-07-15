"""外部模型与 Physics Critic 核心之间的可替换协议。

核心流水线只依赖这些 Protocol，不依赖某个检测框架或 VLM SDK。后续接入模型时
不需要继承基类，只要对象提供相同方法签名即可，因此也便于在测试中注入轻量假实现。
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

from .schemas import (
    CriticRequest,
    Detection,
    Event,
    PhysicsPlan,
    QuestionGraph,
    TrackSequence,
    ViolationCandidate,
    VisualEvidence,
    VLMReview,
)


class ObjectDetector(Protocol):
    """逐帧目标检测协议。

    ``frame_image`` 保持后端原生类型（内置实现为 OpenCV BGR ndarray）；输出必须
    使用统一的 ``Detection``，其中 bbox 坐标约定为 ``[x_min, y_min, x_max, y_max]``。
    """

    def detect(
        self, frame_image: Any, frame_index: int, timestamp_sec: float
    ) -> Sequence[Detection]: ...


class QuestionGraphGenerator(Protocol):
    """从请求中的 prompt/PhysicsPlan 生成原子问题 DAG 的协议。

    第一阶段使用确定性模板实现；后续 VLM QG 只需实现此方法，并将输出交给统一的
    ``QuestionGraphValidator``，即可复用执行、评分和审计层。
    """

    def generate(self, request: CriticRequest) -> QuestionGraph: ...


class PhysicsPlanner(Protocol):
    """把自然语言 prompt 转换为统一 PhysicsPlan 的协议。"""

    def generate(self, prompt: str) -> PhysicsPlan: ...


class StructuredTextModel(Protocol):
    """返回符合调用方 JSON Schema 的文本模型协议。

    OpenAI、DeepSeek 和测试假模型都通过这一窄接口接入，核心图算法不依赖 SDK。
    """

    def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class MultimodalStructuredModel(Protocol):
    """用选定图像证据生成结构化 JSON 的多模态模型协议。"""

    def generate_json_with_images(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_data_urls: Sequence[str],
        schema: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class VisualEvidenceExtractor(Protocol):
    """Grounding/跟踪/光流/外观模型接入五维检查表的协议。"""

    def extract(
        self,
        request: CriticRequest,
        tracks: Sequence[TrackSequence],
        events: Sequence[Event],
    ) -> Sequence[VisualEvidence]: ...


class VLMVerifier(Protocol):
    """候选异常的可选语义复核协议。

    返回 ``None`` 明确表示本次没有 VLM 证据，融合器会仅使用规则分数；不要用 0
    代替“未调用”，否则会被解释成 VLM 明确否定该异常。
    """

    def verify(
        self,
        request: CriticRequest,
        candidate: ViolationCandidate,
        critical_frames: Sequence[int],
    ) -> VLMReview | None: ...
