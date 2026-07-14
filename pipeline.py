"""Physics Critic 的统一编排流水线。

执行顺序固定为：PhysicsPlan→问题图；视频/外部状态→轨迹→事件→规则候选；随后
由 DAG 执行器将图节点路由到观察、事件或规则验证器，再完成可选 VLM 候选复核与
结果融合。编排器不把某个深度学习框架写死在核心流程中，检测器、问题图生成器和
VLM 都通过 Protocol 注入；其余确定性模块可通过配置调整阈值。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .config import CriticConfig
from .detector import ColorBlobDetector
from .event_detector import EventDetector
from .fusion import ResultFusion
from .interfaces import ObjectDetector, QuestionGraphGenerator, VLMVerifier
from .keyframe_selector import KeyframeSelector
from .physics_rules import PhysicsRuleEngine, RuleContext
from .question_executor import QuestionExecutionContext, QuestionGraphExecutor
from .question_generator import TemplateQuestionGraphGenerator
from .question_scoring import QuestionGraphScorer
from .schemas import CriticArtifacts, CriticReport, CriticRequest, FrameState
from .temporal_localizer import TemporalLocalizer
from .tracker import CentroidTracker
from .trajectory import TrajectoryExtractor
from .vlm_verifier import NoOpVLMVerifier


class PhysicsCritic:
    """可组合、可审计且具有确定性规则基线的 Physics Critic。"""

    def __init__(
        self,
        config: CriticConfig | None = None,
        *,
        detector: ObjectDetector | None = None,
        question_graph_generator: QuestionGraphGenerator | None = None,
        vlm_verifier: VLMVerifier | None = None,
    ) -> None:
        """组装一次可复用的 Critic 实例。

        Args:
            config: 完整配置；缺省时使用经过校验的默认值。
            detector: 可选视觉前端；不注入时使用 HSV 红色目标基线。
            question_graph_generator: 可选问题图生成器；缺省时从 PhysicsPlan 生成模板图。
            vlm_verifier: 可选 Video-VLM 复核器；缺省时明确跳过 VLM。
        """

        self.config = config or CriticConfig()
        self.config.validate()
        # 各组件均无跨请求业务状态，同一实例可顺序分析多个视频。
        self.detector = detector or self._default_detector()
        self.tracker = CentroidTracker(self.config.tracker)
        self.trajectory = TrajectoryExtractor(self.config.trajectory)
        self.event_detector = EventDetector(self.config.events)
        self.rule_engine = PhysicsRuleEngine(self.config.rules, self.config.events)
        self.localizer = TemporalLocalizer()
        self.keyframe_selector = KeyframeSelector(self.config.temporal)
        self.question_graph_generator = (
            question_graph_generator
            or TemplateQuestionGraphGenerator(self.config.question_graph)
        )
        self.question_executor = QuestionGraphExecutor(
            enabled_rule_categories=self.config.rules.enabled,
            rule_pass_confidence=self.config.question_graph.rule_pass_confidence,
        )
        self.question_scorer = QuestionGraphScorer()
        self.vlm_verifier = vlm_verifier or NoOpVLMVerifier()
        self.fusion = ResultFusion(self.config.fusion)

    def analyze(
        self,
        request: CriticRequest,
        *,
        observations: Iterable[FrameState] | None = None,
        floor_y: float | None = None,
    ) -> CriticReport:
        """返回精简公开报告。

        传入 ``observations`` 时跳过视频解码和检测/跟踪，适合 Blender 真值、缓存轨迹
        和单元测试；否则读取 ``request.video_path`` 执行完整视觉前端。
        """

        # 精简入口复用 detailed 流程，保证两种 API 不会产生判定差异。
        return self.analyze_detailed(
            request, observations=observations, floor_y=floor_y
        ).report

    def analyze_detailed(
        self,
        request: CriticRequest,
        *,
        observations: Iterable[FrameState] | None = None,
        floor_y: float | None = None,
    ) -> CriticArtifacts:
        """执行完整流水线，并额外返回富化轨迹和离散事件供调试审计。"""

        # 问题图仅依赖请求计划，可与视频解码并行；当前同步实现先生成图以便尽早发现
        # 非法外部 QG 输出。关闭图层时保持原 1.0 规则流水线行为。
        question_graph = (
            self.question_graph_generator.generate(request)
            if self.config.question_graph.enabled
            else None
        )
        if observations is None:
            # 视频路径会同时推断画面高度对应的地面像素坐标。
            states, inferred_floor = self._observe_video(request.video_path)
            floor_y = inferred_floor if floor_y is None else floor_y
        else:
            # 立即物化迭代器，避免后续多个阶段读取时数据已经被消费。
            states = tuple(observations)

        # 以下阶段保持显式顺序，每个中间对象均可序列化并在 artifacts 中审计。
        tracks = self.trajectory.extract(states, floor_y=floor_y)
        events = self.event_detector.detect(tracks)
        context = RuleContext(request=request, tracks=tracks, events=events)
        raw_candidates = self.rule_engine.evaluate(context)
        candidates = tuple(self.localizer.localize(item, events) for item in raw_candidates)

        # 用候选索引关联关键帧与 VLM 结果，支持同对象同类别出现多个异常区间。
        keyframes = {
            index: self.keyframe_selector.select(candidate, tracks)
            for index, candidate in enumerate(candidates)
        }

        if question_graph is not None:
            # 第一阶段节点验证复用同一批轨迹、事件、规则候选和关键帧，不重复运行视觉
            # 前端。节点结果在融合后附加到报告，不改变原有最强违规风险的判定语义。
            node_results = self.question_executor.execute(
                question_graph,
                QuestionExecutionContext(
                    tracks=tracks,
                    events=events,
                    candidates=candidates,
                    candidate_keyframes=keyframes,
                ),
            )
        else:
            node_results = ()

        reviews = {
            index: self.vlm_verifier.verify(request, candidate, keyframes[index])
            for index, candidate in enumerate(candidates)
        }
        report = self.fusion.fuse(candidates, keyframes, reviews)
        if question_graph is not None:
            report = self.question_scorer.enrich_report(
                report,
                question_graph,
                node_results,
            )
        return CriticArtifacts(
            report=report,
            tracks=tracks,
            events=events,
            question_graph=question_graph,
            node_results=node_results,
        )

    def _observe_video(self, video_path: str) -> tuple[tuple[FrameState, ...], float]:
        """解码视频、逐帧检测并跟踪，返回原始状态和推断地面 y 坐标。"""

        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        # 与 detector.py 一致采用延迟导入，观察值模式不需要安装 OpenCV。
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError(
                "Video analysis requires optional dependencies; run "
                "`pip install -e .[video]`."
            ) from exc

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            # 某些容器不写 FPS 元数据；30 FPS 是可预测的回退值并会影响速度尺度。
            fps = 30.0
        timed_detections = []
        height: int | None = None
        index = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                # 高度只需从首个可解码帧读取，视频流正常情况下分辨率保持不变。
                height = int(image.shape[0]) if height is None else height
                timestamp = index / fps
                detections = self.detector.detect(image, index, timestamp)
                timed_detections.append((index, timestamp, detections))
                index += 1
        finally:
            # 即使检测器抛出异常也释放系统视频句柄。
            capture.release()
        if height is None:
            raise RuntimeError(f"Video contains no decodable frames: {path}")
        # 先缓存轻量 Detection 而不是原始图像，控制长视频分析的内存占用。
        states = self.tracker.track_timed(timed_detections)
        floor_y = height * self.config.detector.floor_y_ratio
        return states, floor_y

    def _default_detector(self) -> ObjectDetector:
        """根据配置创建内置检测器，未知后端要求调用方显式注入实现。"""

        if self.config.detector.backend != "color_blob":
            raise ValueError(
                f"No built-in detector backend {self.config.detector.backend!r}; "
                "inject an ObjectDetector when constructing PhysicsCritic."
            )
        return ColorBlobDetector(self.config.detector)
