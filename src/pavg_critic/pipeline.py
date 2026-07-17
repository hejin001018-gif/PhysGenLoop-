"""Physics Critic 的统一编排流水线。

执行顺序固定为：PhysicsPlan→问题图；视频/外部状态→轨迹→事件→规则候选；随后
由 DAG 执行器将图节点路由到观察、事件或规则验证器，再完成可选 VLM 候选复核与
结果融合。编排器不把某个深度学习框架写死在核心流程中，检测器、问题图生成器和
VLM 都通过 Protocol 注入；其余确定性模块可通过配置调整阈值。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .checklist import VideoScienceChecklistEvaluator
from .config import CriticConfig
from .detector import ColorBlobDetector
from .api_models import ModelAPIError
from .event_detector import EventDetector
from .evidence_fusion import (
    CoverageAwareEvidenceFusion,
    hard_violation_override_applied,
)
from .fusion import ResultFusion
from .interfaces import (
    ObjectDetector,
    PhysicsPlanner,
    QuestionGraphGenerator,
    StructuredTextModel,
    VisualEvidenceExtractor,
    VLMVerifier,
)
from .keyframe_selector import KeyframeSelector
from .mechanics import MechanicsEvaluator
from .physics_rules import PhysicsRuleEngine, RuleContext
from .planner import ModelPhysicsPlanner, PhysicsPlanResolver, TemplatePhysicsPlanner
from .pqsg import HybridQuestionGraphGenerator, PQSGQuestionGraphGenerator
from .question_executor import QuestionExecutionContext, QuestionGraphExecutor
from .question_generator import TemplateQuestionGraphGenerator
from .question_scoring import QuestionGraphScorer
from .question_graph import QuestionGraphError
from .schemas import CriticArtifacts, CriticReport, CriticRequest, FrameState, SchemaError
from .temporal_localizer import TemporalLocalizer
from .tracker import CentroidTracker
from .trajectory import TrajectoryExtractor
from .vlm_verifier import NoOpVLMVerifier, with_track_evidence


class PhysicsCritic:
    """可组合、可审计且具有确定性规则基线的 Physics Critic。"""

    def __init__(
        self,
        config: CriticConfig | None = None,
        *,
        detector: ObjectDetector | None = None,
        question_graph_generator: QuestionGraphGenerator | None = None,
        question_model: StructuredTextModel | None = None,
        physics_planner: PhysicsPlanner | None = None,
        planner_model: StructuredTextModel | None = None,
        visual_evidence_extractors: Iterable[VisualEvidenceExtractor] = (),
        vlm_verifier: VLMVerifier | None = None,
    ) -> None:
        """组装一次可复用的 Critic 实例。

        Args:
            config: 完整配置；缺省时使用经过校验的默认值。
            detector: 可选视觉前端；不注入时使用 HSV 红色目标基线。
            question_graph_generator: 可选问题图生成器；缺省时从 PhysicsPlan 生成模板图。
            question_model: 可选结构化文本模型；提供时自动把 PQSG 模型图融合进模板图。
            physics_planner: 自定义 prompt→PhysicsPlan 实现。
            planner_model: Planner 专用结构化模型；缺省时复用 question_model。
            visual_evidence_extractors: 可选 CV 插件；共享轨迹后输出五维检查表证据。
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
        self.checklist = VideoScienceChecklistEvaluator(self.config.checklist)
        self.mechanics = MechanicsEvaluator(self.config.mechanics)
        self.evidence_fusion = CoverageAwareEvidenceFusion(
            self.config.fusion,
            enabled_rules=self.config.rules.enabled,
        )
        self.visual_evidence_extractors = tuple(visual_evidence_extractors)
        if physics_planner is not None and planner_model is not None:
            raise ValueError("Provide either physics_planner or planner_model, not both")
        # 模型优先级：显式 Planner 实现/专用模型 > 复用 QG 模型 > 确定性模板。
        if physics_planner is not None:
            self.physics_plan_resolver = PhysicsPlanResolver(physics_planner)
        elif planner_model is not None:
            self.physics_plan_resolver = PhysicsPlanResolver(
                ModelPhysicsPlanner(planner_model),
                fallback=TemplatePhysicsPlanner(),
                fallback_on_provider_error=True,
            )
        elif question_model is not None:
            self.physics_plan_resolver = PhysicsPlanResolver(
                ModelPhysicsPlanner(question_model),
                fallback=TemplatePhysicsPlanner(),
                fallback_on_provider_error=True,
            )
        else:
            self.physics_plan_resolver = PhysicsPlanResolver(TemplatePhysicsPlanner())
        if question_graph_generator is not None and question_model is not None:
            raise ValueError(
                "Provide either question_graph_generator or question_model, not both"
            )
        template_generator = TemplateQuestionGraphGenerator(self.config.question_graph)
        self.template_question_graph_generator = template_generator
        self.question_model_enabled = question_model is not None
        if question_graph_generator is not None:
            self.question_graph_generator = question_graph_generator
        elif question_model is not None:
            # PAVG 模板负责已有计划/规则覆盖；PQSG 模型只作为增量问题来源。
            self.question_graph_generator = HybridQuestionGraphGenerator(
                template_generator,
                PQSGQuestionGraphGenerator(question_model),
            )
        else:
            self.question_graph_generator = template_generator
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

        # Planner 必须先于问题图执行；后续所有阶段共享同一个 resolved_request，避免
        # 模板图、规则、力学模块和插件看到互相矛盾的计划版本。
        provider_failures: list[dict[str, object]] = []
        resolution = self.physics_plan_resolver.resolve(request)
        resolved_request = replace(request, physics_plan=resolution.plan)
        request = resolved_request
        if resolution.provider_failure is not None:
            provider_failures.append(resolution.provider_failure)

        # 问题图仅依赖已解析请求，可与视频解码并行；当前同步实现先生成图以便尽早发现
        # 非法外部 QG 输出。关闭图层时保持原 1.0 规则流水线行为。
        if self.config.question_graph.enabled:
            try:
                question_graph = self.question_graph_generator.generate(request)
            except _OPTIONAL_PROVIDER_ERRORS as exc:
                if not self.question_model_enabled:
                    raise
                question_graph = self.template_question_graph_generator.generate(request)
                provider_failures.append(_provider_failure("question_graph", exc))
        else:
            question_graph = None
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
        if self.config.mechanics.enabled:
            mechanics_results, mechanics_summary = self.mechanics.evaluate(
                request=request,
                tracks=tracks,
                events=events,
            )
        else:
            mechanics_results, mechanics_summary = (), None
        context = RuleContext(request=request, tracks=tracks, events=events)
        raw_candidates = self.rule_engine.evaluate(context)
        localized_candidates = tuple(
            self.localizer.localize(item, events) for item in raw_candidates
        )
        if isinstance(self.vlm_verifier, NoOpVLMVerifier):
            candidates = localized_candidates
        else:
            candidates = tuple(
                with_track_evidence(candidate, tracks)
                for candidate in localized_candidates
            )

        visual_evidence = tuple(
            evidence
            for extractor in self.visual_evidence_extractors
            for evidence in extractor.extract(request, tracks, events)
        )

        if self.config.checklist.enabled:
            checklist_results, checklist_summary = self.checklist.evaluate(
                request=request,
                tracks=tracks,
                events=events,
                candidates=candidates,
                external_evidence=visual_evidence,
            )
        else:
            checklist_results, checklist_summary = (), None

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

        reviews = {}
        verify_many = getattr(self.vlm_verifier, "verify_many", None)
        if callable(verify_many) and candidates:
            try:
                batch_reviews = verify_many(request, candidates, keyframes)
                reviews.update(
                    {index: batch_reviews.get(index) for index in range(len(candidates))}
                )
            except _OPTIONAL_PROVIDER_ERRORS as exc:
                reviews.update({index: None for index in range(len(candidates))})
                failure = _provider_failure("vlm_review_batch", exc)
                failure["candidate_count"] = len(candidates)
                provider_failures.append(failure)
        else:
            for index, candidate in enumerate(candidates):
                try:
                    reviews[index] = self.vlm_verifier.verify(
                        request, candidate, keyframes[index]
                    )
                except _OPTIONAL_PROVIDER_ERRORS as exc:
                    reviews[index] = None
                    failure = _provider_failure("vlm_review", exc)
                    failure["candidate_index"] = index
                    failure["category"] = candidate.category
                    provider_failures.append(failure)
        report = self.fusion.fuse(candidates, keyframes, reviews)
        if question_graph is not None:
            report = self.question_scorer.enrich_report(
                report,
                question_graph,
                node_results,
            )
        diagnostics = dict(report.diagnostics)
        metadata = request.physics_plan.planner_metadata
        diagnostics["planner"] = {
            "source": metadata.source,
            "confidence": metadata.confidence,
            "fallback_used": metadata.fallback_used,
            "model": metadata.model,
            "resolved_plan": request.physics_plan.to_dict(),
        }
        if provider_failures:
            diagnostics["provider_failures"] = tuple(provider_failures)
        report = replace(report, diagnostics=diagnostics)
        if checklist_summary is not None:
            # 检查表先作为独立诊断写入；阶段 5 的覆盖感知融合再决定它对总分的权重。
            diagnostics = dict(report.diagnostics)
            diagnostics["video_science"] = {
                "summary": checklist_summary,
                "dimensions": checklist_results,
            }
            score_breakdown = dict(report.score_breakdown)
            if checklist_summary.score is not None:
                score_breakdown["checklist"] = checklist_summary.score
            report = replace(
                report,
                diagnostics=diagnostics,
                score_breakdown=score_breakdown,
            )
        if mechanics_summary is not None:
            diagnostics = dict(report.diagnostics)
            diagnostics["morpheus_mechanics"] = {
                "summary": mechanics_summary,
                "evaluators": mechanics_results,
            }
            score_breakdown = dict(report.score_breakdown)
            if mechanics_summary.score is not None:
                score_breakdown["mechanics"] = mechanics_summary.score
            report = replace(
                report,
                diagnostics=diagnostics,
                score_breakdown=score_breakdown,
            )
        pre_evidence_fusion = {
            "decision": report.decision,
            "physics_score": report.physics_score,
            "confidence": report.confidence,
            "coverage": report.coverage,
        }
        report = self.evidence_fusion.enrich(
            report,
            tracks=tracks,
            candidates=candidates,
            reviews=reviews,
            checklist_summary=checklist_summary,
            mechanics_summary=mechanics_summary,
        )
        diagnostics = dict(report.diagnostics)
        diagnostics["pre_evidence_fusion"] = pre_evidence_fusion
        diagnostics["hard_violation_override"] = hard_violation_override_applied(
            report, self.config.fusion
        )
        report = replace(report, diagnostics=diagnostics)
        return CriticArtifacts(
            report=report,
            tracks=tracks,
            events=events,
            candidates=candidates,
            keyframes={
                index: tuple(frames) for index, frames in keyframes.items()
            },
            reviews=dict(reviews),
            question_graph=question_graph,
            node_results=node_results,
            checklist_results=checklist_results,
            checklist_summary=checklist_summary,
            visual_evidence=visual_evidence,
            mechanics_results=mechanics_results,
            mechanics_summary=mechanics_summary,
            resolved_request=resolved_request,
        )

    def observe_video(self, video_path: str) -> tuple[tuple[FrameState, ...], float]:
        """Run only decode, detection and identity tracking.

        The returned states deliberately contain no derived trajectory, event or floor
        features. Benchmark observation caches use this boundary so later ablations do
        not inherit rule-dependent values from an earlier full critic pass.
        """

        return self._observe_video(video_path)

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


_OPTIONAL_PROVIDER_ERRORS = (
    ModelAPIError,
    TimeoutError,
    ConnectionError,
    OSError,
    SchemaError,
    QuestionGraphError,
    KeyError,
    ValueError,
    TypeError,
)


def _provider_failure(stage: str, error: BaseException) -> dict[str, object]:
    """记录可选 provider 故障，不包含请求 header、密钥或图像正文。"""

    return {
        "stage": stage,
        "error_type": type(error).__name__,
        "message": str(error)[:300],
    }
