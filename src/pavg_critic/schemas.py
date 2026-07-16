"""Physics Critic 各阶段共享的 schema 2.0 数据模型。

这里刻意只使用标准库 dataclass，保证 Blender 真值、离线评估脚本和在线服务都能
在不安装深度学习依赖的情况下读取同一格式。外部输入统一通过 ``from_dict`` 进入，
内部计算则传递不可变对象，避免某一阶段原地修改证据导致审计结果不可复现。

坐标约定：bbox 为 ``[x_min, y_min, x_max, y_max]``，图像 y 轴向下为正；速度单位
默认为像素/秒。分数和置信度统一限制在闭区间 [0, 1]。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

# 2.0 将问题图、覆盖率和三态决策纳入稳定报告，同时继续读取 1.x 请求/观察值。
SCHEMA_VERSION = "2.0"
SUPPORTED_SCHEMA_VERSIONS = frozenset({"1.0", "1.1", SCHEMA_VERSION})
DECISIONS = ("physical", "violation", "unknown")
CHECKLIST_DIMENSIONS = (
    "phenomenon_congruency",
    "correct_dynamism",
    "spatiotemporal_continuity",
    "immutability",
    "interaction_realism",
)
CHECKLIST_STATUSES = ("pass", "fail", "unknown")
VLM_CLAIM_STATUSES = ("confirmed", "rejected", "uncertain")
MECHANICS_EVALUATORS = ("freefall", "projectile", "rebound", "collision")
MECHANICS_APPLICABILITY = ("applicable", "not_applicable", "failed")
EVIDENCE_FAMILIES = ("rules", "pqsg", "checklist", "mechanics", "vlm")
EVIDENCE_STATUSES = ("available", "unknown", "not_applicable", "failed")

# 字符串常量比 Enum 更容易与外部 JSON、Blender 脚本及不同模型 SDK 互操作。
QUESTION_CATEGORIES = ("object", "action", "physics")
NODE_STATUSES = ("yes", "no", "blocked", "unknown")
VERIFIER_HINTS = ("observation", "event", "rule", "hybrid")


class SchemaError(ValueError):
    """外部数据不符合受支持 schema 时抛出的可识别异常。"""


def _score(value: float, name: str) -> float:
    """将数值转为 float，并统一执行 [0, 1] 范围检查。"""

    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise SchemaError(f"{name} must be in [0, 1], got {result}")
    return result


def _version(value: Any) -> str:
    """接受当前版本和可迁移的 1.0 输入，同时拒绝未知未来格式。"""

    result = str(value or SCHEMA_VERSION)
    if result not in SUPPORTED_SCHEMA_VERSIONS:
        raise SchemaError(
            f"Unsupported schema_version {result!r}; "
            f"supported versions are {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    return result


def _tuple_of_numbers(value: Iterable[Any], length: int, name: str) -> tuple[float, ...]:
    """规范化外部数值数组，并检查中心点、bbox 等字段的固定维数。"""

    result = tuple(float(item) for item in value)
    if len(result) != length:
        raise SchemaError(f"{name} must have {length} numbers")
    return result


def _jsonable(value: Any) -> Any:
    """递归转换 dataclass/元组，并删除值为 None 的可选字段。

    删除 ``None`` 能让报告保持紧凑，但会保留 0、False 和空列表等具有明确语义的值。
    """

    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items() if item is not None}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _stable_strings(values: Iterable[Any]) -> tuple[str, ...]:
    """清理字符串列表，并按首次出现顺序去重。"""

    result: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw).strip()
        if not value:
            raise SchemaError("physics plan string values must not be empty")
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class PhysicsRelation:
    """PhysicsPlan 中两个对象之间的具名物理关系。"""

    id: str
    subject: str
    relation: str
    object: str

    def __post_init__(self) -> None:
        for name in ("id", "subject", "relation", "object"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise SchemaError("physics relation fields must not be empty")
            object.__setattr__(self, name, value)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhysicsRelation":
        try:
            return cls(
                id=str(data["id"]),
                subject=str(data["subject"]),
                relation=str(data["relation"]),
                object=str(data["object"]),
            )
        except KeyError as exc:
            raise SchemaError(f"physics relation is missing {exc.args[0]!r}") from exc


@dataclass(frozen=True)
class PhysicsConstraint:
    """待验证的物理约束；只表达语义，不猜测数值参数。"""

    id: str
    domain: str
    subjects: tuple[str, ...]
    expectation: str
    condition: str | None = None

    def __post_init__(self) -> None:
        for name in ("id", "domain", "expectation"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise SchemaError("physics constraint id/domain/expectation must not be empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "subjects", _stable_strings(self.subjects))
        if not self.subjects:
            raise SchemaError("physics constraint requires non-empty subjects")
        if self.condition is not None:
            condition = str(self.condition).strip()
            object.__setattr__(self, "condition", condition or None)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhysicsConstraint":
        try:
            return cls(
                id=str(data["id"]),
                domain=str(data["domain"]),
                subjects=tuple(data.get("subjects") or ()),
                expectation=str(data["expectation"]),
                condition=(None if data.get("condition") is None else str(data["condition"])),
            )
        except KeyError as exc:
            raise SchemaError(f"physics constraint is missing {exc.args[0]!r}") from exc


@dataclass(frozen=True)
class PlannerMetadata:
    """系统分配的 PhysicsPlan 来源与降级审计信息。"""

    source: str = "empty"
    confidence: float = 0.0
    fallback_used: bool = False
    model: str | None = None

    def __post_init__(self) -> None:
        allowed = {"explicit", "template", "model", "merged", "template_fallback", "empty"}
        if self.source not in allowed:
            raise SchemaError(f"invalid planner source: {self.source!r}")
        object.__setattr__(self, "confidence", _score(self.confidence, "planner confidence"))
        if self.model is not None:
            model = str(self.model).strip()
            object.__setattr__(self, "model", model or None)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PlannerMetadata":
        data = data or {}
        return cls(
            source=str(data.get("source", "empty")),
            confidence=float(data.get("confidence", 0.0)),
            fallback_used=bool(data.get("fallback_used", False)),
            model=None if data.get("model") is None else str(data["model"]),
        )


@dataclass(frozen=True)
class PhysicsPlan:
    """Planner 输出的对象、事件、关系、约束及可审计来源。"""

    objects: tuple[str, ...] = ()
    expected_events: tuple[str, ...] = ()
    relations: tuple[PhysicsRelation, ...] = ()
    physics_constraints: tuple[PhysicsConstraint, ...] = ()
    planner_metadata: PlannerMetadata = PlannerMetadata()

    def __post_init__(self) -> None:
        object.__setattr__(self, "objects", _stable_strings(self.objects))
        object.__setattr__(self, "expected_events", _stable_strings(self.expected_events))
        self._validate_unique_ids(self.relations, "physics relation")
        self._validate_unique_ids(self.physics_constraints, "physics constraint")

    @staticmethod
    def _validate_unique_ids(items: Iterable[Any], label: str) -> None:
        seen: set[str] = set()
        for item in items:
            if item.id in seen:
                raise SchemaError(f"duplicate {label} id: {item.id!r}")
            seen.add(item.id)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "PhysicsPlan":
        """读取旧版或扩展版计划；缺失及 null 数组均按空集合处理。"""

        data = data or {}
        objects = data.get("objects") or ()
        events = data.get("expected_events") or ()
        if not isinstance(objects, (list, tuple)):
            raise SchemaError("physics plan objects must be an array")
        if not isinstance(events, (list, tuple)):
            raise SchemaError("physics plan expected_events must be an array")
        relations = data.get("relations") or ()
        constraints = data.get("physics_constraints") or ()
        if not all(isinstance(item, Mapping) for item in relations):
            raise SchemaError("physics plan relations must contain JSON objects")
        if not all(isinstance(item, Mapping) for item in constraints):
            raise SchemaError("physics plan physics_constraints must contain JSON objects")
        metadata = data.get("planner_metadata")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise SchemaError("planner_metadata must be a JSON object")
        return cls(
            objects=_stable_strings(objects),
            expected_events=_stable_strings(events),
            relations=tuple(PhysicsRelation.from_dict(item) for item in relations),
            physics_constraints=tuple(
                PhysicsConstraint.from_dict(item) for item in constraints
            ),
            planner_metadata=PlannerMetadata.from_dict(metadata),
        )

    def validate_references(self) -> None:
        """确保关系与约束只引用计划对象，避免下游静默丢失目标。"""

        known = set(self.objects)
        for relation in self.relations:
            unknown = {relation.subject, relation.object} - known
            if unknown:
                raise SchemaError(
                    f"physics relation {relation.id!r} references unknown objects: "
                    f"{sorted(unknown)}"
                )
        for constraint in self.physics_constraints:
            unknown = set(constraint.subjects) - known
            if unknown:
                raise SchemaError(
                    f"physics constraint {constraint.id!r} references unknown objects: "
                    f"{sorted(unknown)}"
                )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class QuestionNode:
    """问题图中的一个原子验证问题。

    ``parent_ids`` 表达回答本节点前必须满足的逻辑前置条件；``target_objects`` 和
    ``expected_events`` 为确定性验证器提供机器可读目标；``rule_ids`` 将 Physics
    问题连接到现有规则 ID。问题文本供人类和未来 VLM 使用，但第一阶段执行不从自然
    语言反向解析语义。
    """

    id: str
    category: str
    question: str
    parent_ids: tuple[str, ...] = ()
    target_objects: tuple[str, ...] = ()
    expected_events: tuple[str, ...] = ()
    physics_domain: str | None = None
    verifier_hint: str = "hybrid"
    rule_ids: tuple[str, ...] = ()
    weight: float = 1.0

    def __post_init__(self) -> None:
        # 节点 ID 是图边和审计输出的稳定主键，禁止空字符串。
        if not self.id.strip():
            raise SchemaError("question node id must not be empty")
        if self.category not in QUESTION_CATEGORIES:
            raise SchemaError(
                f"question category must be one of {QUESTION_CATEGORIES}, got {self.category!r}"
            )
        if not self.question.strip():
            raise SchemaError("question text must not be empty")
        if self.verifier_hint not in VERIFIER_HINTS:
            raise SchemaError(
                f"verifier_hint must be one of {VERIFIER_HINTS}, got {self.verifier_hint!r}"
            )
        if self.weight <= 0:
            raise SchemaError("question node weight must be positive")


@dataclass(frozen=True)
class QuestionGraph:
    """由原子问题和 ``parent_ids`` 隐式边组成的有向无环图。

    dataclass 只验证局部字段；跨节点的引用、边类型与环路检查由
    ``QuestionGraphValidator`` 统一执行，避免 schema 层依赖图算法实现。
    """

    nodes: tuple[QuestionNode, ...] = ()
    source: str = "physics_plan_template"

    def __post_init__(self) -> None:
        if not self.source.strip():
            raise SchemaError("question graph source must not be empty")


@dataclass(frozen=True)
class NodeResult:
    """一个问题节点的可审计执行结果。

    ``direct_score`` 只对真正执行的 yes/no 节点有值；blocked 表示前置条件未满足，
    unknown 表示验证器能力或证据不足。二者都不会被伪装成物理问题的直接 No，评分器
    会根据 prompt fulfillment 与 physics plausibility 的不同语义分别处理。
    """

    node_id: str
    category: str
    status: str
    direct_score: float | None
    confidence: float
    reason: str
    verifier: str
    critical_frames: tuple[int, ...] = ()
    rule_ids: tuple[str, ...] = ()
    blocked_by: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in QUESTION_CATEGORIES:
            raise SchemaError(f"invalid node result category: {self.category!r}")
        if self.status not in NODE_STATUSES:
            raise SchemaError(f"node status must be one of {NODE_STATUSES}")
        _score(self.confidence, "node result confidence")
        if self.direct_score is not None:
            _score(self.direct_score, "node direct_score")
        if self.status in {"yes", "no"} and self.direct_score is None:
            raise SchemaError("answered nodes require direct_score")
        if self.status in {"blocked", "unknown"} and self.direct_score is not None:
            raise SchemaError("blocked/unknown nodes must not have direct_score")
        if self.status == "blocked" and not self.blocked_by:
            raise SchemaError("blocked nodes require at least one blocked_by parent")


@dataclass(frozen=True)
class CategoryEvaluation:
    """一个问题类别的直接评分、prompt fulfillment 与覆盖统计。"""

    category: str
    score: float | None
    fulfillment_score: float
    coverage: float
    total: int
    answered: int
    yes: int
    no: int
    blocked: int
    unknown: int

    def __post_init__(self) -> None:
        if self.category not in QUESTION_CATEGORIES:
            raise SchemaError(f"invalid category evaluation: {self.category!r}")
        if self.score is not None:
            _score(self.score, "category score")
        _score(self.fulfillment_score, "category fulfillment_score")
        _score(self.coverage, "category coverage")
        counts = (self.total, self.answered, self.yes, self.no, self.blocked, self.unknown)
        if any(value < 0 for value in counts):
            raise SchemaError("category counts must be non-negative")
        if self.yes + self.no != self.answered:
            raise SchemaError("category answered count must equal yes + no")
        if self.answered + self.blocked + self.unknown != self.total:
            raise SchemaError("category result counts must sum to total")


@dataclass(frozen=True)
class GraphEvaluationSummary:
    """问题图的全局评分摘要。

    ``prompt_fulfillment_score`` 按 PQSG 语义将 blocked/unknown 视为未满足；
    ``physics_plausibility_score`` 只统计实际回答的 Physics 节点，并必须结合
    ``physics_coverage`` 解读，防止少量可回答节点造成虚高分。
    """

    prompt_fulfillment_score: float
    physics_plausibility_score: float | None
    question_coverage: float
    physics_coverage: float
    categories: dict[str, CategoryEvaluation]
    root_failure_nodes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _score(self.prompt_fulfillment_score, "prompt_fulfillment_score")
        if self.physics_plausibility_score is not None:
            _score(self.physics_plausibility_score, "physics_plausibility_score")
        _score(self.question_coverage, "question_coverage")
        _score(self.physics_coverage, "physics_coverage")


@dataclass(frozen=True)
class VisualEvidence:
    """外部 CV 工具写入检查表的统一证据。

    ``measurements`` 可保存光流残差、掩膜 IoU、外观相似度等后端专属数值，但判定
    层只依赖归一化 score/confidence 和固定维度名。
    """

    dimension: str
    source: str
    score: float
    confidence: float
    critical_frames: tuple[int, ...] = ()
    measurements: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dimension not in CHECKLIST_DIMENSIONS:
            raise SchemaError(f"invalid visual evidence dimension: {self.dimension!r}")
        if not self.source.strip():
            raise SchemaError("visual evidence source must not be empty")
        _score(self.score, "visual evidence score")
        _score(self.confidence, "visual evidence confidence")


@dataclass(frozen=True)
class ChecklistResult:
    """一个 VideoScience 评估维度的证据化判定。"""

    dimension: str
    status: str
    score: float | None
    confidence: float
    reason: str
    critical_frames: tuple[int, ...] = ()
    evidence_sources: tuple[str, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dimension not in CHECKLIST_DIMENSIONS:
            raise SchemaError(f"invalid checklist dimension: {self.dimension!r}")
        if self.status not in CHECKLIST_STATUSES:
            raise SchemaError(f"invalid checklist status: {self.status!r}")
        if self.score is not None:
            _score(self.score, "checklist score")
        if self.status == "unknown" and self.score is not None:
            raise SchemaError("unknown checklist result must not have a score")
        if self.status != "unknown" and self.score is None:
            raise SchemaError("pass/fail checklist result requires a score")
        _score(self.confidence, "checklist confidence")


@dataclass(frozen=True)
class ChecklistSummary:
    """五维检查表的覆盖感知摘要；unknown 不作为零分参与均值。"""

    score: float | None
    coverage: float
    passed: int
    failed: int
    unknown: int

    def __post_init__(self) -> None:
        if self.score is not None:
            _score(self.score, "checklist summary score")
        _score(self.coverage, "checklist coverage")
        if min(self.passed, self.failed, self.unknown) < 0:
            raise SchemaError("checklist counts must be non-negative")
        if self.passed + self.failed + self.unknown != len(CHECKLIST_DIMENSIONS):
            raise SchemaError("checklist counts must cover all dimensions")


@dataclass(frozen=True)
class MechanicsResult:
    """一个力学假设的门控状态、双分数与可审计指标。"""

    evaluator: str
    applicability: str
    invariance_score: float | None
    dynamical_score: float | None
    score: float | None
    is_plausible: bool | None
    reason: str
    critical_frames: tuple[int, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evaluator not in MECHANICS_EVALUATORS:
            raise SchemaError(f"invalid mechanics evaluator: {self.evaluator!r}")
        if self.applicability not in MECHANICS_APPLICABILITY:
            raise SchemaError(f"invalid mechanics applicability: {self.applicability!r}")
        scores = (self.invariance_score, self.dynamical_score, self.score)
        if self.applicability == "applicable":
            if any(value is None for value in scores) or self.is_plausible is None:
                raise SchemaError("applicable mechanics result requires scores and decision")
        elif any(value is not None for value in scores) or self.is_plausible is not None:
            raise SchemaError("non-applicable/failed mechanics result must not contain scores")
        for value in scores:
            if value is not None:
                _score(value, "mechanics score")


@dataclass(frozen=True)
class MechanicsSummary:
    """只聚合适用评估器；not_applicable 不作为零分。"""

    score: float | None
    coverage: float
    applicable: int
    not_applicable: int
    failed: int

    def __post_init__(self) -> None:
        if self.score is not None:
            _score(self.score, "mechanics summary score")
        _score(self.coverage, "mechanics summary coverage")
        if self.applicable + self.not_applicable + self.failed != len(MECHANICS_EVALUATORS):
            raise SchemaError("mechanics counts must cover all evaluators")


@dataclass(frozen=True)
class EvidenceBundle:
    """所有证据家族进入最终融合前的统一语义。

    ``score`` 始终表示物理可信度（1 为可信），不再混用“违规概率”；coverage 与
    confidence 单独保存，使融合权重和能力缺口均可审计。
    """

    family: str
    source: str
    status: str
    score: float | None
    confidence: float
    coverage: float
    critical_frames: tuple[int, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in EVIDENCE_FAMILIES:
            raise SchemaError(f"invalid evidence family: {self.family!r}")
        if self.status not in EVIDENCE_STATUSES:
            raise SchemaError(f"invalid evidence status: {self.status!r}")
        if not self.source.strip():
            raise SchemaError("evidence source must not be empty")
        if self.status == "available" and self.score is None:
            raise SchemaError("available evidence requires a score")
        if self.status != "available" and self.score is not None:
            raise SchemaError("unavailable evidence must not contain a score")
        if self.score is not None:
            _score(self.score, "evidence score")
        _score(self.confidence, "evidence confidence")
        _score(self.coverage, "evidence coverage")


@dataclass(frozen=True)
class CriticRequest:
    """一次 Critic 分析请求，对应 README 中的推荐输入格式。"""

    video_path: str
    prompt: str = ""
    physics_plan: PhysicsPlan = PhysicsPlan()
    reference_simulation: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # 即使调用方直接构造 dataclass，也必须执行与 JSON 入口相同的版本约束。
        _version(self.schema_version)
        if not self.video_path:
            raise SchemaError("video_path must not be empty")
        # 用户显式提供的扩展字段必须立即合法；模型输出的错误则由 Planner 边界降级处理。
        self.physics_plan.validate_references()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticRequest":
        """解析请求字典；只强制要求视频路径，其余字段均有安全默认值。"""

        if "video_path" not in data:
            raise SchemaError("Critic request requires video_path")
        return cls(
            schema_version=_version(data.get("schema_version")),
            video_path=str(data["video_path"]),
            prompt=str(data.get("prompt", "")),
            physics_plan=PhysicsPlan.from_dict(data.get("physics_plan")),
            reference_simulation=(
                None
                if data.get("reference_simulation") is None
                else str(data["reference_simulation"])
            ),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "CriticRequest":
        """从 UTF-8 JSON 文件读取分析请求。"""

        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise SchemaError("Critic request root must be a JSON object")
        return cls.from_dict(raw)

    def to_dict(self) -> dict[str, Any]:
        """转换成可直接交给 ``json.dumps`` 的普通字典。"""

        return _jsonable(self)


@dataclass(frozen=True)
class Detection:
    """单帧检测结果，可携带上游分配的稳定 ``track_id``。"""

    frame: int
    timestamp_sec: float
    object: str
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    confidence: float = 1.0
    track_id: str | None = None

    def __post_init__(self) -> None:
        # 在数据进入跟踪器之前验证坐标，避免非法框破坏距离匹配。
        if self.frame < 0:
            raise SchemaError("frame must be non-negative")
        if self.timestamp_sec < 0:
            raise SchemaError("timestamp_sec must be non-negative")
        _score(self.confidence, "detection confidence")
        if len(self.center) != 2 or len(self.bbox) != 4:
            raise SchemaError("center and bbox must contain 2 and 4 values respectively")
        if self.bbox[2] < self.bbox[0] or self.bbox[3] < self.bbox[1]:
            raise SchemaError("bbox must use [x_min, y_min, x_max, y_max]")
        if self.track_id is not None and not self.track_id.strip():
            raise SchemaError("detection track_id must not be empty")


@dataclass(frozen=True)
class FrameState:
    """一个物体在某一帧的完整状态。

    ``distance_to_floor`` 为 bbox 底边到地面的有符号距离：正值在地面上方，0 表示
    接触，负值表示已经越过地面。``visible=False`` 表示跟踪器预测身份仍存在但当前
    帧没有检测命中，不能等同于物体已被确认删除。
    """

    frame: int
    timestamp_sec: float
    object: str
    center: tuple[float, float]
    bbox: tuple[float, float, float, float]
    visible: bool = True
    confidence: float = 1.0
    track_id: str | None = None
    distance_to_floor: float | None = None
    overlap_with_floor: float | None = None
    velocity: tuple[float, float] | None = None
    acceleration: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        # 此处也覆盖直接构造场景；from_dict 仅负责外部类型转换。
        if self.frame < 0:
            raise SchemaError("frame must be non-negative")
        if self.timestamp_sec < 0:
            raise SchemaError("timestamp_sec must be non-negative")
        _score(self.confidence, "state confidence")
        if len(self.center) != 2 or len(self.bbox) != 4:
            raise SchemaError("center and bbox must contain 2 and 4 values respectively")
        if self.bbox[2] < self.bbox[0] or self.bbox[3] < self.bbox[1]:
            raise SchemaError("bbox must use [x_min, y_min, x_max, y_max]")
        if self.velocity is not None and len(self.velocity) != 2:
            raise SchemaError("velocity must contain 2 values")
        if self.acceleration is not None and len(self.acceleration) != 2:
            raise SchemaError("acceleration must contain 2 values")
        if self.overlap_with_floor is not None:
            _score(self.overlap_with_floor, "overlap_with_floor")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "FrameState":
        """解析 Blender 真值或外部跟踪器导出的逐帧状态。"""

        required = ("frame", "timestamp_sec", "object", "center", "bbox")
        missing = [name for name in required if name not in data]
        if missing:
            raise SchemaError(f"Frame state missing fields: {missing}")
        velocity = data.get("velocity")
        acceleration = data.get("acceleration")
        return cls(
            frame=int(data["frame"]),
            timestamp_sec=float(data["timestamp_sec"]),
            object=str(data["object"]),
            center=_tuple_of_numbers(data["center"], 2, "center"),
            bbox=_tuple_of_numbers(data["bbox"], 4, "bbox"),
            visible=bool(data.get("visible", True)),
            confidence=float(data.get("confidence", 1.0)),
            track_id=None if data.get("track_id") is None else str(data["track_id"]),
            distance_to_floor=(
                None if data.get("distance_to_floor") is None else float(data["distance_to_floor"])
            ),
            overlap_with_floor=(
                None
                if data.get("overlap_with_floor") is None
                else float(data["overlap_with_floor"])
            ),
            velocity=None if velocity is None else _tuple_of_numbers(velocity, 2, "velocity"),
            acceleration=(
                None if acceleration is None else _tuple_of_numbers(acceleration, 2, "acceleration")
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """输出可序列化状态，未计算的可选运动学字段会被省略。"""

        return _jsonable(self)


@dataclass(frozen=True)
class TrackSequence:
    """同一 ``track_id`` 的时间有序状态序列。"""

    track_id: str
    object: str
    states: tuple[FrameState, ...]


@dataclass(frozen=True)
class Event:
    """从连续状态中提取的离散事件及其证据区间。"""

    event_type: str
    object: str
    track_id: str
    start_frame: int
    peak_frame: int
    end_frame: int
    confidence: float
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 所有下游定位与关键帧逻辑都依赖 start <= peak <= end。
        _score(self.confidence, "event confidence")
        if not self.start_frame <= self.peak_frame <= self.end_frame:
            raise SchemaError("event frames must satisfy start <= peak <= end")


@dataclass(frozen=True)
class ViolationCandidate:
    """规则引擎产生、尚未经过 VLM 与融合阈值处理的异常候选。"""

    object: str
    track_id: str
    category: str
    start_frame: int
    peak_frame: int
    end_frame: int
    reason: str
    repair_instruction: str
    detector_score: float
    rules: tuple[str, ...]
    evidence_frames: tuple[int, ...] = ()
    evidence: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _score(self.detector_score, "detector_score")
        if not self.start_frame <= self.peak_frame <= self.end_frame:
            raise SchemaError("candidate frames must satisfy start <= peak <= end")


@dataclass(frozen=True)
class VLMReview:
    """VLM 对一个规则候选的独立复核结果。"""

    score: float
    reason: str = ""
    repair_instruction: str = ""
    model: str = "unknown"
    claim_status: str = "uncertain"

    def __post_init__(self) -> None:
        _score(self.score, "vlm score")
        if self.claim_status not in VLM_CLAIM_STATUSES:
            raise SchemaError(
                f"claim_status must be one of {VLM_CLAIM_STATUSES}, got {self.claim_status!r}"
            )


@dataclass(frozen=True)
class Violation:
    """通过融合阈值、最终写入公开报告的结构化异常。"""

    object: str
    category: str
    start_frame: int
    peak_frame: int
    end_frame: int
    critical_frames: tuple[int, ...]
    reason: str
    repair_instruction: str
    evidence: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.start_frame <= self.peak_frame <= self.end_frame:
            raise SchemaError("violation frames must satisfy start <= peak <= end")


@dataclass(frozen=True)
class CriticReport:
    """对外稳定的 Physics Critic 输出。

    ``decision`` 是 2.0 的规范判定；``is_physical`` 仅为兼容已有调用方而保留。
    ``unknown`` 必须与 ``is_physical=False`` 配合，表示证据不足而非已经确认违规。
    """

    is_physical: bool
    physics_score: float
    confidence: float
    violations: tuple[Violation, ...] = ()
    graph_evaluation: GraphEvaluationSummary | None = None
    node_results: tuple[NodeResult, ...] = ()
    decision: str | None = None
    coverage: float = 1.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    evidence_bundles: tuple[EvidenceBundle, ...] = ()
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _version(self.schema_version)
        _score(self.physics_score, "physics_score")
        _score(self.confidence, "confidence")
        _score(self.coverage, "coverage")
        decision = self.decision or ("physical" if self.is_physical else "violation")
        if decision not in DECISIONS:
            raise SchemaError(f"decision must be one of {DECISIONS}, got {decision!r}")
        if decision == "physical" and not self.is_physical:
            raise SchemaError("physical decision requires is_physical=True")
        if decision != "physical" and self.is_physical:
            raise SchemaError("violation/unknown decision requires is_physical=False")
        for name, value in self.score_breakdown.items():
            _score(value, f"score_breakdown.{name}")
        object.__setattr__(self, "decision", decision)

    def to_dict(self) -> dict[str, Any]:
        """生成 JSON 兼容字典。"""

        return _jsonable(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        """生成 UTF-8 友好的 JSON 字符串，默认使用两空格缩进。"""

        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass(frozen=True)
class CriticArtifacts:
    """调试输出：最终报告、轨迹、事件以及第一阶段问题图执行产物。"""

    report: CriticReport
    tracks: tuple[TrackSequence, ...]
    events: tuple[Event, ...]
    candidates: tuple[ViolationCandidate, ...] = ()
    question_graph: QuestionGraph | None = None
    node_results: tuple[NodeResult, ...] = ()
    checklist_results: tuple[ChecklistResult, ...] = ()
    checklist_summary: ChecklistSummary | None = None
    visual_evidence: tuple[VisualEvidence, ...] = ()
    mechanics_results: tuple[MechanicsResult, ...] = ()
    mechanics_summary: MechanicsSummary | None = None
    resolved_request: CriticRequest | None = None

    def to_dict(self) -> dict[str, Any]:
        """递归转换全部中间产物，供离线审计和可视化使用。"""

        return _jsonable(self)


def load_frame_states(path: str | Path) -> tuple[FrameState, ...]:
    """读取逐帧状态文件。

    文件既可直接使用状态数组，也可使用带 ``schema_version`` 和 ``observations``
    的封装对象；推荐后者，因为它能显式参与版本校验。
    """

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        _version(raw.get("schema_version"))
        raw = raw.get("observations")
    if not isinstance(raw, list):
        raise SchemaError("Observation file must be a list or contain an observations list")
    return tuple(FrameState.from_dict(item) for item in raw)
