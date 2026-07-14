"""Physics Critic 的类型化配置。

本模块只负责“读取、规范化、校验”配置，不在这里实例化模型或执行业务逻辑。
所有配置对象均为冻结 dataclass，目的是避免一次分析过程中阈值被意外修改，保证
实验可复现。外部 JSON 仍使用普通数组；加载时会将需要稳定语义的数组转为元组。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, TypeVar


@dataclass(frozen=True)
class DetectorConfig:
    """内置 HSV 目标检测器及视频坐标系配置。

    OpenCV HSV 色相范围是 0~179，而饱和度和明度是 0~255。红色跨越色相环的
    首尾，因此使用两段阈值。``floor_y_ratio`` 表示地面 y 坐标占画面高度的比例；
    OpenCV 图像坐标向下为正，所以物体 bbox 底边超过该位置时可能发生穿透。
    """

    backend: str = "color_blob"
    object_label: str = "red_ball"
    hsv_lower: tuple[int, int, int] = (0, 90, 70)
    hsv_upper: tuple[int, int, int] = (10, 255, 255)
    hsv_lower_2: tuple[int, int, int] = (170, 90, 70)
    hsv_upper_2: tuple[int, int, int] = (179, 255, 255)
    min_area: float = 80.0
    floor_y_ratio: float = 0.9


@dataclass(frozen=True)
class TrackerConfig:
    """质心跟踪器的匹配距离与短时丢失容忍参数。"""

    max_match_distance_px: float = 80.0
    max_missed_frames: int = 5


@dataclass(frozen=True)
class TrajectoryConfig:
    """轨迹平滑参数；窗口 1 表示完全保留原始检测中心。"""

    smoothing_window: int = 3


@dataclass(frozen=True)
class EventConfig:
    """由连续轨迹触发离散事件时使用的阈值。

    速度单位统一为像素/秒，距离单位统一为像素。真实尺度尚不可得时，调用方应按
    分辨率校准这些阈值，或在后续检测器中将轨迹转换到世界坐标再复用事件接口。
    """

    velocity_epsilon_px_s: float = 8.0
    contact_tolerance_px: float = 3.0
    penetration_tolerance_px: float = 5.0
    min_disappearance_frames: int = 3
    min_upward_frames: int = 2
    teleport_speed_px_s: float = 1000.0


@dataclass(frozen=True)
class RuleConfig:
    """规则开关与接触事件回看窗口。"""

    contact_lookback_frames: int = 2
    gravity_contact_lookback_frames: int = 3
    enabled: tuple[str, ...] = (
        "premature_rebound",
        "surface_penetration",
        "object_disappearance",
        "reverse_gravity",
        "teleportation",
    )


@dataclass(frozen=True)
class TemporalConfig:
    """异常前后证据帧的最大搜索范围。"""

    pre_context_frames: int = 3
    post_context_frames: int = 3


@dataclass(frozen=True)
class QuestionGraphConfig:
    """第一阶段问题图的生成与执行配置。

    ``enabled`` 允许实验中关闭新增图层并复现原规则基线。模板生成器会从扁平
    ``PhysicsPlan`` 推导 Object/Action/Physics 节点；``include_generic_physics``
    控制是否额外检查物体恒存和运动连续性。规则未命中时使用
    ``rule_pass_confidence`` 作为“当前可用规则未发现异常”的保守置信度，它不代表
    物理正确性的绝对概率。
    """

    enabled: bool = True
    include_generic_physics: bool = True
    rule_pass_confidence: float = 0.75


@dataclass(frozen=True)
class ChecklistConfig:
    """VideoScience 风格五维检查表配置。"""

    enabled: bool = True
    pass_confidence: float = 0.75


@dataclass(frozen=True)
class MechanicsConfig:
    """Morpheus 启发的力学评估门控和阈值。"""

    enabled: bool = True
    min_points: int = 4
    plausible_threshold: float = 0.6
    contact_lookback_frames: int = 2


@dataclass(frozen=True)
class FusionConfig:
    """规则/VLM 融合权重以及最终判定阈值。"""

    detector_weight: float = 0.7
    vlm_weight: float = 0.3
    violation_threshold: float = 0.5
    physical_score_threshold: float = 0.6
    clean_confidence: float = 0.75
    rule_family_weight: float = 0.35
    pqsg_family_weight: float = 0.2
    checklist_family_weight: float = 0.2
    mechanics_family_weight: float = 0.2
    vlm_family_weight: float = 0.05
    minimum_coverage: float = 0.35


@dataclass(frozen=True)
class CriticConfig:
    """完整 Critic 配置根对象，各字段对应配置文件的同名 section。"""

    detector: DetectorConfig = DetectorConfig()
    tracker: TrackerConfig = TrackerConfig()
    trajectory: TrajectoryConfig = TrajectoryConfig()
    events: EventConfig = EventConfig()
    rules: RuleConfig = RuleConfig()
    temporal: TemporalConfig = TemporalConfig()
    question_graph: QuestionGraphConfig = QuestionGraphConfig()
    checklist: ChecklistConfig = ChecklistConfig()
    mechanics: MechanicsConfig = MechanicsConfig()
    fusion: FusionConfig = FusionConfig()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CriticConfig":
        """从已解析 JSON 构造配置，并拒绝拼写错误或尚未支持的字段。"""

        # 严格拒绝未知 section，防止错误配置被静默忽略后仍产生看似有效的实验结果。
        known = {item.name for item in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown critic config sections: {sorted(unknown)}")
        config = cls(
            detector=_section(DetectorConfig, data.get("detector", {})),
            tracker=_section(TrackerConfig, data.get("tracker", {})),
            trajectory=_section(TrajectoryConfig, data.get("trajectory", {})),
            events=_section(EventConfig, data.get("events", {})),
            rules=_section(RuleConfig, data.get("rules", {})),
            temporal=_section(TemporalConfig, data.get("temporal", {})),
            question_graph=_section(QuestionGraphConfig, data.get("question_graph", {})),
            checklist=_section(ChecklistConfig, data.get("checklist", {})),
            mechanics=_section(MechanicsConfig, data.get("mechanics", {})),
            fusion=_section(FusionConfig, data.get("fusion", {})),
        )
        config.validate()
        return config

    def validate(self) -> None:
        """校验跨字段语义与数值范围；失败时尽早给出可定位的配置路径。"""

        # 图像前端参数既要满足比例约束，也要满足 OpenCV 阈值的通用取值范围。
        if not 0.0 < self.detector.floor_y_ratio <= 1.0:
            raise ValueError("detector.floor_y_ratio must be in (0, 1]")
        for name in ("hsv_lower", "hsv_upper", "hsv_lower_2", "hsv_upper_2"):
            value = getattr(self.detector, name)
            if len(value) != 3 or any(not 0 <= channel <= 255 for channel in value):
                raise ValueError(f"detector.{name} must contain three values in [0, 255]")
        if self.detector.min_area <= 0:
            raise ValueError("detector.min_area must be positive")
        if self.tracker.max_match_distance_px <= 0:
            raise ValueError("tracker.max_match_distance_px must be positive")
        if self.trajectory.smoothing_window < 1:
            raise ValueError("trajectory.smoothing_window must be >= 1")
        if self.tracker.max_missed_frames < 0:
            raise ValueError("tracker.max_missed_frames must be >= 0")
        if self.events.min_disappearance_frames < 1:
            raise ValueError("events.min_disappearance_frames must be >= 1")
        if self.events.min_upward_frames < 1:
            raise ValueError("events.min_upward_frames must be >= 1")
        if not 0.0 <= self.question_graph.rule_pass_confidence <= 1.0:
            raise ValueError("question_graph.rule_pass_confidence must be in [0, 1]")
        if not 0.0 <= self.checklist.pass_confidence <= 1.0:
            raise ValueError("checklist.pass_confidence must be in [0, 1]")
        if self.mechanics.min_points < 3:
            raise ValueError("mechanics.min_points must be >= 3")
        if not 0.0 <= self.mechanics.plausible_threshold <= 1.0:
            raise ValueError("mechanics.plausible_threshold must be in [0, 1]")
        if self.mechanics.contact_lookback_frames < 0:
            raise ValueError("mechanics.contact_lookback_frames must be >= 0")
        for name in (
            "violation_threshold",
            "physical_score_threshold",
            "clean_confidence",
            "minimum_coverage",
        ):
            value = getattr(self.fusion, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"fusion.{name} must be in [0, 1]")
        if self.fusion.detector_weight < 0 or self.fusion.vlm_weight < 0:
            raise ValueError("fusion weights must be non-negative")
        if self.fusion.detector_weight + self.fusion.vlm_weight == 0:
            raise ValueError("at least one fusion weight must be positive")
        family_weights = (
            self.fusion.rule_family_weight,
            self.fusion.pqsg_family_weight,
            self.fusion.checklist_family_weight,
            self.fusion.mechanics_family_weight,
            self.fusion.vlm_family_weight,
        )
        if any(value < 0 for value in family_weights) or sum(family_weights) <= 0:
            raise ValueError("fusion family weights must be non-negative with a positive sum")


T = TypeVar("T")


def _section(section_type: type[T], raw: Any) -> T:
    """将一个 JSON section 转成指定 dataclass，同时完成列表到元组的规范化。"""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{section_type.__name__} must be a JSON object")
    allowed = {item.name for item in fields(section_type)}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown {section_type.__name__} fields: {sorted(unknown)}")
    # 拷贝后再规范化，确保不会修改调用方持有的原始配置字典。
    normalized = dict(raw)
    for item in fields(section_type):
        if item.name in normalized and item.name.startswith("hsv_"):
            normalized[item.name] = tuple(normalized[item.name])
    if section_type is RuleConfig and "enabled" in normalized:
        normalized["enabled"] = tuple(normalized["enabled"])
    return section_type(**normalized)


def load_config(path: str | Path | None = None) -> CriticConfig:
    """加载 UTF-8 JSON/YAML 配置；未传路径时返回经过相同校验的默认配置。

    YAML 使用 ``safe_load``，避免配置文件构造任意 Python 对象。无法识别的扩展名会
    直接报错，防止一个拼错后缀的文件被意外按另一种格式解释。
    """

    if path is None:
        config = CriticConfig()
        config.validate()
        return config
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        import yaml

        data = yaml.safe_load(text)
    else:
        raise ValueError("Critic config must use a .json, .yaml, or .yml extension")
    if not isinstance(data, Mapping):
        raise ValueError("Critic config root must be an object")
    return CriticConfig.from_dict(data)
