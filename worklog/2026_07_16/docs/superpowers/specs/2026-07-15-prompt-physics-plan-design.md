# Prompt → PhysicsPlan 前置 Planner 设计

日期：2026-07-15
状态：已在对话中逐节确认

## 1. 目标与范围

在 Critic 主流水线之前补全稳定的 Planner 节点：

```text
prompt → resolved PhysicsPlan → Hybrid Question Graph → Critic
```

本次只实现 Prompt 到 PhysicsPlan 的生成、补全、校验、回退和审计，不实现视频生成、Repair Agent 或多轮 Agentic Loop。

成功标准：仅提供 prompt 的 `CriticRequest` 能自动得到对象、事件、关系和物理约束；同一个 resolved request 被问题图、规则、检查表和力学模块共同使用。

## 2. 兼容数据模型

保留现有稳定字段：

```python
objects: tuple[str, ...]
expected_events: tuple[str, ...]
```

新增可选字段：

```python
relations: tuple[PhysicsRelation, ...]
physics_constraints: tuple[PhysicsConstraint, ...]
planner_metadata: PlannerMetadata
```

### 2.1 PhysicsRelation

```python
@dataclass(frozen=True)
class PhysicsRelation:
    id: str
    subject: str
    relation: str
    object: str
```

首批常用关系为 `initially_supported_by`、`interacts_with`、`expected_to_collide_with` 和 `moves_relative_to`。`relation` 保留字符串扩展能力。

### 2.2 PhysicsConstraint

```python
@dataclass(frozen=True)
class PhysicsConstraint:
    id: str
    domain: str
    subjects: tuple[str, ...]
    expectation: str
    condition: str | None = None
```

首批 domain 为 `gravity`、`contact`、`rebound`、`collision`、`continuity` 和 `projectile`。Planner 不生成 prompt 未明确提供的质量、速度、重力常数或恢复系数等精确数值。

### 2.3 PlannerMetadata

```python
@dataclass(frozen=True)
class PlannerMetadata:
    source: str = "empty"
    confidence: float = 0.0
    fallback_used: bool = False
    model: str | None = None
```

source 使用 `explicit`、`template`、`model`、`merged`、`template_fallback` 或 `empty`。可信度由系统按来源给出，不采信模型自报置信度：

| 来源 | confidence |
|---|---:|
| explicit | 1.00 |
| model | 0.80 |
| template | 0.55 |
| template_fallback | 0.40 |
| empty | 0.00 |

## 3. Planner 组件

新增 `src/pavg_critic/planner.py`：

- `PhysicsPlanner`：统一生成协议。
- `TemplatePhysicsPlanner`：无 API 的中英文关键词和事件模板。
- `ModelPhysicsPlanner`：通过 `StructuredTextModel.generate_json()` 生成受 schema 约束的计划。
- `PhysicsPlanResolver`：决定是否调用 Planner，并负责显式计划与生成计划的合并。

## 4. 模型选择规则

`PhysicsCritic` 新增可选参数：

```python
planner_model: StructuredTextModel | None = None
```

选择优先级：

```text
planner_model > question_model > TemplatePhysicsPlanner
```

因此只配置 `question_model` 时，同一模型先生成 PhysicsPlan，再生成 PQSG 图；高级调用方可以为两阶段提供不同模型。

## 5. Template Planner

确定性模板同时识别常用中英文对象与事件：

| Prompt 语义 | 规范输出 |
|---|---|
| red ball / 红球 | `red_ball` |
| ball / 球 | `ball` |
| table / 桌子 | `table` |
| floor, ground / 地面 | `floor` |
| wall / 墙 | `wall` |
| fall, drop / 下落、掉落 | `leave_support → fall` |
| hit floor / 落地、接触地面 | `floor_contact` |
| bounce, rebound / 反弹 | `floor_contact → rebound` |
| throw, launch / 抛出 | `projectile` |
| collide / 相撞 | `collision` |

根据事件派生保守约束：

```text
fall          → gravity/downward_acceleration
floor_contact → contact/no_interpenetration
rebound       → rebound/velocity_reversal_without_energy_gain
collision     → collision/momentum_consistency
projectile    → projectile/parabolic_vertical_motion
```

模板只生成 prompt 明确支持或物理过程必需的内容。

## 6. Model Planner

模型输入包含 prompt 和显式的部分计划，输出必须符合严格 JSON Schema。提示词要求：

- 只规划应该发生的物理过程，不判断视频实际发生了什么；
- 对象 ID 使用稳定 snake_case；
- 事件使用首批规范词表并按时间排序；
- relation/constraint 引用的对象必须存在；
- 不输出精确物理参数；
- 不输出问题答案或违规结论。

模型输出必须先转换为冻结 dataclass 并完成跨引用校验，不能直接进入 Pipeline。

## 7. 显式计划与生成计划合并

核心字段采用“显式非空字段优先”：

```python
objects = explicit.objects or generated.objects
events = explicit.expected_events or generated.expected_events
```

扩展条目按稳定 ID 合并，显式条目覆盖同 ID 的生成条目；不同 ID 保持确定顺序。完整显式计划跳过 Planner API。

校验分两层：

1. 解析时执行字段类型、非空 ID 和 confidence 范围等局部校验；
2. 合并后检查 ID 唯一性和 relation/constraint 对最终 objects 的跨引用。

调用方显式输入无效时直接报 schema 错误，不回退或静默修正。

## 8. Pipeline 数据流

`analyze_detailed()` 在问题图之前执行：

```python
resolved_plan = self.physics_plan_resolver.resolve(request)
resolved_request = replace(request, physics_plan=resolved_plan)
question_graph = self.question_graph_generator.generate(resolved_request)
```

后续视频解码、规则、检查表、力学和 VLM 全部使用 `resolved_request`，不允许不同节点读取不同版本的 PhysicsPlan。

`CriticArtifacts` 新增：

```python
resolved_request: CriticRequest
```

报告诊断新增 `diagnostics.planner`，保存 source、confidence、fallback、model 和完整 resolved plan。

## 9. 错误处理

模型 Planner 的超时、HTTP、畸形 JSON、字段类型或 schema 错误属于可选 provider 故障：

1. 记录 `diagnostics.provider_failures`，stage 为 `physics_planner`；
2. 回退 `TemplatePhysicsPlanner`；
3. 继续原有规则主链。

显式用户 PhysicsPlan 错误不进入此回退路径。

## 10. 测试要求

所有生产代码严格遵循测试先行。新增测试覆盖：

- 旧 PhysicsPlan 请求向后兼容；
- relations、constraints 和 metadata 解析/校验；
- 中英文 fall/contact/rebound 模板；
- projectile 和 collision 模板；
- 空 prompt 不调用模型；
- 合法模型输出、超时、null 数组、非法引用回退；
- 完整显式计划不调用模型；
- 部分显式计划只补空字段；
- 显式扩展条目覆盖同 ID 模型条目；
- `planner_model`、`question_model`、模板三层优先级；
- resolved request 被问题图、规则、检查表和力学共同使用；
- Planner 诊断和 provider failure 可审计；
- 原有测试全部继续通过。

## 11. 文档与示例

README、操作指南和 examples 增加两类示例：

- 仅 prompt、无 API 的 Template Planner；
- `planner_model` 与 `question_model` 共用或分离的 API Planner。

默认 CLI 不配置 API 时自动使用 Template Planner，不产生 API 费用。
