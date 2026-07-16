# PhysGenLoop 整体架构骨架设计

日期：2026-07-15  
状态：已在对话中批准方案 A

## 1. 目标与边界

在不打散现有 `pavg_critic` 稳定实现的前提下，把仓库从单一 Critic 扩展为可逐步演进的 PhysGenLoop 工程骨架，并把已经设计完成的 Prompt → PhysicsPlan Planner 实装到 Critic 前端。

本次交付包括：

- 可运行、可审计的 Prompt → PhysicsPlan Resolver；
- 保持现有 `pavg_critic` API 和测试兼容；
- 新增轻量 `physgenloop` 总控包，定义生成、修复、选择与循环控制契约；
- 提供不依赖 GPU 或外部 API 的 deterministic fake，使最小闭环能够在测试中执行；
- 更新 README，区分已实现能力、框架占位和未来里程碑。

本次不接入真实 HunyuanVideo、Blender、训练流水线、任务队列、持久化数据库或多机调度，也不声称 fake 闭环代表真实生成性能。

## 2. 总体架构

```text
User Prompt
    ↓
PhysicsPlan Resolver       本次实装：template/model/explicit merge
    ↓
VideoGenerator             本次：协议 + deterministic fake
    ↓
PhysicsCritic              已有核心，复用轨迹、事件、规则、PQSG、VLM 与证据融合
    ↓
Repairer                   本次：协议 + 最小提示词修复策略
    ↓
Selector                   本次：按 Critic 报告选择最佳候选
    ↓
LoopController             本次：有界轮次、停止条件、历史审计
```

`pavg_critic` 继续负责物理计划解析和视频物理评价；`physgenloop` 只负责跨组件编排，不复制 Critic 内部逻辑。真实模型以后通过 Protocol 适配器注入。

## 3. 包与文件边界

### 3.1 `pavg_critic`

- `schemas.py`：扩展 `PhysicsPlan`，加入 relation、constraint 和 planner metadata。
- `planner.py`：模板 Planner、结构化模型 Planner、显式/生成计划合并与 provider 回退。
- `pipeline.py`：在问题图生成前解析唯一的 resolved request，并让所有下游阶段共用。
- `pqsg.py`：把扩展后的物理计划上下文传给问题图模型。

Planner 的详细字段、优先级和错误处理继续遵循 `2026-07-15-prompt-physics-plan-design.md`，不在本设计中另造第二套语义。

### 3.2 `physgenloop`

- `contracts.py`：生成请求/结果、候选评价、循环状态与最终结果等冻结数据模型。
- `interfaces.py`：`VideoGenerator`、`PromptRepairer`、`CandidateSelector` 协议。
- `generator.py`：deterministic fake generator；真实 HunyuanVideo 适配器留到后续里程碑。
- `repairer.py`：根据结构化 violation 的 `repair_instruction` 生成下一轮提示词。
- `selector.py`：按 decision、physics score、confidence 和稳定次序选择候选。
- `controller.py`：实现 Best-of-K 和有限轮反馈循环，不包含并行队列或分布式调度。

每个模块只依赖公开契约；`controller.py` 通过构造函数注入 generator、critic、repairer 和 selector。

## 4. 数据流与停止条件

每轮流程为：

1. 为当前 prompt 生成 K 个候选；
2. 用同一个 `PhysicsCritic` 分析所有候选；
3. Selector 选出本轮最佳候选并写入历史；
4. 若最佳候选为 `physical` 且达到配置阈值，则成功停止；
5. 若已达到最大轮数，则返回历史最优候选；
6. 否则由 Repairer 根据最佳候选的结构化违规生成下一轮 prompt。

循环必须有 `max_rounds >= 1` 和 `candidates_per_round >= 1`，不允许无限循环。相同分数按候选生成顺序稳定选择，保证回归测试可重复。

## 5. 错误处理与审计

- 用户显式 PhysicsPlan 无效：直接抛出 schema 错误，不静默修复。
- 可选 Planner provider 超时、HTTP 或畸形输出：记录 `physics_planner` failure，回退模板 Planner。
- generator 或 critic 的不可恢复错误：终止当前运行并保留异常上下文；本版不实现复杂重试。
- 每轮保存 prompt、seed/候选 ID、CriticReport、所选候选和停止原因，便于之后复现实验。
- fake generator 只产出测试用 artifact，不伪装成真实 MP4。

## 6. 测试策略

- Planner：旧 schema 兼容、中英文模板、模型输出、合并优先级、provider 回退、Pipeline 共用 resolved request。
- 框架契约：冻结数据模型、输入校验和稳定序列化。
- Selector/Repairer：确定性排序、无违规回退和 repair instruction 聚合。
- Controller：首轮成功、达到最大轮数、Best-of-K、历史最优保留和错误传播。
- 回归：现有 Critic 测试必须继续通过；测试不访问网络、不需要 GPU。

## 7. README 信息结构

README 按以下顺序重写：

1. 项目定位与当前状态；
2. 最新整体架构图；
3. 已实现 / 框架已搭 / 尚未实现三列表；
4. 安装与 Critic 快速运行；
5. Planner 用法；
6. 最小 fake 闭环示例；
7. 仓库结构、测试和路线图；
8. 已知限制。

文档不得把 HunyuanVideo、Blender 数据生成、Repair 模型或真实端到端性能描述为已经完成。

## 8. 成功标准

- 仅 prompt 的 Critic 请求能得到 resolved PhysicsPlan，并在 diagnostics 中可审计；
- 原有显式 PhysicsPlan 请求保持兼容；
- `physgenloop` 可被导入，最小 fake 闭环能在 CPU 测试中跑通；
- README 的架构、目录和能力状态与代码一致；
- 全部测试、Python 编译检查和 `git diff --check` 通过。
