# PhysGenLoop · Physics-Aware Agentic Video Generation

## 0. 项目概览

构建面向视频生成物理一致性的评估与反馈闭环：Blender 合成带精确物理真值的视频数据 → 训练/组合 Physics Critic → 接入视频生成模型形成「规划 → 生成 → 检测 → 修复 → 再生成」闭环。

**核心成果**：可独立运行的 Physics Planner + Physics Critic，可插拔的 Best-of-K
闭环编排层，可审计的 benchmark 评测流水线，以及 canonical Learning Repair Agent
（Policy、Selector、Executor、Memory、Actual-Trial Runner）。

---

## 1. 对话与工作方式

- 用中文回答；技术术语保留英文。
- 改代码前先讲方案，讨论清楚细节、得到同意后再动手；只做最小必要修改。
- 需要讨论思路时，引导魔尊把细节聊清楚。
- 保持代码整洁、避免冗余；复杂工作分多步进行。
- 信息不确定时先与魔尊确认，不要臆测。
- 先读后写；修改前必须理解上下文。

---

## 2. 数据 / 代码分离（重要）

本地工作目录：

```text
D:\PythonProject\PhysGenLoop-\
```

- 真实数据（视频/图片/权重）绝不进 Git。
- `data/`、`models/`、`checkpoints/`、`outputs/` 留在项目根目录，但必须被 `.gitignore` 忽略。
- 能软链接的尽量软链接，不复制大目录。
- 代码一律走相对路径 `./data ./models ./outputs`，禁止在 pipeline 代码中硬编码绝对路径。
- `models/` 与 `checkpoints/` 默认只读使用，不随意删除或覆盖。
- 运行产物、timing、debug 图片、生成视频都留在 `outputs/`，不进 Git。

---

## 3. 目录边界

只约定**根目录一级边界**，不列具体文件；具体文件以代码实际状态为准。

```text
PhysGenLoop-/
├── README.md            # 本文件：项目约束与目录索引
├── pyproject.toml       # 包定义
├── configs/             # 全局配置（权重、阈值、评测基准）
├── schemas/             # 跨模块契约（JSON Schema，见 §4）
├── src/
│   ├── pavg_critic/     # Physics Planner + Critic，可独立运行
│   └── physgenloop/     # 闭环编排层（Generator / Critic / Repairer / Selector）
├── agents/              # Agentic 闭环独立探索层（prompt 改写 / repair / 视频后端）
├── benchmarks/          # 外域评测入口脚本
├── evaluation/          # 评测数据与 manifests
├── examples/            # 使用示例
├── data/                # 数据集根目录（不入库）
├── data_pipeline/       # 数据处理管道（blender / sam2 数据生产）
├── models/              # 模型权重（不入库，只读）
├── checkpoints/         # 训练检查点（不入库，只读）
├── generators/          # 视频生成探针（wanphysics）
├── experiments/         # 消融实验脚本
├── outputs/             # 生成产物（不入库；benchmarks 运行产物 + repair_training 报告）
├── envs/                # 虚拟环境（不入库：main / vllm / vllm-cu128）
├── tests/               # 测试
└── worklog/             # 迭代日志（按日期分卷，见 §5）
```

**目录边界说明**：

- `src/` 为核心代码包，禁止把探索性代码直接写进包内；探索层放 `agents/`。
- `data/samples/` 存放单样本目录，必须对齐 `schemas/sample.schema.json`。
- `data_pipeline/` 只放数据加载与转换适配器，不放数据本体。
- `outputs/` 根目录不再放其他实验结果或 legacy 入口；每次运行产物按 run 名归档。
- `data_pipeline/` 现含 `blender/`、`sam2/` 两套数据生产管道脚本。
- `envs/` 存放虚拟环境（`main` 主环境、`vllm` / `vllm-cu128` 推理环境），不入库，按 §7 重建。
- 探索性讨论进 `worklog/`。

---

## 4. 接口契约（全组共用，改动须同意）

以 `schemas/` 为权威定义，改动必须全组同意：

- **单样本目录结构**：`sample_id/{video.mp4, config.json, trajectory.json, contacts.json, annotation.json}` — 对齐 `sample.schema.json`。
- **Critic 输出契约**：`critic_output.schema.json` — 定义 `violations`、`critical_frames`、`repair_instruction` 等字段。
- **闭环数据契约**：`src/physgenloop/contracts.py` — 定义 Generator / Critic / Repairer 之间的数据结构。
- **闭环接口协议**：`src/physgenloop/interfaces.py` — 定义 Generator / Critic / Repairer 的 Protocol。

sample_id 按人分段，避免多人写同一目录时撞号。

---

## 5. 工作日志

按日期分目录，push 共享：

```text
worklog/YYYY-MM-DD/
```

- 每完成一个阶段，把工作记录追加到当天目录的 `work-record.md`，记录：实现了哪些功能？遇到了哪些错误？如何解决？当前状态与下一步是什么？
- 需要沉淀的设计细节按主题命名，例如 `pipeline-design.md`、`critic-fusion.md`。
- 每次会话开始，先读当天目录的 `work-record.md` 和相关设计文档，再追加新的状态。
- 稳定项目说明写到根目录 `README.md`；接口契约写在 `schemas/` 与 `src/physgenloop/contracts.py`。
- 技术讨论与犯错反思追加到当天 `worklog/YYYY-MM-DD/work-record.md`。

---

## 6. Git 工作流

- 本地 `main` 为主线分支；个人开发用短期 feature 分支（例如 `lsh`）。
- 多人同时改同一批文件前先沟通，避免覆盖未提交改动。
- 做大改动前先看 `git status --short --branch`，发现别人未提交文件时不要重置、清理或覆盖。
- Commit message 简洁，遵循 Conventional Commits（feat/fix/docs/refactor/…）。
- Commit message 里绝不出现 AI、Claude 等字样；绝不写 Co-Authored-By。
- 禁止 `--force` 推送到 `main`。

---

## 7. Python 环境

- 用 conda 管理依赖。
- 包安装入口：`pip install -e . --no-deps`（基于根目录 `pyproject.toml`）。
- 具体环境名、依赖版本以 `pyproject.toml` 与 `configs/` 为准。

---

## 8. 日志分析

- 运行日志文件很大，用 Python 脚本分析：建议从文件末尾往上看（开头多为初始化）。
- 结合代码里的 print 内容在日志里搜索定位。
- 不要在会话里直接 `cat` 全量日志。

---

## 9. 当前阶段

- Physics Planner / Critic 基线可独立运行。
- Benchmark 评测流水线（含可恢复运行器与分组指标）已实现。
- 闭环编排契约已冻结，Fake Generator 打通 Best-of-K 控制器。
- Learning Repair Agent 已统一到 `src/physgenloop/learning_repair/`；v3 使用 1,200 个
  Blender group、22,200 条 proxy 样本完成训练，当前模型仍是研究基线。
- **尚未接入**：真实 HunyuanVideo 生成服务、真实 Prompt/Global/Local Executor
  rollout 与 Hunyuan calibration/test；Actual Repair Trial 当前为 0。

Learning Repair 的最终架构、命令、指标边界和下一阶段见
[团队交接文档](worklog/2026_07_17/docs/learning-repair-milestones-1-5.md)。
