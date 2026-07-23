# PhysGenLoop · Physics-Aware Agentic Video Generation

## 0. 项目概览

构建面向视频生成物理一致性的严格评估与反馈闭环：Wan 产生候选，Physics Critic 给出完整证据，Strict Enforce Gate 只把 `REJECTED` 交给三动作 Repair Policy，Prompt Repair 或 ProPainter Local Editing 产生的新候选必须 Re-Critic/Re-Gate；`UNAVAILABLE` 作为评价失败收口。


---

## 1. 对话与工作方式

- 用中文回答；技术术语保留英文。
- 改代码前先读上下文，理解现有边界后再动手。
- 只做最小必要修改，避免把探索性代码写进核心包。
- 信息不确定时先确认，不要臆测。
- 复杂工作分阶段完成，每阶段保留可验证状态。

---

## 2. 数据 / 代码分离（重要）

本地工作目录：

```text
D:\PythonProject\PhysGenLoop-\
```

- 真实数据、视频、图片、模型权重、checkpoint 绝不进 Git。
- `data/`、`models/`、`checkpoints/`、`outputs/`、`envs/` 可以留在项目根目录，但必须被 `.gitignore` 忽略。
- 代码默认使用项目根目录下的相对路径；禁止在 pipeline 代码中硬编码个人机器的绝对路径。
- `models/` 与 `checkpoints/` 默认只读使用，不随意删除或覆盖。
- 运行产物、debug 文件、生成视频、mask、日志写入 `outputs/` ，不进 Git。

---

## 3. 目录边界

只约定**根目录一级边界**和当前主线归属；具体文件以代码实际状态为准。

```text
PhysGenLoop-/
├── README.md              # 本文件：项目约束与目录边界
├── pyproject.toml         # 包定义与测试配置
├── configs/               # V2 运行配置
├── schemas/               # V2 跨模块 JSON Schema
├── src/
│   ├── pavg_critic/       # Physics Critic 基础能力
│   └── physgenloop/       # V2 通用契约、selector、最小 repair 契约
├── agents/
│   └── wanphysics/        # V2 CLI 入口与一次性 gen/eval 子进程
├── generators/
│   └── wanphysics/        # Wan、critic、editor 适配层与 V2 编排实现
├── evaluation/
│   └── manifests/         # 评测 manifest
├── tests/
│   └── wanphysics_v2/     # V2 测试
├── data/                # 数据集根目录（不入库）
├── data_pipeline/       # 数据处理管道（blender / sam2 数据生产）
├── models/              # 模型权重（不入库，只读）
├── checkpoints/         # 训练检查点（不入库，只读）
├── outputs/             # 生成产物（不入库；benchmarks 运行产物）
├── envs/                # 虚拟环境（不入库：main / vllm / vllm-cu128）
└── worklog/               # 迭代日志与设计记录
```

**目录边界说明**：

- `src/` 为稳定代码包；探索性入口放在 `agents/`，不要反向污染核心包。
- `generators/wanphysics/v2/` 是当前闭环编排、门禁、executor、artifact、trial 的主边界。
- 本地临时文件、远程同步脚本、备份目录不属于项目代码，不纳入提交。
- 若本地仍存在旧目录或临时目录，以 Git 跟踪状态和当前主线边界为准。

---

## 4. 接口契约（全组共用，改动须同意）

以以下位置为权威定义，改动必须先沟通：

- `schemas/`：V2 JSON Schema。
- `src/physgenloop/contracts.py`：候选、评估等通用数据结构。
- `src/physgenloop/learning_repair/contracts.py`：repair decision、execution request/result 等动作契约。
- `generators/wanphysics/v2/trials.py`：`WanRepairTrialV3` 严格因果与审计结构。

---

## 5. 工作日志

按日期分目录，push 共享：

```text
worklog/YYYY_MM_DD/
```

- 需要沉淀的设计、实施结果和验证结果写入当天目录。
- 稳定项目约束写到根目录 `README.md`。
- 接口契约写在 `schemas/` 与 `src/physgenloop/`。
- 运行输出不要写进 worklog；worklog 只保留设计、结论和必要摘要。

---

## 6. Git 工作流

- `main` 为主线分支。
- 大改动前先看 `git status --short --branch`，不要覆盖别人未提交的变更。
- Commit message 简洁，遵循 Conventional Commits（feat/fix/docs/refactor/...）。
- Commit message 里不要出现 AI、Claude 等字样；不要写 Co-Authored-By。
- 禁止 `--force` 推送到 `main`。

---

## 7. Python 环境

- 用 conda 管理依赖。
- 包安装入口：`pip install -e . --no-deps`。
- 具体依赖和运行参数以 `pyproject.toml` 与 `configs/` 为准。
- 真实 GPU 依赖、模型权重和 checkpoint 不在仓库内。

---

## 8. 验证边界

主线测试：

```bash
pytest tests/wanphysics_v2
```

CPU 级入口验证：

```bash
python agents/wanphysics/run_videophy2_loop_v2.py --dry-run
```

当前在线运行只有 `agents/wanphysics/run_videophy2_loop_v2.py` 一个入口。动作集合固定为 `prompt_repair`、`local_editing`、`reject`；CLI、配置和 manifest 均不得覆盖 Policy 动作。`run_actual_trials_v2.py` 为历史退役入口，新运行会 fail fast。Memory、Global Regeneration、shadow Gate 和旧四动作 proxy checkpoint 不进入在线链路。

真实链路必须在具备模型、checkpoint、GPU 环境的部署机器上验证。
