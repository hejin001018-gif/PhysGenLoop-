# Learning Repair Agent 最终整合记录

## 今日目标

将本地 `learning_repair` 与 `learning_repair_pipeline` 两套实现整合成一套可训练、
可推理、可执行、可记录 Trial 的最终研究版本，并保持团队现有 Critic 和
`LoopController` 行为不变。

## 已完成

1. 以 `src/physgenloop/learning_repair/` 为唯一 canonical package。
2. 统一基础样本契约与 Executor-facing 丰富版 `RepairDecision`。
3. 增加独立 `RepairSelector`，明确执行 capability mask、proxy/value 模式选择、
   置信度 abstention 和 heuristic fallback。
4. 迁入 Prompt、Global、Local、Reject Executor 与 `ExecutorRegistry`。
5. 迁入 `LearningRepairLoopRunner`、Actual Trial campaign、JSONL recorder、
   versioned Memory、R0–R5 评测和 Action-Value 训练。
6. 将 CLI 统一为 `pavg-repair` / `python -m physgenloop.learning_repair`。
7. 旧 `physgenloop.learning_repair_pipeline` 改为 deprecated import shim。
8. 正式配置统一为 `configs/learning_repair/`，增加 v3 可复核 proxy baseline manifest。
9. 整理 v1/v2/v3 报告、v3 diagnostics、cleanup receipts 和 Blender 训练脚本。
10. 未修改团队共享 `LoopController`、Protocol 或 Critic 输出契约。

## 当前训练结果

| 版本 | Groups | Samples | Held-out Macro-F1 | Gain/Value MAE |
|---|---:|---:|---:|---:|
| v1 | 600 | 7,800 | 1.0 | 0.099762 |
| v2 | 900 | 11,700 | 1.0 | 0.058520 |
| v3 | 1,200 | 22,200 | 1.0 | 0.026106 |

v3 模型 ID 为 `repair-value-78304cfff2fa`，模式为 `classification_proxy`。Actual
Executor Trial 数仍为 0，`source_revision=unknown`，所以
`deployment_ready=false`。这些指标不能宣传为 HunyuanVideo 修复成功率。

## 验证结果

- Learning Repair 定向测试：21 passed。
- canonical CLI sample manifest 审计：valid。
- v3 baseline 文件大小与 SHA-256：valid。
- Critic config/schema/feature schema compatibility：valid。
- clean checkout 曾暴露 Windows CRLF 会改变冻结文件 SHA；已用定向
  `.gitattributes` 强制 repair evidence/config 使用 LF，并在第二个全新 checkout 复核通过。
- 全仓可运行测试：180 passed（176 + schema 非 generator 4）。
- 团队分支现有 `tests/test_schemas.py` 中两个 generator schema 测试未运行通过，原因是
  `schemas/generator_request.schema.json` 在当前团队基线不存在；本次没有越权新增共享
  schema，也没有修改对应测试。

## 下一步

1. 团队确定最终 Critic revision 后重新冻结 compatibility manifest。
2. 接入真实 Prompt、Global、Local Executor，采集多动作 `RepairTrialV1`。
3. 用 Actual Trial 重新训练并切换为 `action_value` selection。
4. 分离 Hunyuan calibration/test，评估真实 physics gain、语义保持、质量、成本和失败率。

---

署名：hejin
