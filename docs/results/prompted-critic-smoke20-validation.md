# 带 Prompt 的完整 Critic 服务器验收报告

**日期：** 2026-07-17  
**性质：** Stage A 工程验收，不作为正式 benchmark 性能声明  
**代码：** `sy@473d2cfb457fa42f796844a97214b2b0098fbc77`

## 结论

完整的 prompt-conditioned `M5_FULL` 已在主服务器 A100 上跑通冻结 VideoPhy-2 smoke20：20/20 样本完成，Planner、PQSG 和 VLM Verifier 均实际参与，零推理失败、零 provider fallback、零空计划、零空问题图、零 OOM。测试过程中发现并修复了三个真实的模型边界问题，修复后服务器完整测试为 `384/384` 通过。

这次结果说明完整 Critic 工程链路现在可正常运行，并在同一 smoke20 上达到 Macro-F1 0.697，高于旧直接 VLM D0 的 0.549，与旧 B1 的 0.697 相同。但 20 个样本只用于集成验收，不能据此宣称完整框架在 3,397 个样本上显著优于 baseline。

## 冻结配置

- 数据：既有 smoke20 成员，10 physical / 10 violation、7 个生成器、20 个非空生成 prompt。
- Manifest SHA-256：`921239725003268e2fbd45b931a561e1bee486b3bb53dce674fa9e955fff762f`。
- 模型：本地 `Qwen/Qwen3-VL-8B-Instruct`，vLLM 0.11.0，严格 JSON Schema。
- 视觉：复用官方 SAM2.1 Hiera B+ 的既有完整轨迹缓存，没有重复传播视频。
- 方法：Model Planner + hybrid PQSG + SAM2 轨迹/事件 + 规则 + VideoScience Checklist + mechanics + grouped VLM verification + coverage-aware fusion。
- 服务器目录：`/root/benchmark/pavg-benchmark`，没有在 `/root` 新建顶层项目目录。

## 结果

| 指标 | M5_FULL |
|---|---:|
| 样本数 | 20 |
| Accuracy | 0.700 |
| Balanced Accuracy | 0.700 |
| Macro-F1 | 0.697 |
| Physical recall | 0.800 |
| Violation recall | 0.600 |
| Violation precision | 0.750 |
| Physics Spearman | 0.324 |
| Unknown / failure rate | 0 / 0 |
| 平均 / p50 / p95 延迟 | 25.31 / 7.66 / 117.70 s |

混淆情况：8 个 physical 判对、2 个 physical 判为 violation；6 个 violation 判对、4 个 violation 判为 physical。模型最终输出 12 个 physical、8 个 violation。

同一 smoke20 的历史结果：D0 Macro-F1 0.549，B1 Macro-F1 0.697。本次 M5 相对 D0 为 `+0.148`，相对 B1 为 `0.000`。这只是小样本诊断方向，正式架构结论仍须使用冻结全量 M5-vs-D0 配对评测和 action-group bootstrap。

## 发现并修复的问题

1. 服务器目录移动后，虚拟环境的 editable install 仍指向旧 `/root/pavg-benchmark`，造成子进程无法导入 `pavg_critic` 和 `sam2`。重新绑定到 `/root/benchmark/pavg-benchmark` 后修复。
2. Qwen3 对非空 prompt 返回 schema 合法但四个数组全空的 PhysicsPlan，旧代码静默接受。现在空计划会触发一次明确的修复请求，仍为空时才按原策略失败/降级。
3. Planner 第二次响应仅有一条 relation 引用未声明对象，旧代码会丢弃整份有效计划。现在仅裁掉仍引用未知对象的 relation/constraint，再严格复验其余计划。
4. PQSG 返回 `weight=0`，vLLM 约束解码未执行 `exclusiveMinimum`。现在归一为默认 1.0 并显式标记 graph sanitized。

对应 `sy` 提交：`cb56095`、`168f2de`、`473d2cf`。每项均先复现失败、增加回归测试、再实现修复，并在 Linux 服务器跑完整测试。

## 完整性与资源

- 最终 predictions / diagnostics：20 / 20，键集合完全一致，重复、缺失、额外和 pending 均为 0。
- Planner：20/20 为 model source，对象数 1–4。
- PQSG：每个样本 1–18 个节点，4 张图记录为 sanitized。
- Provider errors / fallbacks：0 / 0。
- GPU：峰值 21,519MiB、利用率 100%、最高 59°C；运行 507 秒。结束后已停止 vLLM，GPU 回到 0MiB / 0%。
- 250 个模型缓存和最终产物通过标签、人类规则、密钥、认证头、图像数据和 raw payload 扫描。

## 本地产物

完整非敏感产物位于 `outputs/benchmarks/prompted-critic-smoke20-qwen3vl8b/`：

- `manifest.json`
- `predictions.jsonl`
- `diagnostics.jsonl`
- `resolved_config.json`
- `summary.json` / `summary.md`
- `gpu.csv`

详细执行命令、哈希、失败证据和修复过程记录在 `docs/superpowers/plans/2026-07-17-prompted-critic-server-validation.md`。
