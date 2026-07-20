# VideoPhy-2 完整 PAVG Critic 评测结果

本报告对应只在 `px-cloud1.matpool.com:27339` 上完成的完整 prompt-conditioned Critic 评测。VideoPhy-1、Generator/Repairer/Selector loop、VideoSearch 和微调均不在本次范围内。

## 数据与完整性

- 冻结 manifest：`evaluation/manifests/videophy2_test_full.json`，SHA-256：`d8be5fe97ddf6902515c09ccbb53f394b25230213db7c3058d61f84748624906`
- 3,397 个样本，1,785 physical / 1,612 violation
- 27,176 条预测键：8 个方法各 3,397 条
- 16,985 条模块诊断键：M1–M5 各 3,397 条
- 2,000 次 action-group bootstrap，seed `20260717`
- 本地完整测试：`388 passed`；远程完整测试：`388 passed`

## 主结果

| 方法 | Accuracy | Balanced Accuracy | Macro-F1 | Physical recall | Violation recall | Failure rate |
|---|---:|---:|---:|---:|---:|---:|
| D0_DIRECT_VLM | 0.551663 | 0.549100 | 0.548897 | 0.599440 | 0.498759 | 0.000000 |
| D1_STRUCTURED_VLM | 0.572270 | 0.567685 | 0.566133 | 0.657703 | 0.477667 | 0.000000 |
| B1_RULE | 0.544598 | 0.544391 | 0.544539 | 0.548459 | 0.540323 | 0.001472 |
| M1_GRAPH | 0.543126 | 0.540706 | 0.541014 | 0.588235 | 0.493176 | 0.001766 |
| M2_CHECKLIST | 0.543126 | 0.540706 | 0.541014 | 0.588235 | 0.493176 | 0.001766 |
| M3_MECHANICS | 0.543126 | 0.540706 | 0.541014 | 0.588235 | 0.493176 | 0.001766 |
| M4_VLM | 0.537827 | 0.515703 | 0.413658 | 0.950140 | 0.081266 | 0.001766 |
| M5_FULL | 0.547836 | 0.532381 | 0.492793 | 0.835854 | 0.228908 | 0.001766 |

M5 相对于 D0 的 Macro-F1 delta 为 `-0.056104`，action-group bootstrap 95% CI 为 `[-0.081351, -0.030257]`。因此完整 Critic 没有通过预设的“Macro-F1 至少提升 0.05 且 bootstrap 下界大于 0”的支持门槛。

## Prompt 诊断子集

300 样本上的 shuffled prompt 运行完成 300/300 且失败为 0；oracle plan 运行经一次同配置全清单恢复后，成功优先合并仍保留 4/300 个 `SchemaError`，失败率 1.33%。

- Correct prompt − shuffled prompt：Macro-F1 delta `+0.005101`，CI `[-0.056647, +0.071709]`
- Oracle plan − correct prompt：Macro-F1 delta `-0.021331`，CI `[-0.060424, +0.018240]`

这些结果属于诊断性证据，不能替代全量主比较，也不能证明当前实现已经有效利用 prompt 语义。

## 解释与限制

SAM2 确实参与了 B1/M1–M5 的 observation 生产和轨迹证据链；本次结果不能归因于“没有使用 SAM2”。主要问题是当前规则、证据融合和 VLM 验证组合使 M5 的 violation recall 大幅下降，整体 Macro-F1 低于 D0。VideoSearch 当前没有实现，因此不能在本报告中声称该模块已被评测。

完整机器可读结果位于 [report-v1](../../outputs/benchmarks/videophy2-full-pavg-qwen3vl8b/report-v1/)，包括 `summary.json`、`prompt_diagnostics.json`、`module_attribution.json`、`slices.json`、合并 predictions/diagnostics 和 artifact audit。
