# JSON Schema

- `critic_output.schema.json`：PAVG Critic 2.0 的稳定输出，包括三态决策、覆盖率、违规、问题图结果和五类证据包。
- `critic_trace.schema.json`：单视频 Critic 的逐节点输入/输出、耗时、降级状态和五类证据融合算术；版本为 `pavg-critic-trace/v1`。
- `sample.schema.json`：保留的评估样本/人工标注契约，继续兼容现有 1.0 数据。

运行 `python -m pytest tests/test_schemas.py -q` 可检查 schema 本身及示例输出。

生成并严格校验 trace：

```powershell
python examples/evaluate_video.py --video 2n.mp4 --prompt "石头滚下坡" `
  --trace --trace-output outputs/2n.trace.json --output outputs/2n.result.json

python examples/validate_pipeline_trace.py outputs/2n.trace.json `
  --require-sam2 --require-model-planner --fail-on-provider-fallback
```

校验器退出码：`0` 表示全部必需条件通过，`1` 表示结构、依赖、隐私或融合算术不一致，`2` 表示文件或调用无效。严格 flags 会把 SAM2 未使用、Planner 降级或 provider fallback 视为失败。
