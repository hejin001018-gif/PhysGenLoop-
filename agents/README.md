# agents/ — Agentic 闭环组件

对齐 `worklog/2026_7_14/框架骨干.md` §六 L4 Agentic 闭环层。

## 组件

| 文件 | 职责 |
|------|------|
| `prompt_rewriter.py` | 用 LLM 把 Critic violations 翻译为新 prompt + physics_hint。支持 Claude / OpenAI / Stub 三 backend |
| `repairer.py` | Repair Agent 决策器：依 physics_score 选 L1/L2/L3/停机，产出 `generator_request.schema.json` 对齐的请求 |
| `video_backend.py` | 视频生成后端抽象：`hunyuan-local` / `replicate` / `fal` / `stub` |

## Repair 决策规则

| physics_score | 决策 | 说明 |
|---------------|------|------|
| `> 0.6` | `prompt_only` | 轻度违规，只重写 prompt 换 seed |
| `(0.4, 0.6]` | `local_inpaint` | 首期关闭；启用需 HunyuanVideo 支持 mask 条件 |
| `≤ 0.4` | `full_regen` | 严重违规，强制换 seed + 强化 prompt |
| `is_physical=True` | `stop` | Critic 认可，闭环完成 |
| `round_idx ≥ max_rounds` | `stop` | 达轮数上限 |
| 连续 2 轮无提升 | `stop` | 提前终止 |

## 用法

### 一：单次修复（无状态）

```python
from agents.repairer import repair_once

critic_output = {  # 来自 Physics Critic
    "schema_version": "1.0",
    "is_physical": False,
    "physics_score": 0.35,
    "confidence": 0.9,
    "violations": [{
        "object": "red_ball",
        "category": "premature_rebound",
        "start_frame": 47, "end_frame": 53,
        "critical_frames": [44, 47, 49, 53],
        "reason": "The ball reverses direction before contacting the floor.",
        "repair_instruction": "Keep the ball moving downward until visible floor contact.",
    }],
}

decision = repair_once(critic_output, original_prompt="A red ball falls from a table.")
print(decision.action)              # "full_regen"
print(decision.generator_request)   # 契合 generator_request.schema.json 的 dict
```

### 二：多轮闭环（有状态）

```python
from agents.repairer import decide, RepairState, RepairConfig
from agents.video_backend import make_backend

state = RepairState(original_prompt="A red ball falls from a table.")
cfg = RepairConfig(max_rounds=2)
backend = make_backend("stub")  # 换成 "hunyuan-local" / "replicate" / "fal"

critic_output = ...  # 首轮 Critic 结果
for _ in range(cfg.max_rounds + 1):
    d = decide(critic_output, state, cfg, output_path=f"outputs/round_{state.round_idx}.mp4")
    if d.action == "stop":
        break
    gen_result = backend.generate(d.generator_request)
    critic_output = run_critic(gen_result.output_path)  # 你的 Critic
```

## LLM Backend 切换

```bash
# 默认 stub，无需 API Key
python -c "from agents.repairer import repair_once; ..."

# Claude
export ANTHROPIC_API_KEY=sk-ant-...
# 代码中：make_client("claude")

# OpenAI
export OPENAI_API_KEY=sk-...
# 代码中：make_client("openai")
```

## 视频生成 Backend 切换

| backend | 环境变量 | 依赖 |
|---------|----------|------|
| `stub` | 无 | 无（只写空文件，用于跑通闭环） |
| `hunyuan-local` | 无 | 本地安装 HunyuanVideo-1.5 |
| `replicate` | `REPLICATE_API_TOKEN` | `pip install replicate requests` |
| `fal` | `FAL_KEY` | `pip install fal-client requests` |

```bash
export PAVG_VIDEO_BACKEND=replicate
export REPLICATE_API_TOKEN=r8_xxx
```

## 首期取舍

- **L2 局部 inpainting 不启用**：HunyuanVideo-1.5 首个版本不支持 mask 条件视频 inpainting。可用替代路径：（a）改用支持 video inpainting 的模型如 CogVideoX-Fun / VACE；（b）沿用 full_regen 但通过 `critical_frames` 引导 prompt 强化。
- **API 优先建议**：如果本地没 24GB+ GPU，先用 `replicate` 或 `fal` 打通闭环，再切本地。
- **Seed 策略**：`prompt_only` 沿用旧 seed（只改 prompt），`full_regen` 强制换 seed。

## 待办

- [ ] `agents/controller.py` — 完整多轮闭环编排（Planner + Repair + Selector）
- [ ] `agents/selector.py` — 历史候选选最优
- [ ] LLM prompt 模板迭代与 few-shot 例子
- [ ] Repair 决策的 A/B 消融实验骨架
