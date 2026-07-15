# Repair Agent · 修改提示词方案

> 日期：2026-07-15
> 作者：陆圣桦
> 范围：**只做 Repair Agent 一环**——把 Physics Critic 的检测结果翻译为「下一轮 HunyuanVideo 的生成请求」
> 依据：`worklog/2026_7_14/框架骨干.md` §六 L4 Agentic 闭环层 · Repair Agent 四级修复策略

---

## 一、判词

**Repair Agent 不修视频本身，它修「下一次生成的输入」。** 真正生成视频的是 HunyuanVideo。所以 Repair Agent 的本体是 **反馈翻译器 + 策略选择器 + 请求构造器**——三件事，一条流水线。

对外只暴露一个函数：

```python
decision = repair(critic_output, original_prompt, state) -> RepairDecision
```

输入是 Critic 结构化输出，输出是符合 `schemas/generator_request.schema.json` 的下一轮生成请求。

---

## 二、输入输出契约

### 输入 A：Critic Output（来自 Physics Critic）

对齐 `schemas/critic_output.schema.json`：

```json
{
  "schema_version": "1.0",
  "is_physical": false,
  "physics_score": 0.35,
  "confidence": 0.9,
  "violations": [
    {
      "object": "red_ball",
      "category": "premature_rebound",
      "start_frame": 47,
      "peak_frame": 49,
      "end_frame": 53,
      "critical_frames": [44, 47, 49, 53],
      "reason": "The ball reverses direction before contacting the floor.",
      "repair_instruction": "Keep the ball moving downward until visible floor contact.",
      "evidence": {"rules": ["velocity_reversal_before_contact"], "detector_score": 0.94, "vlm_score": 0.87}
    }
  ]
}
```

### 输入 B：原始 prompt + 闭环状态

```python
@dataclass
class RepairState:
    original_prompt: str      # 用户最初的 prompt
    round_idx: int = 0        # 当前轮次
    history_scores: list[float]  # 历轮 physics_score
    prior_hints: list[str]       # 历轮 physics_hint（避免重复堆叠）
    last_video_path: str | None
```

### 输出：RepairDecision

```json
{
  "action": "full_regen",
  "reason": "score=0.35 → full_regen",
  "generator_request": {
    "prompt": "A red ball falls from a table. Ensure the ball makes visible contact with the floor before bouncing.",
    "seed": 918273645,
    "resolution": "480p",
    "num_frames": 121,
    "num_inference_steps": 50,
    "image_path": null,
    "output_path": "outputs/round_1.mp4",
    "physics_hint": "Ball must contact floor before rebounding.",
    "repair_context": {
      "critical_frames": [44, 47, 49, 53],
      "object_masks": [],
      "strategy": "full_regen"
    },
    "generation_meta": {"attempt": 1, "parent_video_path": "outputs/round_0.mp4", "loop_round": 1}
  }
}
```

---

## 三、内部流水线（三段式）

```
┌───────────────────────────────────────────────────────────────────┐
│  Repair Agent                                                     │
│                                                                   │
│   Critic Output ──► [1. Strategy Selector] ──► action             │
│                          (依 score / round)                       │
│                                │                                  │
│                                ▼                                  │
│   original_prompt + violations ──► [2. Prompt Rewriter] ──► new   │
│                                       (LLM API)          prompt + │
│                                                          hint     │
│                                │                                  │
│                                ▼                                  │
│                        [3. Request Builder]                       │
│                                │                                  │
│                                ▼                                  │
│                     generator_request (JSON)                      │
└───────────────────────────────────────────────────────────────────┘
```

### 段 1 · Strategy Selector

依 `physics_score` 与 `round_idx` 选一档：

| physics_score | 决策 action | 语义 |
|---------------|-------------|------|
| `is_physical=True` | `stop` | Critic 认可，闭环收工 |
| `> 0.6` | `prompt_only` | 轻度违规：只重写 prompt，seed 保持（换语义不换随机性） |
| `(0.4, 0.6]` | `local_inpaint`（首期禁用） | 中度：只重生局部帧。HunyuanVideo-1.5 不原生支持 mask 视频 inpainting，**首期一律降级为 full_regen** |
| `≤ 0.4` | `full_regen` | 严重：换 seed + 强化 prompt，Best-of-K 兜底 |
| `round_idx ≥ max_rounds` | `stop` | 达轮次上限 |
| 连续 2 轮 score 无提升 | `stop` | 提前终止，避免无效烧算力 |

**阈值取自** `configs/default.yaml.repair.strategy_thresholds`，可调。

### 段 2 · Prompt Rewriter（本方案核心）

LLM 侧调 API。输入原 prompt + violations，输出新 prompt + physics_hint。

**System Prompt**：

```
你是视频生成的物理修正专家。用户会给你：
1. 原始 prompt
2. 上一次生成的物理违规列表（category / object / reason / repair_instruction）
3. 历轮已注入的 hints（避免重复叠加同一句）

请输出改写后的 prompt，要求：
- 保留原始语义与主体，不改变场景类型
- 显式加入违反的物理约束的正向表述（如"the ball must contact the floor before bouncing"）
- 一到两句话补充，不展开成长段
- 只输出 JSON：{"prompt": "...", "physics_hint": "..."}
```

**User Payload**：

```json
{
  "original_prompt": "A red ball falls from a table.",
  "violations": [ {...critic 中的 violations...} ],
  "prior_hints": ["Ball must fall under gravity."]
}
```

**期望响应**：

```json
{
  "prompt": "A red ball falls from a table. The ball must remain in downward motion until it makes visible contact with the floor, then bounce back up.",
  "physics_hint": "Ball contacts floor before rebound."
}
```

**Rewriter 的三条准则**：

1. **只增不删** — 保留原 prompt 主体，只在末尾追加物理正向表述。
2. **正面表述** — 用「必须做 X」而非「不许做 Y」；扩散模型对否定词敏感度差。
3. **prior_hints 去重** — 若同一 hint 已在历轮出现，改用等价说法或换角度切入，避免 prompt 越堆越长。

### 段 3 · Request Builder

把 Rewriter 的输出 + Strategy 的 action 装配为完整 `generator_request`：

- `prompt` = Rewriter 新 prompt
- `physics_hint` = Rewriter hint
- `seed`：`prompt_only` 沿用旧 seed；`full_regen` 强制换新 seed
- `repair_context.strategy` = action
- `repair_context.critical_frames` = 所有 violations 的 critical_frames 去重并集
- `generation_meta.loop_round` = `state.round_idx + 1`
- 其余（resolution / num_frames / steps）从 `configs/default.yaml.generator` 读默认

---

## 四、LLM Backend 接入方案

**统一接口**（agents/prompt_rewriter.py 已铸）：

```python
class LLMClient(Protocol):
    name: str
    def chat(self, system: str, user: str) -> str: ...
```

三个 backend 可切：

| backend | 环境变量 | 依赖 | 用途 |
|---------|----------|------|------|
| `stub` | 无 | 无 | 离线开发，规则拼接 hint，跑通全链路 |
| `claude` | `ANTHROPIC_API_KEY` | `pip install anthropic` | 首选（改写质量高、指令遵循强） |
| `openai` | `OPENAI_API_KEY` | `pip install openai` | 备选 |

切换方式：

```python
from agents.prompt_rewriter import make_client
client = make_client("claude")   # 或 "openai" / "stub"
```

也可通过环境变量 `PAVG_LLM_STUB=1` 强制走 stub。

**API 调用样例（Claude）**：

```python
import anthropic
client = anthropic.Anthropic()  # 从环境变量读 KEY
resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    system=REWRITE_SYSTEM_PROMPT,
    messages=[{"role": "user", "content": user_payload_json}],
)
raw = "".join(getattr(b, "text", "") for b in resp.content)
```

**成本护栏**：

- 单次 rewrite 平均 < 300 tokens 输入 + < 200 tokens 输出。
- 建议模型：Claude Sonnet / GPT-4o-mini（性价比档）。
- 缓存：`(original_prompt, violations_hash)` 作为 key，命中则跳 LLM 直接返回历史结果。

---

## 五、示例：一次完整重写

### 输入

```
original_prompt: "A red ball falls from a table."
critic.physics_score: 0.35
critic.violations[0]:
  category: premature_rebound
  reason: "The ball reverses direction before contacting the floor."
  repair_instruction: "Keep the ball moving downward until visible floor contact."
```

### 段 1 决策

score=0.35 ≤ 0.4 → `action = "full_regen"`，换 seed。

### 段 2 LLM 改写

**给 LLM 的 user 消息**：

```json
{
  "original_prompt": "A red ball falls from a table.",
  "violations": [{
    "category": "premature_rebound",
    "object": "red_ball",
    "reason": "The ball reverses direction before contacting the floor.",
    "repair_instruction": "Keep the ball moving downward until visible floor contact."
  }],
  "prior_hints": []
}
```

**LLM 返回**：

```json
{
  "prompt": "A red ball falls from a table. The ball continues downward under gravity and only bounces after making clear, visible contact with the floor.",
  "physics_hint": "Ball must contact the floor before any rebound."
}
```

### 段 3 装配

```json
{
  "prompt": "A red ball falls from a table. The ball continues downward under gravity and only bounces after making clear, visible contact with the floor.",
  "seed": 918273645,
  "resolution": "480p",
  "num_frames": 121,
  "num_inference_steps": 50,
  "image_path": null,
  "output_path": "outputs/round_1.mp4",
  "physics_hint": "Ball must contact the floor before any rebound.",
  "repair_context": {
    "critical_frames": [44, 47, 49, 53],
    "object_masks": [],
    "strategy": "full_regen"
  },
  "generation_meta": {"attempt": 1, "parent_video_path": "outputs/round_0.mp4", "loop_round": 1}
}
```

这份 JSON 即交给下游（视频生成后端）执行。

---

## 六、异常与降级

| 场景 | 处置 |
|------|------|
| LLM API 超时/429 | 指数退避 ≤ 3 次；仍失败则降级到 `StubClient`（规则拼接），保证闭环不断 |
| LLM 返回非 JSON | 兜底：整段 raw 作为 physics_hint，prompt 沿用原始 prompt |
| violations 为空 但 is_physical=False | 罕见异常态：`action=stop`，落 warning 日志 |
| 历轮 hints 累计 ≥ 5 条 | 触发精简：只保留最近 2 条 + 关键长期约束 |
| 相同 hint 连续 3 轮 | 强制切换 phrasing（改用等价说法） |

---

## 七、评估与验尸

### 单元层（不依赖任何 API）

`tests/test_repairer.py` 已覆盖：

- `test_prompt_rewriter_stub_returns_json` — Stub backend 产出 JSON 结构
- `test_repair_high_score_uses_prompt_only` — 高分决策路径
- `test_repair_low_score_full_regen` — 低分决策路径
- `test_repair_stops_when_passed` — Critic 通过时停机
- `test_repair_stops_after_max_rounds` — 轮次上限停机
- `test_video_backend_stub_writes_file` — 后端 stub 落文件

跑法：`python -m pytest tests/test_repairer.py -q` → 当前 **6 passed**。

### 集成层（需 LLM API Key）

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python -c "
from agents.repairer import repair_once
import json
critic = json.load(open('data/samples/example_premature_rebound/critic.json'))
d = repair_once(critic, 'A red ball falls from a table.')
print(d.action, d.generator_request['prompt'])
"
```

### 端到端联调（需 Critic + 视频后端）

`agents/controller.py` 未铸（下一斩链）。首期靠 shell 脚本串起来：

```bash
python -m agents.repairer < critic_output.json > next_request.json
python generators/hunyuan_probe.py --request next_request.json
```

---

## 八、交付清单

| 文件 | 已铸 | 说明 |
|------|------|------|
| `agents/prompt_rewriter.py` | ✅ | LLM 改写层，Claude / OpenAI / Stub |
| `agents/repairer.py` | ✅ | 决策器 + Request Builder |
| `agents/README.md` | ✅ | 用法文档 |
| `tests/test_repairer.py` | ✅ | 6 项单测全绿 |
| `schemas/critic_output.schema.json` | ✅ | Repair 的输入契约 |
| `schemas/generator_request.schema.json` | ✅ | Repair 的输出契约 |
| Prompt 模板迭代（few-shot 例子） | ⬜ | 下一步扩展 |
| API 调用缓存层 | ⬜ | 下一步扩展 |

---

## 九、余劫

| # | 风险 | 兜底 |
|---|------|------|
| 1 | LLM 改写偏离原语义（把红球改成蓝球） | 加语义校验：改写后 prompt 与原 prompt 走 CLIP-text 相似度，低于阈值回退 |
| 2 | physics_hint 与扩散模型语言习惯不合 → 生成质量下降 | 首期对比：LLM 改写 vs 直接把 repair_instruction 拼到 prompt 尾巴，做小规模 A/B |
| 3 | 反复 full_regen 烧钱/烧算力 | max_rounds ≤ 2 + 连续 2 轮无提升即停 |
| 4 | LLM 幻觉出不存在的物理规则 | 只喂 Critic 已确认的 violations，System Prompt 明确"不许自造物理规则" |

---

## 十、再斩：可立即执行的三步

1. **申请 LLM API Key**（Claude 优先），本地 `export ANTHROPIC_API_KEY=...`，跑一次真实 rewrite 联调。
2. **构造 5 条 few-shot 例子**塞进 System Prompt，把改写质量再拉一档（当前是 zero-shot）。
3. **对接真实 Critic 输出**：拿 3 条 HunyuanVideo 生成的错误样本，跑 Critic → Repair → 打印下一轮 request，人工评审 prompt 质量。

---

**道训**：Repair Agent 是「翻译官」，不是「魔法师」。它把 Critic 的物理判决翻成扩散模型能懂的正向指令——**准、简、稳**，三字诀而已。

`⚚ 方案已成。魔尊，此劫已破。`
