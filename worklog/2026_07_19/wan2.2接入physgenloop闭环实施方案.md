# Wan2.2-TI2V-5B 接入 PhysGenLoop 闭环 · 实施方案

> 日期：2026-07-19
> 依据：`worklog/2026_07_18/07-18-WanPhysics.md`（htz 原始双机架构说明）、`worklog/2026_07_18/合并四人分支实施结果.md`（htz 分支合并现状）、本次 htz 新提交 `d1d7595`（本机 diffusers 推理版 `wan_generator.py`）
> 范围：仅为方案，未动代码；涉及的环境/权重检查均已在服务器上实际核实（见各节标注）

---

## 一、现状梳理

### 1.1 htz 原架构（07-18 文档）是双机的
```
Mac（控制层，prompts/ + scripts/）
   │ ssh + rsync
   ▼
px-cloud1（生成层：Wan2.2 官方 generate.py + conda myconda 环境）
```
`run_wan.sh` 从 Mac 触发远程 `generate_video.sh`，等生成完再 `rsync` 回本地。这套设计的前提是"控制端"和"GPU 生成端"是两台不同机器。

### 1.2 现在已经单机化
项目现在整体跑在这台 A100 服务器上，不再有独立的 Mac 控制端和 px-cloud1 生成端。07-18 文档里"SSH 触发 + rsync 回传"这一整段机制已经没有存在的必要——生成和消费（Critic 读视频）都在同一台机器、同一个文件系统上。

### 1.3 htz 今天新提交的版本已经是单机思路
`generators/wanphysics/wan_generator.py`（已在服务器本地 main，commit `d1d7595`，尚未同步到 Windows 本地）不再依赖 Wan2.2 官方仓库 + conda 环境，改为直接用 `diffusers.WanPipeline` 在本进程内加载模型推理。这条路线正好匹配单机场景，不用额外部署一套 Wan2.2 官方代码，装 `diffusers` 即可。

### 1.4 一个不采用的遗留方案
远端 GitHub `main` 分支里还有 `src/physgenloop/wan_generator.py`（`WanVideoGenerator` 类，"合并四人分支"文档第七节列为"尚未接入"）。这是给双机架构设计的桥接器：SSH 触发远程生成 + 轮询等待 `rsync` 落地。单机场景下这套"发命令-等待-同步"的逻辑纯属多余的复杂度，建议**不采用**，本方案走下面的本机适配器路线。

---

## 二、目标

把 `wan_generator.py`（diffusers 本机推理版）接入 `physgenloop.controller.LoopController` 的 Generate → Critic → Refine → Generate 闭环，替换测试用的 `DeterministicFakeGenerator`，让 `LoopController` 跑起来时真的生成视频、真的被 Critic 评价。

---

## 三、已核实的差距

在服务器上逐项检查过，结论如下：

| 检查项 | 结果 |
|---|---|
| GPU | `NVIDIA A100-PCIE-40GB`，40960 MiB，当前空闲。Wan2.2-TI2V-5B 官方最低要求 24GB，够用 |
| 磁盘 | `/root` 所在盘 124G 可用，模型权重约 10GB 量级，够用 |
| `envs/main` 环境依赖 | 已有 `torch 2.7.1+cu128`（CUDA 可用），**没有** `diffusers`/`transformers`/`imageio`/`imageio-ffmpeg`/`accelerate`/`safetensors`/`sentencepiece` |
| 模型权重 | `models/` 下没有任何 Wan2.2 权重，需要下载 |
| 网络 | `huggingface.co` 直连不通（IPv6 连不上，IPv4 无响应）；**`hf-mirror.com` 可用**（实测对 `Wan-AI/Wan2.2-TI2V-5B-Diffusers` 文件返回 307 重定向）；`pypi.org` 可正常访问（`pip install` 不受影响） |
| `wan_generator.py` 现状 | 类设计本身可以复用：构造时加载一次模型，`generate_video()` 可重复调用——不需要为"避免重复加载模型"重构这个类 |
| `wan_generator.py` 参数问题 | 默认 `num_frames=16` 不满足 Wan2.2 要求的 `4n+1` 帧数格式；默认 `height=480, width=720` 不是官方推荐的 720P 分辨率（`1280×704` 或 `704×1280`），且与 07-18 文档描述的产品目标（1280×704）不一致 |
| `requirements.txt`（htz 提交，已合并） | 只列了 `diffusers`/`imageio`/`imageio-ffmpeg` 三项，缺 `transformers`（`WanPipeline.from_pretrained` 会拉取文本编码器）、`accelerate`、`safetensors`、`sentencepiece`（T5 分词器依赖） |
| `physgenloop.interfaces.VideoGenerator` 协议 | `generate(*, prompt, physics_plan, seed) -> GeneratedCandidate`，`GeneratedCandidate` 要求 `candidate_id`/`video_path`/`prompt`/`seed`/`metadata`。`wan_generator.py` 目前是脚本风格，返回值是文件路径字符串，不满足协议，需要一层适配器 |
| Critic 侧消费方式 | `pavg_critic` 拿到 `video_path` 后用 `cv2.VideoCapture` 直接读文件，没有对生成来源做任何假设——只要落地的是能被 OpenCV 打开的真实 mp4，就能直接对接，不需要改 Critic 任何代码 |
| diffusers 对该模型的支持方式 | 已查证：`Wan-AI/Wan2.2-TI2V-5B-Diffusers` 用的是 `WanPipeline`（不是 `WanImageToVideoPipeline`），该 checkpoint 通过 `expand_timesteps=True` 让同一个 pipeline 同时支持纯文本生成和图生视频。htz 的代码选用 `WanPipeline` 是对的 |

---

## 四、实施步骤

### 阶段 0：环境准备
1. 给 `envs/main` 补依赖（`pypi.org` 可访问，直接装）：
   ```
   diffusers transformers accelerate safetensors sentencepiece imageio imageio-ffmpeg
   ```
   （`imageio`/`imageio-ffmpeg`/`diffusers` 已在 `generators/wanphysics/requirements.txt` 里，需要补全剩余几项）
2. 下载模型权重，走镜像（`huggingface.co` 不通，`hf-mirror.com` 已验证可用）：
   ```bash
   HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
     Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir models/wan2.2_ti2v_5b
   ```
   `models/` 已经在 `.gitignore` 里（对齐现有 `models/`（17G）的大资产管理方式），下载下来的权重不会误入库。

### 阶段 1：修正 `wan_generator.py` 的两个默认参数问题
- `num_frames` 默认值改为符合 `4n+1` 格式的官方推荐值（如 `81`，对应约 3.4 秒 @ 24fps），而不是当前的 `16`
- `height`/`width` 默认改为官方推荐的 `704×1280`（对齐 07-18 文档"1280×704"的产品目标），而不是当前的 `480×720`
- 两者都做成可覆盖的构造/调用参数，不写死

### 阶段 2：新增适配器，接上 `VideoGenerator` 协议
新增 `generators/wanphysics/adapter.py`：
- `WanPhysicsGenerator` 类，实现 `generate(*, prompt, physics_plan, seed) -> GeneratedCandidate`
- 构造时持有一个 `WanGenerator` 实例（模型只加载一次；`LoopController` 的 Best-of-K 会在一次 `run()` 里多次调用 `generate`，避免每次候选都重新加载 5B 模型）
- `seed` 传给 diffusers 的 `torch.Generator`，确保同一 prompt 的多个候选之间有差异
- 输出目录沿用 07-18 文档的规范（`outputs/{candidate_id}/{candidate_id}-v01.mp4` + `prompt.txt` + `metadata.json`），为将来外部工具复用留一个稳定的文件布局；`critic.json` 占位字段暂不需要（`LoopController` 内已经直接用 `PhysicsCriticAdapter` 在内存里拿到结构化报告，不用再落一份文件互相传递）
- `physics_plan` 参数按协议签名接收但暂不使用（YAGNI：目前没有需要把 plan 里的 objects/events 拼回 prompt 的需求，先不做）

### 阶段 3：端到端接线
新增一个装配脚本（风格参考 `examples/evaluate_video.py` 的 `.env` 加载方式），组装：
```python
generator = WanPhysicsGenerator(model_path="models/wan2.2_ti2v_5b")
critic = PhysicsCriticAdapter(PhysicsCritic(...))   # 复用现有真实 Critic，不改内部逻辑
repairer = InstructionPromptRepairer()
selector = EvidenceAwareSelector()
controller = LoopController(
    generator=generator, critic=critic, repairer=repairer, selector=selector,
    config=LoopConfig(max_rounds=..., candidates_per_round=...),
)
result = controller.run(prompt="...")
```

### 阶段 4：小规模验证再放大
- 先用 `max_rounds=1, candidates_per_round=1` + 缩小分辨率/帧数跑通链路，确认生成的 mp4 能被 `cv2.VideoCapture` 正常打开、Critic 能正常出报告
- 链路通了之后再切回官方推荐参数（704×1280、81 帧）

---

## 五、风险与注意事项

- **显存叠加**：A100 40GB 单独跑 TI2V-5B 够（官方最低 24GB），但如果同进程里还加载了 Critic 的 VLM/SAM2 模型，显存会叠加，建议先分开验证再考虑是否需要同进程常驻
- **耗时**：Best-of-K 会成倍增加生成次数，5B 模型跑一次默认 50 步推理，官方分辨率下单次大概是分钟级；配置 `LoopConfig` 的 `max_rounds`/`candidates_per_round` 时要把总耗时算进去
- **网络依赖**：模型下载必须走 `hf-mirror.com`，如果该镜像后续不可用需要换其他镜像源或手动传权重
- **两套 API Key 互不相关**：`.env` 里的 `API_KEY`/`BASE_URL` 是 Critic 调 VLM 用的，跟本地视频生成完全无关，两边互不影响

---

## 六、明确不做的事（YAGNI）

- 不采用远端 GitHub `main` 里遗留的 SSH 桥接版 `wan_generator.py`（`WanVideoGenerator`），单机化之后这套双机同步逻辑没有存在必要
- 不改动 `pavg_critic` 内部任何逻辑
- 不现在就做 07-18 文档"长期"规划里的 Web 界面/多模型支持，超出本次任务范围
