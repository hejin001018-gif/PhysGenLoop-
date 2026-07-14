# PAVG Critic 操作指南

本文面向两种情况：在当前电脑逐步确认 Critic 是否正常，以及在一台全新电脑上从零部署。命令以 Windows PowerShell 为主，Linux/macOS 的差异单独标出。

## 1. 先选择运行档位

| 档位 | 能力 | 需要什么 |
|---|---|---|
| A：轨迹自检 | 规则、问题图、检查表、力学、融合 | Python；不需要 OpenCV/API |
| B：真实视频 | A + 视频解码 + 默认颜色检测/跟踪 | NumPy、OpenCV；不需要 API |
| C：DeepSeek | B + 模型生成 PQSG 文本问题图 | `DEEPSEEK_API_KEY` |
| D：OpenAI | B + PQSG 图 + 关键帧多模态复核 | `OPENAI_API_KEY`、显式模型名 |
| E：自定义 CV | B + 通用检测/跟踪/光流/外观证据 | 自行注入 detector 或 `VisualEvidenceExtractor` |

建议按 A → B → C/D 顺序检查。前一档失败时不要继续增加 API 或模型变量。

## 2. 新电脑从零部署

### 2.1 安装基础软件

安装：

1. Git。
2. 64 位 Python 3.10–3.12；当前验证环境为 Python 3.12。
3. 可选：VS Code/PyCharm。

确认命令：

```powershell
git --version
python --version
```

如果 Windows 的 `python` 指向错误版本，可使用 `py -3.12` 替代下面命令中的 `python`。

### 2.2 克隆仓库

```powershell
git clone https://github.com/hejin001018-gif/PhysGenLoop-.git
cd PhysGenLoop-
git status
```

确认根目录存在 `pyproject.toml` 和 `src/pavg_critic/`。如果 Critic 2.0 尚未合并到默认分支，应切换到实际保存这些文件的分支。

### 2.3 创建独立虚拟环境

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Linux/macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

看到命令行前缀出现 `(.venv)` 后再继续。

### 2.4 安装项目

只跑轨迹和单元测试：

```powershell
python -m pip install -e ".[test]"
```

需要真实视频：

```powershell
python -m pip install -e ".[video,test]"
```

项目已删除旧 `requirements.txt`，`pyproject.toml` 是唯一依赖源。不要同时维护两份依赖列表。

### 2.5 安装后健康检查

```powershell
python -m pip check
python -c "import pavg_critic; print(pavg_critic.__version__)"
python -c "from pavg_critic import PhysicsCritic; print('import ok')"
```

预期版本是 `0.3.0`，`pip check` 应显示没有破损依赖。

## 3. 按顺序检测代码是否正常

### 第一步：运行全部自动化测试

```powershell
python -m pytest -q
```

当前版本预期 62 个测试通过。未来增加测试后数量可能变多，判断标准是退出码 0 且没有 failed/error。

若失败，先保存完整 traceback，不要直接运行 API。

### 第二步：编译检查

```powershell
python -m compileall -q src tests benchmarks
```

命令无输出且退出码为 0 表示所有 Python 文件可编译。

### 第三步：校验默认配置

```powershell
python -c "from pavg_critic import load_config; c=load_config('configs/default.yaml'); print(c)"
```

配置是严格模式：未知 section 或拼错字段会直接报错，避免实验悄悄使用默认值。

### 第四步：运行不需要视频/API 的标准示例

```powershell
python -m pavg_critic `
  --request examples/critic_request.json `
  --observations examples/observations.json `
  --config configs/default.yaml `
  --floor-y 100 `
  --output outputs/example_report.json
```

检查摘要：

```powershell
python -c "import json; r=json.load(open('outputs/example_report.json')); print(r['decision'], r['physics_score'], r['confidence'], r['coverage'])"
```

当前示例预期 `decision=physical`、`coverage=0.85`。它证明轨迹→事件→规则→问题图→检查表→力学→融合链路可运行。

### 第五步：校验输出 schema

```powershell
python -c "import json,jsonschema; s=json.load(open('schemas/critic_output.schema.json')); r=json.load(open('outputs/example_report.json')); jsonschema.validate(r,s); print('schema ok')"
```

### 第六步：运行冻结评估

```powershell
python benchmarks/evaluate_critic.py --mode B1_RULE --output outputs/eval_b1.json
python benchmarks/evaluate_critic.py --mode M3_MECHANICS --output outputs/eval_m3.json
```

查看指标：

```powershell
python -c "import json; print(json.load(open('outputs/eval_b1.json'))['metrics'])"
```

冻结 6 样例用于回归，预期 accuracy/F1 为 1.0；它不能作为论文 benchmark 性能。

### 第七步：确认真实视频依赖

```powershell
python -c "import cv2,numpy; print(cv2.__version__, numpy.__version__)"
```

检查自己的视频：

```powershell
python -c "import cv2; p=r'C:\path\to\video.mp4'; c=cv2.VideoCapture(p); print(c.isOpened(), int(c.get(cv2.CAP_PROP_FRAME_COUNT)), c.get(cv2.CAP_PROP_FPS)); c.release()"
```

`isOpened()` 必须为 `True`，帧数和 FPS 应大于 0。

### 第八步：运行真实视频

先创建请求，例如 `my_request.json`：

```json
{
  "schema_version": "2.0",
  "video_path": "C:/videos/red_ball.mp4",
  "prompt": "A red ball falls, contacts the floor, and rebounds.",
  "physics_plan": {
    "objects": ["red_ball", "floor"],
    "expected_events": ["fall", "floor_contact", "rebound"]
  }
}
```

运行：

```powershell
python -m pavg_critic --request my_request.json --config configs/default.yaml --output outputs/my_report.json
```

重要：默认 detector 是受控红球 HSV baseline。它不适合任意行人、车辆或复杂背景视频；通用视频出现大量碎片轨迹/违规时，应先替换 detector/tracker，而不是调低最终阈值掩盖问题。

## 4. 配置 DeepSeek API

DeepSeek 当前只接入结构化文本路径，用于生成增量 PQSG 问题图；关键帧图像复核请使用支持图像输入的适配器。

### 4.1 当前 PowerShell 会话临时设置

```powershell
$env:DEEPSEEK_API_KEY="你的密钥"
$env:DEEPSEEK_MODEL="deepseek-chat"
```

只检查是否存在，不显示密钥：

```powershell
python -c "import os; assert os.getenv('DEEPSEEK_API_KEY'); print('DeepSeek key configured')"
```

### 4.2 运行

```python
from pavg_critic import CriticRequest, DeepSeekChatModel, PhysicsCritic

model = DeepSeekChatModel.from_env()
critic = PhysicsCritic(question_model=model)
report = critic.analyze(CriticRequest.from_json("my_request.json"))
print(report.to_json())
```

第一次实际调用会产生网络请求和可能的 API 费用。先用小视频/少量样例验证，不要直接跑完整 benchmark。

## 5. 配置 OpenAI API

OpenAI 适配器使用 Responses API 的结构化输出；同一个支持图像输入的模型可用于 PQSG 图和关键帧 VLM。

### 5.1 设置环境变量

```powershell
$env:OPENAI_API_KEY="你的密钥"
$env:OPENAI_MODEL="你的可用模型名"
```

项目故意不硬编码“最新模型”。`OPENAI_MODEL` 必须由你按账户权限选择一个支持结构化输出的模型；启用关键帧 VLM 时还必须支持图像输入。

检查变量：

```powershell
python -c "import os; assert os.getenv('OPENAI_API_KEY'); assert os.getenv('OPENAI_MODEL'); print('OpenAI environment configured')"
```

### 5.2 仅生成 PQSG 图

```python
from pavg_critic import OpenAIResponsesModel, PhysicsCritic

model = OpenAIResponsesModel.from_env()
critic = PhysicsCritic(question_model=model)
```

### 5.3 PQSG 图 + 关键帧 VLM

```python
from pavg_critic import (
    CriticRequest,
    EvidenceGroundedVLMVerifier,
    OpenAIResponsesModel,
    PhysicsCritic,
)

model = OpenAIResponsesModel.from_env()
critic = PhysicsCritic(
    question_model=model,
    vlm_verifier=EvidenceGroundedVLMVerifier(model, model_name=model.model),
)
report = critic.analyze(CriticRequest.from_json("my_request.json"))
print(report.to_json())
```

VLM 只读取定位后的关键帧；没有违规候选或没有关键帧时不会调用图像 API。

## 6. 密钥的持久化和安全

临时 `$env:...` 只对当前 PowerShell 有效，最安全也最容易清理。若必须持久化：

```powershell
setx OPENAI_API_KEY "你的密钥"
setx OPENAI_MODEL "你的模型名"
```

`setx` 后需要关闭并重新打开终端；当前窗口不会自动更新。DeepSeek 同理。

安全要求：

- 不要把密钥写入 `configs/default.yaml`、JSON 请求、Python 源码或截图。
- `.env` 已被 gitignore，但项目不会自动加载 `.env`；如自行引入 dotenv，要保持密钥不入库。
- CI/CD 使用 GitHub Actions Secrets、系统环境变量或云端 secret manager。
- 日志只记录 provider/model，不记录 Authorization header。
- API `base_url` 必须为绝对 HTTPS 地址，防止 Bearer 密钥经明文 HTTP 传输。
- 密钥一旦误提交，立即在供应商后台撤销并重新生成；仅删除 Git 文件不够。

## 7. 如何阅读检测结果

优先按以下顺序：

1. `decision`：最终三态。
2. `coverage`：低于默认 0.35 时通常为 `unknown`。
3. `violations`：类别、对象、区间、关键帧、原因和修复建议。
4. `evidence_bundles`：五个家族是否 `available/unknown/not_applicable/failed`。
5. `diagnostics.video_science`：五维检查表。
6. `diagnostics.morpheus_mechanics`：适用的力学模型、NMSE/恢复系数等。
7. `graph_evaluation` 和 `node_results`：O/A/P 问题的回答、blocked 根因与覆盖率。
8. `diagnostics.provider_failures`：可选 API 超时/响应错误；出现时 Critic 已降级到模板、规则或无 VLM 路径。

处理原则：

- `physical`：证据覆盖达到阈值且融合分数通过。
- `violation`：规则已确认违规，或充分覆盖下融合分数低于阈值。
- `unknown`：不要当成物理正确；先补检测、轨迹、问题或模型证据。

## 8. B0/B1/M1–M5 正确运行方式

- B0_PQSG：必须使用官方 PQSG 仓库实际生成的 `psg/answers/score`，再用 `load_pqsg_evaluation_records()` 读取。
- B1_RULE：`python benchmarks/evaluate_critic.py --mode B1_RULE`。
- M1_GRAPH：离线模板问题图。
- M2_CHECKLIST：M1 + 五维检查表。
- M3_MECHANICS：M2 + 四类力学。
- M4_VLM：必须显式注入真实多模态 verifier。
- M5_FULL：必须同时注入模型 PQSG 与 VLM。

代码会拒绝在 NoOp 模型下把 B0/M4/M5 标成正式实验。

## 9. 外部评估资产（可选）

外部文件位于 `evaluation/external/`，不进入 Git。`evaluation/external_manifest.json` 记录来源、大小和 SHA-256。

验证已有文件：

```powershell
Get-FileHash -Algorithm SHA256 evaluation\external\opencv_vtest.avi
Get-FileHash -Algorithm SHA256 evaluation\external\pqsg-main.zip
```

OpenCV 视频只用于解码烟雾测试，不用于报告物理精度；官方 PQSG 快照只用于核对 B0 输出和 tree-score 语义。不要在这些文件上写标注或缓存。

## 10. 常见故障

### `No module named pavg_critic`

确认虚拟环境已激活，并重新执行：

```powershell
python -m pip install -e ".[video,test]"
```

### 找不到 `pavg-critic` 命令

虚拟环境未激活，或 Scripts 不在 PATH。直接使用：

```powershell
python -m pavg_critic --help
```

### `No module named cv2`

```powershell
python -m pip install -e ".[video]"
```

### `Set OPENAI_API_KEY` / `Set DEEPSEEK_API_KEY`

环境变量没有进入运行 Python 的那个终端。用“不打印值”的检查命令确认，并检查 IDE 的 Run Configuration 是否继承了环境变量。

### OpenAI 返回模型/结构化输出错误

检查 `OPENAI_MODEL` 是否属于当前账户，并同时支持结构化输出；启用 VLM 时还需支持图像输入。不要仅通过修改解析器吞掉服务端错误。

### 视频输出大量错误违规

先检查 detector/tracker 是否适配场景。默认红球 HSV baseline 在通用视频上会过检，这是感知前端问题，不是把 `violation_threshold` 调高就能科学解决的问题。

### 报告是 `unknown`

检查每个 `evidence_bundle.status` 和 `coverage`。常见原因是没有轨迹、PhysicsPlan 为空、问题节点无法回答、力学不适用或 API 未启用。

### pip 自身损坏

先尝试：

```powershell
python -m ensurepip --upgrade
python -m pip --version
```

如果仍失败，删除虚拟环境并重新创建通常比修补全局 Python 更安全。不要在损坏环境中继续安装模型依赖。

## 11. 每次换电脑后的最短验收清单

依次确认：

- [ ] 虚拟环境激活，Python 3.10+。
- [ ] `pip check` 通过。
- [ ] `import pavg_critic` 显示 0.3.0。
- [ ] `pytest -q` 无失败。
- [ ] `compileall` 无错误。
- [ ] observation 示例输出通过 schema 2.0。
- [ ] B1/M3 冻结评估可运行。
- [ ] OpenCV 能打开目标视频。
- [ ] 默认 detector 与目标场景匹配，或已替换。
- [ ] API 环境变量只存在于 secret/环境，不在仓库。
- [ ] 小样本 API smoke 通过后再运行付费批量实验。
