# 目录整理方案（B：物理自包含）

> 日期：2026-07-18
> 范围：整理 `/root/PhysGenLoop-` 目录结构 + 把 `/root/benchmark/pavg-benchmark` 的资产物理搬入
> 原则：README §2 数据/代码分离、§3 目录边界、§7 conda/venv 环境
> 状态：**待审阅，未执行**。审阅同意后逐条执行，全程不删文件（清理项只列清单，手动删）

---

## 一、目标结构

```text
/root/PhysGenLoop-/
├── README.md  pyproject.toml  .gitignore  .gitattributes  .env.example
├── configs/            # 保持
├── schemas/            # 保持
├── src/{pavg_critic, physgenloop}   # 保持
├── benchmarks/  evaluation/  examples/  tests/  docs/  worklog/   # 保持
├── generators/wanphysics/           # 保持（htz 已就位）
├── data_pipeline/                   # 🆕
│   ├── blender/        # ← Blender_video/（git mv）
│   └── sam2/           # ← SAM2_video/（git mv）
├── outputs/            # 运行产物（不入库）
│   ├── benchmarks/     # sy 全量评测产物（已入库部分保留）
│   │   └── runs/       # ← benchmark/runs（mv，不入库）
│   └── repair_training/ # ← artifacts/（git mv 后转为不入库）
├── data/               # ← benchmark/data（mv，17→3.8G，不入库）
│   └── videophy2/
├── models/             # ← benchmark/models（mv，17G，不入库）
│   └── sam2-src/  Qwen3-VL-8B-Instruct/  sam2.1_hiera_base_plus.pt
└── envs/               # 🆕 重建的虚拟环境（不入库）
    ├── main/           # 主环境（重建，指向本项目 src）
    ├── vllm/           # vllm 推理环境（重建）
    └── vllm-cu128/     # vllm cu128 环境（重建）
```

---

## 二、执行步骤（分 4 阶段）

### 阶段 A：PhysGenLoop- 内部违规目录归位（git，会上传 GitHub）

在 integration 分支上：

```bash
cd /root/PhysGenLoop-
mkdir -p data_pipeline
git mv Blender_video data_pipeline/blender
git mv SAM2_video   data_pipeline/sam2
# artifacts → outputs（先移动，再由 .gitignore 忽略）
mkdir -p outputs
git mv artifacts outputs/repair_training
```

改 `.gitattributes`（路径同步）：
- `/Blender_video/*.sh`        → `/data_pipeline/blender/*.sh`
- `/Blender_video/scripts/*.sh` → `/data_pipeline/blender/scripts/*.sh`
- `/artifacts/repair_training/**` → `/outputs/repair_training/**`

改 `.gitignore`：新增 `outputs/repair_training/` 之前，确认 `outputs/` 已整体忽略。
注意：`outputs/repair_training` 一旦被忽略，hj 的训练报告 JSON 将从 Git/GitHub 移除（用户已确认可接受）。
需 `git rm -r --cached outputs/repair_training` 让其脱离跟踪但保留本地文件。

同步内部路径引用：检查 Blender_video/SAM2_video 脚本内是否互相硬编码旧路径。

### 阶段 B：大资产物理搬入（mv，同分区秒移，不入库）

```bash
cd /root/PhysGenLoop-
mv /root/benchmark/pavg-benchmark/models  ./models
mv /root/benchmark/pavg-benchmark/data    ./data
mkdir -p outputs/benchmarks
mv /root/benchmark/pavg-benchmark/runs    ./outputs/benchmarks/runs
```

`.gitignore` 确认忽略：`models/  data/  outputs/`（现有规则已覆盖）。

### 阶段 C：重建虚拟环境（envs/，不入库）

用 uv（`/root/.local/bin/uv`）+ 19G uv 缓存（秒装），Python 3.12.13。

主环境：
```bash
cd /root/PhysGenLoop-
mkdir -p envs
/root/.local/bin/uv venv envs/main --python /root/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12
envs/main/bin/python -m pip install -e ".[video,env,test,sam2]" --no-build-isolation
# SAM2 官方源码 editable（指向搬入后的新位置）
envs/main/bin/python -m pip install -e ./models/sam2-src
```

vllm 两个环境（推理服务用，按需重建）：
```bash
/root/.local/bin/uv venv envs/vllm       --python .../python3.12
/root/.local/bin/uv venv envs/vllm-cu128 --python .../python3.12
# 各自 pip install vllm==0.25.1 等（依旧环境 freeze 清单）
```

`.gitignore` 确认忽略 `envs/`（或复用 `.venv/`、`venv/` 规则 → 建议新增 `envs/`）。

验证：
```bash
PYTHONPATH=src envs/main/bin/python -m pytest -q   # 期望 405 passed
envs/main/bin/python -c "import pavg_critic, physgenloop, torch, cv2, sam2"
```

### 阶段 D：验证 + 收尾

- 全量测试 405 passed
- `models/data/outputs/envs` 均被 gitignore（`git status` 不显示）
- git 结构：`git status` 只剩 data_pipeline 改名 + artifacts 转移 + .gitattributes/.gitignore

---

## 三、留在 /root/benchmark 不动（用户已确认）

| 目录 | 大小 | 处置 |
|------|------|------|
| `artifacts/`（依赖 tar/bundle/retry32-videos） | 4.6G | **留原地**，主结构搬完后再单独决定 |

## 四、清理清单（冗余，仅列出，手动删，不自动删）

前置检查已确认：三个 src 快照 HEAD 均已在 integration，dirty 的 `src`(2210e16) 改动已被 sy 正式提交且 integration 更完整，无独有代码。

| 目录/项 | 大小 | 为何冗余 |
|------|------|----------|
| `benchmark/src` | 13M | 过时快照(2210e16)，改动已进 integration（且更旧，含已废弃的硬编码 VLM 后端） |
| `benchmark/src-sy-c8d1810` | 34M | 过时快照(473d2cf)，已进 integration |
| `benchmark/report-src` | 6.1M | 过时快照(f371981)，已进 integration |
| `benchmark/venv`（旧） | 6.6G | 已按 B2 在 envs/main 重建，旧的可删 |
| `benchmark/vllm-venv`（旧） | 9.1G | 重建后旧的可删 |
| `benchmark/vllm-cu128-venv`（旧） | 9.4G | 重建后旧的可删 |
| `benchmark/logs tmp report-pytest-tmp report-*` | ~8M | 临时物 |

搬走 models/data/runs + 重建 venv 后，`/root/benchmark` 基本只剩 artifacts(4.6G) + 上述可删项。

## 五、风险与回滚

| 风险 | 兜底 |
|------|------|
| mv 大目录中断 | 同分区 mv 是原子改指针，不会半拷贝；万一中断，目标不存在则源仍完整 |
| venv 重建失败 | 旧 3 个 venv 暂不删（清理清单待手动），失败可回退用旧 venv |
| SAM2 editable 路径错 | 重装 `pip install -e ./models/sam2-src` 即可修正 |
| artifacts 转不入库后报告丢失 | 本地文件仍在 outputs/repair_training/，只是不上 GitHub（用户已确认） |
| git mv 后测试引用旧路径 | 阶段 D 全量 pytest 兜底 |

## 六、执行顺序建议

A（git 结构）→ 提交一次 → B（搬大资产）→ C（重建 env）→ D（验证）→ 用户确认后再处理清理清单。
