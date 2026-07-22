# PhysGenLoop ProPainter 补齐与方案实施对齐报告

审查时间：2026-07-21 17:27:39 +08:00

服务器项目：`/root/PhysGenLoop-`

服务器 HEAD：`13c6289`

对齐基准：本地 `PhysGenLoop_全链路问题修复方案.md`

本报告只新增本地文档，不修改原始修复方案文档。服务器端本轮只完成 ProPainter 资产补齐、验证与日志归档，不改 V2 业务代码。

---

## 1. 本轮 ProPainter 资产补齐结论

ProPainter 资产层已经补齐，可以认为“ProPainter 后端环境就绪”。

但这不等于“Local Editing 全链路已完整符合方案”。原因是 V2 代码里的 mask manifest 到 ProPainter editor 的严格逐帧接线仍未完成，且 `local_editing.enabled` 配置尚未被 `build_backends.py` 严格消费。

### 1.1 已完成事项

| 项目 | 结果 |
|---|---|
| ProPainter repo 位置 | 已在 `/root/PhysGenLoop-/models/ProPainter`，与 `models/sam2-src` 同层 |
| ProPainter repo revision | `e870e79321c31b733e2031af5aa2fb1fe3ac7eec` |
| 推理脚本 | `/root/PhysGenLoop-/models/ProPainter/inference_propainter.py` 存在 |
| 依赖 | `envs/main/bin/pip check` 通过：`No broken requirements found.` |
| 配置路径 | `configs/loop_v2.yaml` 已指向 `/root/PhysGenLoop-/models/ProPainter` |
| 权重 | 三份 `.pth` 已补齐到 `/root/PhysGenLoop-/models/ProPainter/weights/` |
| 旧 `.part` | 已移出 weights，归档到 `worklog/2026_07_21/physgenloop_propainter_recurrent_flow_completion.pth.part.stale_20260721` |
| `--help` | `/root/PhysGenLoop-/envs/main/bin/python inference_propainter.py --help` 正常 |
| 模型加载 | RAFT / RecurrentFlowCompleteNet / InpaintGenerator 均成功加载 |
| V2 preflight | `all_ok=true`，`capability_mask.local_editing=true` |
| 最小 smoke | 8 帧 128x128 合成输入 + 逐帧 mask，真实 ProPainter 推理成功 |

### 1.2 服务器端权重 SHA256

| 文件 | 大小 | SHA256 |
|---|---:|---|
| `raft-things.pth` | 21,108,000 | `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1` |
| `recurrent_flow_completion.pth` | 20,348,681 | `22939a1a7900da878dbe1ccd011d646b1bfb30b8290039d8ff0e0c2fefbfd283` |
| `ProPainter.pth` | 157,780,510 | `12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591` |

本地下载目录：`E:\PAVG\propainter_weights_transfer\`

上传目标：`/root/PhysGenLoop-/models/ProPainter/weights/`

### 1.3 服务器验证日志

所有服务器侧日志和验证记录均放在 `worklog/2026_07_21/`：

| 日志/记录 | 用途 |
|---|---|
| `physgenloop_propainter_pip_20260721.log` | ProPainter requirements 安装日志 |
| `physgenloop_propainter_pip_20260721.status` | pip 安装状态，值为 `0` |
| `physgenloop_propainter_weights_20260721.log` | 早期服务器直连下载日志 |
| `physgenloop_propainter_weights_proxy_20260721.log` | proxy 下载尝试日志 |
| `physgenloop_propainter_weights_complete_20260721.log` | 兼容下载器尝试日志 |
| `physgenloop_propainter_verify_20260721.log` | 权重 SHA、import、help、模型加载、V2 preflight 验证 |
| `physgenloop_propainter_smoke_20260721.log` | 最小 ProPainter 推理 smoke |
| `propainter_smoke_20260721/` | smoke 输入帧、mask 与输出帧/mp4 |

### 1.4 ProPainter smoke 结果

Smoke 输入：

- 8 张 128x128 PNG 帧；
- 8 张 128x128 PNG mask；
- frame 2-5 为有效局部 mask，其余为空 mask；
- 使用 GPU 0，`--fp16`，`--raft_iter 2`。

Smoke 输出：

- `worklog/2026_07_21/propainter_smoke_20260721/result/frames/frames/0000.png` 至 `0007.png`
- `worklog/2026_07_21/propainter_smoke_20260721/result/frames/inpaint_out.mp4`
- `worklog/2026_07_21/propainter_smoke_20260721/result/frames/masked_in.mp4`

结论：ProPainter 资产和基础推理链路可用。

---

## 2. 方案与当前实现对齐总览

| 方案条目 | 当前实现程度 | 结论 |
|---|---|---|
| Phase 0 / W0b 迁移就绪门禁 | Wan/Qwen/SAM2/ProPainter 资产均已具备，ProPainter 已 smoke | 基本符合，但需要把 ProPainter 版本/SHA 写入正式 run manifest |
| P0-1 CriticReport round-trip | `v2/critic_codec.py` 已恢复 `violations/critical_frames/evidence/mask_uri` | 基本符合 |
| P0-2 mask evidence 传递 | SAM2 已落盘 `mask_uri/mask_uris`，V2 已生成 `mask_manifest.json` | 部分符合，manifest 未写回 evidence，Local target 仍指向单张 mask |
| P0-3 Local Editing backend preflight | ProPainter 资产补齐后 preflight 通过 | 资产符合，代码门控仍需加强 |
| P0-4 Action-aware Executor 闭环 | `runner.py` 已实现 executor 后立即 re-critic/re-gate | 基本符合 |
| P0-5 Prompt Executor 二次 Policy | `DecisionPromptRepairExecutor` 不再调用 Policy | 基本符合 |
| P1-1 Proxy Memory schema | `memory_adapter.py` 已做格式识别，默认 disabled | 部分符合，尚未和正式 policy 训练闭环验收 |
| P1-2 RepairTrace / RepairTrial | V2 trace 和 `WanRepairTrialV2` 已有 | 部分符合，forced actual trial 仍偏骨架 |
| P1-3 Semantic / Quality gate | `guardrails.py` 已支持 shadow/enforce | 部分符合，真实 semantic scorer 仍需 vLLM 实测 |
| P1-4 vLLM 进程所有权 | `resource_coordinator.py` 已有 owner manifest 逻辑 | 部分符合，真实 run 的 owner/stop 行为还需 smoke |
| P2 pilot300 | 未执行 | 未符合 |

当前最准确表述：

> ProPainter 资产已补齐并通过最小真实推理；V2 框架核心机制已有；但 Local Editing 的严格逐帧 mask 实现、配置门控、actual trial 和 full-chain GPU 验收仍未达到最终 Definition of Done。

---

## 3. 当前最关键的不对齐点

### 3.1 `local_editing.enabled=false` 目前没有真正控制 capability

当前 `configs/loop_v2.yaml`：

```yaml
local_editing:
  enabled: false
  require_mask: true
  allow_full_frame_fallback: false
  propainter_repo: /root/PhysGenLoop-/models/ProPainter
  propainter_script: /root/PhysGenLoop-/models/ProPainter/inference_propainter.py
  propainter_weights: /root/PhysGenLoop-/models/ProPainter/weights
  python: /root/PhysGenLoop-/envs/main/bin/python
```

但 `generators/wanphysics/v2/build_backends.py` 当前在 preflight 后直接使用：

```python
caps = dict(preflight.capability_mask)
```

由于 ProPainter 现在已安装，preflight 返回：

```json
{
  "all_ok": true,
  "capability_mask": {
    "prompt_repair": true,
    "global_regeneration": true,
    "local_editing": true,
    "reject": true
  }
}
```

这会导致即使配置里 `local_editing.enabled=false`，运行时 capability 仍可能暴露 `local_editing=true`。这不符合方案“默认保守关闭，显式开启”的 rollout 要求。

必须修改位置：

- `generators/wanphysics/v2/build_backends.py`
- 在 `preflight = run_preflight(...)` 后，`caps = dict(preflight.capability_mask)` 附近。

建议最小改法：

```python
preflight = run_preflight(
    propainter_repo=local_cfg.get("propainter_repo", "/root/PhysGenLoop-/models/ProPainter"),
    vllm_host=vllm_cfg.get("host", "127.0.0.1"),
    vllm_port=vllm_cfg.get("port", 18000),
    require_local_editing=bool(local_cfg.get("enabled", False)),
)
caps = dict(preflight.capability_mask)
if not bool(local_cfg.get("enabled", False)):
    caps["local_editing"] = False
```

原则：

- `preflight` 只回答“后端是否具备”；
- `local_editing.enabled` 回答“本 run 是否允许使用”；
- 二者必须同时为 true，最终 capability 才能是 true。

### 3.2 ProPainter preflight 仍应改为精确三权重检查

当前 `generators/wanphysics/v2/preflight.py` 已能识别 `.part`，这是正确补充。

但当前逻辑仍是“weights 目录中只要存在非空 `.pth` 即通过”。在历史状态里只有 `raft-things.pth` 时，这种逻辑会误报 ProPainter ready。

必须修改位置：

- `generators/wanphysics/v2/preflight.py`
- `check_propainter()` 函数。

建议替换为精确权重检查：

```python
REQUIRED_PROPAINTER_WEIGHTS = {
    "raft-things.pth": 20_000_000,
    "recurrent_flow_completion.pth": 20_000_000,
    "ProPainter.pth": 150_000_000,
}


def check_propainter(repo: str | None) -> CheckResult:
    """ProPainter 仓库 + 推理脚本 + 三权重完整性。"""

    if not _exists(repo):
        return CheckResult("propainter_repo", False, f"missing: {repo}")
    root = Path(repo)
    script = root / "inference_propainter.py"
    if not script.exists():
        return CheckResult("propainter_script", False, f"missing: {script}")

    weights = root / "weights"
    if not weights.exists():
        return CheckResult("propainter_weights", False, f"missing: {weights}")

    parts = sorted(p.name for p in weights.glob("*.part"))
    if parts:
        return CheckResult("propainter_weights", False, f"incomplete downloads: {parts}")

    missing: list[str] = []
    too_small: list[str] = []
    for name, min_bytes in REQUIRED_PROPAINTER_WEIGHTS.items():
        p = weights / name
        if not p.exists():
            missing.append(name)
            continue
        if p.stat().st_size < min_bytes:
            too_small.append(f"{name}:{p.stat().st_size}")

    if missing or too_small:
        return CheckResult(
            "propainter_weights",
            False,
            f"missing={missing}; too_small={too_small}",
        )
    return CheckResult("propainter", True, str(repo))
```

测试需要补：

- `tests/wanphysics_v2/test_preflight.py`
- 新增“只有一份 `.pth` 不能通过”、“三份完整 `.pth` 才能通过”、“存在 `.part` 必须失败”。

### 3.3 `mask_manifest.py` 注释与行为不一致

当前 `generators/wanphysics/v2/mask_manifest.py` 的 `build_local_edit_target()` 注释写：

```python
Local Editor 需要逐帧 mask，因此 mask_uri 指向 manifest（editor 据此按帧取 mask）
```

但实际代码：

```python
mask_uri=valid_paths[int(critical[0])],
```

也就是仍把第一张有效 mask 当作 `mask_uri`。这直接违反方案第 13 节：

> Local Editor 必须读取每帧 mask，不能把第一张 mask 复制到全部帧。

必须修改位置：

- `generators/wanphysics/v2/mask_manifest.py`
- `build_local_edit_target()`

建议最小改法：不改 canonical `LocalEditTarget` dataclass，仅让 V2 strict local editor 约定 `mask_uri` 指向 manifest 文件。

```python
def build_local_edit_target(
    *,
    parent_candidate_id: str,
    violation: Any,
    manifest: MaskManifest,
    manifest_uri: str,
) -> LocalEditTarget | None:
    valid_paths = manifest.valid_frame_paths(getattr(violation, "object", ""))
    critical = tuple(
        int(f)
        for f in (getattr(violation, "critical_frames", ()) or ())
        if int(f) in valid_paths
    )
    if not critical:
        return None
    return LocalEditTarget(
        parent_candidate_id=parent_candidate_id,
        objects=(str(getattr(violation, "object", "")),),
        start_frame=int(getattr(violation, "start_frame", min(critical))),
        end_frame=int(getattr(violation, "end_frame", max(critical))),
        critical_frames=critical,
        mask_uri=str(manifest_uri),
    )
```

测试需要改：

- `tests/wanphysics_v2/test_mask_manifest.py`
- 把 `assert target.mask_uri.endswith("baseball_00012.png")` 改成 `assert target.mask_uri.endswith("mask_manifest.json")`。

### 3.4 需要新增 strict ProPainter V2 editor，而不是继续扩写 legacy editor

当前 `generators/wanphysics/local_editor.py` 仍有两个方案冲突点：

```python
ref_mask = cv2.imread(str(target.mask_uri), cv2.IMREAD_GRAYSCALE)
```

它只读一张 mask。

```python
ref_mask = np.ones((h, w), dtype=np.uint8) * 255
```

无 mask 时使用全白 fallback。

```python
mask = dilated if i in active_frames else empty_mask
```

它把同一张 mask 复制到所有 active frames。

不建议继续把 strict manifest 逻辑塞进这个 legacy editor。那会形成“单张 mask / manifest / fallback / strict / 非 strict”混在一个类里的屎山。

建议新增文件：

- `generators/wanphysics/v2/propainter_strict_editor.py`

推荐实现方式：继承 legacy `ProPainterLocalEditor`，只覆盖 `_build_masks()`，保持已有抽帧、调用 ProPainter、编码视频逻辑复用。

代码骨架：

```python
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from generators.wanphysics.local_editor import ProPainterLocalEditor
from .mask_manifest import MaskManifest, verify_manifest


class StrictProPainterLocalEditor(ProPainterLocalEditor):
    """V2 strict local editor: target.mask_uri must point to mask_manifest.json."""

    def _build_masks(self, masks_dir: Path, frame_count: int, target, video_path: Path) -> None:
        if not target.mask_uri:
            raise RuntimeError("strict local editing requires mask manifest uri")

        manifest_path = Path(target.mask_uri)
        if not manifest_path.exists():
            raise RuntimeError(f"mask manifest missing: {manifest_path}")

        manifest = MaskManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
        ok, problems = verify_manifest(manifest, check_sha=True)
        if not ok:
            raise RuntimeError(f"mask manifest invalid: {problems}")

        cap = cv2.VideoCapture(str(video_path))
        ok_frame, first = cap.read()
        cap.release()
        if not ok_frame:
            raise RuntimeError(f"cannot read first frame: {video_path}")
        h, w = first.shape[:2]

        active = {int(f) for f in target.critical_frames}
        if not active:
            raise RuntimeError("strict local editing requires critical_frames")
        if any(f < 0 or f >= frame_count for f in active):
            raise RuntimeError(f"critical frame out of range: {sorted(active)}")

        object_names = target.objects or tuple(obj.name for obj in manifest.objects)
        frame_to_mask: dict[int, np.ndarray] = {}
        for object_name in object_names:
            for frame_index, frame in manifest.frames_for(object_name).items():
                if frame_index not in active:
                    continue
                if not frame.valid:
                    continue
                mask = cv2.imread(frame.path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise RuntimeError(f"cannot read mask: {frame.path}")
                if mask.shape[:2] != (h, w):
                    raise RuntimeError(f"mask size mismatch: {frame.path}")
                frame_to_mask[frame_index] = np.maximum(
                    frame_to_mask.get(frame_index, np.zeros((h, w), dtype=np.uint8)),
                    mask,
                )

        missing = sorted(active - set(frame_to_mask))
        if missing:
            raise RuntimeError(f"missing valid masks for critical frames: {missing}")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        empty = np.zeros((h, w), dtype=np.uint8)
        for i in range(frame_count):
            mask = cv2.dilate(frame_to_mask[i], kernel) if i in frame_to_mask else empty
            cv2.imwrite(str(masks_dir / f"{i:05d}.png"), mask)
```

注意：

- legacy `ProPainterLocalEditor` 保留不动，旧链路继续可用；
- V2 只用 `StrictProPainterLocalEditor`；
- 无 manifest / 无 mask / 空 mask / 尺寸不匹配 / frame 越界全部 fail closed；
- 不允许全白 fallback。

### 3.5 `build_backends.py` 需要从 manifest 生成 LocalEditTarget

当前 `build_backends.py` 已写 `mask_manifest.json` 并记录 `_mask_valid[cid]`，但没有保存 manifest path，也没有让 local decision 使用 manifest。

必须修改位置：

- `generators/wanphysics/v2/build_backends.py`
- `_ArtifactingCritic.evaluate()`
- `build_v2_runner()`
- `ActionAwareRunnerV2` 构造处

建议让 `RunArtifacts.write_mask_manifest()` 返回 path：

文件：`generators/wanphysics/v2/artifacts.py`

```python
def write_mask_manifest(self, sample_id: str, candidate_id: str, payload: dict[str, Any]) -> Path:
    path = self.candidate_dir(sample_id, candidate_id) / "mask_manifest.json"
    write_json(path, payload)
    return path
```

然后在 `_ArtifactingCritic` 中缓存 manifest 和 path：

```python
from .mask_manifest import build_manifest, has_valid_masks, build_local_edit_target

self._mask_manifest: dict[str, MaskManifest] = {}
self._mask_manifest_path: dict[str, Path] = {}
```

```python
manifest_path = self._artifacts.write_mask_manifest(self._sample_id, cid, manifest.to_dict())
self._mask_valid[cid] = has_valid_masks(manifest)
self._mask_manifest[cid] = manifest
self._mask_manifest_path[cid] = manifest_path
```

新增方法：

```python
def local_target(self, report: Any, candidate: Any):
    cid = str(getattr(candidate, "candidate_id", ""))
    manifest = self._mask_manifest.get(cid)
    manifest_path = self._mask_manifest_path.get(cid)
    if manifest is None or manifest_path is None:
        return None
    for violation in getattr(report, "violations", ()) or ():
        target = build_local_edit_target(
            parent_candidate_id=cid,
            violation=violation,
            manifest=manifest,
            manifest_uri=str(manifest_path),
        )
        if target is not None:
            return target
    return None
```

### 3.6 Runner 需要在 guard 决定 local 后注入 strict LocalEditTarget

当前 `runner.py` 的 `_with_action()` 只替换 action，不会给“被 guard 覆盖成 local_editing”的 decision 补 `local_target`。

必须修改位置：

- `generators/wanphysics/v2/runner.py`
- `ActionAwareRunnerV2.__init__`
- `run()` 中 `final_action = RepairAction(guard.final_action)` 后，构造 `ExecutionRequest` 前

建议最小改法：

```python
def __init__(..., local_target_fn: Callable[[Any, Any], Any | None] | None = None, ...):
    ...
    self.local_target_fn = local_target_fn
```

新增 helper：

```python
def _with_local_target(decision: Any, target: Any) -> Any:
    try:
        from dataclasses import replace
        return replace(decision, local_target=target)
    except Exception:
        return decision
```

执行前：

```python
decision_for_execution = _with_action(decision, final_action)
if final_action is RepairAction.LOCAL_EDITING:
    target = self.local_target_fn(report, current_candidate) if self.local_target_fn else None
    if target is None:
        rec.state = "EXECUTOR_FAILED"
        rec.terminal_reason = "local_target_missing"
        result.rounds.append(rec)
        result.stop_reason = "executor_failed"
        result.final_state = "EXECUTOR_FAILED"
        break
    decision_for_execution = _with_local_target(decision_for_execution, target)
```

再把 `ExecutionRequest(decision=...)` 改成：

```python
exec_request = ExecutionRequest(
    decision=decision_for_execution,
    ...
)
```

在 `build_backends.py` 构造 runner 时传入：

```python
local_target_fn=critic.local_target,
```

这样做的好处：

- Policy 仍然只决策一次；
- scope guard 仍然负责 action override；
- Local target 由 critic/artifacts 侧的 manifest 统一生成；
- runner 不需要知道 manifest 细节。

### 3.7 `build_backends.py` 应使用 strict editor 并避免无效注册

当前：

```python
editor = ProPainterLocalEditor(...)
registry = ExecutorRegistry(
    executors=[
        ...,
        MaskSequenceLocalEditingExecutor(editor=editor),
        ...
    ]
)
```

建议改为：

```python
from .propainter_strict_editor import StrictProPainterLocalEditor

editor = StrictProPainterLocalEditor(
    propainter_repo=local_cfg.get("propainter_repo", "/root/PhysGenLoop-/models/ProPainter"),
    python=local_cfg.get("python", python),
    output_root=sample_dir,
)

executors = [
    DecisionPromptRepairExecutor(generator=generator),
    OriginalPromptGlobalRegenerationExecutor(generator=generator),
    AuditedRejectExecutor(selector=EvidenceAwareSelector()),
]
if caps.get("local_editing", False):
    executors.insert(2, MaskSequenceLocalEditingExecutor(editor=editor))

registry = ExecutorRegistry(executors=executors)
```

如果要保留“注册但 capability=false 不执行”的测试便利，可以保留注册；但正式 run 更建议按 capability 条件注册，避免误触发时才发现后端不可用。

### 3.8 Forced actual trials 需要完整 WanRepairTrialV2 unavailable 记录

当前 `agents/wanphysics/run_actual_trials_v2.py` 在 local unavailable 时写的是简化 dict：

```python
{
    "trial_id": f"{sid}-unavailable",
    "action": action,
    "status": "unavailable",
    ...
}
```

这不满足方案第 20 节“每个正式 Trial 必须包含 before、decision、executor、after、score、failure reason、mask、critic backend 和 compatibility manifest”的结构化要求。

必须修改位置：

- `agents/wanphysics/run_actual_trials_v2.py`
- `_real_trials()` 中 local unavailable 分支

建议改为构造 `WanRepairTrialV2`，即使 unavailable 也保留完整 schema：

```python
trial = WanRepairTrialV2(
    trial_id=f"{sid}-unavailable",
    group_id=sid,
    source_candidate=CandidateRecord(
        candidate_id="unavailable_before",
        video_path="",
        prompt=prompt,
        seed=int(cfg.get("loop", {}).get("base_seed", 42)),
    ),
    prompt=prompt,
    critic_before={"status": "not_run", "reason": "local_editing capability masked by preflight"},
    decision=RepairDecision(
        action=RepairAction.LOCAL_EDITING,
        confidence=0.0,
        instruction="forced:local_editing",
        action_probabilities={a.value: (1.0 if a is RepairAction.LOCAL_EDITING else 0.0) for a in RepairAction},
        per_action_values={a.value: 0.0 for a in RepairAction},
        source="force_action_unavailable",
    ),
    execution={"status": "unavailable", "backend_id": "v2-mask-sequence-local-editor"},
    before_scores=ScoreBundle(physics=0.0),
    successful=False,
    failure_reason="local_editing capability masked by preflight",
)
artifacts.append_trial(sid, trial.to_dict())
```

如果 `CandidateRecord.video_path` 不允许空字符串，则应先生成 before candidate 并评估，再根据 action unavailable 写完整 before 信息。这是更严格的做法。

---

## 4. 当前不建议直接做的事

1. 不建议直接把 `configs/loop_v2.yaml` 的 `local_editing.enabled` 改成 `true`。

   ProPainter 资产已 ready，但 strict manifest editor 还没接好。现在打开会让 V2 有机会走 legacy 单张 mask editor，违反方案的逐帧 mask 要求。

2. 不建议在 `generators/wanphysics/local_editor.py` 里继续堆参数开关。

   这个类现在是 legacy 单张 mask editor。继续塞 `manifest_uri`、`allow_full_frame_fallback`、`strict`、`per_frame` 等分支，会把局部修复路径变成难审计的混合逻辑。

3. 不建议把 ProPainter smoke 当作 full-chain smoke。

   本轮 smoke 只证明 ProPainter 后端可运行；尚未证明 `SAM2 mask -> manifest -> strict editor -> repaired video -> re-critic -> gate` 的全链路。

---

## 5. 推荐下一步执行顺序

### Step A：先修最小门控

修改：

- `generators/wanphysics/v2/preflight.py`
- `generators/wanphysics/v2/build_backends.py`
- `tests/wanphysics_v2/test_preflight.py`
- `tests/wanphysics_v2/test_build_backends.py`

目标：

- 三权重精确检查；
- `local_editing.enabled=false` 时最终 capability 必须 false；
- `enabled=true` 且 preflight 全通过时 capability 才 true。

### Step B：新增 strict editor，不碰 legacy editor

新增：

- `generators/wanphysics/v2/propainter_strict_editor.py`
- `tests/wanphysics_v2/test_propainter_strict_editor.py`

目标：

- `target.mask_uri` 指向 `mask_manifest.json`；
- 逐帧读取每个 critical frame 的 mask；
- 无 mask、空 mask、全白 mask、尺寸错、帧越界全部 fail closed；
- 不允许全白 fallback。

### Step C：把 manifest path 注入 local target

修改：

- `generators/wanphysics/v2/artifacts.py`
- `generators/wanphysics/v2/mask_manifest.py`
- `generators/wanphysics/v2/build_backends.py`
- `generators/wanphysics/v2/runner.py`
- `tests/wanphysics_v2/test_mask_manifest.py`
- `tests/wanphysics_v2/test_runner.py`

目标：

- `mask_manifest.json` 路径可追踪；
- guard 覆盖为 local editing 时，也能生成合法 `LocalEditTarget`；
- executor 看到的是 manifest，不是第一张 mask。

### Step D：再开 `local_editing.enabled=true` 做 forced local smoke

只有 Step A-C 通过后，才建议改：

```yaml
local_editing:
  enabled: true
```

然后执行单样本 forced local smoke，验收证据必须包含：

- `run_manifest.json` 中 ProPainter repo revision 和三权重 SHA；
- `critic_report.json` 中 `violations/critical_frames/mask_uri/mask_uris` 不丢；
- `mask_manifest.json` 校验通过；
- `repair_decision.json` 中 `final_action=local_editing`；
- `repair_trace.jsonl` 中 `EXECUTING -> RE_EVALUATING`；
- repaired video 存在；
- after `critic_report.json` 存在；
- `loop_result.json` 中 after candidate 进入候选链。

---

## 6. 当前最终判断

ProPainter 部分按资产和基础推理验证已经完成：

- repo 位置正确；
- 配置路径正确；
- 依赖正确；
- 三权重补齐；
- 权重 SHA 可追溯；
- 模型类加载通过；
- 最小真实推理通过；
- V2 preflight 在资产层通过。

但按照 `PhysGenLoop_全链路问题修复方案.md` 的完整要求，服务器端代码仍未完全对齐：

- strict per-frame mask editor 未实现；
- manifest path 未真正进入 LocalEditTarget；
- `local_editing.enabled` 未严格控制 final capability；
- actual trial unavailable 记录仍不是完整 WanRepairTrialV2；
- full-chain forced local editing smoke 尚未完成。

因此当前应标记为：

```text
ProPainter backend readiness: PASS
V2 Local Editing strict integration: PARTIAL
Full-chain方案最终验收: NOT YET
```

