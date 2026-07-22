"""WanPhysics V2 Sidecar.

一条与现有 ``ActionAwareRunnerV2`` / canonical Learning Repair 完全隔离的闭环实现，
用于在不删除、不覆盖旧模块的前提下修复全链路的证据丢失、动作未执行、Trial 不
规范和接受门禁缺失问题。

设计约束（对齐 worklog/2026_07_21 修复方案 V2）：

1. 只新增、不删除；不修改 ``src/pavg_critic/schemas.py`` 或任何旧入口。
2. 所有新行为默认关闭，通过 ``configs/loop_v2.yaml`` 或 CLI 显式开启。
3. CriticReport 不允许静默丢字段；无 mask 时禁用 local editing。
4. Policy 只决策一次，Executor 不再次调用 Policy。
5. proxy 与 actual trial 分开；proxy checkpoint 需显式 research 开关。

本子包位于 ``generators/wanphysics/v2/``，与 ``repairer.py`` /
``executor_factory.py`` 同层，符合 README §3 目录边界（generators=视频生成探针）。
"""

from __future__ import annotations

__all__ = ["V2_PIPELINE_VERSION"]

V2_PIPELINE_VERSION = "v2"
