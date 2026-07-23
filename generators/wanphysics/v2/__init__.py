"""WanPhysics V2 Sidecar.

当前唯一在线闭环实现：单入口、三动作 Policy、Strict Enforce Gate、Re-Critic/Re-Gate
和 WanRepairTrialV3 审计。

设计约束（对齐 worklog/2026_07_21 修复方案 V2）：

1. 不删除历史文件；旧 forced/proxy 路径保留只读但不进入 runtime。
2. 真实运行只通过 ``run_videophy2_loop_v2.py --enable`` 开启。
3. CriticReport 不允许静默丢字段；无 mask 时禁用 local editing。
4. Policy 只决策一次，Executor 不再次调用 Policy。
5. Memory、四动作 proxy checkpoint 和 CLI action override 不进入在线决策。

本子包位于 ``generators/wanphysics/v2/``，与 ``repairer.py`` /
``executor_factory.py`` 同层，符合 README §3 目录边界（generators=视频生成探针）。
"""

from __future__ import annotations

__all__ = ["V2_PIPELINE_VERSION"]

V2_PIPELINE_VERSION = "v2"
