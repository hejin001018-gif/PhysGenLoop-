"""默认 VLM 复核边界。

真实 Video-VLM 适配器应实现 ``interfaces.VLMVerifier``，根据 request 中的视频和
``critical_frames`` 读取证据帧，并返回结构化 ``VLMReview``。核心包不会把未调用
VLM 的情况伪装成一个模型分数。
"""

from __future__ import annotations

from typing import Sequence

from .schemas import CriticRequest, ViolationCandidate, VLMReview


class NoOpVLMVerifier:
    """显式返回“无 VLM 证据”，用于纯规则基线和消融实验。"""

    def verify(
        self,
        request: CriticRequest,
        candidate: ViolationCandidate,
        critical_frames: Sequence[int],
    ) -> VLMReview | None:
        """返回 ``None``，让融合器仅以 detector/rule 分数作出判断。"""

        return None
