"""把结构化 Critic 违规转换为下一轮提示词。"""

from __future__ import annotations

from pavg_critic.schemas import CriticReport


class InstructionPromptRepairer:
    """按报告顺序追加去重后的 repair_instruction。"""

    def repair(self, *, prompt: str, report: CriticReport) -> str:
        instructions = tuple(
            dict.fromkeys(
                item.repair_instruction.strip()
                for item in report.violations
                if item.repair_instruction.strip()
            )
        )
        if not instructions:
            return prompt
        return f"{prompt}\nPhysics correction: {' '.join(instructions)}"
