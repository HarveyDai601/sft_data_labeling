"""工具调用决策评估器 — tool call 类型/顺序是否合理。"""

from __future__ import annotations

import logging
from typing import Any

from core.models import CleaningAction, EvalResult
from evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class ToolCallEvaluator(BaseEvaluator):
    name = "tool_call_decision"

    _EXPECTED_ORDER = {
        "script_retrieval": 0,
        "step_retrieval": 1,
        "script_completion": 2,
        "consistency_check": 3,
    }

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        calls = [
            (mi, tc.get("function", {}).get("name", ""))
            for mi, msg in enumerate(messages)
            if msg.get("role") == "assistant"
            for tc in msg.get("tool_calls", [])
            if tc.get("function", {}).get("name", "") in self._EXPECTED_ORDER
        ]

        if len(calls) < 2:
            return results

        logger.info(f"[{self.name}] 调用序列: {' → '.join(f'#{mi}:{n}' for mi, n in calls)}")

        # 顺序检查
        for i in range(1, len(calls)):
            prev_mi, prev_name = calls[i - 1]
            curr_mi, curr_name = calls[i]
            if self._EXPECTED_ORDER[curr_name] < self._EXPECTED_ORDER[prev_name]:
                results.append(EvalResult(
                    message_index=curr_mi, dimension=self.name,
                    finding=f"顺序错误: {prev_name} 后调用 {curr_name}",
                    confidence=0.5,
                    action=CleaningAction.KEEP,
                    action_reason="顺序错误不自动修改，仅记录",
                ))

        # 重复检查
        seen: dict[str, int] = {}
        for mi, name in calls:
            if name in seen and name != "consistency_check":
                results.append(EvalResult(
                    message_index=mi, dimension=self.name,
                    finding=f"重复调用 {name}（首次 #{seen[name]}）",
                    confidence=0.4,
                    action=CleaningAction.KEEP,
                    action_reason="重复调用不自动删除，仅记录",
                ))
            else:
                seen[name] = mi

        return results
