"""工具调用决策评估器 — tool call 类型/顺序是否合理。"""

from __future__ import annotations

import logging
from typing import Any

from core.models import EvalResult
from evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class ToolCallEvaluator(BaseEvaluator):
    name = "tool_call_decision"

    _EXPECTED_ORDER = {
        "file_retrieval": 0,
        "step_retrieval": 1,
        "code_completion": 2,
        "consistency_check": 3,
    }

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        call_history: list[tuple[int, str]] = []

        for mi, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls", [])
            for tc in tool_calls:
                func_name = tc.get("function", {}).get("name", "")
                if func_name in self._EXPECTED_ORDER:
                    call_history.append((mi, func_name))

        if len(call_history) < 2:
            logger.info(f"[{self.name}] 调用次数 {len(call_history)} < 2，跳过顺序检查")
            return results

        logger.info(
            f"[{self.name}] 调用序列: "
            + " → ".join(f"#{mi}:{name}" for mi, name in call_history)
        )

        # 检查调用顺序
        for i in range(1, len(call_history)):
            prev_idx, prev_name = call_history[i - 1]
            curr_idx, curr_name = call_history[i]
            prev_order = self._EXPECTED_ORDER.get(prev_name, -1)
            curr_order = self._EXPECTED_ORDER.get(curr_name, -1)

            if curr_order < prev_order:
                logger.info(
                    f"[{self.name}] 顺序错误: {prev_name}(#{prev_idx}) → {curr_name}(#{curr_idx})"
                )
                results.append(
                    EvalResult(
                        message_index=curr_idx,
                        dimension=self.name,
                        verdict="bad",
                        reason=f"调用顺序不合理: {prev_name} 后不应调用 {curr_name}",
                        severity=0.5,
                    )
                )

        # 检查重复调用
        seen: dict[str, int] = {}
        for mi, name in call_history:
            if name in seen and name != "consistency_check":
                logger.info(f"[{self.name}] 重复调用: {name} (首次 #{seen[name]}, 再次 #{mi})")
                results.append(
                    EvalResult(
                        message_index=mi,
                        dimension=self.name,
                        verdict="bad",
                        reason=f"重复调用 {name}（首次在消息 #{seen[name]}）",
                        severity=0.4,
                    )
                )
            else:
                seen[name] = mi

        logger.info(f"[{self.name}] 产出 {len(results)} 条评估结果")
        return results
