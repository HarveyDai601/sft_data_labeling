"""函数准召率评估器 — reference 中的函数是否出现在 trace 中。"""

from __future__ import annotations

import re
from typing import Any

from core.models import EvalResult
from evaluators.base import BaseEvaluator


class FunctionRecallEvaluator(BaseEvaluator):
    name = "function_recall"

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        ref_funcs = self._extract_functions(reference)
        if not ref_funcs:
            return []

        # 收集 trace 中所有出现的函数名
        trace_text = self._collect_text(messages)
        trace_funcs = self._extract_functions(trace_text)

        missing = ref_funcs - trace_funcs
        extra = trace_funcs - ref_funcs

        results: list[EvalResult] = []

        if missing:
            # 找到应该出现但没出现的位置（在最后一条 assistant message 标注）
            last_assistant = -1
            for mi, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    last_assistant = mi
            if last_assistant >= 0:
                results.append(
                    EvalResult(
                        message_index=last_assistant,
                        dimension=self.name,
                        verdict="missing",
                        reason=f"reference 中的函数未出现: {', '.join(sorted(missing))}",
                        severity=min(1.0, len(missing) * 0.3),
                        suggested_fix={"missing_functions": sorted(missing)},
                    )
                )

        if extra:
            # 找到多余函数首次出现的位置
            for mi, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue
                msg_text = self._msg_text(msg)
                found_extra = [f for f in extra if f in msg_text]
                if found_extra:
                    results.append(
                        EvalResult(
                            message_index=mi,
                            dimension=self.name,
                            verdict="bad",
                            reason=f"多余的函数: {', '.join(sorted(found_extra))}",
                            severity=min(1.0, len(found_extra) * 0.2),
                        )
                    )

        return results

    # ── 内部方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_functions(text: str) -> set[str]:
        """从代码文本中提取函数名。"""
        # Python: def func_name(
        # 也匹配调用: func_name(
        defs = set(re.findall(r"\bdef\s+(\w+)\s*\(", text))
        calls = set(re.findall(r"\b(\w+)\s*\(", text))
        # 过滤内置/关键字
        _builtin = {"print", "len", "range", "str", "int", "float", "list", "dict",
                     "set", "tuple", "type", "isinstance", "getattr", "setattr",
                     "hasattr", "super", "classmethod", "staticmethod", "property",
                     "if", "for", "while", "with", "return", "yield", "raise"}
        return (defs | calls) - _builtin

    @staticmethod
    def _collect_text(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for msg in messages:
            parts.append(FunctionRecallEvaluator._msg_text(msg))
        return "\n".join(parts)

    @staticmethod
    def _msg_text(msg: dict[str, Any]) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        return ""
