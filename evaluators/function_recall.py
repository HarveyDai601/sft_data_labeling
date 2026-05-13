"""函数准召率评估器 — reference 中的函数是否出现在 trace 中。

对 main trace：只检查含代码块的 assistant message，跳过调度指令。
对 code_completion sub：检查所有 assistant message。
"""

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

        is_main = context.get("trace_type") == "main"

        # main trace：只收集含代码块的 assistant message 的文本
        if is_main:
            code_assistant_indices = [
                mi for mi, msg in enumerate(messages)
                if msg.get("role") == "assistant" and self._has_code_block(msg)
            ]
            if not code_assistant_indices:
                return []
            trace_text = "\n".join(
                self._msg_text(messages[mi]) for mi in code_assistant_indices
            )
        else:
            code_assistant_indices = [
                mi for mi, msg in enumerate(messages)
                if msg.get("role") == "assistant"
            ]
            trace_text = self._collect_text(messages)

        trace_funcs = self._extract_functions(trace_text)

        missing = ref_funcs - trace_funcs
        extra = trace_funcs - ref_funcs

        results: list[EvalResult] = []

        if missing and code_assistant_indices:
            # 标注在最后一条含代码的 assistant message
            results.append(
                EvalResult(
                    message_index=code_assistant_indices[-1],
                    dimension=self.name,
                    verdict="missing",
                    reason=f"reference 中的函数未出现: {', '.join(sorted(missing))}",
                    severity=min(1.0, len(missing) * 0.3),
                    suggested_fix={"missing_functions": sorted(missing)},
                )
            )

        if extra:
            for mi in code_assistant_indices:
                msg_text = self._msg_text(messages[mi])
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
    def _has_code_block(msg: dict[str, Any]) -> bool:
        """判断 message 是否包含代码块。"""
        content = msg.get("content", "")
        if isinstance(content, str):
            return "```" in content
        if isinstance(content, list):
            return any(
                item.get("type") == "code"
                or "```" in (item.get("text", "") or "")
                for item in content
                if isinstance(item, dict)
            )
        return False

    @staticmethod
    def _extract_functions(text: str) -> set[str]:
        """从代码文本中提取函数名。"""
        defs = set(re.findall(r"\bdef\s+(\w+)\s*\(", text))
        calls = set(re.findall(r"\b(\w+)\s*\(", text))
        _builtin = {"print", "len", "range", "str", "int", "float", "list", "dict",
                     "set", "tuple", "type", "isinstance", "getattr", "setattr",
                     "hasattr", "super", "classmethod", "staticmethod", "property",
                     "if", "for", "while", "with", "return", "yield", "raise"}
        return (defs | calls) - _builtin

    @staticmethod
    def _collect_text(messages: list[dict[str, Any]]) -> str:
        return "\n".join(FunctionRecallEvaluator._msg_text(msg) for msg in messages)

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
