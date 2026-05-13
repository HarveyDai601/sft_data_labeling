"""参数准召率评估器 — reference 中的参数名是否被正确使用。"""

from __future__ import annotations

import ast
import re
from typing import Any

from core.models import EvalResult
from evaluators.base import BaseEvaluator


class ParamRecallEvaluator(BaseEvaluator):
    name = "param_recall"

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        ref_params = self._extract_params(reference)
        if not ref_params:
            return []

        trace_text = self._collect_text(messages)
        trace_params = self._extract_params(trace_text)

        missing = ref_params - trace_params
        extra = trace_params - ref_params

        results: list[EvalResult] = []

        if missing:
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
                        reason=f"reference 中的参数未使用: {', '.join(sorted(missing))}",
                        severity=min(1.0, len(missing) * 0.25),
                        suggested_fix={"missing_params": sorted(missing)},
                    )
                )

        if extra:
            for mi, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue
                msg_text = self._msg_text(msg)
                found_extra = [p for p in extra if p in msg_text]
                if found_extra:
                    results.append(
                        EvalResult(
                            message_index=mi,
                            dimension=self.name,
                            verdict="bad",
                            reason=f"多余的参数: {', '.join(sorted(found_extra))}",
                            severity=min(1.0, len(found_extra) * 0.15),
                        )
                    )

        return results

    # ── 内部方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_params(text: str) -> set[str]:
        """提取函数定义中的参数名 + 普通变量赋值。"""
        params: set[str] = set()
        # 函数参数: def f(a, b, c=1, *args, **kwargs)
        for m in re.finditer(r"def\s+\w+\s*\(([^)]*)\)", text):
            sig = m.group(1)
            for arg in re.findall(r"[\*]*(\w+)", sig):
                if arg not in ("self", "cls"):
                    params.add(arg)
        # 变量赋值: x = ..., x: int = ...
        for m in re.finditer(r"^(\s*)(\w+)\s*(?::\s*\w+\s*)?=", text, re.MULTILINE):
            var = m.group(2)
            if var not in ("def", "class", "if", "for", "while", "with", "return"):
                params.add(var)
        return params

    @staticmethod
    def _collect_text(messages: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for msg in messages:
            parts.append(ParamRecallEvaluator._msg_text(msg))
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
