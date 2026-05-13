"""参数准召率评估器 — reference 中的参数名是否被正确使用。

对 main trace：只检查含代码块的 assistant message，跳过调度指令。
对 code_completion sub：检查所有 assistant message。
"""

from __future__ import annotations

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

        is_main = context.get("trace_type") == "main"

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

        trace_params = self._extract_params(trace_text)

        missing = ref_params - trace_params
        extra = trace_params - ref_params

        results: list[EvalResult] = []

        if missing and code_assistant_indices:
            results.append(
                EvalResult(
                    message_index=code_assistant_indices[-1],
                    dimension=self.name,
                    verdict="missing",
                    reason=f"reference 中的参数未使用: {', '.join(sorted(missing))}",
                    severity=min(1.0, len(missing) * 0.25),
                    suggested_fix={"missing_params": sorted(missing)},
                )
            )

        if extra:
            for mi in code_assistant_indices:
                msg_text = self._msg_text(messages[mi])
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
    def _has_code_block(msg: dict[str, Any]) -> bool:
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
    def _extract_params(text: str) -> set[str]:
        params: set[str] = set()
        for m in re.finditer(r"def\s+\w+\s*\(([^)]*)\)", text):
            sig = m.group(1)
            for arg in re.findall(r"[\*]*(\w+)", sig):
                if arg not in ("self", "cls"):
                    params.add(arg)
        for m in re.finditer(r"^(\s*)(\w+)\s*(?::\s*\w+\s*)?=", text, re.MULTILINE):
            var = m.group(2)
            if var not in ("def", "class", "if", "for", "while", "with", "return"):
                params.add(var)
        return params

    @staticmethod
    def _collect_text(messages: list[dict[str, Any]]) -> str:
        return "\n".join(ParamRecallEvaluator._msg_text(msg) for msg in messages)

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
