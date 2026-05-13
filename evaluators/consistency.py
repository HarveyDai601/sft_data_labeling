"""一致性检查评估器 — 检查结果是否正确识别了不一致。"""

from __future__ import annotations

from typing import Any

from core.models import EvalResult
from evaluators.base import BaseEvaluator
from llm.client import LLMClient


class ConsistencyEvaluator(BaseEvaluator):
    name = "consistency"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        results: list[EvalResult] = []

        for mi, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            # tool response 来自 consistency check
            content = self._msg_text(msg)
            if not content:
                continue

            # 检查一致性检查结论
            check_result = self._parse_consistency_result(content)
            if check_result is None:
                continue

            # 如果有 LLM，用 LLM 验证判断是否正确
            if self.llm:
                verdict = self._llm_verify(content, reference)
                if verdict:
                    results.append(
                        EvalResult(
                            message_index=mi,
                            dimension=self.name,
                            verdict=verdict["verdict"],
                            reason=verdict["reason"],
                            severity=verdict["severity"],
                        )
                    )
            else:
                # 无 LLM 时做基本检查
                if check_result.get("consistent") and "error" in content.lower():
                    results.append(
                        EvalResult(
                            message_index=mi,
                            dimension=self.name,
                            verdict="bad",
                            reason="一致性检查声称一致但内容包含错误标记",
                            severity=0.6,
                        )
                    )

        return results

    # ── 内部方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_consistency_result(content: str) -> dict[str, Any] | None:
        """尝试解析一致性检查结论。"""
        content_lower = content.lower()
        if "consistent" in content_lower or "inconsistent" in content_lower:
            return {"consistent": "inconsistent" not in content_lower}
        return None

    def _llm_verify(self, check_content: str, reference: str) -> dict[str, Any] | None:
        """用 LLM 验证一致性检查是否正确。"""
        prompt = f"""以下是一次代码一致性检查的结果。请判断这个检查结论是否正确。

一致性检查结果:
{check_content}

Reference 代码:
```
{reference}
```

请用 JSON 回答:
{{"correct": true/false, "reason": "原因", "severity": 0.0-1.0}}"""

        try:
            resp = self.llm.chat_json([{"role": "user", "content": prompt}])
            if resp.get("correct"):
                return None
            return {
                "verdict": "bad",
                "reason": f"一致性检查结论错误: {resp.get('reason', '')}",
                "severity": resp.get("severity", 0.5),
            }
        except Exception:
            return None

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
