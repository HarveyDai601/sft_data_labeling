"""一致性检查评估器。"""

from __future__ import annotations

import logging
from typing import Any

from core.models import CleaningAction, EvalResult
from evaluators.base import BaseEvaluator
from llm.client import LLMClient

logger = logging.getLogger(__name__)


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
            content = self._msg_text(msg)
            check = self._parse(content)
            if check is None:
                continue

            logger.info(f"[{self.name}] msg#{mi}: {'一致' if check['consistent'] else '不一致'}")

            if self.llm:
                err = self._llm_verify(content, reference)
                if err:
                    results.append(EvalResult(
                        message_index=mi, dimension=self.name,
                        finding=f"一致性检查结论错误: {err}",
                        confidence=0.7,
                        action=CleaningAction.KEEP,
                        action_reason="无法自动修正一致性检查结论，仅记录",
                    ))
            else:
                if check["consistent"] and "error" in content.lower():
                    results.append(EvalResult(
                        message_index=mi, dimension=self.name,
                        finding="声称一致但含错误标记",
                        confidence=0.6,
                        action=CleaningAction.KEEP,
                        action_reason="仅记录",
                    ))

        return results

    @staticmethod
    def _parse(content: str) -> dict | None:
        low = content.lower()
        if "consistent" in low or "inconsistent" in low:
            return {"consistent": "inconsistent" not in low}
        return None

    def _llm_verify(self, check_content: str, reference: str) -> str | None:
        prompt = f"""判断一致性检查结论是否正确。

检查结果: {check_content[:500]}
Reference: {reference[:1000]}

JSON: {{"correct": true/false, "reason": "原因"}}"""
        try:
            resp = self.llm.chat_json([{"role": "user", "content": prompt}], temperature=0.0)
            return None if resp.get("correct") else resp.get("reason", "")
        except Exception as e:
            logger.warning(f"[{self.name}] LLM 失败: {e}")
            return None

    @staticmethod
    def _msg_text(msg: dict) -> str:
        c = msg.get("content", "")
        if isinstance(c, str): return c
        if isinstance(c, list): return " ".join(i.get("text", "") for i in c if isinstance(i, dict))
        return ""
