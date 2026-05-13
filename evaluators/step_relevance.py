"""步骤相关性评估器 — 检索的步骤是否是 reference 实现所需。"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.models import CleaningAction, EvalResult
from evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class StepRelevanceEvaluator(BaseEvaluator):
    name = "step_relevance"

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        ref_steps = self._extract_steps(reference)
        if not ref_steps:
            return []

        results: list[EvalResult] = []
        for mi, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            retrieved = self._extract_retrieved_steps(msg)
            if not retrieved:
                continue

            irrelevant = retrieved - ref_steps
            missing = ref_steps - retrieved

            logger.info(
                f"[{self.name}] msg#{mi}: 检索 {len(retrieved)}, "
                f"不相关 {len(irrelevant)}, 缺失 {len(missing)}"
            )

            if irrelevant:
                results.append(EvalResult(
                    message_index=mi, dimension=self.name,
                    finding=f"不相关步骤: {', '.join(sorted(irrelevant)[:5])}",
                    confidence=min(1.0, len(irrelevant) * 0.2),
                    action=CleaningAction.KEEP,
                    action_reason="不相关步骤保留，供人工判断",
                ))
            if missing:
                results.append(EvalResult(
                    message_index=mi, dimension=self.name,
                    finding=f"缺失步骤: {', '.join(sorted(missing)[:5])}",
                    confidence=min(1.0, len(missing) * 0.3),
                    action=CleaningAction.KEEP,
                    action_reason="无法自动生成检索结果，仅记录",
                ))

        return results

    @staticmethod
    def _extract_steps(code: str) -> set[str]:
        steps = set()
        for m in re.finditer(r"def\s+(\w+)", code): steps.add(m.group(1))
        for m in re.finditer(r"class\s+(\w+)", code): steps.add(m.group(1))
        for kw in ("import", "open", "read", "write", "parse", "process", "validate", "transform"):
            if kw in code: steps.add(kw)
        return steps

    @staticmethod
    def _extract_retrieved_steps(msg: dict) -> set[str]:
        steps = set()
        content = msg.get("content", "")
        text = content if isinstance(content, str) else " ".join(
            i.get("text", "") for i in content if isinstance(i, dict)
        ) if isinstance(content, list) else ""
        for m in re.finditer(r"\b(\w+(?:\s+\w+)?)\b", text.lower()):
            steps.add(m.group(1))
        return steps
