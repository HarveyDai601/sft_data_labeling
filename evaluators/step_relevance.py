"""步骤相关性评估器 — 检索的步骤是否是 reference 实现所需。"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.models import EvalResult
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
        ref_steps = self._extract_implementation_steps(reference)
        if not ref_steps:
            logger.info(f"[{self.name}] reference 中未提取到实现步骤，跳过")
            return []

        results: list[EvalResult] = []

        for mi, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            retrieved_steps = self._extract_retrieved_steps(msg)
            if not retrieved_steps:
                continue

            relevant = retrieved_steps & ref_steps
            irrelevant = retrieved_steps - ref_steps
            missing = ref_steps - retrieved_steps

            logger.info(
                f"[{self.name}] msg#{mi}: 检索 {len(retrieved_steps)} 个步骤, "
                f"相关 {len(relevant)}, 不相关 {len(irrelevant)}, 缺失 {len(missing)}"
            )

            if irrelevant:
                snippet = ", ".join(sorted(irrelevant)[:5])
                logger.info(f"[{self.name}]   不相关: {snippet}")
                results.append(
                    EvalResult(
                        message_index=mi,
                        dimension=self.name,
                        verdict="bad",
                        reason=f"不相关的步骤: {snippet}",
                        severity=min(1.0, len(irrelevant) * 0.2),
                    )
                )
            if missing:
                snippet = ", ".join(sorted(missing)[:5])
                logger.info(f"[{self.name}]   缺失: {snippet}")
                results.append(
                    EvalResult(
                        message_index=mi,
                        dimension=self.name,
                        verdict="missing",
                        reason=f"缺失的步骤: {snippet}",
                        severity=min(1.0, len(missing) * 0.3),
                        suggested_fix={"missing_steps": sorted(missing)},
                    )
                )

        logger.info(f"[{self.name}] 产出 {len(results)} 条评估结果")
        return results

    # ── 内部方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_implementation_steps(code: str) -> set[str]:
        steps: set[str] = set()
        for m in re.finditer(r"def\s+(\w+)", code):
            steps.add(m.group(1))
        for m in re.finditer(r"class\s+(\w+)", code):
            steps.add(m.group(1))
        for kw in ("import", "open", "read", "write", "parse", "process",
                    "validate", "transform", "create", "delete", "update"):
            if kw in code:
                steps.add(kw)
        return steps

    @staticmethod
    def _extract_retrieved_steps(msg: dict[str, Any]) -> set[str]:
        steps: set[str] = set()
        content = msg.get("content", "")
        if isinstance(content, str):
            for m in re.finditer(r"\b(\w+(?:\s+\w+)?)\b", content.lower()):
                steps.add(m.group(1))
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text", "") or str(item.get("content", ""))
                    for m in re.finditer(r"\b(\w+(?:\s+\w+)?)\b", text.lower()):
                        steps.add(m.group(1))
        return steps
