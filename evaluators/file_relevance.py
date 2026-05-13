"""文件相关性评估器 — 检索的文件是否在 reference 的 import/依赖中。"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.models import CleaningAction, EvalResult
from evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class FileRelevanceEvaluator(BaseEvaluator):
    name = "file_relevance"

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        ref_files = self._extract_imported_files(reference)
        if not ref_files:
            return []

        results: list[EvalResult] = []
        for mi, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            retrieved = self._extract_retrieved_files(msg)
            if not retrieved:
                continue

            irrelevant = retrieved - ref_files
            missing = ref_files - retrieved

            logger.info(
                f"[{self.name}] msg#{mi}: 检索 {len(retrieved)}, "
                f"不相关 {len(irrelevant)}, 缺失 {len(missing)}"
            )

            if irrelevant:
                results.append(EvalResult(
                    message_index=mi, dimension=self.name,
                    finding=f"不相关文件: {', '.join(sorted(irrelevant))}",
                    confidence=min(1.0, len(irrelevant) * 0.3),
                    action=CleaningAction.KEEP,
                    action_reason="不相关文件保留，供人工判断",
                ))
            if missing:
                results.append(EvalResult(
                    message_index=mi, dimension=self.name,
                    finding=f"缺失文件: {', '.join(sorted(missing))}",
                    confidence=min(1.0, len(missing) * 0.4),
                    action=CleaningAction.KEEP,
                    action_reason="无法自动生成检索结果，仅记录",
                ))

        return results

    @staticmethod
    def _extract_imported_files(code: str) -> set[str]:
        files = set()
        for m in re.finditer(r"(?:from|import)\s+([\w.]+)", code):
            mod = m.group(1)
            files.add(mod.replace(".", "/") + ".py")
            files.add(mod)
        return files

    @staticmethod
    def _extract_retrieved_files(msg: dict) -> set[str]:
        files = set()
        content = msg.get("content", "")
        text = content if isinstance(content, str) else " ".join(
            i.get("text", "") for i in content if isinstance(i, dict)
        ) if isinstance(content, list) else ""
        for m in re.finditer(r"[\w/\\]+\.\w+", text):
            files.add(m.group())
        return files
