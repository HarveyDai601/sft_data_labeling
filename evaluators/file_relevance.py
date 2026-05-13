"""文件相关性评估器 — 检索的文件是否在 reference 的 import/依赖中。"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.models import EvalResult
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
            logger.info(f"[{self.name}] reference 中未提取到 import 文件，跳过")
            return []

        results: list[EvalResult] = []

        for mi, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            retrieved_files = self._extract_retrieved_files(msg)
            if not retrieved_files:
                continue

            relevant = retrieved_files & ref_files
            irrelevant = retrieved_files - ref_files
            missing = ref_files - retrieved_files

            logger.info(
                f"[{self.name}] msg#{mi}: 检索 {len(retrieved_files)} 个文件, "
                f"相关 {len(relevant)}, 不相关 {len(irrelevant)}, 缺失 {len(missing)}"
            )

            if irrelevant:
                logger.info(f"[{self.name}]   不相关: {sorted(irrelevant)}")
                results.append(
                    EvalResult(
                        message_index=mi,
                        dimension=self.name,
                        verdict="bad",
                        reason=f"检索了不相关的文件: {', '.join(sorted(irrelevant))}",
                        severity=min(1.0, len(irrelevant) * 0.3),
                    )
                )
            if missing:
                logger.info(f"[{self.name}]   缺失: {sorted(missing)}")
                results.append(
                    EvalResult(
                        message_index=mi,
                        dimension=self.name,
                        verdict="missing",
                        reason=f"未检索到需要的文件: {', '.join(sorted(missing))}",
                        severity=min(1.0, len(missing) * 0.4),
                        suggested_fix={"missing_files": sorted(missing)},
                    )
                )

        logger.info(f"[{self.name}] 产出 {len(results)} 条评估结果")
        return results

    # ── 内部方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_imported_files(code: str) -> set[str]:
        files: set[str] = set()
        for m in re.finditer(r"(?:from|import)\s+([\w.]+)", code):
            mod = m.group(1)
            files.add(mod.replace(".", "/") + ".py")
            files.add(mod)
        return files

    @staticmethod
    def _extract_retrieved_files(msg: dict[str, Any]) -> set[str]:
        files: set[str] = set()
        content = msg.get("content", "")
        if isinstance(content, str):
            for m in re.finditer(r"[\w/\\]+\.\w+", content):
                files.add(m.group())
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text", "") or str(item.get("content", ""))
                    for m in re.finditer(r"[\w/\\]+\.\w+", text):
                        files.add(m.group())
        return files
