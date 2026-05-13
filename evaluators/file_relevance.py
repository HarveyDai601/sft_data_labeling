"""文件相关性评估器 — 检索的文件是否在 reference 的 import/依赖中。"""

from __future__ import annotations

import re
from typing import Any

from core.models import EvalResult
from evaluators.base import BaseEvaluator


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
            retrieved_files = self._extract_retrieved_files(msg)
            if not retrieved_files:
                continue

            relevant = retrieved_files & ref_files
            irrelevant = retrieved_files - ref_files
            missing = ref_files - retrieved_files

            if irrelevant:
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

        return results

    # ── 内部方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_imported_files(code: str) -> set[str]:
        """从 Python 代码中提取 import 的模块/文件。"""
        files: set[str] = set()
        # import xxx / from xxx import yyy
        for m in re.finditer(r"(?:from|import)\s+([\w.]+)", code):
            mod = m.group(1)
            # 转换模块路径到文件路径
            files.add(mod.replace(".", "/") + ".py")
            files.add(mod)
        return files

    @staticmethod
    def _extract_retrieved_files(msg: dict[str, Any]) -> set[str]:
        """从 tool response 中提取检索到的文件路径。"""
        files: set[str] = set()
        content = msg.get("content", "")
        if isinstance(content, str):
            # 尝试匹配文件路径
            for m in re.finditer(r"[\w/\\]+\.\w+", content):
                files.add(m.group())
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text", "") or str(item.get("content", ""))
                    for m in re.finditer(r"[\w/\\]+\.\w+", text):
                        files.add(m.group())
        return files
