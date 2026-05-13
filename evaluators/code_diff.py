"""代码差异评估器 — predict vs reference 逐行 diff + LLM 语义判断。"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Any

from core.models import EvalResult
from evaluators.base import BaseEvaluator
from llm.client import LLMClient

logger = logging.getLogger(__name__)


class CodeDiffEvaluator(BaseEvaluator):
    name = "code_diff"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        results: list[EvalResult] = []
        total_blocks = 0
        diff_blocks = 0

        for mi, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            code_blocks = self._extract_code_blocks(msg)
            for ii, code in enumerate(code_blocks):
                total_blocks += 1
                verdicts = self._compare_code(code, reference)
                if verdicts:
                    diff_blocks += 1
                    logger.info(
                        f"[{self.name}] msg#{mi} item#{ii}: 发现 {len(verdicts)} 个差异"
                    )
                for v in verdicts:
                    results.append(
                        EvalResult(
                            message_index=mi,
                            item_index=ii,
                            dimension=self.name,
                            verdict=v["verdict"],
                            reason=v["reason"],
                            severity=v["severity"],
                            suggested_fix=v.get("suggested_fix"),
                            debug_info={
                                "predict_code_preview": code[:500],
                                "reference_preview": reference[:500],
                                "diff_lines": v.get("diff_lines", []),
                                "llm_used": self.llm is not None,
                            },
                        )
                    )

        logger.info(
            f"[{self.name}] 扫描 {total_blocks} 个代码块, "
            f"{diff_blocks} 个有差异, 产出 {len(results)} 条评估结果"
        )
        return results

    # ── 内部方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_code_blocks(msg: dict) -> list[str]:
        blocks: list[str] = []
        content = msg.get("content", "")
        if isinstance(content, str):
            for m in re.finditer(r"```(?:\w*\n)?(.*?)```", content, re.DOTALL):
                blocks.append(m.group(1).strip())
            if not blocks and content.strip():
                blocks.append(content.strip())
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "code":
                        blocks.append(item.get("text", "") or item.get("content", ""))
                    elif item.get("type") == "text":
                        text = item.get("text", "")
                        for m in re.finditer(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL):
                            blocks.append(m.group(1).strip())
        return blocks

    def _compare_code(self, predict: str, reference: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []

        predict_lines = predict.splitlines(keepends=True)
        ref_lines = reference.splitlines(keepends=True)

        diff = list(difflib.unified_diff(ref_lines, predict_lines, lineterm=""))
        if not diff:
            return []

        added, removed = [], []
        for line in diff:
            if line.startswith("+") and not line.startswith("+++"):
                added.append(line[1:])
            elif line.startswith("-") and not line.startswith("---"):
                removed.append(line[1:])

        if removed:
            snippet = "".join(removed[:3]).strip()[:100]
            logger.info(f"[{self.name}]   缺少: {snippet}...")
            results.append({
                "verdict": "bad",
                "reason": f"缺少 reference 中的代码: {snippet}...",
                "severity": min(1.0, len(removed) * 0.2),
                "suggested_fix": {"type": "insert", "content": "".join(removed)},
                "diff_lines": [l.rstrip() for l in diff],
            })
        if added:
            snippet = "".join(added[:3]).strip()[:100]
            logger.info(f"[{self.name}]   多余: {snippet}...")
            results.append({
                "verdict": "bad",
                "reason": f"多余的代码: {snippet}...",
                "severity": min(1.0, len(added) * 0.15),
                "suggested_fix": {"type": "delete_lines", "content": "".join(added)},
                "diff_lines": [l.rstrip() for l in diff],
            })

        if self.llm and results:
            semantic = self._llm_semantic_check(predict, reference)
            if semantic:
                results.append(semantic)

        return results

    def _llm_semantic_check(self, predict: str, reference: str) -> dict[str, Any] | None:
        prompt = f"""比较以下两段代码，判断它们是否功能等价。

Reference 代码:
```
{reference}
```

Predict 代码:
```
{predict}
```

请用 JSON 回答:
{{"equivalent": true/false, "reason": "原因", "severity": 0.0-1.0}}"""

        try:
            resp = self.llm.chat_json([{"role": "user", "content": prompt}])
            if resp.get("equivalent"):
                logger.info(f"[{self.name}] LLM 判断: 功能等价")
                return None
            reason = resp.get("reason", "")
            logger.info(f"[{self.name}] LLM 判断: 功能不等价 — {reason}")
            return {
                "verdict": "bad",
                "reason": f"LLM 判断功能不等价: {reason}",
                "severity": resp.get("severity", 0.5),
            }
        except Exception as e:
            logger.warning(f"[{self.name}] LLM 语义检查失败: {e}")
            return None
