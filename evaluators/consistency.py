"""一致性检查评估器 — 检查结果是否正确识别了不一致。"""

from __future__ import annotations

import logging
from typing import Any

from core.models import EvalResult
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
        checks = 0

        for mi, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            content = self._msg_text(msg)
            if not content:
                continue

            check_result = self._parse_consistency_result(content)
            if check_result is None:
                continue

            checks += 1
            logger.info(
                f"[{self.name}] msg#{mi}: 一致性检查结论 — "
                f"{'一致' if check_result.get('consistent') else '不一致'}"
            )

            if self.llm:
                verdict = self._llm_verify(content, reference)
                if verdict:
                    logger.info(f"[{self.name}]   LLM 判断: 结论错误 — {verdict['reason']}")
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
                    logger.info(f"[{self.name}]   LLM 判断: 结论正确")
            else:
                if check_result.get("consistent") and "error" in content.lower():
                    logger.info(f"[{self.name}]   启发式判断: 声称一致但含错误标记")
                    results.append(
                        EvalResult(
                            message_index=mi,
                            dimension=self.name,
                            verdict="bad",
                            reason="一致性检查声称一致但内容包含错误标记",
                            severity=0.6,
                        )
                    )

        logger.info(f"[{self.name}] 扫描 {checks} 次一致性检查, 产出 {len(results)} 条评估结果")
        return results

    # ── 内部方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_consistency_result(content: str) -> dict[str, Any] | None:
        content_lower = content.lower()
        if "consistent" in content_lower or "inconsistent" in content_lower:
            return {"consistent": "inconsistent" not in content_lower}
        return None

    def _llm_verify(self, check_content: str, reference: str) -> dict[str, Any] | None:
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
        except Exception as e:
            logger.warning(f"[{self.name}] LLM 验证失败: {e}")
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
