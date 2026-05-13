"""代码差异评估器 — predict vs reference 逐行 diff + LLM 语义判断。

逻辑：
- 从 write/edit 工具的 arguments 中提取 agent 实际写出的代码（predict）
- 与 reference.py 逐行 diff
- 无差异 → KEEP
- 有多余代码 → KEEP（记录供人工检查）
- 有缺失代码 → REPLACE（用 reference 替换 predict）
- LLM 判断功能等价 → KEEP
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from typing import Any

from core.models import CleaningAction, EvalResult
from evaluators.base import BaseEvaluator
from llm.client import LLMClient

logger = logging.getLogger(__name__)

# 写代码的工具 → arguments 中包含文件内容的 key
_WRITE_TOOL_CONTENT_KEYS: dict[str, list[str]] = {
    "write": ["content", "file_text", "text", "code"],
    "edit": ["content", "new_content", "file_text", "text", "code"],
    "create": ["content", "file_text", "text", "code"],
    "create_file": ["content", "file_text", "text", "code"],
    "write_file": ["content", "file_text", "text", "code"],
}


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
        if not reference.strip():
            return []

        results: list[EvalResult] = []
        total_blocks = 0

        for mi, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue

            # 1. 优先从 write/edit 工具调用中提取 predict code
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                if tool_name not in _WRITE_TOOL_CONTENT_KEYS:
                    continue
                code = self._extract_from_write_args(fn.get("arguments", ""), tool_name)
                if not code:
                    continue
                total_blocks += 1
                results.extend(self._compare_block(mi, None, code, reference))

            # 2. 如果没有 write 工具，回退到 content 中的代码块
            #    （适用于 main trace 中 assistant 直接输出代码的场景）
            if not msg.get("tool_calls"):
                for ii, code in enumerate(self._extract_code_blocks(msg)):
                    total_blocks += 1
                    results.extend(self._compare_block(mi, ii, code, reference))

        logger.info(f"[{self.name}] 扫描 {total_blocks} 个代码块, 产出 {len(results)} 条结果")
        return results

    def _compare_block(
        self, mi: int, ii: int | None, predict: str, reference: str
    ) -> list[EvalResult]:
        """对比一个代码块与 reference。"""
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

        results: list[EvalResult] = []

        # 缺少 reference 中的代码 → 用 reference 替换 predict
        if removed:
            missing_code = "".join(removed)
            snippet = missing_code.strip()[:100]
            logger.info(f"[{self.name}]   msg#{mi} 缺少: {snippet}...")

            # LLM 判断是否功能等价
            if self.llm:
                if self._llm_is_equivalent(predict, reference):
                    logger.info(f"[{self.name}]   LLM 判断: 功能等价，跳过")
                    return []

            results.append(
                EvalResult(
                    message_index=mi,
                    item_index=ii,
                    dimension=self.name,
                    finding=f"缺少 reference 中的代码: {snippet}...",
                    confidence=min(1.0, len(removed) * 0.2),
                    action=CleaningAction.REPLACE,
                    action_reason="用 reference 代码替换不完整的实现",
                    new_content=reference,
                    debug_info={
                        "predict_code_preview": predict[:500],
                        "reference_preview": reference[:500],
                        "diff_lines": [l.rstrip() for l in diff],
                        "added_count": len(added),
                        "removed_count": len(removed),
                    },
                )
            )

        # 多余的代码 → 记录但不自动删
        if added and not removed:
            extra_code = "".join(added)
            snippet = extra_code.strip()[:100]
            logger.info(f"[{self.name}]   msg#{mi} 多余: {snippet}...")

            results.append(
                EvalResult(
                    message_index=mi,
                    item_index=ii,
                    dimension=self.name,
                    finding=f"多余的代码: {snippet}...",
                    confidence=min(1.0, len(added) * 0.15),
                    action=CleaningAction.KEEP,
                    action_reason="多余的代码保留，供人工判断是否删除",
                    debug_info={
                        "predict_code_preview": predict[:500],
                        "reference_preview": reference[:500],
                        "diff_lines": [l.rstrip() for l in diff],
                        "extra_lines": [l.rstrip() for l in added],
                    },
                )
            )

        return results

    def _llm_is_equivalent(self, predict: str, reference: str) -> bool:
        """LLM 判断两段代码是否功能等价。"""
        prompt = f"""比较以下两段代码，判断它们是否功能等价（实现方式不同但结果相同）。

Reference:
```
{reference[:2000]}
```

Predict:
```
{predict[:2000]}
```

只回答 JSON: {{"equivalent": true/false, "reason": "一句话原因"}}"""

        try:
            resp = self.llm.chat_json([{"role": "user", "content": prompt}], temperature=0.0)
            return resp.get("equivalent", False)
        except Exception as e:
            logger.warning(f"[{self.name}] LLM 判断失败: {e}")
            return False

    # ── 工具方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_from_write_args(arguments: str, tool_name: str) -> str:
        """从 write/edit 工具的 arguments 中提取文件内容。"""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(args, dict):
            return ""

        keys = _WRITE_TOOL_CONTENT_KEYS.get(tool_name, [])
        for key in keys:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    @staticmethod
    def _extract_code_blocks(msg: dict) -> list[str]:
        """从 assistant content 中提取 ``` 代码块（回退方案）。"""
        blocks: list[str] = []
        content = msg.get("content", "")
        if isinstance(content, str):
            for m in re.finditer(r"```(?:\w*\n)?(.*?)```", content, re.DOTALL):
                blocks.append(m.group(1).strip())
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
