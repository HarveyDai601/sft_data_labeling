"""函数准召率评估器 — AST 初筛 + LLM 终判。

评估逻辑：
- 所有 API 都出现 → KEEP
- 有缺失 → KEEP（观察记录，不自动修改代码）
  - 因为无法自动生成缺失函数的实现代码

debug_info 中提供完整的中间数据供人工验证。
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

from core.models import CleaningAction, EvalResult
from evaluators.base import BaseEvaluator
from llm.client import LLMClient

logger = logging.getLogger(__name__)


class FunctionRecallEvaluator(BaseEvaluator):
    name = "function_recall"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        required_apis = self._extract_apis_from_code(reference)
        if not required_apis:
            logger.info(f"[{self.name}] reference 中未提取到 API，跳过")
            return []

        is_main = context.get("trace_type") == "main"
        target_indices = self._get_target_indices(messages, is_main)
        if not target_indices:
            logger.info(f"[{self.name}] 无目标 assistant message，跳过")
            return []

        # 提取 trace 中的 API
        trace_code_blocks: list[str] = []
        for mi in target_indices:
            trace_code_blocks.extend(self._extract_code_blocks(messages[mi]))

        trace_apis: set[str] = set()
        for code in trace_code_blocks:
            trace_apis.update(self._extract_apis_from_code(code))

        logger.info(
            f"[{self.name}] reference API: {sorted(required_apis)} | "
            f"trace API: {sorted(trace_apis)}"
        )

        # 差集 = 疑点
        suspects = required_apis - trace_apis
        if not suspects:
            logger.info(f"[{self.name}] 所有 API 均已出现，无缺失")
            return []

        logger.info(f"[{self.name}] 疑点（AST 差集）: {sorted(suspects)}")

        # LLM 终判
        llm_rejected: set[str] = set()
        if self.llm:
            confirmed, rejected = self._llm_verify(
                reference, "\n".join(trace_code_blocks), suspects
            )
            llm_rejected = rejected
            if rejected:
                logger.info(f"[{self.name}] LLM 过滤掉的误报: {sorted(rejected)}")
            if not confirmed:
                logger.info(f"[{self.name}] LLM 判断全部为等价实现，无真正缺失")
                return []
            suspects = confirmed
            logger.info(f"[{self.name}] LLM 确认真正缺失: {sorted(suspects)}")
        else:
            ref_defs = self._extract_top_level_defs(reference)
            before = suspects.copy()
            suspects = suspects & ref_defs
            llm_rejected = before - suspects
            if llm_rejected:
                logger.info(f"[{self.name}] 启发式过滤: {sorted(llm_rejected)}")

        if not suspects:
            return []

        # 输出：只观察，不执行清洗
        missing_ratio = len(suspects) / len(required_apis)
        confidence = min(1.0, missing_ratio * 1.5)

        logger.info(
            f"[{self.name}] 最终: 缺失 {sorted(suspects)}, "
            f"confidence={confidence:.2f}"
        )

        return [
            EvalResult(
                message_index=target_indices[-1],
                dimension=self.name,
                finding=f"核心 API 缺失: {', '.join(sorted(suspects))}",
                confidence=round(confidence, 2),
                action=CleaningAction.KEEP,
                action_reason="无法自动生成缺失函数实现，仅记录",
                debug_info={
                    "reference_apis": sorted(required_apis),
                    "trace_apis": sorted(trace_apis),
                    "suspects_ast": sorted(required_apis - trace_apis),
                    "llm_rejected": sorted(llm_rejected),
                    "confirmed_missing": sorted(suspects),
                    "llm_used": self.llm is not None,
                    "target_messages": target_indices,
                    "trace_code_blocks_count": len(trace_code_blocks),
                },
            )
        ]

    # ── LLM 终判 ──────────────────────────────────────────────────────────

    def _llm_verify(
        self, reference: str, trace_code: str, suspects: set[str],
    ) -> tuple[set[str], set[str]]:
        suspects_list = ", ".join(sorted(suspects))
        prompt = f"""你是一个代码审查专家。以下是 reference 代码和 agent 实际产出的代码。

Reference 代码:
```python
{reference}
```

Agent 产出的代码:
```python
{trace_code}
```

AST 分析发现以下函数/API 在 agent 代码中未出现:
{suspects_list}

请逐个判断：这些缺失的 API 是否是**真正必须的**？

判断标准：
- 如果 reference 中定义了该函数且它是核心逻辑 → 必须
- 如果 trace 用了不同的库/方法实现了相同功能 → 不必须（等价实现）
- 如果是工具性调用且 trace 用了替代方案 → 不必须
- 如果是 reference 中的核心业务函数且 trace 完全没实现 → 必须

请用 JSON 回答:
{{"confirmed": ["真正缺失的API列表"], "rejected": ["有等价实现的API列表"], "reasons": {{"API名": "判断原因"}}}}"""

        try:
            resp = self.llm.chat_json([{"role": "user", "content": prompt}], temperature=0.0)
            confirmed = set(resp.get("confirmed", []))
            rejected = set(resp.get("rejected", []))
            reasons = resp.get("reasons", {})
            for api, reason in reasons.items():
                logger.info(f"[{self.name}] LLM 判断 {api}: {reason}")
            return confirmed, rejected
        except Exception as e:
            logger.warning(f"[{self.name}] LLM 验证失败: {e}")
            ref_defs = self._extract_top_level_defs(reference)
            return suspects & ref_defs, set()

    # ── AST 解析 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_apis_from_code(code: str) -> set[str]:
        apis: set[str] = set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set(re.findall(r"\bdef\s+(\w+)\s*\(", code))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                apis.add(node.name)
            elif isinstance(node, ast.Call):
                name = FunctionRecallEvaluator._get_call_name(node.func)
                if name:
                    apis.add(name)
        return apis

    @staticmethod
    def _extract_top_level_defs(code: str) -> set[str]:
        defs: set[str] = set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set(re.findall(r"\bdef\s+(\w+)\s*\(", code))
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defs.add(node.name)
        return defs

    @staticmethod
    def _get_call_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    # ── 工具方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_target_indices(messages: list[dict], is_main: bool) -> list[int]:
        if is_main:
            return [
                mi for mi, msg in enumerate(messages)
                if msg.get("role") == "assistant" and FunctionRecallEvaluator._has_code_block(msg)
            ]
        return [mi for mi, msg in enumerate(messages) if msg.get("role") == "assistant"]

    @staticmethod
    def _has_code_block(msg: dict[str, Any]) -> bool:
        content = msg.get("content", "")
        if isinstance(content, str):
            return "```" in content
        if isinstance(content, list):
            return any(
                item.get("type") == "code" or "```" in (item.get("text", "") or "")
                for item in content if isinstance(item, dict)
            )
        return False

    @staticmethod
    def _extract_code_blocks(msg: dict[str, Any]) -> list[str]:
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
                        for m in re.finditer(r"```(?:\w*\n)?(.*?)```", item.get("text", ""), re.DOTALL):
                            blocks.append(m.group(1).strip())
        return blocks
