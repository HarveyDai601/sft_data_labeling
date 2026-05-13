"""步骤相关性评估器 — 检索的步骤是否是 reference 实现所需。

逻辑：
1. 遍历 assistant 的 tool_calls，识别步骤检索类工具（step_retrieval、grep、search 等）
2. 从 arguments 中提取查询关键词
3. 通过 tool_call_id 关联 tool response，提取实际返回的步骤
4. 与 reference.py 的函数/类/关键操作比对，判断相关性
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from core.models import CleaningAction, EvalResult
from evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

# sub-agent 类型中涉及步骤检索的
_STEP_SUBAGENT_TYPES = {"step_retrieval"}

# 实际步骤检索工具 → 从 arguments 提取查询的 key
_STEP_TOOL_KEYS: dict[str, list[str]] = {
    "grep": ["pattern", "query"],
    "rg": ["pattern", "query"],
    "search": ["query", "keyword", "pattern"],
}


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

        # 构建 tool_call_id → tool response 的映射
        tool_responses: dict[str, dict] = {}
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                tool_responses[msg["tool_call_id"]] = msg

        results: list[EvalResult] = []
        for mi, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                tool_name = fn.get("name", "")
                tc_id = tc.get("id", "")

                # 两种情况：
                # 1. main trace: function.name == "task", arguments.subagent_type 指定类型
                # 2. sub-agent trace: function.name 是实际工具（grep、search 等）
                subagent_type = None
                if tool_name == "task":
                    subagent_type = self._get_subagent_type(fn.get("arguments", ""))
                    if subagent_type not in _STEP_SUBAGENT_TYPES:
                        continue
                elif tool_name not in _STEP_TOOL_KEYS:
                    continue

                # 从 arguments 提取查询关键词
                query_terms = self._extract_query_terms(
                    fn.get("arguments", ""), tool_name, subagent_type
                )

                # 从 tool response 提取实际返回的步骤
                response_content = ""
                if tc_id in tool_responses:
                    resp = tool_responses[tc_id]
                    content = resp.get("content", "")
                    if isinstance(content, str):
                        response_content = content
                    elif isinstance(content, list):
                        response_content = " ".join(
                            i.get("text", "") for i in content if isinstance(i, dict)
                        )

                retrieved = self._extract_retrieved_steps(response_content)
                if not retrieved:
                    continue

                irrelevant = retrieved - ref_steps
                missing = ref_steps - retrieved

                logger.info(
                    f"[{self.name}] msg#{mi} tool={tool_name}: "
                    f"查询 {query_terms}, 检索 {len(retrieved)} 步骤, "
                    f"不相关 {len(irrelevant)}, 缺失 {len(missing)}"
                )

                debug = {
                    "tool_name": tool_name,
                    "tool_call_id": tc_id,
                    "query_terms": sorted(query_terms),
                    "retrieved_steps": sorted(retrieved)[:20],
                    "irrelevant": sorted(irrelevant)[:10],
                    "missing": sorted(missing)[:10],
                    "ref_steps": sorted(ref_steps)[:20],
                }

                if irrelevant:
                    results.append(EvalResult(
                        message_index=mi, dimension=self.name,
                        finding=f"不相关步骤: {', '.join(sorted(irrelevant)[:5])}",
                        confidence=min(1.0, len(irrelevant) * 0.2),
                        action=CleaningAction.KEEP,
                        action_reason="不相关步骤保留，供人工判断",
                        debug_info=debug,
                    ))
                if missing:
                    results.append(EvalResult(
                        message_index=mi, dimension=self.name,
                        finding=f"缺失步骤: {', '.join(sorted(missing)[:5])}",
                        confidence=min(1.0, len(missing) * 0.3),
                        action=CleaningAction.KEEP,
                        action_reason="无法自动生成检索结果，仅记录",
                        debug_info=debug,
                    ))

        return results

    # ── 提取方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_subagent_type(arguments: str) -> str | None:
        """从 task 工具调用的 arguments 中提取 subagent_type。"""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(args, dict):
            return args.get("subagent_type") or args.get("sub_agent_type")
        return None

    @staticmethod
    def _extract_query_terms(
        arguments: str, tool_name: str, subagent_type: str | None = None
    ) -> set[str]:
        """从 tool_call 的 arguments 中提取查询关键词。"""
        terms: set[str] = set()
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            return terms

        if not isinstance(args, dict):
            return terms

        # main trace 的 task 调用
        if tool_name == "task" and subagent_type:
            for key in ("topic", "query", "keyword", "search", "task_input"):
                val = args.get(key)
                if isinstance(val, str) and val:
                    terms.add(val)
            task_input = args.get("task_input") or args.get("input") or {}
            if isinstance(task_input, dict):
                for key in ("topic", "query", "keyword"):
                    val = task_input.get(key)
                    if isinstance(val, str) and val:
                        terms.add(val)
            return terms

        # sub-agent trace 的实际工具调用
        keys = _STEP_TOOL_KEYS.get(tool_name, [])
        for key in keys:
            val = args.get(key)
            if isinstance(val, str) and val:
                terms.add(val)

        return terms

    @staticmethod
    def _extract_retrieved_steps(content: str) -> set[str]:
        """从 tool response content 中提取返回的步骤/函数/类名。"""
        steps: set[str] = set()
        if not content:
            return steps

        # 提取函数定义
        for m in re.finditer(r"def\s+(\w+)", content):
            steps.add(m.group(1))
        # 提取类定义
        for m in re.finditer(r"class\s+(\w+)", content):
            steps.add(m.group(1))
        # 提取关键操作词
        for kw in ("import", "open", "read", "write", "parse", "process",
                    "validate", "transform", "create", "delete", "update"):
            if kw in content.lower():
                steps.add(kw)

        return steps

    @staticmethod
    def _extract_steps(code: str) -> set[str]:
        """从 reference.py 提取期望的步骤集合。"""
        steps: set[str] = set()
        for m in re.finditer(r"def\s+(\w+)", code):
            steps.add(m.group(1))
        for m in re.finditer(r"class\s+(\w+)", code):
            steps.add(m.group(1))
        for kw in ("import", "open", "read", "write", "parse", "process",
                    "validate", "transform", "create", "delete", "update"):
            if kw in code.lower():
                steps.add(kw)
        return steps
