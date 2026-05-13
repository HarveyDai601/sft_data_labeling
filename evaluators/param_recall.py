"""参数准召率评估器 — 用 AST 提取函数签名中的参数，过滤 docstring。

评估逻辑：
- 只提取函数定义签名中的参数（def xxx(a, b, c)）
- 用 AST 解析，不匹配 docstring 中的参数示例
- 对 main trace：只检查含代码块的 assistant message
"""

from __future__ import annotations

import ast
import re
from typing import Any

from core.models import EvalResult
from evaluators.base import BaseEvaluator


class ParamRecallEvaluator(BaseEvaluator):
    name = "param_recall"

    # 应忽略的常见参数名
    _IGNORED_PARAMS = {"self", "cls", "args", "kwargs"}

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        ref_params = self._extract_params(reference)
        if not ref_params:
            return []

        is_main = context.get("trace_type") == "main"

        if is_main:
            target_indices = [
                mi for mi, msg in enumerate(messages)
                if msg.get("role") == "assistant" and self._has_code_block(msg)
            ]
        else:
            target_indices = [
                mi for mi, msg in enumerate(messages)
                if msg.get("role") == "assistant"
            ]

        if not target_indices:
            return []

        # 收集 trace 中所有出现的参数
        trace_params: set[str] = set()
        for mi in target_indices:
            code_blocks = self._extract_code_blocks(messages[mi])
            for code in code_blocks:
                trace_params.update(self._extract_params(code))

        missing = ref_params - trace_params
        results: list[EvalResult] = []

        if missing and target_indices:
            missing_ratio = len(missing) / len(ref_params)
            severity = min(1.0, missing_ratio * 1.5)

            results.append(
                EvalResult(
                    message_index=target_indices[-1],
                    dimension=self.name,
                    verdict="missing",
                    reason=f"核心参数未使用: {', '.join(sorted(missing))}",
                    severity=round(severity, 2),
                    suggested_fix={"missing_params": sorted(missing)},
                )
            )

        return results

    # ── AST 解析 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_params(code: str) -> set[str]:
        """用 AST 提取函数**定义**签名中的参数名（不匹配 docstring，不匹配调用）。"""
        params: set[str] = set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ParamRecallEvaluator._extract_params_regex(code)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 只从函数定义中提取参数，不从函数调用中提取
                for arg in node.args.args:
                    if arg.arg not in ParamRecallEvaluator._IGNORED_PARAMS:
                        params.add(arg.arg)
                for arg in node.args.kwonlyargs:
                    if arg.arg not in ParamRecallEvaluator._IGNORED_PARAMS:
                        params.add(arg.arg)
                if node.args.vararg and node.args.vararg.arg not in ParamRecallEvaluator._IGNORED_PARAMS:
                    params.add(node.args.vararg.arg)
                if node.args.kwarg and node.args.kwarg.arg not in ParamRecallEvaluator._IGNORED_PARAMS:
                    params.add(node.args.kwarg.arg)

        return params

    @staticmethod
    def _extract_params_regex(code: str) -> set[str]:
        """fallback：用正则提取（过滤 docstring）。"""
        # 去掉 docstring
        code = re.sub(r'"""[\s\S]*?"""', '', code)
        code = re.sub(r"'''[\s\S]*?'''", '', code)
        # 去掉行内注释
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)

        params: set[str] = set()
        for m in re.finditer(r"def\s+\w+\s*\(([^)]*)\)", code):
            sig = m.group(1)
            for arg in re.findall(r"[\*]*(\w+)", sig):
                if arg not in ParamRecallEvaluator._IGNORED_PARAMS:
                    params.add(arg)
        return params

    # ── 工具方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _has_code_block(msg: dict[str, Any]) -> bool:
        content = msg.get("content", "")
        if isinstance(content, str):
            return "```" in content
        if isinstance(content, list):
            return any(
                item.get("type") == "code"
                or "```" in (item.get("text", "") or "")
                for item in content
                if isinstance(item, dict)
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
                        text = item.get("text", "")
                        for m in re.finditer(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL):
                            blocks.append(m.group(1).strip())
        return blocks
