"""函数准召率评估器 — 用 AST 解析 reference 中必须调用的核心 API。

评估逻辑：
- 从 reference 中提取「必须的」函数定义和外部 API 调用
- 只关注核心 API 是否被调用，忽略代码风格差异
- 对 main trace：只检查含代码块的 assistant message
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

from core.models import EvalResult
from evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class FunctionRecallEvaluator(BaseEvaluator):
    name = "function_recall"

    # 不应视为"必须 API"的内置/常见函数
    _BUILTIN_CALLS = {
        # Python 内置函数
        "print", "len", "range", "str", "int", "float", "list", "dict",
        "set", "tuple", "type", "isinstance", "getattr", "setattr",
        "hasattr", "super", "classmethod", "staticmethod", "property",
        "abs", "max", "min", "sum", "sorted", "reversed", "enumerate",
        "zip", "map", "filter", "any", "all", "open", "input", "format",
        "repr", "id", "hash", "bool", "bytes", "bytearray", "memoryview",
        "object", "None", "True", "False", "NotImplemented", "Ellipsis",
        "__import__", "compile", "eval", "exec", "globals", "locals",
        "vars", "dir", "help", "iter", "next", "slice",
        "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
        "AttributeError", "RuntimeError", "StopIteration", "OSError",
        "IOError", "FileNotFoundError",
        # 常见字符串/列表方法（不应视为必须 API）
        "strip", "lstrip", "rstrip", "split", "rsplit", "join",
        "replace", "upper", "lower", "startswith", "endswith",
        "find", "rfind", "count", "encode", "decode",
        "append", "extend", "insert", "remove", "pop", "clear",
        "copy", "index", "sort", "reverse", "keys", "values", "items",
        "get", "update", "setdefault", "popitem",
        "read", "write", "readline", "readlines", "writelines",
        "close", "flush", "seek", "tell",
        # 常见类型转换
        "dict", "list", "set", "tuple", "frozenset", "complex",
    }

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        # 从 reference 中提取必须的 API
        required_apis = self._extract_required_apis(reference)
        if not required_apis:
            return []

        is_main = context.get("trace_type") == "main"

        # 确定要检查的 assistant message 索引
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

        # 收集 trace 中所有出现的函数调用
        trace_calls: set[str] = set()
        for mi in target_indices:
            code_blocks = self._extract_code_blocks(messages[mi])
            for code in code_blocks:
                trace_calls.update(self._extract_calls_from_code(code))

        missing = required_apis - trace_calls
        results: list[EvalResult] = []

        if missing and target_indices:
            # severity 基于缺失比例，不是绝对数量
            missing_ratio = len(missing) / len(required_apis)
            severity = min(1.0, missing_ratio * 1.5)  # 比例 * 1.5，上限 1.0

            results.append(
                EvalResult(
                    message_index=target_indices[-1],
                    dimension=self.name,
                    verdict="missing",
                    reason=f"核心 API 未调用: {', '.join(sorted(missing))}",
                    severity=round(severity, 2),
                    suggested_fix={"missing_functions": sorted(missing)},
                )
            )

        return results

    # ── AST 解析 ──────────────────────────────────────────────────────────

    def _extract_required_apis(self, code: str) -> set[str]:
        """从 reference 中提取必须调用的外部 API。

        策略：
        - 函数定义（def xxx）→ 视为必须 API
        - 外部调用（非内置）→ 视为必须 API
        - 内置函数 → 忽略
        """
        apis: set[str] = set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # fallback: 只提取 def 定义
            return self._extract_defs_regex(code)

        for node in ast.walk(tree):
            # 函数定义
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                apis.add(node.name)
            # 函数调用
            elif isinstance(node, ast.Call):
                func_name = self._get_call_name(node.func)
                if func_name and func_name not in self._BUILTIN_CALLS:
                    apis.add(func_name)

        return apis

    @staticmethod
    def _get_call_name(node: ast.expr) -> str | None:
        """从 AST 节点提取函数调用名。"""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            # module.func → 返回 func
            return node.attr
        return None

    @staticmethod
    def _extract_calls_from_code(code: str) -> set[str]:
        """从代码字符串中提取函数定义名和外部调用名（AST 解析）。"""
        names: set[str] = set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return names

        for node in ast.walk(tree):
            # 函数定义
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            # 函数调用
            elif isinstance(node, ast.Call):
                name = FunctionRecallEvaluator._get_call_name(node.func)
                if name and name not in FunctionRecallEvaluator._BUILTIN_CALLS:
                    names.add(name)
        return names

    @staticmethod
    def _extract_defs_regex(code: str) -> set[str]:
        """fallback：用正则提取 def 定义。"""
        return set(re.findall(r"\bdef\s+(\w+)\s*\(", code))

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
