"""信息流追踪评估器 — read 的东西用没用？缺没缺？

核心问题：agent 读的文件/符号，后来在 write 的代码中用到了吗？

逻辑：
1. 构建 tool_call_id → tool_response 映射
2. 提取所有读操作（read/grep/glob/find），记录读取目标和返回内容
3. 提取所有写操作（write/edit），AST 解析用到的符号
4. 交叉比对：读了但没用到？→ 冗余探索
            应该读但没读？→ 缺失依赖

只产出观测结果（KEEP）。
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from core.models import CleaningAction, EvalResult
from evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

# 读类工具
_READ_TOOLS = {"read", "cat", "glob", "grep", "rg", "find", "search"}

# 写类工具
_WRITE_TOOLS = {"write", "edit", "create", "create_file", "write_file"}
_WRITE_CONTENT_KEYS = ["content", "file_text", "text", "code"]

# 读工具的 arguments 中表示路径/目标的 key
_READ_TARGET_KEYS = {
    "read": ["file_path", "path"],
    "cat": ["file_path", "path"],
    "glob": ["pattern", "path"],
    "grep": ["pattern", "path", "glob"],
    "rg": ["pattern", "path", "glob"],
    "find": ["path", "name", "pattern"],
    "search": ["query", "pattern", "keyword"],
}


class InfoUsageEvaluator(BaseEvaluator):
    name = "info_usage"

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        # 1. 构建 response 映射
        responses: dict[str, str] = {}
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("tool_call_id"):
                c = msg.get("content", "")
                if isinstance(c, str):
                    responses[msg["tool_call_id"]] = c
                elif isinstance(c, list):
                    responses[msg["tool_call_id"]] = " ".join(
                        i.get("text", "") for i in c if isinstance(i, dict)
                    )

        # 2. 提取读操作
        reads: list[dict] = []  # {msg_index, tool, target, symbols_in_response}
        for mi, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                tool = fn.get("name", "")
                if tool not in _READ_TOOLS:
                    continue
                tc_id = tc.get("id", "")
                target = self._extract_read_target(fn.get("arguments", ""), tool)
                resp_text = responses.get(tc_id, "")
                symbols = self._extract_symbols_from_text(resp_text)
                reads.append({
                    "msg_index": mi,
                    "tool": tool,
                    "target": target,
                    "tc_id": tc_id,
                    "symbols": symbols,
                    "response_preview": resp_text[:200],
                })

        # 3. 提取写操作 + 代码中使用的符号
        writes: list[dict] = []  # {msg_index, code, symbols_used}
        for mi, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                if fn.get("name", "") not in _WRITE_TOOLS:
                    continue
                code = self._extract_write_code(fn.get("arguments", ""))
                if not code:
                    continue
                symbols = self._extract_symbols_from_code(code)
                writes.append({
                    "msg_index": mi,
                    "code_preview": code[:500],
                    "symbols": symbols,
                })

        if not reads and not writes:
            return []

        # 4. 交叉比对
        results: list[EvalResult] = []

        # 4a. 读了但没用到？
        if reads and writes:
            all_write_symbols = set().union(*(w["symbols"] for w in writes))
            logger.info(
                f"[{self.name}] 写操作符号: {sorted(all_write_symbols)[:30]}"
            )

            for r in reads:
                if not r["symbols"]:
                    continue
                used = r["symbols"] & all_write_symbols
                unused = r["symbols"] - all_write_symbols

                logger.info(
                    f"[{self.name}] msg#{r['msg_index']} {r['tool']}({r['target']}): "
                    f"返回 {len(r['symbols'])} 符号, "
                    f"用到 {len(used)}, 未用到 {len(unused)}"
                )

                if unused and len(r["symbols"]) >= 3:
                    unused_ratio = len(unused) / len(r["symbols"])
                    confidence = round(min(1.0, unused_ratio * 1.5), 2)
                    sample = sorted(unused)[:10]
                    results.append(EvalResult(
                        message_index=r["msg_index"], dimension=self.name,
                        finding=f"读取但未使用: {', '.join(sample)}（{len(unused)}/{len(r['symbols'])} 未用）",
                        confidence=confidence,
                        action=CleaningAction.KEEP,
                        action_reason="未使用的读取可能为合理探索，仅记录",
                        debug_info={
                            "read_tool": r["tool"],
                            "read_target": r["target"],
                            "retrieved_symbols": sorted(r["symbols"]),
                            "used_symbols": sorted(used),
                            "unused_symbols": sorted(unused),
                        },
                    ))

        # 4b. reference 的 import → agent 读了吗？
        if reference:
            ref_imports = self._extract_reference_modules(reference)
            read_targets = set()
            for r in reads:
                read_targets.add(r["target"])
                read_targets.update(r["symbols"])

            missing_reads = set()
            for mod in ref_imports:
                # 检查 reference 中的模块是否被读取
                mod_parts = mod.replace(".", "/")
                found = any(
                    mod_parts in rt or mod in rt
                    for rt in read_targets
                )
                if not found:
                    missing_reads.add(mod)

            if missing_reads:
                logger.info(f"[{self.name}] 可能缺失的读取: {sorted(missing_reads)}")
                results.append(EvalResult(
                    message_index=reads[-1]["msg_index"] if reads else 0,
                    dimension=self.name,
                    finding=f"reference 依赖但未读取: {', '.join(sorted(missing_reads))}",
                    confidence=round(min(1.0, len(missing_reads) * 0.3), 2),
                    action=CleaningAction.KEEP,
                    action_reason="无法自动补全读取操作",
                    debug_info={
                        "ref_imports": sorted(ref_imports),
                        "read_targets": sorted(read_targets),
                        "missing_reads": sorted(missing_reads),
                    },
                ))

        return results

    # ── 提取方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_read_target(arguments: str, tool_name: str) -> str:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(args, dict):
            return ""
        keys = _READ_TARGET_KEYS.get(tool_name, ["file_path", "path", "query"])
        for key in keys:
            val = args.get(key)
            if isinstance(val, str) and val:
                return val
        return ""

    @staticmethod
    def _extract_symbols_from_text(text: str) -> set[str]:
        """从文本中提取标识符（函数名、类名、模块路径）。"""
        symbols: set[str] = set()
        if not text:
            return symbols
        # Python 标识符
        for m in re.finditer(r"\bdef\s+(\w+)", text):
            symbols.add(m.group(1))
        for m in re.finditer(r"\bclass\s+(\w+)", text):
            symbols.add(m.group(1))
        for m in re.finditer(r"\bimport\s+([\w.]+)", text):
            symbols.add(m.group(1))
        for m in re.finditer(r"\bfrom\s+([\w.]+)\s+import", text):
            symbols.add(m.group(1))
        # 文件路径
        for m in re.finditer(r"[\w./\\-]+\.py", text):
            symbols.add(m.group())
        return symbols

    @staticmethod
    def _extract_symbols_from_code(code: str) -> set[str]:
        """AST 解析代码中使用的符号（函数调用、属性访问、import）。"""
        symbols: set[str] = set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return InfoUsageEvaluator._extract_symbols_from_text(code)

        for node in ast.walk(tree):
            # 函数调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    symbols.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    symbols.add(node.func.attr)
            # import
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    symbols.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    symbols.add(node.module.split(".")[0])
            # 属性访问
            elif isinstance(node, ast.Attribute):
                symbols.add(node.attr)

        return symbols

    @staticmethod
    def _extract_write_code(arguments: str) -> str:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(args, dict):
            return ""
        for key in _WRITE_CONTENT_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    @staticmethod
    def _extract_reference_modules(code: str) -> set[str]:
        """提取 reference 的 import 模块名称。"""
        modules: set[str] = set()
        for m in re.finditer(r"(?:from|import)\s+([\w.]+)", code):
            modules.add(m.group(1).split(".")[0])
        return modules
