"""文件相关性评估器 — 检索的文件是否在 reference 的 import/依赖中。

逻辑：
1. 遍历 assistant 的 tool_calls，识别文件操作类工具（read、grep、glob 等）
2. 从 arguments 中提取请求的文件路径
3. 通过 tool_call_id 关联 tool response，确认实际返回了什么
4. 与 reference.py 的 import 列表比对，判断相关性
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from core.models import CleaningAction, EvalResult
from evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

# sub-agent 类型中涉及文件检索的
_FILE_SUBAGENT_TYPES = {"script_retrieval", "file_retrieval"}

# 实际文件操作工具 → 从 arguments 提取路径的 key
_FILE_TOOL_KEYS: dict[str, list[str]] = {
    "read": ["file_path", "path", "file"],
    "cat": ["file_path", "path", "file"],
    "write": ["file_path", "path", "file"],
    "edit": ["file_path", "path", "file"],
    "glob": ["pattern", "path"],
    "grep": ["pattern", "path", "glob"],
    "rg": ["pattern", "path", "glob"],
    "find": ["path", "name"],
}


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
                # 2. sub-agent trace: function.name 是实际工具（read、grep 等）
                subagent_type = None
                if tool_name == "task":
                    subagent_type = self._get_subagent_type(fn.get("arguments", ""))
                    if subagent_type not in _FILE_SUBAGENT_TYPES:
                        continue
                elif tool_name not in _FILE_TOOL_KEYS:
                    continue

                # 从 arguments 提取请求的文件路径
                requested = self._extract_from_arguments(
                    fn.get("arguments", ""), tool_name, subagent_type
                )

                # 从 tool response 提取实际返回的文件信息
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

                # 合并：arguments 中的路径 + response 中可推断的路径
                retrieved = requested.copy()
                retrieved.update(self._extract_from_response(response_content))

                if not retrieved:
                    continue

                # 只检查 .py 文件的路径匹配
                retrieved_py = {f for f in retrieved if f.endswith(".py") or "/" in f}
                if not retrieved_py:
                    continue

                # 归一化路径后比对
                retrieved_norm = {self._normalize(p) for p in retrieved_py}
                ref_norm = {self._normalize(p) for p in ref_files}

                irrelevant = retrieved_norm - ref_norm
                missing = ref_norm - retrieved_norm

                logger.info(
                    f"[{self.name}] msg#{mi} tool={tool_name}: "
                    f"请求 {len(requested)}, 实际 {len(retrieved_py)}, "
                    f"不相关 {len(irrelevant)}, 缺失 {len(missing)}"
                )

                debug = {
                    "tool_name": tool_name,
                    "tool_call_id": tc_id,
                    "requested_files": sorted(requested),
                    "retrieved_files": sorted(retrieved_py),
                    "irrelevant": sorted(irrelevant),
                    "missing": sorted(missing),
                    "ref_files": sorted(ref_files),
                }

                if irrelevant:
                    results.append(EvalResult(
                        message_index=mi, dimension=self.name,
                        finding=f"不相关文件: {', '.join(sorted(irrelevant))}",
                        confidence=min(1.0, len(irrelevant) * 0.3),
                        action=CleaningAction.KEEP,
                        action_reason="不相关文件保留，供人工判断",
                        debug_info=debug,
                    ))
                if missing:
                    results.append(EvalResult(
                        message_index=mi, dimension=self.name,
                        finding=f"缺失文件: {', '.join(sorted(missing))}",
                        confidence=min(1.0, len(missing) * 0.4),
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
    def _extract_from_arguments(
        arguments: str, tool_name: str, subagent_type: str | None = None
    ) -> set[str]:
        """从 tool_call 的 arguments JSON 中提取文件路径。"""
        files: set[str] = set()
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            return files

        if not isinstance(args, dict):
            return files

        # main trace 的 task 调用：从 subagent 相关字段提取
        if tool_name == "task" and subagent_type:
            for key in ("file_path", "path", "file", "query", "topic"):
                val = args.get(key)
                if isinstance(val, str) and val:
                    files.add(val)
            # 也检查 task_input 等嵌套字段
            task_input = args.get("task_input") or args.get("input") or {}
            if isinstance(task_input, dict):
                for key in ("file_path", "path", "file"):
                    val = task_input.get(key)
                    if isinstance(val, str) and val:
                        files.add(val)
            return files

        # sub-agent trace 的实际工具调用
        keys = _FILE_TOOL_KEYS.get(tool_name, [])
        for key in keys:
            val = args.get(key)
            if isinstance(val, str) and val:
                files.add(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        files.add(item)

        # 通用：遍历所有值找路径
        for val in args.values():
            if isinstance(val, str) and ("/" in val or val.endswith(".py")):
                files.add(val)

        return files

    @staticmethod
    def _extract_from_response(content: str) -> set[str]:
        """从 tool response content 中推断涉及的文件路径。

        常见模式：
        - 文件头注释 "# File: path/to/file.py"
        - diff 格式 "--- a/path/to/file.py"
        - grep 输出 "path/to/file.py:123:..."
        """
        files: set[str] = set()
        if not content:
            return files

        # grep 风格: path/to/file.py:行号:
        for m in re.finditer(r"(?:^|\n)([\w./\\-]+\.py)(?::\d+|:)", content):
            files.add(m.group(1))

        # diff 风格: --- a/path 或 +++ b/path
        for m in re.finditer(r"(?:---|\+\+\+)\s+[ab]/([\w./\\-]+)", content):
            files.add(m.group(1))

        # 文件头: # File: path 或 // File: path
        for m in re.finditer(r"(?:^|\n)[#/]+\s*File:\s*([\w./\\-]+)", content):
            files.add(m.group(1))

        return files

    @staticmethod
    def _extract_imported_files(code: str) -> set[str]:
        """从 reference.py 的 import 语句提取文件列表。"""
        files: set[str] = set()
        for m in re.finditer(r"(?:from|import)\s+([\w.]+)", code):
            mod = m.group(1)
            files.add(mod.replace(".", "/") + ".py")
            files.add(mod)
        return files

    @staticmethod
    def _normalize(path: str) -> str:
        """归一化路径：去前缀 ./、统一斜杠。"""
        path = path.replace("\\", "/")
        if path.startswith("./"):
            path = path[2:]
        # 去掉末尾的 .py 后缀用于模糊匹配
        return path
