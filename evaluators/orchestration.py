"""步骤编排评估器 — 因果链是否完整？

逻辑：
1. Main trace: 提取 sub-agent 调用序列，检查顺序和重复
2. Sub-agent trace: 检查 read → write → verify 因果链
   - reference 的每个函数 → agent 读了依赖吗？写了吗？验证了吗？
3. 缺失环 = 步骤不完整；多余环 = 记录但不确定是错

只产出观测结果（KEEP）。
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from core.models import CleaningAction, EvalResult
from evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)

_CALL_ORDER = {
    "script_retrieval": 0,
    "step_retrieval": 1,
    "script_complete": 2,
    "consistency_check": 3,
}

_READ_TOOLS = {"read", "cat", "grep", "rg", "glob", "find"}
_WRITE_TOOLS = {"write", "edit", "create", "create_file", "write_file"}
_VERIFY_TOOLS = {"grep", "rg", "consistency_check", "check"}


class OrchestrationEvaluator(BaseEvaluator):
    name = "orchestration"

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        is_main = context.get("trace_type") == "main"

        if is_main:
            return self._evaluate_main(messages)
        else:
            return self._evaluate_sub(messages, reference)

    # ── Main trace ────────────────────────────────────────────────────────

    def _evaluate_main(self, messages: list[dict]) -> list[EvalResult]:
        """检查 main trace 的 sub-agent 调用序列。"""
        results: list[EvalResult] = []

        # 提取 sub-agent 调用
        calls: list[tuple[int, str, str]] = []  # (msg_index, subagent_type, tc_id)
        for mi, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                if fn.get("name") == "task":
                    st = self._get_subagent_type(fn.get("arguments", ""))
                    if st and st in _CALL_ORDER:
                        calls.append((mi, st, tc.get("id", "")))

        logger.info(
            f"[{self.name}] main 调用链: {' → '.join(f'#{mi}:{t}' for mi, t, _ in calls)}"
        )

        if len(calls) < 2:
            return results

        # 顺序检查
        for i in range(1, len(calls)):
            prev_mi, prev_t, _ = calls[i - 1]
            curr_mi, curr_t, _ = calls[i]
            if _CALL_ORDER.get(curr_t, 0) < _CALL_ORDER.get(prev_t, 0):
                results.append(EvalResult(
                    message_index=curr_mi, dimension=self.name,
                    finding=f"调用顺序异常: {prev_t}(#{prev_mi}) → {curr_t}(#{curr_mi})",
                    confidence=0.5,
                    action=CleaningAction.KEEP,
                    action_reason="顺序异常仅记录",
                ))

        # 重复检查
        seen: dict[str, int] = {}
        for mi, st, _ in calls:
            if st in seen and st != "consistency_check":
                results.append(EvalResult(
                    message_index=mi, dimension=self.name,
                    finding=f"重复调用 {st}（首次 #{seen[st]}）",
                    confidence=0.4,
                    action=CleaningAction.KEEP,
                    action_reason="重复调用记录，不确定是错",
                ))
            else:
                seen[st] = mi

        # 完整性检查：缺了哪些 sub-agent 类型？
        called_types = {st for _, st, _ in calls}
        # 从 reference 推断是否需要 consistency_check
        if "script_retrieval" not in called_types:
            results.append(EvalResult(
                message_index=0, dimension=self.name,
                finding="未调用 script_retrieval，可能缺少依赖文件检索",
                confidence=0.6,
                action=CleaningAction.KEEP,
                action_reason="无法自动补全调用",
            ))
        if "consistency_check" not in called_types:
            results.append(EvalResult(
                message_index=0, dimension=self.name,
                finding="未调用 consistency_check，可能缺少自检步骤",
                confidence=0.3,
                action=CleaningAction.KEEP,
                action_reason="自检步骤可选，仅记录",
            ))

        return results

    # ── Sub-agent trace ───────────────────────────────────────────────────

    def _evaluate_sub(
        self, messages: list[dict], reference: str,
    ) -> list[EvalResult]:
        """检查 sub-agent trace 的因果链：read → write → verify。"""
        results: list[EvalResult] = []

        # 构建 response 映射
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

        # 提取操作
        reads: list[dict] = []   # {msg_index, tool, target, tc_id}
        writes: list[dict] = []  # {msg_index, tool, code}
        verifies: list[dict] = []  # {msg_index, tool}

        for mi, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                tool = fn.get("name", "")
                args_str = fn.get("arguments", "")
                tc_id = tc.get("id", "")

                if tool in _READ_TOOLS:
                    reads.append({
                        "msg_index": mi,
                        "tool": tool,
                        "target": self._extract_target(args_str, tool),
                        "tc_id": tc_id,
                    })
                elif tool in _WRITE_TOOLS:
                    code = self._extract_write_code(args_str)
                    writes.append({
                        "msg_index": mi,
                        "tool": tool,
                        "code": code,
                    })
                elif tool in _VERIFY_TOOLS:
                    verifies.append({
                        "msg_index": mi,
                        "tool": tool,
                    })

        logger.info(
            f"[{self.name}] sub 因果链: "
            f"{len(reads)} reads, {len(writes)} writes, {len(verifies)} verifies"
        )

        # 检查 1：有 write 但没有 read？→ 可能缺上下文
        if writes and not reads:
            results.append(EvalResult(
                message_index=writes[0]["msg_index"],
                dimension=self.name,
                finding="有 write 但无 read：可能缺少上下文检索",
                confidence=0.5,
                action=CleaningAction.KEEP,
                action_reason="仅记录",
            ))

        # 检查 2：有 read 有 write 但没有 verify？→ 可能缺自检
        if reads and writes and not verifies:
            results.append(EvalResult(
                message_index=writes[-1]["msg_index"],
                dimension=self.name,
                finding="有 read+write 但无 verify：可能缺少自检步骤",
                confidence=0.3,
                action=CleaningAction.KEEP,
                action_reason="自检可选，仅记录",
            ))

        # 检查 3：从 reference 推断哪些 read 可能缺失
        if reference:
            ref_modules = self._extract_reference_modules(reference)
            read_about = set()
            for r in reads:
                read_about.add(r["target"])
            for r in reads:
                if r["tc_id"] in responses:
                    read_about.update(
                        re.findall(r"\b([\w.]+)\.py", responses[r["tc_id"]])
                    )

            unread = set()
            for mod in ref_modules:
                mod_norm = mod.replace(".", "/")
                found = any(
                    mod_norm in t or mod in t
                    for t in read_about
                )
                if not found:
                    unread.add(mod)

            if unread:
                results.append(EvalResult(
                    message_index=reads[-1]["msg_index"] if reads else 0,
                    dimension=self.name,
                    finding=f"reference 依赖但未检索: {', '.join(sorted(unread))}",
                    confidence=round(min(1.0, len(unread) * 0.35), 2),
                    action=CleaningAction.KEEP,
                    action_reason="无法自动补全读取",
                ))

        return results

    # ── 提取方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_subagent_type(arguments: str) -> str | None:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(args, dict):
            return args.get("subagent_type") or args.get("sub_agent_type")
        return None

    @staticmethod
    def _extract_target(arguments: str, tool_name: str) -> str:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(args, dict):
            return ""
        for key in ("file_path", "path", "pattern", "query", "name"):
            val = args.get(key)
            if isinstance(val, str) and val:
                return val
        return ""

    @staticmethod
    def _extract_write_code(arguments: str) -> str:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(args, dict):
            return ""
        for key in ("content", "file_text", "text", "code"):
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""

    @staticmethod
    def _extract_reference_modules(code: str) -> set[str]:
        modules: set[str] = set()
        for m in re.finditer(r"(?:from|import)\s+([\w.]+)", code):
            modules.add(m.group(1).split(".")[0])
        return modules
