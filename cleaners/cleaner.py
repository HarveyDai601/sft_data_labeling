"""Cleaner — 执行评估器建议的清洗动作。

职责划分：
- 评估器：检测问题 + 建议动作（EvalResult.action）
- Cleaner：执行动作（不做二次判断）
"""

from __future__ import annotations

import json
import logging
from typing import Any

from core.models import CleaningAction, CleaningPlan, EvalResult

logger = logging.getLogger(__name__)


class Cleaner:

    def __init__(self, conflict_resolution: str = "severity_priority"):
        self.conflict_resolution = conflict_resolution

    def plan(
        self,
        messages: list[dict[str, Any]],
        eval_results: list[EvalResult],
    ) -> list[CleaningPlan]:
        """从评估结果生成清洗计划。直接使用评估器建议的 action。"""
        plans: list[CleaningPlan] = []

        for er in eval_results:
            if er.action == CleaningAction.KEEP:
                continue

            plans.append(CleaningPlan(
                message_index=er.message_index,
                item_index=er.item_index,
                action=er.action,
                reason=er.finding,
                new_content=er.new_content,
                source=er.dimension,
                tool_call_replace=er.tool_call_replace,
            ))

        plans = self._upgrade_to_delete_message(messages, plans)

        logger.info(f"生成 {len(plans)} 条清洗计划（KEEP 的已跳过）")
        return plans

    def execute(
        self,
        messages: list[dict[str, Any]],
        plans: list[CleaningPlan],
    ) -> list[dict[str, Any]]:
        """按清洗计划执行。"""
        plan_by_msg: dict[int, list[CleaningPlan]] = {}
        for p in plans:
            plan_by_msg.setdefault(p.message_index, []).append(p)

        result: list[dict[str, Any]] = []
        for mi, msg in enumerate(messages):
            if mi not in plan_by_msg:
                result.append(self._deep_copy_msg(msg))
                continue

            msg_plans = plan_by_msg[mi]

            if any(p.action == CleaningAction.DELETE_MESSAGE for p in msg_plans):
                logger.info(f"DELETE message #{mi}")
                continue

            new_msg = self._apply_plans(msg, msg_plans)
            if new_msg is not None:
                result.append(new_msg)

        return result

    # ── 内部方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _apply_plans(
        msg: dict[str, Any], plans: list[CleaningPlan],
    ) -> dict[str, Any] | None:
        """对一条 message 应用所有清洗计划。"""
        new_msg = Cleaner._deep_copy_msg(msg)

        # 1. 先处理 tool_call_replace（修改 tool_call arguments）
        for p in plans:
            if p.tool_call_replace and p.action == CleaningAction.REPLACE:
                Cleaner._apply_tool_call_replace(new_msg, p.tool_call_replace)

        # 2. 再处理 message content 级别的操作
        content = new_msg.get("content", "")
        has_tool_calls = bool(new_msg.get("tool_calls"))
        has_tool_call_id = bool(new_msg.get("tool_call_id"))

        # 如果有 tool_call_replace，content 级别的 REPLACE 不需要再执行
        has_tool_replace = any(p.tool_call_replace for p in plans if p.action == CleaningAction.REPLACE)
        content_plans = [p for p in plans if not p.tool_call_replace or p.action != CleaningAction.REPLACE]

        if isinstance(content, str):
            if any(p.action == CleaningAction.DELETE for p in content_plans):
                if has_tool_calls or has_tool_call_id:
                    new_msg["content"] = ""
                    return new_msg
                return None
            if not has_tool_replace and any(p.action == CleaningAction.REPLACE for p in content_plans):
                new_msg["content"] = content_plans[0].new_content or content
                return new_msg
            if any(p.action == CleaningAction.REWRITE for p in content_plans):
                rewrite = next(p for p in content_plans if p.action == CleaningAction.REWRITE)
                new_msg["content"] = rewrite.new_content or content
                return new_msg
            return new_msg

        if not isinstance(content, list):
            return new_msg

        plan_by_item = {p.item_index: p for p in content_plans if p.item_index is not None}
        new_content = []

        for ii, item in enumerate(content):
            p = plan_by_item.get(ii)
            if p is None:
                new_content.append(item)
            elif p.action == CleaningAction.DELETE:
                logger.info(f"DELETE item #{ii}")
            elif p.action == CleaningAction.REPLACE:
                new_content.append(p.new_content or item)
            elif p.action == CleaningAction.INSERT_AFTER:
                new_content.append(item)
                if p.new_content:
                    new_content.append(p.new_content)
            else:
                new_content.append(item)

        if not new_content and (has_tool_calls or has_tool_call_id):
            new_msg["content"] = ""
            return new_msg

        new_msg["content"] = new_content
        return new_msg if new_content else None

    @staticmethod
    def _apply_tool_call_replace(msg: dict[str, Any], replace_info: dict[str, Any]) -> None:
        """修改 tool_call arguments 中的指定字段。原地修改。"""
        tc_id = replace_info.get("tool_call_id", "")
        arg_key = replace_info.get("arg_key", "content")
        new_value = replace_info.get("new_value", "")

        for tc in msg.get("tool_calls", []):
            if tc.get("id") == tc_id:
                fn = tc.get("function", {})
                args_str = fn.get("arguments", "{}")
                try:
                    args = json.loads(args_str) if isinstance(args_str, str) else args_str
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if isinstance(args, dict):
                    args[arg_key] = new_value
                    fn["arguments"] = json.dumps(args, ensure_ascii=False)
                    logger.info(f"REPLACE tool_call {tc_id} .{arg_key}")
                return

        logger.warning(f"tool_call {tc_id} 未找到，跳过替换")

    @staticmethod
    def _upgrade_to_delete_message(
        messages: list[dict], plans: list[CleaningPlan],
    ) -> list[CleaningPlan]:
        plan_by_msg: dict[int, list[CleaningPlan]] = {}
        for p in plans:
            plan_by_msg.setdefault(p.message_index, []).append(p)

        upgraded = []
        seen_msg: set[int] = set()
        for p in plans:
            if p.message_index in seen_msg:
                continue
            msg_plans = plan_by_msg.get(p.message_index, [])
            all_delete = all(
                cp.action == CleaningAction.DELETE
                for cp in msg_plans if cp.item_index is not None
            )
            has_items = any(cp.item_index is not None for cp in msg_plans)

            if all_delete and has_items and len(msg_plans) > 1:
                seen_msg.add(p.message_index)
                upgraded.append(CleaningPlan(
                    message_index=p.message_index,
                    action=CleaningAction.DELETE_MESSAGE,
                    reason="所有 item 均被删除",
                    source=p.source,
                ))
            else:
                upgraded.append(p)
        return upgraded

    @staticmethod
    def _deep_copy_msg(msg: dict[str, Any]) -> dict[str, Any]:
        """深拷贝 message，避免修改原始数据。"""
        new_msg = dict(msg)
        if "tool_calls" in msg and isinstance(msg["tool_calls"], list):
            new_msg["tool_calls"] = [
                {**tc, "function": {**tc.get("function", {})}}
                if isinstance(tc, dict) else tc
                for tc in msg["tool_calls"]
            ]
        return new_msg
