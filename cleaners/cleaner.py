"""Cleaner — 根据评估结果生成清洗计划并执行。"""

from __future__ import annotations

import logging
from typing import Any

from core.models import CleaningAction, CleaningPlan, EvalResult

logger = logging.getLogger(__name__)


class Cleaner:
    """规则驱动的清洗器，工作在 content item 级别。"""

    def __init__(self, conflict_resolution: str = "severity_priority"):
        self.conflict_resolution = conflict_resolution

    def plan(
        self,
        messages: list[dict[str, Any]],
        eval_results: list[EvalResult],
    ) -> list[CleaningPlan]:
        """根据评估结果生成清洗计划。"""
        # 按 (message_index, item_index) 分组
        grouped: dict[tuple[int, int | None], list[EvalResult]] = {}
        for er in eval_results:
            key = (er.message_index, er.item_index)
            grouped.setdefault(key, []).append(er)

        plans: list[CleaningPlan] = []

        for (mi, ii), results in sorted(grouped.items()):
            action, reason, source, new_content = self._resolve_conflicts(results)
            plans.append(
                CleaningPlan(
                    message_index=mi,
                    item_index=ii,
                    action=action,
                    reason=reason,
                    new_content=new_content,
                    source=source,
                )
            )

        # 升级：如果一条 message 的所有 item 都被 DELETE → DELETE_MESSAGE
        plans = self._upgrade_to_delete_message(messages, plans)

        return plans

    def execute(
        self,
        messages: list[dict[str, Any]],
        plans: list[CleaningPlan],
    ) -> list[dict[str, Any]]:
        """按清洗计划执行，返回清洗后的 messages。"""
        # 按 message_index 分组
        plan_by_msg: dict[int, list[CleaningPlan]] = {}
        for p in plans:
            plan_by_msg.setdefault(p.message_index, []).append(p)

        result: list[dict[str, Any]] = []

        for mi, msg in enumerate(messages):
            if mi not in plan_by_msg:
                result.append(msg)
                continue

            msg_plans = plan_by_msg[mi]

            # 检查是否有 DELETE_MESSAGE
            if any(p.action == CleaningAction.DELETE_MESSAGE for p in msg_plans):
                logger.info(f"DELETE message #{mi}: {msg_plans[0].reason}")
                continue

            # item 级别处理
            new_msg = self._apply_item_plans(msg, msg_plans)
            if new_msg is not None:
                result.append(new_msg)

        return result

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _resolve_conflicts(
        self, results: list[EvalResult]
    ) -> tuple[CleaningAction, str, str, Any]:
        """合并多条评估结果，返回最终 action。"""
        if not results:
            return CleaningAction.KEEP, "", "", None

        # 按 severity 降序排序
        results.sort(key=lambda r: r.severity, reverse=True)

        # 保守策略：bad > missing > good
        _priority = {"bad": 0, "missing": 1, "good": 2}
        results.sort(key=lambda r: _priority.get(r.verdict, 99))

        top = results[0]

        if top.verdict == "good":
            return CleaningAction.KEEP, top.reason, top.dimension, None

        if top.verdict == "bad":
            if top.suggested_fix and top.suggested_fix.get("type") == "delete_lines":
                return CleaningAction.DELETE, top.reason, top.dimension, None
            if top.suggested_fix and top.suggested_fix.get("type") == "replace":
                return CleaningAction.REPLACE, top.reason, top.dimension, top.suggested_fix.get("content")
            if top.severity >= 0.8:
                return CleaningAction.REWRITE, top.reason, top.dimension, None
            return CleaningAction.DELETE, top.reason, top.dimension, None

        if top.verdict == "missing":
            return CleaningAction.INSERT_AFTER, top.reason, top.dimension, top.suggested_fix

        return CleaningAction.KEEP, "", "", None

    @staticmethod
    def _upgrade_to_delete_message(
        messages: list[dict[str, Any]],
        plans: list[CleaningPlan],
    ) -> list[CleaningPlan]:
        """如果一条 message 的所有 item 都被 DELETE，升级为 DELETE_MESSAGE。"""
        # 按 message_index 分组
        plan_by_msg: dict[int, list[CleaningPlan]] = {}
        for p in plans:
            plan_by_msg.setdefault(p.message_index, []).append(p)

        upgraded: list[CleaningPlan] = []
        for p in plans:
            msg_plans = plan_by_msg.get(p.message_index, [])
            # 检查是否所有 item 都是 DELETE
            all_delete = all(
                cp.action == CleaningAction.DELETE
                for cp in msg_plans
                if cp.item_index is not None
            )
            has_item_plans = any(cp.item_index is not None for cp in msg_plans)

            if all_delete and has_item_plans and len(msg_plans) > 1:
                # 升级为 DELETE_MESSAGE，只保留一条
                if p == msg_plans[0]:
                    upgraded.append(
                        CleaningPlan(
                            message_index=p.message_index,
                            action=CleaningAction.DELETE_MESSAGE,
                            reason=f"所有 item 均被删除: {p.reason}",
                            source=p.source,
                        )
                    )
            else:
                upgraded.append(p)

        return upgraded

    @staticmethod
    def _apply_item_plans(
        msg: dict[str, Any],
        plans: list[CleaningPlan],
    ) -> dict[str, Any] | None:
        """对单条 message 应用 item 级别的清洗计划。"""
        content = msg.get("content", "")
        has_tool_calls = bool(msg.get("tool_calls"))
        has_tool_call_id = bool(msg.get("tool_call_id"))  # tool response

        # ── string content ────────────────────────────────────────────────
        if isinstance(content, str):
            if any(p.action == CleaningAction.DELETE for p in plans):
                # 有 tool_calls 的 assistant message 或 tool response → 只清空 content，保留结构
                if has_tool_calls or has_tool_call_id:
                    new_msg = dict(msg)
                    new_msg["content"] = ""
                    return new_msg
                return None
            if any(p.action == CleaningAction.REPLACE for p in plans):
                new_msg = dict(msg)
                new_msg["content"] = plans[0].new_content or content
                return new_msg
            return msg

        # ── list content ──────────────────────────────────────────────────
        if not isinstance(content, list):
            return msg

        plan_by_item: dict[int, CleaningPlan] = {
            p.item_index: p for p in plans if p.item_index is not None
        }

        new_content: list[Any] = []
        for ii, item in enumerate(content):
            p = plan_by_item.get(ii)
            if p is None:
                new_content.append(item)
                continue

            if p.action == CleaningAction.KEEP:
                new_content.append(item)
            elif p.action == CleaningAction.DELETE:
                logger.info(f"DELETE item #{ii} in message: {p.reason}")
            elif p.action == CleaningAction.REPLACE:
                new_content.append(p.new_content or item)
            elif p.action == CleaningAction.REWRITE:
                new_content.append(p.new_content or item)
            elif p.action == CleaningAction.INSERT_BEFORE:
                if p.new_content:
                    new_content.append(p.new_content)
                new_content.append(item)
            elif p.action == CleaningAction.INSERT_AFTER:
                new_content.append(item)
                if p.new_content:
                    new_content.append(p.new_content)
            else:
                new_content.append(item)

        # 如果 content 被清空，但有 tool_calls → 保留 message 结构
        if not new_content and (has_tool_calls or has_tool_call_id):
            new_msg = dict(msg)
            new_msg["content"] = ""
            return new_msg

        new_msg = dict(msg)
        new_msg["content"] = new_content
        return new_msg if new_content else None
