"""Cleaner — 执行评估器建议的清洗动作。

职责划分：
- 评估器：检测问题 + 建议动作（EvalResult.action）
- Cleaner：执行动作（不做二次判断）
"""

from __future__ import annotations

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
            # 只处理需要执行的动作
            if er.action == CleaningAction.KEEP:
                continue

            plans.append(CleaningPlan(
                message_index=er.message_index,
                item_index=er.item_index,
                action=er.action,
                reason=er.finding,
                new_content=er.new_content,
                source=er.dimension,
            ))

        # 升级：如果一条 message 的所有 item 都被 DELETE → DELETE_MESSAGE
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
                result.append(msg)
                continue

            msg_plans = plan_by_msg[mi]

            if any(p.action == CleaningAction.DELETE_MESSAGE for p in msg_plans):
                logger.info(f"DELETE message #{mi}")
                continue

            new_msg = self._apply_item_plans(msg, msg_plans)
            if new_msg is not None:
                result.append(new_msg)

        return result

    # ── 内部方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _upgrade_to_delete_message(
        messages: list[dict], plans: list[CleaningPlan],
    ) -> list[CleaningPlan]:
        plan_by_msg: dict[int, list[CleaningPlan]] = {}
        for p in plans:
            plan_by_msg.setdefault(p.message_index, []).append(p)

        upgraded = []
        for p in plans:
            msg_plans = plan_by_msg.get(p.message_index, [])
            all_delete = all(
                cp.action == CleaningAction.DELETE
                for cp in msg_plans if cp.item_index is not None
            )
            has_items = any(cp.item_index is not None for cp in msg_plans)

            if all_delete and has_items and len(msg_plans) > 1:
                if p == msg_plans[0]:
                    upgraded.append(CleaningPlan(
                        message_index=p.message_index,
                        action=CleaningAction.DELETE_MESSAGE,
                        reason=f"所有 item 均被删除",
                        source=p.source,
                    ))
            else:
                upgraded.append(p)
        return upgraded

    @staticmethod
    def _apply_item_plans(
        msg: dict[str, Any], plans: list[CleaningPlan],
    ) -> dict[str, Any] | None:
        content = msg.get("content", "")
        has_tool_calls = bool(msg.get("tool_calls"))
        has_tool_call_id = bool(msg.get("tool_call_id"))

        if isinstance(content, str):
            if any(p.action == CleaningAction.DELETE for p in plans):
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

        if not isinstance(content, list):
            return msg

        plan_by_item = {p.item_index: p for p in plans if p.item_index is not None}
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
            new_msg = dict(msg)
            new_msg["content"] = ""
            return new_msg

        new_msg = dict(msg)
        new_msg["content"] = new_content
        return new_msg if new_content else None
