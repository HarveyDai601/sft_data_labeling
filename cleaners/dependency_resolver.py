"""DependencyResolver — 检测清洗操作间的依赖关系，对受影响的后续消息调用 LLM 重新处理。"""

from __future__ import annotations

import logging
import re
from typing import Any

from core.models import CleaningAction, CleaningPlan, EvalResult
from llm.client import LLMClient

logger = logging.getLogger(__name__)


class DependencyResolver:
    """检测 cleaning 操作之间的依赖关系。"""

    def __init__(
        self,
        llm: LLMClient | None = None,
        max_depth: int = 3,
    ):
        self.llm = llm
        self.max_depth = max_depth

    def resolve(
        self,
        messages: list[dict[str, Any]],
        plans: list[CleaningPlan],
    ) -> list[CleaningPlan]:
        """分析依赖关系，对受影响的后续消息追加清洗计划。"""
        if not plans:
            return plans

        # 收集被修改的消息索引
        modified_indices: set[int] = set()
        for p in plans:
            if p.action in (
                CleaningAction.DELETE,
                CleaningAction.REPLACE,
                CleaningAction.REWRITE,
                CleaningAction.DELETE_MESSAGE,
            ):
                modified_indices.add(p.message_index)

        if not modified_indices:
            return plans

        # 检测后续消息的依赖
        additional_plans: list[CleaningPlan] = []

        for depth in range(self.max_depth):
            new_modified: set[int] = set()
            for mi in sorted(modified_indices):
                affected = self._find_affected(messages, mi, plans)
                for affected_mi in affected:
                    if affected_mi not in modified_indices:
                        new_modified.add(affected_mi)
                        # 让 LLM 重新生成受影响的消息
                        new_plan = self._rewrite_affected(messages, affected_mi, plans)
                        if new_plan:
                            additional_plans.append(new_plan)
                            logger.info(
                                f"依赖处理: 消息 #{mi} 被修改 → 消息 #{affected_mi} 需要重写"
                            )
            modified_indices.update(new_modified)
            if not new_modified:
                break

        return plans + additional_plans

    # ── 内部方法 ──────────────────────────────────────────────────────────

    def _find_affected(
        self,
        messages: list[dict[str, Any]],
        modified_mi: int,
        plans: list[CleaningPlan],
    ) -> list[int]:
        """找到依赖被修改消息的后续消息。"""
        affected: list[int] = []
        modified_content = self._get_message_text(messages[modified_mi])

        for mi in range(modified_mi + 1, len(messages)):
            msg = messages[mi]
            msg_text = self._get_message_text(msg)

            # 引用检测：后续消息是否包含被修改消息的关键词
            if self._has_reference(modified_content, msg_text):
                affected.append(mi)
                continue

            # tool_call 链：后续 tool_call 的参数是否依赖被修改的 tool response
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    args = tc.get("function", {}).get("arguments", "")
                    if self._has_reference(modified_content, args):
                        affected.append(mi)
                        break

        return affected

    def _rewrite_affected(
        self,
        messages: list[dict[str, Any]],
        affected_mi: int,
        plans: list[CleaningPlan],
    ) -> CleaningPlan | None:
        """调用 LLM 重新生成受影响的消息。"""
        if not self.llm:
            return None

        # 收集上下文：受影响消息之前的所有已清洗消息
        context_messages = messages[:affected_mi]
        affected_msg = messages[affected_mi]

        # 构建 prompt
        prompt = self._build_rewrite_prompt(context_messages, affected_msg, plans)
        try:
            new_content = self.llm.chat([{"role": "user", "content": prompt}])
            return CleaningPlan(
                message_index=affected_mi,
                action=CleaningAction.REWRITE,
                reason="依赖关系：前面的消息被修改，此消息需要重新生成",
                new_content=new_content,
                source="dependency_resolver",
            )
        except Exception as e:
            logger.warning(f"LLM 重写失败: {e}")
            return None

    @staticmethod
    def _build_rewrite_prompt(
        context: list[dict[str, Any]],
        affected: dict[str, Any],
        plans: list[CleaningPlan],
    ) -> str:
        """构建重写提示。"""
        # 简化上下文
        context_summary = []
        for i, msg in enumerate(context):
            role = msg.get("role", "")
            text = DependencyResolver._get_message_text(msg)
            context_summary.append(f"[{i}] {role}: {text[:200]}")

        changes = []
        for p in plans:
            if p.action in (CleaningAction.DELETE, CleaningAction.REPLACE, CleaningAction.REWRITE):
                changes.append(f"消息 #{p.message_index}: {p.action.value} — {p.reason}")

        affected_text = DependencyResolver._get_message_text(affected)

        return f"""你是一个代码 agent trace 清洗助手。以下 trace 的前面部分被修改了，你需要重新生成受影响的后续消息。

## 前面的消息（已修改）
{chr(10).join(context_summary[-5:])}

## 修改记录
{chr(10).join(changes)}

## 需要重新生成的消息
角色: {affected.get('role', '')}
原始内容: {affected_text[:500]}

请直接输出新的消息内容（保持原始格式），使它与前面的修改保持一致。只输出内容，不要解释。"""

    @staticmethod
    def _get_message_text(msg: dict[str, Any]) -> str:
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            )
        return ""

    @staticmethod
    def _has_reference(source: str, target: str) -> bool:
        """简单引用检测：source 中的关键词是否出现在 target 中。"""
        if not source or not target:
            return False
        # 提取有意义的标识符
        identifiers = set(re.findall(r"\b[a-zA-Z_]\w{2,}\b", source))
        if not identifiers:
            return False
        # 检查是否有重叠
        overlap = identifiers & set(re.findall(r"\b[a-zA-Z_]\w{2,}\b", target))
        return len(overlap) >= 2  # 至少 2 个标识符重叠才算依赖
