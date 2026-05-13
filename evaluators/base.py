"""评估器基类。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from core.models import EvalResult


class BaseEvaluator(ABC):
    """所有评估器实现此接口。"""

    name: str = "base"

    @abstractmethod
    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        """
        评估 trace messages。

        Args:
            messages: trace 的 messages 数组
            reference: reference.py 代码内容
            context: 额外上下文（如 trace_type, task_id 等）

        Returns:
            评估结果列表（只包含触发了情况的消息，好的不需要输出）
        """
