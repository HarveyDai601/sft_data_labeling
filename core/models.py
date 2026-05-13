from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── 清洗动作 ──────────────────────────────────────────────────────────────


class CleaningAction(str, Enum):
    KEEP = "keep"               # 保留原样
    DELETE = "delete"           # 删除该 item / message
    REPLACE = "replace"         # 替换内容（必须提供 new_content）
    INSERT_AFTER = "insert_after"  # 在后面插入（必须提供 new_content）
    DELETE_MESSAGE = "delete_message"  # 删除整条 message


# ── 评估结果 ──────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """单条评估结果。

    评估器的职责：检测问题 + 建议清洗动作。
    Cleaner 的职责：执行建议的动作（不做二次判断）。
    """
    message_index: int
    item_index: int | None = None       # None → 整条 message 级别
    dimension: str = ""                 # 评估维度名

    # ── 评估结论 ──────────────────────────────────────────────────────────
    finding: str = ""                   # 发现了什么（如 "缺少 process_data 函数"）
    confidence: float = 0.0             # 置信度 0-1（评估器对这个判断有多确定）

    # ── 清洗建议（评估器直接告诉 Cleaner 该怎么做）────────────────────────
    action: CleaningAction = CleaningAction.KEEP
    action_reason: str = ""             # 为什么建议这个动作
    new_content: Any = None             # REPLACE / INSERT 时的新内容

    # ── Debug ─────────────────────────────────────────────────────────────
    debug_info: dict[str, Any] | None = None


# ── 清洗计划 ──────────────────────────────────────────────────────────────


@dataclass
class CleaningPlan:
    """一条清洗计划项 — Cleaner 从多个 EvalResult 合并后产出。"""

    message_index: int
    action: CleaningAction
    reason: str
    item_index: int | None = None
    new_content: Any = None
    source: str = ""                    # 来自哪个评估器


# ── 任务数据 ──────────────────────────────────────────────────────────────


@dataclass
class SubTrace:
    """一个 sub agent 的 trace。"""

    name: str                           # e.g. "script_retrieval_0"
    trace_type: str                     # "script_retrieval" | "step_retrieval" | "script_complete" | "consistency_check"
    index: int                          # 同类型第几次调用 (0‑based)
    messages: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskData:
    """一个 task 的全部数据。"""

    task_id: str
    main_messages: list[dict[str, Any]] = field(default_factory=list)
    sub_traces: list[SubTrace] = field(default_factory=list)
    reference_code: str = ""
    context: str = ""

    @property
    def trace_type(self) -> str:
        return "main"


def trace_type_from_filename(filename: str) -> tuple[str, int]:
    """从文件名解析 trace 类型和索引，如 'script_retrieval_0' → ('script_retrieval', 0)。"""
    for prefix in ("script_retrieval", "step_retrieval", "script_complete", "consistency_check"):
        if filename.startswith(prefix):
            idx = int(filename[len(prefix) + 1 :])
            return prefix, idx
    raise ValueError(f"无法解析 trace 类型: {filename}")
