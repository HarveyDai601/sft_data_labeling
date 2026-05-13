from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ── 评估结果 ──────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    """单条评估结果 — 由评估器产出，喂给 Cleaner。"""

    message_index: int
    item_index: int | None = None       # None → 整条 message 级别
    dimension: str = ""                 # 评估维度名
    verdict: str = "good"               # "good" | "bad" | "missing"
    reason: str = ""
    severity: float = 0.0               # 0‑1
    suggested_fix: dict[str, Any] | None = None


# ── 清洗动作 ──────────────────────────────────────────────────────────────


class CleaningAction(str, Enum):
    KEEP = "keep"
    DELETE = "delete"
    REPLACE = "replace"
    INSERT_BEFORE = "insert_before"
    INSERT_AFTER = "insert_after"
    REWRITE = "rewrite"
    DELETE_MESSAGE = "delete_message"


@dataclass
class CleaningPlan:
    """一条清洗计划项。"""

    message_index: int
    action: CleaningAction
    reason: str
    item_index: int | None = None
    new_content: Any = None             # REPLACE / INSERT 时的新内容
    source: str = ""                    # 来自哪个评估器


# ── 任务数据 ──────────────────────────────────────────────────────────────


@dataclass
class SubTrace:
    """一个 sub agent 的 trace。"""

    name: str                           # e.g. "file_retrieval_0"
    trace_type: str                     # "file_retrieval" | "step_retrieval" | "code_completion" | "consistency_check"
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
    """从文件名解析 trace 类型和索引，如 'file_retrieval_0' → ('file_retrieval', 0)。"""
    for prefix in ("file_retrieval", "step_retrieval", "code_completion", "consistency_check"):
        if filename.startswith(prefix):
            idx = int(filename[len(prefix) + 1 :])
            return prefix, idx
    raise ValueError(f"无法解析 trace 类型: {filename}")
