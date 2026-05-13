"""评估器注册表 — 新增评估器只需在这里导入并添加到 REGISTRY。"""

from __future__ import annotations

from evaluators.base import BaseEvaluator
from evaluators.code_diff import CodeDiffEvaluator
from evaluators.consistency import ConsistencyEvaluator
from evaluators.file_relevance import FileRelevanceEvaluator
from evaluators.function_recall import FunctionRecallEvaluator
from evaluators.param_recall import ParamRecallEvaluator
from evaluators.step_relevance import StepRelevanceEvaluator
from evaluators.tool_call_decision import ToolCallEvaluator

# ── 注册表：name → class ─────────────────────────────────────────────────
REGISTRY: dict[str, type[BaseEvaluator]] = {
    "code_diff": CodeDiffEvaluator,
    "function_recall": FunctionRecallEvaluator,
    "param_recall": ParamRecallEvaluator,
    "file_relevance": FileRelevanceEvaluator,
    "step_relevance": StepRelevanceEvaluator,
    "consistency": ConsistencyEvaluator,
    "tool_call_decision": ToolCallEvaluator,
}

__all__ = [
    "BaseEvaluator",
    "CodeDiffEvaluator",
    "ConsistencyEvaluator",
    "FileRelevanceEvaluator",
    "FunctionRecallEvaluator",
    "ParamRecallEvaluator",
    "StepRelevanceEvaluator",
    "ToolCallEvaluator",
    "REGISTRY",
]
