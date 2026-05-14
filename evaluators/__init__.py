"""评估器注册表 — 新增评估器只需在这里导入并添加到 REGISTRY。"""

from __future__ import annotations

from evaluators.base import BaseEvaluator
from evaluators.code_correctness import CodeCorrectnessEvaluator
from evaluators.info_usage import InfoUsageEvaluator
from evaluators.orchestration import OrchestrationEvaluator
from evaluators.consistency import ConsistencyEvaluator

# ── 注册表：name → class ─────────────────────────────────────────────────
REGISTRY: dict[str, type[BaseEvaluator]] = {
    "code_correctness": CodeCorrectnessEvaluator,
    "info_usage": InfoUsageEvaluator,
    "orchestration": OrchestrationEvaluator,
    "consistency": ConsistencyEvaluator,
}

__all__ = [
    "BaseEvaluator",
    "CodeCorrectnessEvaluator",
    "InfoUsageEvaluator",
    "OrchestrationEvaluator",
    "ConsistencyEvaluator",
    "REGISTRY",
]
