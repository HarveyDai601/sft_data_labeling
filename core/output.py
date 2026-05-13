"""输出模块 — 生成清洗后的 JSON 和报告文件。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from core.models import CleaningAction, CleaningPlan, EvalResult, TaskData

logger = logging.getLogger(__name__)


def write_cleaned_json(path: Path, messages: list[dict[str, Any]]) -> None:
    """写入清洗后的 messages JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"写入清洗文件: {path}")


def write_report(
    path: Path,
    trace_name: str,
    eval_results: list[EvalResult],
    cleaning_plans: list[CleaningPlan],
    total_messages: int,
) -> None:
    """写入单个 trace 的处理报告（markdown + debug JSON）。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    # ── Markdown 报告 ────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append(f"# {trace_name} 处理报告\n")

    # 评估结果（按维度）
    lines.append("## 评估结果（按维度）\n")
    by_dimension: dict[str, list[EvalResult]] = {}
    for er in eval_results:
        by_dimension.setdefault(er.dimension, []).append(er)

    for dim, results in by_dimension.items():
        lines.append(f"### {dim}\n")
        for r in results:
            item_str = f" item#{r.item_index}" if r.item_index is not None else ""
            lines.append(
                f"- 消息 #{r.message_index}{item_str}: "
                f"**{r.action.value.upper()}** (置信度: {r.confidence:.1f}) — "
                f"{r.finding}"
            )
        lines.append("")

    # 清洗操作（实际执行的动作）
    lines.append("## 清洗操作\n")
    action_count: dict[str, int] = {}
    for p in cleaning_plans:
        if p.action == CleaningAction.KEEP:
            continue
        action_count[p.action.value] = action_count.get(p.action.value, 0) + 1
        item_str = f" item#{p.item_index}" if p.item_index is not None else ""
        lines.append(
            f"- 消息 #{p.message_index}{item_str}: **{p.action.value}** — "
            f"{p.reason} (来源: {p.source})"
        )
    if not action_count:
        lines.append("（无实际清洗操作）")
    lines.append("")

    # 观察记录（KEEP 的评估结果，来自 eval_results 而非 plans）
    observations = [r for r in eval_results if r.action == CleaningAction.KEEP]
    if observations:
        lines.append("## 观察记录（未执行清洗）\n")
        for r in observations:
            item_str = f" item#{r.item_index}" if r.item_index is not None else ""
            lines.append(
                f"- 消息 #{r.message_index}{item_str}: {r.finding} (来源: {r.dimension})"
            )
        lines.append("")

    # 统计
    lines.append("## 统计\n")
    lines.append(f"- 总消息数: {total_messages}")
    lines.append(f"- 评估结果数: {len(eval_results)}")
    lines.append(f"- 清洗操作数: {len(cleaning_plans)}")
    for action, count in action_count.items():
        lines.append(f"  - {action}: {count}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"写入报告: {path}")

    # ── Debug JSON ───────────────────────────────────────────────────────
    debug_path = path.with_suffix(".debug.json")
    debug_data = {
        "trace_name": trace_name,
        "total_messages": total_messages,
        "eval_results": [
            {
                "message_index": r.message_index,
                "item_index": r.item_index,
                "dimension": r.dimension,
                "finding": r.finding,
                "confidence": r.confidence,
                "action": r.action.value,
                "action_reason": r.action_reason,
                "new_content_preview": str(r.new_content)[:200] if r.new_content else None,
                "debug_info": r.debug_info,
            }
            for r in eval_results
        ],
        "cleaning_plans": [
            {
                "message_index": p.message_index,
                "item_index": p.item_index,
                "action": p.action.value,
                "reason": p.reason,
                "source": p.source,
            }
            for p in cleaning_plans
        ],
    }
    debug_path.write_text(json.dumps(debug_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"写入 debug JSON: {debug_path}")


def write_summary(
    path: Path,
    task_id: str,
    traces: dict[str, dict[str, Any]],
) -> None:
    """写入整体汇总报告。"""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append(f"# Task {task_id} 汇总报告\n")

    total_messages = 0
    total_evals = 0
    total_plans = 0

    for trace_name, info in traces.items():
        total_messages += info["total_messages"]
        total_evals += info["total_evals"]
        total_plans += info["total_plans"]
        lines.append(f"## {trace_name}")
        lines.append(f"- 消息数: {info['total_messages']}")
        lines.append(f"- 评估结果: {info['total_evals']}")
        lines.append(f"- 清洗操作: {info['total_plans']}")
        lines.append("")

    lines.append("---")
    lines.append(f"**总计**: {total_messages} 消息, {total_evals} 评估, {total_plans} 清洗")

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"写入汇总: {path}")
