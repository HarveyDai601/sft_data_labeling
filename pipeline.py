"""主 Pipeline — 加载 → 评估 → 清洗 → 输出。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from cleaners.cleaner import Cleaner
from cleaners.dependency_resolver import DependencyResolver
from core.loader import load_all_tasks, load_task
from core.models import EvalResult, TaskData
from core.output import write_cleaned_json, write_report, write_summary
from evaluators import REGISTRY as EVALUATOR_REGISTRY
from evaluators.base import BaseEvaluator
from llm.client import LLMClient

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── 配置加载 ──────────────────────────────────────────────────────────────


def load_config(config_path: Path) -> dict[str, Any]:
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {}


# ── 评估器实例化 ──────────────────────────────────────────────────────────


def build_evaluators(
    config: dict[str, Any],
    llm: LLMClient | None,
    trace_type: str,
) -> list[BaseEvaluator]:
    """根据配置和 trace 类型，实例化需要的评估器。"""
    evaluators: list[BaseEvaluator] = []
    eval_cfg = config.get("evaluators", {})

    for name, cls in EVALUATOR_REGISTRY.items():
        cfg = eval_cfg.get(name, {})
        if not cfg.get("enabled", True):
            continue
        applies_to = cfg.get("applies_to", [])
        if applies_to and trace_type not in applies_to:
            continue

        # 部分评估器需要 LLM 客户端
        if name in ("code_diff", "consistency", "function_recall", "param_recall"):
            evaluators.append(cls(llm=llm))
        else:
            evaluators.append(cls())

    return evaluators


# ── 单个 trace 处理 ──────────────────────────────────────────────────────


def process_trace(
    messages: list[dict[str, Any]],
    reference: str,
    trace_type: str,
    trace_name: str,
    evaluators: list[BaseEvaluator],
    cleaner: Cleaner,
    dep_resolver: DependencyResolver,
) -> tuple[list[dict[str, Any]], list[EvalResult], list[CleaningPlan]]:
    """处理单个 trace，返回 (cleaned_messages, eval_results, plans)。"""
    context: dict[str, Any] = {"trace_type": trace_type}

    # Step 1: 评估
    all_eval_results: list[EvalResult] = []
    for evaluator in evaluators:
        results = evaluator.evaluate(messages, reference, context)
        all_eval_results.extend(results)
        logger.info(f"  [{evaluator.name}] 产出 {len(results)} 条评估结果")

    # Step 2: 生成清洗计划
    plans = cleaner.plan(messages, all_eval_results)
    logger.info(f"  生成 {len(plans)} 条清洗计划")

    # Step 3: 依赖解析
    plans = dep_resolver.resolve(messages, plans)
    logger.info(f"  依赖解析后 {len(plans)} 条清洗计划")

    # Step 4: 执行清洗
    cleaned = cleaner.execute(messages, plans)

    return cleaned, all_eval_results, plans


# ── 单个 task 处理 ──────────────────────────────────────────────────────


def process_task(
    task: TaskData,
    config: dict[str, Any],
    llm: LLMClient | None,
    output_dir: Path,
) -> dict[str, Any]:
    """处理单个 task 的所有 trace。"""
    logger.info(f"\n{'='*60}")
    logger.info(f"处理 task: {task.task_id}")
    logger.info(f"{'='*60}")

    cleaner_cfg = config.get("cleaner", {})
    cleaner = Cleaner(conflict_resolution=cleaner_cfg.get("conflict_resolution", "severity_priority"))
    dep_resolver = DependencyResolver(
        llm=llm,
        max_depth=cleaner_cfg.get("max_dependency_depth", 3),
    )

    task_output = output_dir / task.task_id
    task_output.mkdir(parents=True, exist_ok=True)

    traces_info: dict[str, dict[str, Any]] = {}

    # 处理 main trace
    logger.info(f"\n--- main trace ({len(task.main_messages)} messages) ---")
    main_evaluators = build_evaluators(config, llm, "main")
    cleaned_main, evals, plans = process_trace(
        task.main_messages,
        task.reference_code,
        "main",
        "main",
        main_evaluators,
        cleaner,
        dep_resolver,
    )
    write_cleaned_json(task_output / "main_cleaned.json", cleaned_main)
    write_report(task_output / "main_report.md", "main", evals, plans, len(task.main_messages))
    traces_info["main"] = {
        "total_messages": len(task.main_messages),
        "total_evals": len(evals),
        "total_plans": len(plans),
    }

    # 处理每个 sub trace
    for sub in task.sub_traces:
        logger.info(f"\n--- {sub.name} ({len(sub.messages)} messages) ---")
        sub_evaluators = build_evaluators(config, llm, sub.trace_type)
        cleaned_sub, evals, plans = process_trace(
            sub.messages,
            task.reference_code,
            sub.trace_type,
            sub.name,
            sub_evaluators,
            cleaner,
            dep_resolver,
        )
        write_cleaned_json(task_output / f"{sub.name}_cleaned.json", cleaned_sub)
        write_report(task_output / f"{sub.name}_report.md", sub.name, evals, plans, len(sub.messages))
        traces_info[sub.name] = {
            "total_messages": len(sub.messages),
            "total_evals": len(evals),
            "total_plans": len(plans),
        }

    # 写汇总
    write_summary(task_output / "summary.md", task.task_id, traces_info)

    return traces_info


# ── 入口 ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace Labeling & Cleaning Pipeline")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="输入数据目录")
    parser.add_argument("--output_dir", type=Path, default=Path("output"), help="输出目录")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="配置文件")
    parser.add_argument("--task", type=str, default=None, help="只处理指定 task")
    parser.add_argument("--eval_only", action="store_true", help="只评估不清洗")
    parser.add_argument("--verbose", "-v", action="store_true", help="显示详细评估日志")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config = load_config(args.config)

    # 初始化 LLM 客户端
    llm_cfg = config.get("llm", {})
    llm = None
    if llm_cfg.get("base_url"):
        llm = LLMClient(
            base_url=llm_cfg["base_url"],
            api_key=llm_cfg.get("api_key", "sk-none"),
            model=llm_cfg.get("model", "gpt-4"),
            temperature=llm_cfg.get("temperature", 0.1),
            timeout=llm_cfg.get("timeout", 120),
            max_retries=llm_cfg.get("max_retries", 3),
        )
        logger.info(f"LLM 客户端: {llm_cfg['base_url']} / {llm_cfg.get('model', 'gpt-4')}")
    else:
        logger.warning("未配置 LLM，部分评估器将跳过语义判断")

    # 加载 tasks
    if args.task:
        task_dir = args.data_dir / args.task
        tasks = [load_task(task_dir)]
    else:
        tasks = load_all_tasks(args.data_dir)

    logger.info(f"加载 {len(tasks)} 个 task")

    # 处理
    for task in tasks:
        process_task(task, config, llm, args.output_dir)

    logger.info("\n全部完成!")


if __name__ == "__main__":
    main()
