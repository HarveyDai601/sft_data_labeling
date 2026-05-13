"""加载 trace JSON + reference.py，构建 TaskData。"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from core.models import SubTrace, TaskData, trace_type_from_filename

logger = logging.getLogger(__name__)

_SUB_PREFIXES = ("script_retrieval_", "step_retrieval_", "script_completion_", "consistency_check_")


def load_task(task_dir: Path) -> TaskData:
    """加载单个 task 目录。"""
    task_id = task_dir.name

    # main.json
    main_path = task_dir / "main.json"
    if not main_path.exists():
        raise FileNotFoundError(f"缺少 main.json: {main_path}")
    main_messages = json.loads(main_path.read_text(encoding="utf-8"))
    if isinstance(main_messages, dict) and "messages" in main_messages:
        main_messages = main_messages["messages"]

    # reference.py
    ref_path = task_dir / "reference.py"
    reference_code = ref_path.read_text(encoding="utf-8") if ref_path.exists() else ""

    # context.md
    ctx_path = task_dir / "context.md"
    context = ctx_path.read_text(encoding="utf-8") if ctx_path.exists() else ""

    # sub agent traces
    sub_traces: list[SubTrace] = []
    for f in sorted(task_dir.iterdir()):
        if not f.name.endswith(".json") or f.name == "main.json":
            continue
        stem = f.stem  # e.g. "script_retrieval_0"
        if not any(stem.startswith(p) for p in _SUB_PREFIXES):
            continue
        trace_type, idx = trace_type_from_filename(stem)
        messages = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(messages, dict) and "messages" in messages:
            messages = messages["messages"]
        sub_traces.append(
            SubTrace(name=stem, trace_type=trace_type, index=idx, messages=messages)
        )

    return TaskData(
        task_id=task_id,
        main_messages=main_messages,
        sub_traces=sub_traces,
        reference_code=reference_code,
        context=context,
    )


def load_all_tasks(data_dir: Path) -> list[TaskData]:
    """加载 data_dir 下所有 task。"""
    tasks: list[TaskData] = []
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and (d / "main.json").exists():
            tasks.append(load_task(d))
    return tasks
