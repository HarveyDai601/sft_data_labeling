"""诊断脚本 — 逐条检查每个 evaluator 为什么返回 0。

用法:
    python diagnose.py --data_dir data/ --task task_001
    python diagnose.py --data_dir data/  # 检查所有 task
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from core.loader import load_task
from core.models import TaskData
from evaluators import REGISTRY

logging.basicConfig(level=logging.DEBUG, format="%(name)s | %(levelname)s: %(message)s")
logger = logging.getLogger("diagnose")


def diagnose_trace(
    messages: list[dict[str, Any]],
    reference: str,
    trace_type: str,
    trace_name: str,
) -> None:
    """诊断单个 trace。"""
    print(f"\n{'='*70}")
    print(f"  诊断: {trace_name} (type={trace_type}, {len(messages)} messages)")
    print(f"{'='*70}")

    # ── 1. 检查 messages 结构 ────────────────────────────────────────────
    print(f"\n--- 消息结构 ---")
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id", "")

        content_type = type(content).__name__
        content_preview = ""
        if isinstance(content, str):
            content_preview = content[:80].replace("\n", "\\n")
        elif isinstance(content, list):
            content_preview = f"[{len(content)} items: {', '.join(item.get('type', '?') for item in content if isinstance(item, dict))}]"

        tc_str = f", tool_calls={len(tool_calls)}" if tool_calls else ""
        tci_str = f", tool_call_id={tool_call_id[:20]}" if tool_call_id else ""

        print(f"  [{i}] {role} | content({content_type}): {content_preview}{tc_str}{tci_str}")

    # ── 2. 检查代码块提取 ────────────────────────────────────────────────
    print(f"\n--- 代码块提取 ---")
    total_code_blocks = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        blocks = _extract_code_blocks(msg)
        if blocks:
            total_code_blocks += len(blocks)
            for j, block in enumerate(blocks):
                preview = block[:100].replace("\n", "\\n")
                print(f"  [{i}] assistant item#{j}: {len(block)} chars | {preview}...")
        else:
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                print(f"  [{i}] assistant: 无代码块, content={content[:60].replace(chr(10), '\\n')}...")
            elif isinstance(content, list):
                types = [item.get("type", "?") for item in content if isinstance(item, dict)]
                print(f"  [{i}] assistant: 无代码块, content items: {types}")

    if total_code_blocks == 0:
        print("  ⚠️ 未提取到任何代码块！这是返回 0 的主要原因。")
        print("  可能原因:")
        print("    - trace 中没有 ``` 包裹的代码")
        print("    - trace 的代码在 content list 的 code type item 中")
        print("    - trace 的代码在 tool response 中而非 assistant 中")

    # ── 3. 检查 AST 解析 ────────────────────────────────────────────────
    print(f"\n--- AST 解析 ---")
    if reference:
        ref_apis = _extract_apis(reference)
        print(f"  reference 提取到 {len(ref_apis)} 个 API: {sorted(ref_apis)[:20]}")

        ref_sigs = _extract_sigs(reference)
        print(f"  reference 提取到 {len(ref_sigs)} 个函数签名: { {k: sorted(v) for k, v in ref_sigs.items()} }")
    else:
        print("  ⚠️ reference 为空！")
        ref_apis = set()
        ref_sigs = {}

    # ── 4. 逐个 evaluator 诊断 ──────────────────────────────────────────
    print(f"\n--- 逐个 evaluator 诊断 ---")
    for name, cls in REGISTRY.items():
        print(f"\n  [{name}]")

        # 检查是否适用于此 trace type
        # (这里硬编码，和 config.yaml 一致)
        _applies = {
            "code_diff": ["script_complete"],
            "function_recall": ["script_complete", "main"],
            "param_recall": ["script_complete", "main"],
            "file_relevance": ["script_retrieval"],
            "step_relevance": ["step_retrieval"],
            "consistency": ["consistency_check"],
            "tool_call_decision": ["main"],
        }
        applies = _applies.get(name, [])
        if trace_type not in applies:
            print(f"    跳过: 不适用于 {trace_type} (只适用于 {applies})")
            continue

        # 检查 target indices
        is_main = trace_type == "main"
        if name in ("function_recall", "param_recall"):
            if is_main:
                targets = [i for i, m in enumerate(messages) if m.get("role") == "assistant" and _has_code_block(m)]
                print(f"    target_indices (main, 含代码块): {targets}")
            else:
                targets = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
                print(f"    target_indices (sub, 所有 assistant): {targets}")

            if not targets:
                print(f"    ⚠️ 无目标消息 → 返回 0")
                continue

            # 检查差集
            trace_apis = set()
            for mi in targets:
                for block in _extract_code_blocks(messages[mi]):
                    trace_apis.update(_extract_apis(block))
            missing = ref_apis - trace_apis
            print(f"    trace API: {sorted(trace_apis)[:20]}")
            print(f"    差集(疑点): {sorted(missing)[:20]}")
            if not missing:
                print(f"    无疑点 → 返回 0")

        elif name == "code_diff":
            blocks_found = 0
            for i, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue
                blocks = _extract_code_blocks(msg)
                blocks_found += len(blocks)
            print(f"    找到 {blocks_found} 个代码块")
            if blocks_found == 0:
                print(f"    ⚠️ 无代码块可比对 → 返回 0")

        elif name == "tool_call_decision":
            calls = []
            for i, msg in enumerate(messages):
                if msg.get("role") != "assistant":
                    continue
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {}).get("name", "")
                    if fn in ("script_retrieval", "step_retrieval", "script_complete", "consistency_check"):
                        calls.append((i, fn))
            print(f"    tool calls: {calls}")
            if len(calls) < 2:
                print(f"    调用次数 < 2 → 跳过顺序检查")

        elif name == "file_relevance":
            tool_msgs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
            print(f"    tool response 消息: {tool_msgs}")
            if not tool_msgs:
                print(f"    ⚠️ 无 tool response → 返回 0")

        elif name == "step_relevance":
            tool_msgs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
            print(f"    tool response 消息: {tool_msgs}")
            if not tool_msgs:
                print(f"    ⚠️ 无 tool response → 返回 0")

        elif name == "consistency":
            tool_msgs = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
            print(f"    tool response 消息: {tool_msgs}")
            if not tool_msgs:
                print(f"    ⚠️ 无 tool response → 返回 0")


# ── 工具函数（从 evaluator 中复制，用于诊断）──────────────────────────────


def _has_code_block(msg: dict[str, Any]) -> bool:
    content = msg.get("content", "")
    if isinstance(content, str):
        return "```" in content
    if isinstance(content, list):
        return any(
            item.get("type") == "code" or "```" in (item.get("text", "") or "")
            for item in content if isinstance(item, dict)
        )
    return False


def _extract_code_blocks(msg: dict[str, Any]) -> list[str]:
    blocks: list[str] = []
    content = msg.get("content", "")
    if isinstance(content, str):
        for m in re.finditer(r"```(?:\w*\n)?(.*?)```", content, re.DOTALL):
            blocks.append(m.group(1).strip())
        if not blocks and content.strip():
            blocks.append(content.strip())
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "code":
                    blocks.append(item.get("text", "") or item.get("content", ""))
                elif item.get("type") == "text":
                    text = item.get("text", "")
                    for m in re.finditer(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL):
                        blocks.append(m.group(1).strip())
    return blocks


def _extract_apis(code: str) -> set[str]:
    import ast
    apis: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(re.findall(r"\bdef\s+(\w+)\s*\(", code))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            apis.add(node.name)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                apis.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                apis.add(node.func.attr)
    return apis


def _extract_sigs(code: str) -> dict[str, set[str]]:
    import ast
    sigs: dict[str, set[str]] = {}
    _ignored = {"self", "cls", "args", "kwargs"}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = set()
            for arg in node.args.args:
                if arg.arg not in _ignored:
                    params.add(arg.arg)
            sigs[node.name] = params
    return sigs


# ── 入口 ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="诊断评估器为何返回 0")
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--task", type=str, default=None)
    args = parser.parse_args()

    if args.task:
        task = load_task(args.data_dir / args.task)
        tasks = [task]
    else:
        from core.loader import load_all_tasks
        tasks = load_all_tasks(args.data_dir)

    if not tasks:
        print("未找到任何 task，请检查 --data_dir 路径")
        return

    for task in tasks:
        print(f"\n{'#'*70}")
        print(f"  Task: {task.task_id}")
        print(f"  main messages: {len(task.main_messages)}")
        print(f"  sub traces: {len(task.sub_traces)}")
        print(f"  reference: {'有' if task.reference_code else '无'} ({len(task.reference_code)} chars)")
        print(f"{'#'*70}")

        # 诊断 main
        diagnose_trace(task.main_messages, task.reference_code, "main", "main")

        # 诊断每个 sub
        for sub in task.sub_traces:
            diagnose_trace(sub.messages, task.reference_code, sub.trace_type, sub.name)


if __name__ == "__main__":
    main()
