"""参数准召率评估器 — AST 初筛 + LLM 终判。

评估逻辑：
- 从 write/edit 工具的 arguments 中提取 agent 实际写出的代码
- AST 解析函数签名参数，与 reference 比对
- 参数匹配 → KEEP
- 参数不匹配 → KEEP（观察记录，无法自动生成正确参数名）
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

from core.models import CleaningAction, EvalResult
from evaluators.base import BaseEvaluator
from llm.client import LLMClient

logger = logging.getLogger(__name__)


class ParamRecallEvaluator(BaseEvaluator):
    name = "param_recall"

    _IGNORED_PARAMS = {"self", "cls", "args", "kwargs"}

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        ref_sigs = self._extract_func_signatures(reference)
        if not ref_sigs:
            logger.info(f"[{self.name}] reference 中未提取到函数签名，跳过")
            return []

        is_main = context.get("trace_type") == "main"
        target_indices = self._get_target_indices(messages, is_main)
        if not target_indices:
            return []

        # 从 write/edit 工具调用中提取 trace 中实际写出的代码
        trace_code_blocks: list[str] = []
        for mi in target_indices:
            msg = messages[mi]
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                code = self._extract_from_write_args(fn.get("arguments", ""), fn.get("name", ""))
                if code:
                    trace_code_blocks.append(code)

        trace_code = "\n".join(trace_code_blocks)
        trace_sigs = self._extract_func_signatures(trace_code)

        logger.info(
            f"[{self.name}] reference 签名: { {k: sorted(v) for k, v in ref_sigs.items()} } | "
            f"trace 签名: { {k: sorted(v) for k, v in trace_sigs.items()} }"
        )

        suspects: list[dict[str, Any]] = []
        for func_name, ref_params in ref_sigs.items():
            if func_name not in trace_sigs:
                continue
            trace_params = trace_sigs[func_name]
            missing = ref_params - trace_params
            if missing:
                suspects.append({
                    "function": func_name,
                    "ref_params": sorted(ref_params),
                    "trace_params": sorted(trace_params),
                    "missing": sorted(missing),
                })

        if not suspects:
            return []

        logger.info(
            f"[{self.name}] 疑点: "
            + "; ".join(f"{s['function']} 缺失 {s['missing']}" for s in suspects)
        )

        if self.llm:
            suspects = self._llm_verify(trace_code, suspects)
            if not suspects:
                return []

        details = "; ".join(f"{s['function']}({', '.join(s['missing'])})" for s in suspects)
        total_ref = sum(len(v) for v in ref_sigs.values())
        flat = [p for s in suspects for p in s["missing"]]
        confidence = min(1.0, len(flat) / max(total_ref, 1) * 1.5)

        return [
            EvalResult(
                message_index=target_indices[-1],
                dimension=self.name,
                finding=f"参数名不匹配: {details}",
                confidence=round(confidence, 2),
                action=CleaningAction.KEEP,
                action_reason="无法自动生成正确参数名，仅记录",
                debug_info={
                    "reference_signatures": {k: sorted(v) for k, v in ref_sigs.items()},
                    "trace_signatures": {k: sorted(v) for k, v in trace_sigs.items()},
                    "suspects": suspects,
                    "llm_used": self.llm is not None,
                },
            )
        ]

    def _llm_verify(self, trace_code: str, suspects: list[dict]) -> list[dict]:
        desc = "\n".join(
            f"- {s['function']}: ref {s['ref_params']}, trace {s['trace_params']}, 缺失 {s['missing']}"
            for s in suspects
        )
        prompt = f"""判断以下参数差异是否为真正缺失。

Agent 代码:
```python
{trace_code}
```

参数差异:
{desc}

判断标准：参数名不同但语义等价（如 data/input_data）→ 不是缺失。

JSON: {{"results": [{{"function": "函数名", "real_missing": ["真正缺失的参数"], "equivalent": ["语义等价的参数"]}}]}}"""

        try:
            resp = self.llm.chat_json([{"role": "user", "content": prompt}], temperature=0.0)
            confirmed = []
            for r in resp.get("results", []):
                equiv = r.get("equivalent", [])
                if equiv:
                    logger.info(f"[{self.name}] LLM 判断 {r['function']} 中 {equiv} 为语义等价")
                real = set(r.get("real_missing", []))
                if real:
                    for s in suspects:
                        if s["function"] == r["function"]:
                            confirmed.append({**s, "missing": sorted(real)})
            return confirmed
        except Exception as e:
            logger.warning(f"[{self.name}] LLM 验证失败: {e}")
            return suspects

    @staticmethod
    def _extract_func_signatures(code: str) -> dict[str, set[str]]:
        sigs: dict[str, set[str]] = {}
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ParamRecallEvaluator._extract_func_signatures_regex(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = set()
                for arg in node.args.args:
                    if arg.arg not in ParamRecallEvaluator._IGNORED_PARAMS:
                        params.add(arg.arg)
                for arg in node.args.kwonlyargs:
                    if arg.arg not in ParamRecallEvaluator._IGNORED_PARAMS:
                        params.add(arg.arg)
                if node.args.vararg and node.args.vararg.arg not in ParamRecallEvaluator._IGNORED_PARAMS:
                    params.add(node.args.vararg.arg)
                if node.args.kwarg and node.args.kwarg.arg not in ParamRecallEvaluator._IGNORED_PARAMS:
                    params.add(node.args.kwarg.arg)
                sigs[node.name] = params
        return sigs

    @staticmethod
    def _extract_func_signatures_regex(code: str) -> dict[str, set[str]]:
        code = re.sub(r'"""[\s\S]*?"""', '', code)
        code = re.sub(r"'''[\s\S]*?'''", '', code)
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        sigs = {}
        for m in re.finditer(r"def\s+(\w+)\s*\(([^)]*)\)", code):
            params = {a for a in re.findall(r"[\*]*(\w+)", m.group(2)) if a not in ParamRecallEvaluator._IGNORED_PARAMS}
            sigs[m.group(1)] = params
        return sigs

    _WRITE_TOOLS = {"write", "edit", "create", "create_file", "write_file"}
    _WRITE_CONTENT_KEYS = ["content", "file_text", "text", "code"]

    @staticmethod
    def _get_target_indices(messages: list[dict], is_main: bool) -> list[int]:
        """找出包含 write/edit 工具调用的 assistant 消息。"""
        indices = []
        for mi, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                if fn.get("name", "") in ParamRecallEvaluator._WRITE_TOOLS:
                    indices.append(mi)
                    break
        return indices

    @staticmethod
    def _extract_from_write_args(arguments: str, tool_name: str) -> str:
        """从 write/edit 工具的 arguments 中提取文件内容。"""
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(args, dict):
            return ""
        for key in ParamRecallEvaluator._WRITE_CONTENT_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""
