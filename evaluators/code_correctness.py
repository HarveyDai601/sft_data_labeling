"""代码正确性评估器 — 函数级语义比对。

逻辑：
1. AST 提取 reference 的函数/类定义 → 期望定义集合
2. 从 write/edit 工具 arguments 提取 agent 实际写出的代码
3. AST 提取 trace 代码的函数定义+调用 → 实际 API 集合
4. reference 定义但 trace 未定义 → LLM 判断功能等价
5. 函数签名参数：逐函数比对参数名语义等价

只产出观测结果（KEEP），不自动修改代码。
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

_WRITE_TOOLS = {"write", "edit", "create", "create_file", "write_file"}
_WRITE_CONTENT_KEYS = ["content", "file_text", "text", "code"]
_IGNORED_PARAMS = {"self", "cls", "args", "kwargs"}


class CodeCorrectnessEvaluator(BaseEvaluator):
    name = "code_correctness"

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def evaluate(
        self,
        messages: list[dict[str, Any]],
        reference: str,
        context: dict[str, Any],
    ) -> list[EvalResult]:
        if not reference.strip():
            return []

        # 1. 提取 reference 的期望定义
        ref_defs = self._extract_defs(reference)
        if not ref_defs:
            logger.info(f"[{self.name}] reference 中未提取到函数定义，跳过")
            return []

        # 2. 提取 reference 的函数签名（参数检查用）
        ref_sigs = self._extract_sigs(reference)

        # 3. 收集所有 write 工具写出的代码
        trace_blocks: list[tuple[int, str]] = []  # (msg_index, code)
        for mi, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                if fn.get("name", "") not in _WRITE_TOOLS:
                    continue
                code = self._extract_from_write_args(fn.get("arguments", ""))
                if code:
                    trace_blocks.append((mi, code))

        if not trace_blocks:
            logger.info(f"[{self.name}] 未找到 write 调用，跳过")
            return []

        trace_code = "\n\n".join(code for _, code in trace_blocks)

        # 4. 提取 trace 代码的 API
        trace_defs = self._extract_defs(trace_code)
        trace_apis = self._extract_apis(trace_code)

        logger.info(
            f"[{self.name}] ref 定义: {sorted(ref_defs)} | "
            f"trace 定义: {sorted(trace_defs)} | "
            f"trace API: {sorted(trace_apis)[:20]}"
        )

        # 5. 缺失的函数定义
        missing_defs = ref_defs - trace_defs
        if missing_defs:
            logger.info(f"[{self.name}] AST 缺失定义: {sorted(missing_defs)}")
            if self.llm:
                missing_defs = self._llm_verify_defs(
                    reference, trace_code, missing_defs, trace_apis
                )
                if not missing_defs:
                    logger.info(f"[{self.name}] LLM 判断均为等价实现")

        # 6. 缺失的参数
        param_issues: list[dict[str, Any]] = []
        if ref_sigs:
            trace_sigs = self._extract_sigs(trace_code)
            for func_name, ref_params in ref_sigs.items():
                if func_name not in trace_sigs:
                    continue
                trace_params = trace_sigs[func_name]
                missing_params = ref_params - trace_params
                if missing_params:
                    param_issues.append({
                        "function": func_name,
                        "ref_params": sorted(ref_params),
                        "trace_params": sorted(trace_params),
                        "missing": sorted(missing_params),
                    })

        if param_issues and self.llm:
            param_issues = self._llm_verify_params(trace_code, param_issues)

        # 7. 产出结果
        results: list[EvalResult] = []
        target_mi = trace_blocks[-1][0] if trace_blocks else 0

        if missing_defs:
            defs_ratio = len(missing_defs) / max(len(ref_defs), 1)
            confidence = round(min(1.0, defs_ratio * 1.5), 2)
            results.append(EvalResult(
                message_index=target_mi,
                dimension=self.name,
                finding=f"缺失函数: {', '.join(sorted(missing_defs))}",
                confidence=confidence,
                action=CleaningAction.KEEP,
                action_reason="无法自动生成缺失函数实现",
                debug_info={
                    "ref_defs": sorted(ref_defs),
                    "trace_defs": sorted(trace_defs),
                    "trace_apis": sorted(trace_apis)[:30],
                    "missing_defs": sorted(missing_defs),
                    "trace_blocks_count": len(trace_blocks),
                },
            ))

        if param_issues:
            all_missing = [p for s in param_issues for p in s["missing"]]
            total_ref_params = sum(len(v) for v in ref_sigs.values())
            confidence = round(min(1.0, len(all_missing) / max(total_ref_params, 1) * 1.5), 2)
            details = "; ".join(
                f"{s['function']}({', '.join(s['missing'])})" for s in param_issues
            )
            results.append(EvalResult(
                message_index=target_mi,
                dimension=self.name,
                finding=f"参数不匹配: {details}",
                confidence=confidence,
                action=CleaningAction.KEEP,
                action_reason="无法自动修正参数名",
                debug_info={
                    "param_issues": param_issues,
                    "ref_sigs": {k: sorted(v) for k, v in ref_sigs.items()},
                },
            ))

        if not results:
            logger.info(f"[{self.name}] 代码正确性检查通过")

        return results

    # ── LLM 验证 ──────────────────────────────────────────────────────────

    def _llm_verify_defs(
        self,
        reference: str,
        trace_code: str,
        missing_defs: set[str],
        trace_apis: set[str],
    ) -> set[str]:
        prompt = f"""判断以下缺失的函数定义是否是真的缺失。

Reference:
```python
{reference[:2000]}
```

Agent 代码:
```python
{trace_code[:2000]}
```

AST 发现 reference 定义了但 agent 未定义的函数: {', '.join(sorted(missing_defs))}
Agent 代码中出现了: {', '.join(sorted(trace_apis)[:30])}

逐个判断：
- 如果 agent 用了不同的函数名实现了相同逻辑 → 等价实现，不缺失
- 如果 agent 把逻辑内联到其他函数/类里 → 等价实现，不缺失
- 如果 reference 中的函数是核心业务逻辑且 agent 完全没实现 → 真正缺失
- 如果 reference 中的函数只被使用不要求实现 → 不缺失

JSON: {{"confirmed": ["真正缺失的"], "rejected": ["等价的"], "reasons": {{"func": "原因"}}}}"""

        try:
            resp = self.llm.chat_json([{"role": "user", "content": prompt}], temperature=0.0)
            confirmed = set(resp.get("confirmed", []))
            if resp.get("reasons"):
                for func, reason in resp["reasons"].items():
                    logger.info(f"[{self.name}] LLM: {func} → {reason}")
            return confirmed
        except Exception as e:
            logger.warning(f"[{self.name}] LLM 验证失败: {e}")
            return missing_defs

    def _llm_verify_params(
        self, trace_code: str, issues: list[dict],
    ) -> list[dict]:
        desc = "\n".join(
            f"- {s['function']}: ref {s['ref_params']}, trace {s['trace_params']}, 缺失 {s['missing']}"
            for s in issues
        )
        prompt = f"""判断参数差异是否真正缺失。

Agent 代码:
```python
{trace_code[:1500]}
```

参数差异:
{desc}

标准：参数名不同但语义等价（data/input_data, config/configuration）→ 不缺失。

JSON: {{"results": [{{"function": "...", "real_missing": ["..."], "equivalent": ["..."]}}]}}"""

        try:
            resp = self.llm.chat_json([{"role": "user", "content": prompt}], temperature=0.0)
            confirmed = []
            for r in resp.get("results", []):
                equiv = r.get("equivalent", [])
                if equiv:
                    logger.info(f"[{self.name}] LLM: {r['function']} {equiv} 语义等价")
                real = set(r.get("real_missing", []))
                if real:
                    for s in issues:
                        if s["function"] == r["function"]:
                            confirmed.append({**s, "missing": sorted(real)})
            return confirmed
        except Exception as e:
            logger.warning(f"[{self.name}] LLM 参数验证失败: {e}")
            return issues

    # ── AST 解析 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_defs(code: str) -> set[str]:
        """提取函数/类定义（不包括调用，避免内置函数噪声）。"""
        defs: set[str] = set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set(re.findall(r"\bdef\s+(\w+)\s*\(", code))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defs.add(node.name)
        return defs

    @staticmethod
    def _extract_apis(code: str) -> set[str]:
        """提取函数定义+调用。"""
        apis: set[str] = set()
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set(re.findall(r"\bdef\s+(\w+)\s*\(", code))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                apis.add(node.name)
            elif isinstance(node, ast.Call):
                name = CodeCorrectnessEvaluator._get_call_name(node.func)
                if name:
                    apis.add(name)
        return apis

    @staticmethod
    def _extract_sigs(code: str) -> dict[str, set[str]]:
        """提取函数签名（参数名）。"""
        sigs: dict[str, set[str]] = {}
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return sigs
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params = set()
                for arg in node.args.args:
                    if arg.arg not in _IGNORED_PARAMS:
                        params.add(arg.arg)
                for arg in node.args.kwonlyargs:
                    if arg.arg not in _IGNORED_PARAMS:
                        params.add(arg.arg)
                if node.args.vararg and node.args.vararg.arg not in _IGNORED_PARAMS:
                    params.add(node.args.vararg.arg)
                sigs[node.name] = params
        return sigs

    @staticmethod
    def _get_call_name(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    # ── 工具方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_from_write_args(arguments: str) -> str:
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            return ""
        if not isinstance(args, dict):
            return ""
        for key in _WRITE_CONTENT_KEYS:
            val = args.get(key)
            if isinstance(val, str) and val.strip():
                return val
        return ""
