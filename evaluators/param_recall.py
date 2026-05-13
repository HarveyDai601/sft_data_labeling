"""参数准召率评估器 — AST 初筛 + LLM 终判。

流程：
1. AST 从 reference 函数定义中提取参数名
2. AST 从 trace 函数定义中提取参数名
3. 差集 = 疑点（可能缺失/不同的参数）
4. LLM 对疑点做语义判断：参数名不同但语义等价？

无 LLM 时退化为纯 AST 规则。
"""

from __future__ import annotations

import ast
import logging
import re
from typing import Any

from core.models import EvalResult
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
        # ── Step 1: AST 提取 reference 函数签名 ───────────────────────────
        ref_sigs = self._extract_func_signatures(reference)
        if not ref_sigs:
            logger.info(f"[{self.name}] reference 中未提取到函数签名，跳过")
            return []

        is_main = context.get("trace_type") == "main"
        target_indices = self._get_target_indices(messages, is_main)
        if not target_indices:
            logger.info(f"[{self.name}] 无目标 assistant message，跳过")
            return []

        # ── Step 2: AST 提取 trace 函数签名 ───────────────────────────────
        trace_code = "\n".join(
            code
            for mi in target_indices
            for code in self._extract_code_blocks(messages[mi])
        )
        trace_sigs = self._extract_func_signatures(trace_code)

        logger.info(
            f"[{self.name}] reference 签名: { {k: sorted(v) for k, v in ref_sigs.items()} } | "
            f"trace 签名: { {k: sorted(v) for k, v in trace_sigs.items()} }"
        )

        # ── Step 3: 按函数配对，找参数差异 ────────────────────────────────
        suspects: list[dict[str, Any]] = []
        for func_name, ref_params in ref_sigs.items():
            if func_name not in trace_sigs:
                logger.info(f"[{self.name}] 函数 {func_name} 在 trace 中未定义（由 function_recall 处理）")
                continue
            trace_params = trace_sigs[func_name]
            missing_params = ref_params - trace_params
            if missing_params:
                suspects.append({
                    "function": func_name,
                    "ref_params": sorted(ref_params),
                    "trace_params": sorted(trace_params),
                    "missing": sorted(missing_params),
                })

        if not suspects:
            logger.info(f"[{self.name}] 所有函数参数匹配，无差异")
            return []

        logger.info(
            f"[{self.name}] 疑点（AST 差集）: "
            + "; ".join(f"{s['function']} 缺失 {s['missing']}" for s in suspects)
        )

        # ── Step 4: LLM 终判 ─────────────────────────────────────────────
        if self.llm:
            before_count = sum(len(s["missing"]) for s in suspects)
            suspects = self._llm_verify(trace_code, suspects)
            after_count = sum(len(s["missing"]) for s in suspects) if suspects else 0
            if before_count > after_count:
                logger.info(f"[{self.name}] LLM 过滤掉 {before_count - after_count} 个语义等价的参数")
            if not suspects:
                logger.info(f"[{self.name}] LLM 判断全部为语义等价，无真正缺失")
                return []

        # ── 生成结果 ─────────────────────────────────────────────────────
        all_missing = [s["missing"] for s in suspects]
        flat_missing = [p for group in all_missing for p in group]
        total_ref_params = sum(len(params) for params in ref_sigs.values())
        missing_ratio = len(flat_missing) / max(total_ref_params, 1)
        severity = min(1.0, missing_ratio * 1.5)

        details = "; ".join(
            f"{s['function']}({', '.join(s['missing'])})"
            for s in suspects
        )

        logger.info(
            f"[{self.name}] 最终结果: {details}, "
            f"missing_ratio={missing_ratio:.2f}, severity={severity:.2f}"
        )

        return [
            EvalResult(
                message_index=target_indices[-1],
                dimension=self.name,
                verdict="missing",
                reason=f"参数名不匹配: {details}",
                severity=round(severity, 2),
                suggested_fix={"suspects": suspects},
            )
        ]

    # ── LLM 终判 ──────────────────────────────────────────────────────────

    def _llm_verify(
        self,
        trace_code: str,
        suspects: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        suspect_desc = "\n".join(
            f"- {s['function']}: reference 参数 {s['ref_params']}, "
            f"trace 参数 {s['trace_params']}, 缺失 {s['missing']}"
            for s in suspects
        )

        prompt = f"""你是一个代码审查专家。以下是 agent 产出的代码和参数差异分析。

Agent 代码:
```python
{trace_code}
```

参数差异:
{suspect_desc}

请判断：每个函数的参数名差异是否是**真正的缺失**？

判断标准：
- 如果 trace 的参数名与 reference 不同但语义等价（如 data/input_data, config/cfg）→ 不是缺失
- 如果 trace 确实漏掉了功能上必须的参数 → 是缺失
- 如果 trace 的实现方式不同，不需要该参数 → 不是缺失

请用 JSON 回答:
{{"results": [{{"function": "函数名", "real_missing": ["真正缺失的参数"], "equivalent": ["语义等价的参数"]}}]}}"""

        try:
            resp = self.llm.chat_json(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            confirmed: list[dict[str, Any]] = []
            for r in resp.get("results", []):
                real_missing = set(r.get("real_missing", []))
                equiv = r.get("equivalent", [])
                if equiv:
                    logger.info(f"[{self.name}] LLM 判断 {r['function']} 中 {equiv} 为语义等价")
                if real_missing:
                    for s in suspects:
                        if s["function"] == r["function"]:
                            confirmed.append({
                                "function": s["function"],
                                "ref_params": s["ref_params"],
                                "trace_params": s["trace_params"],
                                "missing": sorted(real_missing),
                            })
            return confirmed
        except Exception as e:
            logger.warning(f"[{self.name}] LLM 验证失败，退化为 AST 规则: {e}")
            return suspects

    # ── AST 解析 ──────────────────────────────────────────────────────────

    @staticmethod
    def _extract_func_signatures(code: str) -> dict[str, set[str]]:
        sigs: dict[str, set[str]] = {}
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return ParamRecallEvaluator._extract_func_signatures_regex(code)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                params: set[str] = set()
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

        sigs: dict[str, set[str]] = {}
        for m in re.finditer(r"def\s+(\w+)\s*\(([^)]*)\)", code):
            func_name = m.group(1)
            sig = m.group(2)
            params = {
                arg for arg in re.findall(r"[\*]*(\w+)", sig)
                if arg not in ParamRecallEvaluator._IGNORED_PARAMS
            }
            sigs[func_name] = params
        return sigs

    # ── 工具方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _get_target_indices(messages: list[dict], is_main: bool) -> list[int]:
        if is_main:
            return [
                mi for mi, msg in enumerate(messages)
                if msg.get("role") == "assistant" and ParamRecallEvaluator._has_code_block(msg)
            ]
        return [
            mi for mi, msg in enumerate(messages)
            if msg.get("role") == "assistant"
        ]

    @staticmethod
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

    @staticmethod
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
                        for m in re.finditer(r"```(?:\w*\n)?(.*?)```", item.get("text", ""), re.DOTALL):
                            blocks.append(m.group(1).strip())
        return blocks
