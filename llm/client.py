"""LLM 客户端 — 兼容任何 OpenAI 兼容 API（公司内网、vLLM、Ollama 等）。"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)


class LLMClient:
    """薄封装：只做 chat completions，支持自定义 base_url。"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "sk-none",
        model: str = "gpt-4",
        temperature: float = 0.1,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )

    # ── 核心调用 ──────────────────────────────────────────────────────────

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        response_format: dict | None = None,
    ) -> str:
        """发送 chat completion 请求，返回纯文本。"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if response_format:
            kwargs["response_format"] = response_format
        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """发送请求并解析 JSON 响应。"""
        raw = self.chat(
            messages,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return json.loads(raw)
