from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .config import AgentConfig


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    wall_ms: int
    model_reported: str = ""
    error: str | None = None
    tool_calls: list = None
    finish_reason: str = "stop"


class LLMProvider(Protocol):
    def complete(self, system: str, user: str, cfg: AgentConfig) -> LLMResponse: ...
    def complete_with_tools(
        self, messages: list, tools: list, cfg: AgentConfig
    ) -> LLMResponse: ...


class InfrastructureError(Exception):
    pass


class ProviderConfigError(Exception):
    pass


_OPENAI_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o":         (2.50, 10.00),
    "gpt-4o-mini":    (0.15,  0.60),
    "gpt-4.1":        (2.00,  8.00),
    "gpt-4.1-mini":   (0.40,  1.60),
    "gpt-5":          (5.00, 15.00),
    "gpt-5-mini":     (0.50,  2.00),
    "o1":             (15.00, 60.00),
    "o1-mini":        (1.10,  4.40),
    "o3":             (10.00, 40.00),
    "o3-mini":        (1.10,  4.40),
}


def openai_cost_usd(model: str, tokens_in: int, tokens_out: int) -> float | None:
    key = model.split(":")[0]
    if key not in _OPENAI_PRICING:
        for k, v in _OPENAI_PRICING.items():
            if model.startswith(k):
                pin, pout = v
                return (tokens_in / 1_000_000) * pin + (tokens_out / 1_000_000) * pout
        return None
    pin, pout = _OPENAI_PRICING[key]
    return (tokens_in / 1_000_000) * pin + (tokens_out / 1_000_000) * pout


class OpenAIProvider:
    def __init__(self, api_key: str | None = None, timeout_s: float = 60.0):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ProviderConfigError("openai package not installed — `uv sync`") from e
        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ProviderConfigError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=key, timeout=timeout_s)

    def complete(self, system: str, user: str, cfg: AgentConfig) -> LLMResponse:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_output_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                **cfg.provider_kwargs,
            )
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            raise InfrastructureError(f"{type(e).__name__}: {e}") from e
        except Exception as e:
            raise ProviderConfigError(f"{type(e).__name__}: {e}") from e

        wall_ms = int((time.perf_counter() - t0) * 1000)
        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = getattr(resp, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0) if usage else 0
        tokens_out = getattr(usage, "completion_tokens", 0) if usage else 0
        return LLMResponse(
            text=text,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            wall_ms=wall_ms,
            model_reported=getattr(resp, "model", cfg.model),
        )

    def complete_with_tools(
        self, messages: list, tools: list, cfg: AgentConfig
    ) -> LLMResponse:
        from openai import APIConnectionError, APITimeoutError, RateLimitError
        import json as _json
        t0 = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=cfg.model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_output_tokens,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                **cfg.provider_kwargs,
            )
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            raise InfrastructureError(f"{type(e).__name__}: {e}") from e
        except Exception as e:
            raise ProviderConfigError(f"{type(e).__name__}: {e}") from e

        wall_ms = int((time.perf_counter() - t0) * 1000)
        choice = resp.choices[0]
        msg = choice.message
        text = msg.content or ""
        raw_calls = getattr(msg, "tool_calls", None) or []
        parsed_calls = []
        for tc in raw_calls:
            fn = getattr(tc, "function", None)
            if fn is None:
                continue
            try:
                args = _json.loads(fn.arguments) if fn.arguments else {}
            except Exception:
                args = {"_raw": fn.arguments}
            parsed_calls.append({"id": tc.id, "name": fn.name, "arguments": args})
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=text,
            tokens_in=getattr(usage, "prompt_tokens", 0) if usage else 0,
            tokens_out=getattr(usage, "completion_tokens", 0) if usage else 0,
            wall_ms=wall_ms,
            model_reported=getattr(resp, "model", cfg.model),
            tool_calls=parsed_calls if parsed_calls else None,
            finish_reason=choice.finish_reason or "stop",
        )


class MockProvider:
    def __init__(
        self,
        responder: Callable[[str, str, AgentConfig], str] | None = None,
        latency_ms: int = 5,
        seed: int = 0,
    ):
        self._responder = responder or (lambda s, u, c: "FINAL PATH:")
        self._latency_ms = latency_ms
        self._rng = random.Random(seed)

    def complete(self, system: str, user: str, cfg: AgentConfig) -> LLMResponse:
        time.sleep(self._latency_ms / 1000.0)
        text = self._responder(system, user, cfg)
        return LLMResponse(
            text=text,
            tokens_in=len(system) // 4 + len(user) // 4,
            tokens_out=len(text) // 4,
            wall_ms=self._latency_ms + self._rng.randint(0, 3),
            model_reported=f"mock:{cfg.model}",
        )

    def complete_with_tools(
        self, messages: list, tools: list, cfg: AgentConfig
    ) -> LLMResponse:
        import json as _json, uuid as _uuid
        time.sleep(self._latency_ms / 1000.0)
        last = messages[-1] if messages else {"role": "user", "content": ""}
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_text = last.get("content") or ""
        raw = self._responder(system_msg, user_text, cfg)

        parsed_calls = None
        finish = "stop"
        if isinstance(raw, dict) and "tool_calls" in raw:
            parsed_calls = [
                {"id": f"call_{_uuid.uuid4().hex[:8]}", "name": tc["name"], "arguments": tc.get("arguments", {})}
                for tc in raw["tool_calls"]
            ]
            text = ""
            finish = "tool_calls"
        else:
            text = raw if isinstance(raw, str) else _json.dumps(raw)

        return LLMResponse(
            text=text,
            tokens_in=sum(len(m.get("content") or "") for m in messages) // 4,
            tokens_out=len(text) // 4 if text else 0,
            wall_ms=self._latency_ms + self._rng.randint(0, 3),
            model_reported=f"mock:{cfg.model}",
            tool_calls=parsed_calls,
            finish_reason=finish,
        )
