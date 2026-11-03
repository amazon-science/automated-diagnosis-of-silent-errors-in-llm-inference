"""Chat-model factory supporting Amazon Bedrock and the Anthropic API.

Select the provider with the ``provider`` key in a diagnosis config (``bedrock`` or
``anthropic``), or override it for every run with ``EKKA_LLM_PROVIDER``. The model
can likewise be overridden with ``EKKA_MODEL_ID``.

Bedrock model IDs (for example ``global.anthropic.claude-sonnet-4-5-20250929-v1:0``)
are translated to Anthropic API model names (``claude-sonnet-4-5-20250929``)
automatically when the provider is ``anthropic``, so the same config works with
either backend.

Credentials:
* bedrock: the standard AWS credential chain (``AWS_PROFILE``, ``AWS_ACCESS_KEY_ID`` /
  ``AWS_SECRET_ACCESS_KEY``, an instance role, ...) and ``AWS_REGION`` (or
  ``AWS_DEFAULT_REGION``). Requires ``langchain-aws``.
* anthropic: ``ANTHROPIC_API_KEY``. Requires ``langchain-anthropic``.

Newer Claude models (Sonnet 5 and later) reject requests whose last message is an
assistant turn ("assistant message prefill"). LangGraph's ``create_react_agent`` sends
exactly that when it asks for the final ``response_format`` output, so the models built
here append a short user turn whenever a request would otherwise end on an assistant
message.

Bedrock calls use a long read timeout (``EKKA_BEDROCK_READ_TIMEOUT`` seconds, default
1800) because large ``max_tokens`` responses are returned in one piece and easily exceed
botocore's 60 s default.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional

from .paths import setup_env

DEFAULT_MODEL_ID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
DEFAULT_MODEL_KWARGS: Dict[str, Any] = {"max_tokens": 64000}
SUPPORTED_PROVIDERS = ("bedrock", "anthropic")

_FINAL_ANSWER_PROMPT = "Provide your final answer in the requested output format."

_BEDROCK_ID_RE = re.compile(r"^(?:(?:global|us|eu|apac|jp|au)\.)?anthropic\.(?P<name>.+?)(?:-v\d+(?::\d+)?)?$")


def to_anthropic_model_name(model_id: str) -> str:
    """Convert a Bedrock Claude model ID to the Anthropic API model name (no-op otherwise)."""
    m = _BEDROCK_ID_RE.match(model_id)
    return m.group("name") if m else model_id


def _end_with_user_turn(messages):
    """Append a user message if the conversation ends on an assistant turn (no prefill)."""
    from langchain_core.messages import AIMessage, HumanMessage

    if messages and isinstance(messages[-1], AIMessage):
        return [*messages, HumanMessage(content=_FINAL_ANSWER_PROMPT)]
    return messages


def _no_prefill(cls):
    """Subclass a LangChain chat model so every request ends with a user message."""

    class NoPrefill(cls):  # type: ignore[misc, valid-type]
        def _generate(self, messages, *args, **kwargs):
            return super()._generate(_end_with_user_turn(messages), *args, **kwargs)

        def _stream(self, messages, *args, **kwargs):
            return super()._stream(_end_with_user_turn(messages), *args, **kwargs)

        async def _agenerate(self, messages, *args, **kwargs):
            return await super()._agenerate(_end_with_user_turn(messages), *args, **kwargs)

        def _astream(self, messages, *args, **kwargs):
            return super()._astream(_end_with_user_turn(messages), *args, **kwargs)

    NoPrefill.__name__ = cls.__name__
    NoPrefill.__qualname__ = cls.__qualname__
    return NoPrefill


def make_chat_model(
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    region: Optional[str] = None,
):
    """Build a LangChain chat model for the requested provider."""
    setup_env()  # pick up EKKA_* / AWS_* overrides from .env when called outside load_config
    provider = (os.environ.get("EKKA_LLM_PROVIDER") or provider or "bedrock").lower()
    model_id = os.environ.get("EKKA_MODEL_ID") or model_id or DEFAULT_MODEL_ID
    model_kwargs = dict(model_kwargs if model_kwargs is not None else DEFAULT_MODEL_KWARGS)

    if provider == "bedrock":
        from botocore.config import Config
        from langchain_aws import ChatBedrock

        region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        boto_config = Config(
            read_timeout=int(os.environ.get("EKKA_BEDROCK_READ_TIMEOUT", "1800")),
            retries={"max_attempts": 5, "mode": "adaptive"},
        )
        kwargs: Dict[str, Any] = {"model_id": model_id, "model_kwargs": model_kwargs, "config": boto_config}
        if region:
            kwargs["region_name"] = region
        return _no_prefill(ChatBedrock)(**kwargs)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        max_tokens = model_kwargs.pop("max_tokens", 64000)
        # Large max_tokens values require streaming with the Anthropic SDK.
        return _no_prefill(ChatAnthropic)(
            model=to_anthropic_model_name(model_id),
            max_tokens=max_tokens,
            streaming=True,
            **model_kwargs,
        )

    raise ValueError(f"Unsupported LLM provider '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}")
