"""A thinking provider needs ``reasoning_content`` on *every* tool turn.

DeepSeek (and the other thinking dialects) reject the whole request with
``The `reasoning_content` in the thinking mode must be passed back to the
API`` as soon as one assistant turn carrying ``tool_calls`` arrives without
that field — even when that particular round produced no reasoning of its
own, which the model decides per round. Multi-round exploration hits this on
the second or third tool call, long after the first round's reasoning proved
the conversation is in thinking mode.

The repair is symmetric: the field is restored only once the same request
already carries reasoning from another turn, so a provider that never speaks
the dialect receives no new key.
"""

from __future__ import annotations

import asyncio
from typing import Any

from deeptutor.runtime.agentic.client import _sanitize_chat_client
from deeptutor.services.llm.provider_core.openai_compat_provider import (
    OpenAICompatProvider,
)

_TOOL_CALLS = [
    {
        "id": "call_000abcdefg",
        "type": "function",
        "function": {"name": "rag", "arguments": '{"query": "customer value"}'},
    }
]


def _history(*, thought_turn: bool, silent_tool_turn: bool) -> list[dict[str, Any]]:
    thought = {"role": "assistant", "content": "TOOL 先检索。", "tool_calls": _TOOL_CALLS}
    if thought_turn:
        thought["reasoning_content"] = "先想清楚要查哪些维度。"
    silent = {"role": "assistant", "content": "TOOL 再查一次。", "tool_calls": _TOOL_CALLS}
    if silent_tool_turn:
        silent["reasoning_content"] = "第二轮也想了。"
    return [
        {"role": "system", "content": "explore"},
        {"role": "user", "content": "出三道题"},
        thought,
        {"role": "tool", "tool_call_id": "call_000abcdefg", "content": "检索结果"},
        silent,
        {"role": "tool", "tool_call_id": "call_000abcdefg", "content": "检索结果"},
        {"role": "user", "content": "继续"},
    ]


def test_missing_tool_turn_reasoning_is_restored_as_empty_string() -> None:
    sanitized = OpenAICompatProvider._sanitize_messages(
        _history(thought_turn=True, silent_tool_turn=False)
    )

    assistants = [m for m in sanitized if m["role"] == "assistant"]
    assert assistants[0]["reasoning_content"] == "先想清楚要查哪些维度。"
    assert assistants[1]["reasoning_content"] == ""


def test_no_reasoning_anywhere_adds_no_new_key() -> None:
    """A non-thinking provider must not see the field appear out of nowhere."""
    sanitized = OpenAICompatProvider._sanitize_messages(
        _history(thought_turn=False, silent_tool_turn=False)
    )

    for message in sanitized:
        assert "reasoning_content" not in message


def test_plain_text_turns_are_left_alone() -> None:
    """Only tool-calling turns are checked by the provider."""
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "普通回答", "reasoning_content": "想了一下"},
        {"role": "assistant", "content": "没有推理的普通回答"},
    ]

    sanitized = OpenAICompatProvider._sanitize_messages(messages)

    assert sanitized[2]["content"] == "没有推理的普通回答"
    assert "reasoning_content" not in sanitized[2]


def test_responses_api_keeps_its_own_replay_format() -> None:
    """The Responses dialect replays native output items, not this field."""
    messages = [
        {"role": "assistant", "content": "TOOL 先检索。", "reasoning_content": "想了"},
        {"role": "assistant", "content": "TOOL 再查。", "tool_calls": _TOOL_CALLS},
    ]

    sanitized = OpenAICompatProvider._sanitize_messages(messages, responses_api=True)

    assert "reasoning_content" not in sanitized[1]


def test_sdk_bridge_projects_the_same_repair() -> None:
    """Agent loops call the SDK directly; the wrapper must repair too."""
    sent: dict[str, Any] = {}

    class _Completions:
        async def create(self, **kwargs: Any) -> Any:
            sent.update(kwargs)
            return "ok"

    client = type(
        "Client",
        (),
        {"chat": type("Chat", (), {"completions": _Completions()})()},
    )()

    asyncio.run(
        _sanitize_chat_client(client).chat.completions.create(
            model="deepseek-flash",
            messages=_history(thought_turn=True, silent_tool_turn=False),
        )
    )

    assistants = [m for m in sent["messages"] if m["role"] == "assistant"]
    assert assistants[1]["reasoning_content"] == ""
