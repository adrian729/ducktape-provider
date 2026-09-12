"""Adapter for OpenAI's Responses API."""

import http.client
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from ..adapter import Adapter
from ..streaming import _iter_sse
from ..types import Block, Message, Response, StopReason, StreamEvent, ToolDef


class OpenAIAdapter(Adapter):
    _MODELS_URL = "https://api.openai.com/v1/models"
    _RESPONSES_URL = "https://api.openai.com/v1/responses"
    _CHAT_MODEL_RE = re.compile(r"^(gpt-|chatgpt-|o\d)")
    _NON_CHAT_RE = re.compile(r"-(audio|realtime|transcribe|tts|search)|^gpt-image")
    _MODELS_TTL = 60
    _CHAT_TIMEOUT = 120

    def __init__(self):
        self._models_cache: set[str] | None = None
        self._cache_time = 0.0

    def is_available(self) -> bool:
        return bool(os.environ.get("OPENAI_API_KEY"))

    def models(self) -> set[str]:
        now = time.monotonic()
        if self._models_cache is not None and now - self._cache_time < self._MODELS_TTL:
            return set(self._models_cache)
        req = urllib.request.Request(
            self._MODELS_URL,
            headers={"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.load(resp)
            model_ids = {
                m["id"]
                for m in data.get("data", [])
                if self._CHAT_MODEL_RE.match(m["id"])
                and not self._NON_CHAT_RE.search(m["id"])
            }
        except (
            OSError,
            ValueError,
            AttributeError,
            KeyError,
            TypeError,
            http.client.HTTPException,
        ):
            return set()
        self._models_cache = model_ids
        self._cache_time = now
        return set(model_ids)

    def _serialize(self, messages: list[Message]) -> list[dict[str, Any]]:
        """The Responses API's "input" is a flat, order-significant list of items, not messages
        grouping content blocks — tool_use/tool_result blocks become their own top-level
        function_call/function_call_output items, interleaved in the order blocks appear."""
        serialized: list[dict[str, Any]] = []
        for message in messages:
            content: list[dict[str, Any]] = []

            def flush(role: str = message["role"], buf: list[dict[str, Any]] = content) -> None:
                if buf:
                    serialized.append({"type": "message", "role": role, "content": list(buf)})
                    buf.clear()

            for block in message["content"]:
                if block["type"] == "text":
                    content.append({"type": "input_text", "text": block["text"]})
                elif block["type"] == "image":
                    content.append(
                        {
                            "type": "input_image",
                            "image_url": f"data:{block['media_type']};base64,{block['data']}",
                        }
                    )
                elif block["type"] == "tool_use":
                    flush()
                    serialized.append(
                        {
                            "type": "function_call",
                            "call_id": block["id"],
                            "name": block["name"],
                            "arguments": json.dumps(block["input"]),
                        }
                    )
                elif block["type"] == "tool_result":
                    flush()
                    output = block["content"]
                    if block.get("is_error"):
                        output = f"ERROR: {output}"
                    serialized.append(
                        {
                            "type": "function_call_output",
                            "call_id": block["tool_use_id"],
                            "output": output,
                        }
                    )
            flush()
        return serialized

    def _serialize_tools(self, tools: list[ToolDef]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            }
            for t in tools
        ]

    def _deserialize(self, data: dict[str, Any]) -> Response:
        blocks: list[Block] = []
        has_refusal = False
        for item in data.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "output_text":
                        blocks.append({"type": "text", "text": part["text"]})
                    elif part.get("type") == "refusal":
                        blocks.append({"type": "text", "text": part["refusal"]})
                        has_refusal = True
            elif item.get("type") == "function_call":
                try:
                    args = json.loads(item["arguments"])
                except json.JSONDecodeError:
                    args = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": item["call_id"],
                        "name": item["name"],
                        "input": args,
                    }
                )

        has_tool_use = any(b["type"] == "tool_use" for b in blocks)
        status = data.get("status", "")
        incomplete_reason = (data.get("incomplete_details") or {}).get("reason", "")
        if incomplete_reason == "max_output_tokens":
            stop_reason: StopReason = "max_tokens"
        elif incomplete_reason == "content_filter":
            stop_reason = "content_filter"
        elif has_refusal:
            stop_reason = "refusal"
        elif has_tool_use:
            stop_reason = "tool_use"
        elif status == "completed":
            stop_reason = "end_turn"
        else:
            stop_reason = "other"

        usage = data.get("usage") or {}
        return {
            "content": blocks,
            "stop_reason": stop_reason,
            "raw_stop_reason": incomplete_reason or status,
            "usage": {
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
            "raw": data,
        }

    def _build_request(
        self,
        model: str,
        messages: list[Message],
        system: str | None,
        tools: list[ToolDef] | None,
        config: dict[str, Any] | None,
        stream: bool,
    ) -> tuple[urllib.request.Request, float]:
        payload: dict[str, Any] = {
            "model": model,
            "input": self._serialize(messages),
            "store": False,
            "stream": stream,
        }
        if system:
            payload["instructions"] = system
        if tools:
            payload["tools"] = self._serialize_tools(tools)
        payload.update(config or {})
        timeout = payload.pop("timeout", self._CHAT_TIMEOUT)
        req = urllib.request.Request(
            self._RESPONSES_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
                "Content-Type": "application/json",
            },
        )
        return req, timeout

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response:
        req, timeout = self._build_request(
            model, messages, system, tools, config, stream=False
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(f"openai chat failed: {e.code} {body}") from e
        if data.get("status") == "failed":
            raise RuntimeError(f"openai chat failed: {data.get('error')}")
        return self._deserialize(data)

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        req, timeout = self._build_request(
            model, messages, system, tools, config, stream=True
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                yield from self._stream_events(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            raise RuntimeError(f"openai chat failed: {e.code} {body}") from e

    def _stream_events(self, resp: http.client.HTTPResponse) -> Iterator[StreamEvent]:
        block_index_by_item: dict[str, int] = {}
        next_index = 0

        def block_index(item_id: str) -> int:
            nonlocal next_index
            if item_id not in block_index_by_item:
                block_index_by_item[item_id] = next_index
                next_index += 1
            return block_index_by_item[item_id]

        for event in _iter_sse(resp):
            etype = event.get("type")
            if etype == "response.output_item.added":
                item = event["item"]
                if item.get("type") == "function_call":
                    yield {
                        "type": "tool_use_start",
                        "index": block_index(item["id"]),
                        "id": item.get("call_id", ""),
                        "name": item.get("name", ""),
                    }
            elif etype == "response.output_text.delta":
                yield {
                    "type": "text_delta",
                    "index": block_index(event["item_id"]),
                    "text": event["delta"],
                }
            elif etype == "response.function_call_arguments.delta":
                yield {
                    "type": "tool_use_delta",
                    "index": block_index(event["item_id"]),
                    "partial_json": event["delta"],
                }
            elif etype == "response.output_item.done":
                item = event["item"]
                if item.get("type") != "reasoning":
                    yield {"type": "block_stop", "index": block_index(item["id"])}
            elif etype in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                data = event["response"]
                if data.get("status") == "failed":
                    raise RuntimeError(f"openai chat failed: {data.get('error')}")
                yield {"type": "message_stop", "response": self._deserialize(data)}
