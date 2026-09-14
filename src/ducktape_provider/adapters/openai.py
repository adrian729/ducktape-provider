"""Adapter for OpenAI's Responses API."""

import json
import logging
import os
import re
import time
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any, NoReturn

from .. import errors
from ..adapter import Adapter, _merge_config, _validate_headers
from ..streaming import (
    _PROBE_ERRORS,
    _iter_sse,
    _loads_tool_input,
    _read_json,
    _request_json,
    _shape_checked,
    _stream_request,
    _StreamTimer,
)
from ..types import Block, Message, Response, StopReason, StreamEvent, ToolDef, Usage

logger = logging.getLogger(__name__)

# HTTP status OpenAI documents for each error code or type, so failures reported
# inside a 200 response map to the same exception class as the HTTP error would.
# Sources: the Responses API ResponseErrorCode enum (server_error,
# rate_limit_exceeded) and the error-codes guide (the 429 quota/spend-limit codes,
# 503 server_is_overloaded); the remaining ResponseErrorCode values (invalid_prompt,
# invalid_image, ...) are request problems and fall through to a plain APIError.
_ERROR_CODE_STATUS = {
    "invalid_request_error": 400,
    "context_length_exceeded": 400,
    "rate_limit_exceeded": 429,
    "rate_limit_error": 429,
    "slow_down": 429,
    "insufficient_quota": 429,
    "credit_balance_exhausted": 429,
    "organization_spend_limit_exceeded": 429,
    "project_spend_limit_exceeded": 429,
    "organization_usage_limit_exceeded": 429,
    "server_error": 500,
    "service_unavailable_error": 503,
    "server_is_overloaded": 503,
}


class OpenAIAdapter(Adapter):
    _MODELS_URL = "https://api.openai.com/v1/models"
    _RESPONSES_URL = "https://api.openai.com/v1/responses"
    _CHAT_MODEL_RE = re.compile(r"^(gpt-|chatgpt-|o\d)")
    _NON_CHAT_RE = re.compile(r"-(audio|realtime|transcribe|tts|search)|^gpt-image")
    _MODELS_TTL = 60
    _CHAT_TIMEOUT = 120
    _RESERVED_CONFIG = frozenset({"model", "input", "stream"})

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
                data = _read_json(resp, "openai")
            model_ids = {
                m["id"]
                for m in data.get("data", [])
                if self._CHAT_MODEL_RE.match(m["id"])
                and not self._NON_CHAT_RE.search(m["id"])
            }
        except _PROBE_ERRORS:
            return set()
        self._models_cache = model_ids
        self._cache_time = now
        return set(model_ids)

    def _invalidate_models_cache_on_404(self, e: errors.APIError) -> None:
        # A 404 means the model itself is gone, so Provider's auto-match must not
        # keep re-picking this adapter off a stale models() list for up to _MODELS_TTL.
        if e.status == 404:
            self._models_cache = None

    def _serialize(self, messages: list[Message]) -> list[dict[str, Any]]:
        """The Responses API's "input" is a flat, order-significant list of items, not messages
        grouping content blocks — tool_use/tool_result blocks become their own top-level
        function_call/function_call_output items, interleaved in the order blocks appear."""
        serialized: list[dict[str, Any]] = []
        dropped_thinking = 0
        for message in messages:
            content: list[dict[str, Any]] = []

            def flush(
                role: str = message["role"], buf: list[dict[str, Any]] = content
            ) -> None:
                if buf:
                    serialized.append(
                        {"type": "message", "role": role, "content": list(buf)}
                    )
                    buf.clear()

            for block in message["content"]:
                if block["type"] == "text":
                    content.append({"type": "input_text", "text": block["text"]})
                elif block["type"] == "image":
                    image_url = (
                        block["url"]
                        if block["source"] == "url"
                        else f"data:{block['media_type']};base64,{block['data']}"
                    )
                    content.append({"type": "input_image", "image_url": image_url})
                elif block["type"] == "document":
                    logger.warning(
                        "openai adapter does not support document blocks yet — "
                        "dropping one from the request"
                    )
                elif block["type"] == "thinking":
                    # Round-tripping reasoning needs the item id and encrypted
                    # content, which a ThinkingBlock doesn't carry.
                    dropped_thinking += 1
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
        if dropped_thinking:
            logger.warning(
                "openai adapter cannot send thinking blocks back as reasoning "
                "items — dropping %d from the request",
                dropped_thinking,
            )
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

    def _deserialize(
        self, data: dict[str, Any], latency_ms: float, ttft_ms: float | None = None
    ) -> Response:
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
                args = _loads_tool_input(item["arguments"])
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": item["call_id"],
                        "name": item["name"],
                        "input": args,
                    }
                )
            elif item.get("type") == "reasoning":
                thinking = "\n".join(
                    part["text"]
                    for part in item.get("summary", [])
                    if part.get("type") == "summary_text"
                )
                if thinking:
                    blocks.append({"type": "thinking", "thinking": thinking})

        has_tool_use = any(b["type"] == "tool_use" for b in blocks)
        status = data.get("status") or ""
        incomplete_reason = (data.get("incomplete_details") or {}).get("reason") or ""
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
        normalized_usage: Usage = {
            "input_tokens": usage.get("input_tokens") or 0,
            "output_tokens": usage.get("output_tokens") or 0,
        }
        input_tokens_details = usage.get("input_tokens_details") or {}
        if (cached := input_tokens_details.get("cached_tokens")) is not None:
            normalized_usage["cache_read_tokens"] = cached
        if (cache_write := input_tokens_details.get("cache_write_tokens")) is not None:
            normalized_usage["cache_write_tokens"] = cache_write
        response: Response = {
            "content": blocks,
            "stop_reason": stop_reason,
            "raw_stop_reason": incomplete_reason or status,
            "usage": normalized_usage,
            "raw": data,
            "latency_ms": latency_ms,
        }
        if ttft_ms is not None:
            response["ttft_ms"] = ttft_ms
        return response

    def _raise_for_error(self, error: dict[str, Any] | None) -> NoReturn:
        error = error or {}
        code = error.get("code") or ""
        error_type = error.get("type") or ""
        message = error.get("message") or "unknown error"
        errors.raise_for_vendor_error(
            "openai",
            f"{code}: {message}" if code else message,
            status=_ERROR_CODE_STATUS.get(code) or _ERROR_CODE_STATUS.get(error_type),
            body=json.dumps(error),
        )

    def _build_request(
        self,
        model: str,
        messages: list[Message],
        system: str | None,
        tools: list[ToolDef] | None,
        config: dict[str, Any] | None,
        stream: bool,
    ) -> tuple[urllib.request.Request, float | None]:
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
        timeout, extra_headers = _merge_config(
            "openai", payload, config, self._RESERVED_CONFIG, self._CHAT_TIMEOUT
        )
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
            "Content-Type": "application/json",
        }
        headers.update(extra_headers)
        _validate_headers("openai", headers)
        req = urllib.request.Request(
            self._RESPONSES_URL,
            data=json.dumps(payload).encode(),
            headers=headers,
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
            data, latency_ms = _request_json("openai", req, timeout)
            with _shape_checked("openai"):
                if data.get("status") == "failed":
                    self._raise_for_error(data.get("error"))
                return self._deserialize(data, latency_ms)
        except errors.APIError as e:
            self._invalidate_models_cache_on_404(e)
            raise

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
            yield from _stream_request(
                "openai",
                req,
                timeout,
                frames=_iter_sse,
                handler=self._stream_handler,
                terminal="response.completed",
            )
        except errors.APIError as e:
            self._invalidate_models_cache_on_404(e)
            raise

    def _stream_handler(
        self, timer: _StreamTimer
    ) -> Callable[[dict[str, Any]], Iterator[StreamEvent]]:
        """Numbers streamed blocks to match the final content: one index per
        function call, per reasoning item (all its summary parts), and per text or
        refusal part of a message item, allocated in the order they first appear."""
        index_by_part: dict[tuple[str, int], int] = {}
        indices_by_item: dict[str, list[int]] = {}
        # Only indices that produced a start or delta event get a block_stop.
        surfaced: set[int] = set()
        summary_index_by_item: dict[str, int] = {}

        def block_index(item_id: str, part: int = 0) -> int:
            key = (item_id, part)
            if key not in index_by_part:
                index_by_part[key] = len(index_by_part)
                indices_by_item.setdefault(item_id, []).append(index_by_part[key])
            return index_by_part[key]

        def handle(event: dict[str, Any]) -> Iterator[StreamEvent]:
            etype = event.get("type")
            if etype == "response.output_item.added":
                item = event["item"]
                if item.get("type") == "function_call":
                    index = block_index(item["id"])
                    surfaced.add(index)
                    yield {
                        "type": "tool_use_start",
                        "index": index,
                        "id": item.get("call_id", ""),
                        "name": item.get("name", ""),
                    }
            elif etype == "response.content_part.added":
                # Reserved even if no delta follows, since the final content still
                # holds a (possibly empty) block for this part.
                if event["part"].get("type") in ("output_text", "refusal"):
                    block_index(event["item_id"], event["content_index"])
            elif etype in ("response.output_text.delta", "response.refusal.delta"):
                index = block_index(event["item_id"], event.get("content_index", 0))
                surfaced.add(index)
                yield {"type": "text_delta", "index": index, "text": event["delta"]}
            elif etype == "response.function_call_arguments.delta":
                index = block_index(event["item_id"])
                surfaced.add(index)
                yield {
                    "type": "tool_use_delta",
                    "index": index,
                    "partial_json": event["delta"],
                }
            elif etype == "response.reasoning_summary_text.delta":
                item_id = event["item_id"]
                text = event["delta"]
                summary_index = event.get("summary_index", 0)
                previous = summary_index_by_item.get(item_id)
                # Matches the "\n" the buffered path joins summary parts with.
                if previous is not None and previous != summary_index:
                    text = "\n" + text
                summary_index_by_item[item_id] = summary_index
                index = block_index(item_id)
                surfaced.add(index)
                yield {"type": "thinking_delta", "index": index, "thinking": text}
            elif etype == "response.output_item.done":
                for index in indices_by_item.get(event["item"]["id"], []):
                    if index in surfaced:
                        yield {"type": "block_stop", "index": index}
            elif etype == "error":
                # Documented flat, but accept the fields nested like other errors.
                nested = event.get("error")
                self._raise_for_error(nested if isinstance(nested, dict) else event)
            elif etype == "response.failed":
                self._raise_for_error((event.get("response") or {}).get("error"))
            elif etype in ("response.completed", "response.incomplete"):
                yield {
                    "type": "message_stop",
                    "response": self._deserialize(
                        event["response"], timer.latency_ms(), timer.ttft_ms()
                    ),
                }

        return handle
