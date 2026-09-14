"""Adapter for Anthropic's Claude Messages API."""

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

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

# HTTP status Anthropic documents for each error type, so mid-stream error
# events map to the same exception class as the equivalent HTTP error.
_ERROR_TYPE_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "conflict_error": 409,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


class ClaudeAdapter(Adapter):
    _MODELS_URL = "https://api.anthropic.com/v1/models"
    _MESSAGES_URL = "https://api.anthropic.com/v1/messages"
    _MODELS_TTL = 60
    _CHAT_TIMEOUT = 120
    _MAX_TOKENS = 4096
    _RESERVED_CONFIG = frozenset({"model", "messages", "stream"})

    def __init__(self):
        self._models_cache: set[str] | None = None
        self._cache_time = 0.0

    def is_available(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def models(self) -> set[str]:
        now = time.monotonic()
        if self._models_cache is not None and now - self._cache_time < self._MODELS_TTL:
            return set(self._models_cache)
        model_ids: set[str] = set()
        after_id = None
        try:
            while True:
                url = self._MODELS_URL
                if after_id:
                    url = f"{url}?{urllib.parse.urlencode({'after_id': after_id})}"
                req = urllib.request.Request(
                    url,
                    headers={
                        "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                        "anthropic-version": "2023-06-01",
                    },
                )
                with urllib.request.urlopen(req, timeout=3) as resp:
                    data = _read_json(resp, "claude")
                model_ids.update(m["id"] for m in data.get("data", []))
                next_after_id = data.get("last_id")
                if not data.get("has_more") or not next_after_id:
                    break
                if next_after_id == after_id:
                    break
                after_id = next_after_id
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
        serialized: list[dict[str, Any]] = []
        dropped_thinking = 0
        for message in messages:
            content: list[dict[str, Any]] = []
            dropped_before = dropped_thinking
            for block in message["content"]:
                if block["type"] == "image":
                    if block["source"] == "url":
                        content.append(
                            {
                                "type": "image",
                                "source": {"type": "url", "url": block["url"]},
                            }
                        )
                    else:
                        content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": block["media_type"],
                                    "data": block["data"],
                                },
                            }
                        )
                elif block["type"] == "document":
                    if block["source"] == "url":
                        content.append(
                            {
                                "type": "document",
                                "source": {"type": "url", "url": block["url"]},
                            }
                        )
                    else:
                        content.append(
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": block["media_type"],
                                    "data": block["data"],
                                },
                            }
                        )
                elif block["type"] == "tool_result":
                    entry: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": block["tool_use_id"],
                        "content": block["content"],
                    }
                    if block.get("is_error"):
                        entry["is_error"] = True
                    content.append(entry)
                elif block["type"] == "thinking" and not block.get("signature"):
                    # Only Claude-issued thinking blocks carry a signature, and the
                    # API rejects unsigned ones (e.g. OpenAI/Ollama reasoning).
                    dropped_thinking += 1
                else:
                    content.append(dict(block))
            # The API rejects empty content, and dropping the message instead is
            # safe because it merges the consecutive same-role turns this leaves.
            if not content and dropped_thinking > dropped_before:
                continue
            serialized.append({"role": message["role"], "content": content})
        if dropped_thinking:
            logger.warning(
                "claude adapter cannot send unsigned thinking blocks — "
                "dropping %d from the request",
                dropped_thinking,
            )
        return serialized

    def _serialize_tools(self, tools: list[ToolDef]) -> list[dict[str, Any]]:
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "input_schema": t["parameters"],
            }
            for t in tools
        ]

    def _deserialize(
        self, data: dict[str, Any], latency_ms: float, ttft_ms: float | None = None
    ) -> Response:
        blocks: list[Block] = data.get("content", [])

        raw_reason = data.get("stop_reason") or ""
        if raw_reason == "tool_use":
            stop_reason: StopReason = "tool_use"
        elif raw_reason == "max_tokens":
            stop_reason = "max_tokens"
        elif raw_reason == "stop_sequence":
            stop_reason = "stop_sequence"
        elif raw_reason == "refusal":
            stop_reason = "refusal"
        elif raw_reason == "end_turn":
            stop_reason = "end_turn"
        elif raw_reason == "pause_turn":
            stop_reason = "pause_turn"
        else:
            stop_reason = "other"

        usage = data.get("usage") or {}
        cache_read = usage.get("cache_read_input_tokens")
        cache_write = usage.get("cache_creation_input_tokens")
        normalized_usage: Usage = {
            "input_tokens": (usage.get("input_tokens") or 0)
            + (cache_read or 0)
            + (cache_write or 0),
            "output_tokens": usage.get("output_tokens") or 0,
        }
        if cache_read is not None:
            normalized_usage["cache_read_tokens"] = cache_read
        if cache_write is not None:
            normalized_usage["cache_write_tokens"] = cache_write
        response: Response = {
            "content": blocks,
            "stop_reason": stop_reason,
            "raw_stop_reason": raw_reason,
            "usage": normalized_usage,
            "raw": data,
            "latency_ms": latency_ms,
        }
        if ttft_ms is not None:
            response["ttft_ms"] = ttft_ms
        return response

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
            "messages": self._serialize(messages),
            "max_tokens": self._MAX_TOKENS,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._serialize_tools(tools)
        timeout, extra_headers = _merge_config(
            "claude", payload, config, self._RESERVED_CONFIG, self._CHAT_TIMEOUT
        )
        headers = {
            "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        headers.update(extra_headers)
        _validate_headers("claude", headers)
        req = urllib.request.Request(
            self._MESSAGES_URL,
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
            data, latency_ms = _request_json("claude", req, timeout)
            with _shape_checked("claude"):
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
                "claude",
                req,
                timeout,
                frames=_iter_sse,
                handler=self._stream_handler,
                terminal="message_stop",
            )
        except errors.APIError as e:
            self._invalidate_models_cache_on_404(e)
            raise

    def _stream_handler(
        self, timer: _StreamTimer
    ) -> Callable[[dict[str, Any]], Iterator[StreamEvent]]:
        # Keyed by wire index, so a delta for a block that never started is a
        # KeyError (malformed) rather than silently growing a padded list.
        blocks: dict[int, dict[str, Any]] = {}
        # Deltas collected per block and joined once at content_block_stop, rather
        # than accumulated with `+=` on every delta: str concatenation on a dict
        # value isn't CPython's in-place-append fast path, so `+=` here is O(n^2)
        # over a long stream.
        text_parts: dict[int, list[str]] = {}
        thinking_parts: dict[int, list[str]] = {}
        signature_parts: dict[int, list[str]] = {}
        json_buffers: dict[int, list[str]] = {}
        # Indices that produced a start or delta event; only those get block_stop,
        # so a block the caller never saw (e.g. server_tool_use) has no lone stop.
        surfaced: set[int] = set()
        stream_usage: dict[str, Any] = {}
        stop_reason = ""

        def flush(index: int) -> None:
            # Shared by content_block_stop and message_stop, since a stream can
            # end mid-block (no content_block_stop) and its buffered deltas
            # must still land in the final response rather than vanish.
            block = blocks[index]
            if index in text_parts:
                block["text"] = "".join(text_parts.pop(index))
            if index in thinking_parts:
                block["thinking"] = "".join(thinking_parts.pop(index))
            if index in signature_parts:
                block["signature"] = "".join(signature_parts.pop(index))
            if json_buffers.get(index):
                block["input"] = _loads_tool_input("".join(json_buffers.pop(index)))

        def handle(event: dict[str, Any]) -> Iterator[StreamEvent]:
            nonlocal stream_usage, stop_reason
            etype = event.get("type")
            if etype == "message_start":
                stream_usage = dict(event["message"].get("usage") or {})
            elif etype == "content_block_start":
                index = event["index"]
                block = blocks[index] = dict(event["content_block"])
                # server_tool_use/mcp_tool_use stream their input the same way.
                if "input" in block:
                    json_buffers[index] = []
                if block["type"] == "tool_use":
                    surfaced.add(index)
                    yield {
                        "type": "tool_use_start",
                        "index": index,
                        "id": block["id"],
                        "name": block["name"],
                    }
            elif etype == "content_block_delta":
                index = event["index"]
                block = blocks[index]
                delta = event["delta"]
                dtype = delta.get("type")
                if dtype == "text_delta":
                    text_parts.setdefault(index, []).append(delta["text"])
                    surfaced.add(index)
                    yield {"type": "text_delta", "index": index, "text": delta["text"]}
                elif dtype == "thinking_delta":
                    thinking_parts.setdefault(index, []).append(delta["thinking"])
                    surfaced.add(index)
                    yield {
                        "type": "thinking_delta",
                        "index": index,
                        "thinking": delta["thinking"],
                    }
                elif dtype == "signature_delta":
                    signature_parts.setdefault(index, []).append(delta["signature"])
                elif dtype == "input_json_delta":
                    json_buffers.setdefault(index, []).append(delta["partial_json"])
                    # Server-executed tool calls aren't the caller's to run, so
                    # only client tool_use blocks surface as tool_use events.
                    if block.get("type") == "tool_use":
                        yield {
                            "type": "tool_use_delta",
                            "index": index,
                            "partial_json": delta["partial_json"],
                        }
            elif etype == "content_block_stop":
                index = event["index"]
                flush(index)
                if index in surfaced:
                    yield {"type": "block_stop", "index": index}
            elif etype == "message_delta":
                stop_reason = event["delta"].get("stop_reason") or stop_reason
                # message_delta usage is cumulative and, with server tools, also
                # revises input/cache counts — not just output_tokens.
                stream_usage.update(
                    {
                        k: v
                        for k, v in (event.get("usage") or {}).items()
                        if v is not None
                    }
                )
            elif etype == "error":
                error = event.get("error") or {}
                error_type = error.get("type") or ""
                errors.raise_for_vendor_error(
                    "claude",
                    f"{error_type}: {error.get('message', '')}",
                    status=_ERROR_TYPE_STATUS.get(error_type),
                    body=json.dumps(event),
                )
            elif etype == "message_stop":
                for index in blocks:
                    flush(index)
                data = {
                    "content": [blocks[i] for i in sorted(blocks)],
                    "stop_reason": stop_reason,
                    "usage": stream_usage,
                }
                yield {
                    "type": "message_stop",
                    "response": self._deserialize(
                        data, timer.latency_ms(), timer.ttft_ms()
                    ),
                }

        return handle
