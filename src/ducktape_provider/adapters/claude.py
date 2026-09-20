"""Adapter for Anthropic's Claude Messages API."""

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from .. import errors
from ..adapter import (
    Adapter,
    _key_url_allowed,
    _merge_config,
    _new_request,
    _request_key,
    _resolve_headers,
    _resolve_probe_auth,
    _Secret,
    _validate_headers,
    _warn_headers_refused,
)
from ..streaming import (
    _PROBE_ERRORS,
    _CancellableStream,
    _clear_tracebacks,
    _ErrorHookedStream,
    _iter_sse,
    _loads_tool_input,
    _read_json,
    _request_json,
    _shape_checked,
    _stream_request,
    _StreamTimer,
)
from ..types import (
    Block,
    DocumentBlock,
    ImageBlock,
    Message,
    ModelInfo,
    Response,
    StopReason,
    StreamEvent,
    TextBlock,
    ToolDef,
    Usage,
)

logger = logging.getLogger(__name__)

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
    _AUTH_HEADER = "x-api-key"
    _key_source: _Secret | None = None
    _provider_headers: tuple[_Secret, ...] = ()
    _provider_name: str | None = None
    _warned_transport = False

    def __init__(self, api_key: str | Callable[[], str | None] | None = None):
        """`api_key` is the key, or a function called for it on every request.

        Either way `ANTHROPIC_API_KEY` is then never read (fall back to it with
        `lambda: vault_key() or os.environ.get("ANTHROPIC_API_KEY")`); left out,
        that env var is read per request. A function may be called from several
        threads at once: `Provider`'s async executor workers and stream reader
        threads, and concurrent `async_models()` probes. A subclass overriding
        `_build_request` or `models()` must send the key and any configured
        Provider headers (`self._provider_headers`) itself.
        """
        source = None if api_key is None else _Secret(api_key)
        api_key = None
        self._models_cache: dict[str, ModelInfo] | None = None
        self._cache_time = 0.0
        if source is not None:
            source.validate("claude")
            self._key_source = source

    def is_available(self) -> bool:
        if self._key_source is not None:
            return True
        if any(h.sets({self._AUTH_HEADER}) for h in self._provider_headers):
            return True
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def models(self) -> set[str]:
        now = time.monotonic()
        if self._models_cache is not None and now - self._cache_time < self._MODELS_TTL:
            return set(self._models_cache)
        base = self._MODELS_URL
        resolved = _resolve_probe_auth(
            "claude",
            base,
            self._key_source,
            "ANTHROPIC_API_KEY",
            self._AUTH_HEADER,
            self._provider_headers,
        )
        if resolved is None:
            if self._provider_headers and not _key_url_allowed(base):
                _warn_headers_refused(self, logger, "claude")
            return set()
        key, headers = resolved
        model_infos: dict[str, ModelInfo] = {}
        after_id = None
        stop = sys.exception()
        try:
            while True:
                url = base
                if after_id:
                    url = f"{url}?{urllib.parse.urlencode({'after_id': after_id})}"
                with urllib.request.urlopen(
                    _new_request(
                        url,
                        defaults={"anthropic-version": "2023-06-01"},
                        resolved=headers,
                        auth=None if key is None else ("x-api-key", "", lambda: key),
                    ),
                    timeout=3,
                ) as resp:
                    data = _read_json(resp, "claude", operation="models")
                for m in data.get("data", []):
                    model_infos[m["id"]] = {
                        "context_window": m.get("max_input_tokens"),
                        "max_output_tokens": m.get("max_tokens"),
                    }
                next_after_id = data.get("last_id")
                if not data.get("has_more") or not next_after_id:
                    break
                if next_after_id == after_id:
                    break
                after_id = next_after_id
        except urllib.error.HTTPError as e:
            e.close()
            return set()
        except _PROBE_ERRORS:
            return set()
        except BaseException as e:
            _clear_tracebacks(e, stop)
            raise
        self._models_cache = model_infos
        self._cache_time = now
        return set(model_infos)

    def model_info(self, model: str) -> ModelInfo | None:
        """Context window and max output for `model`, read from the same
        paginated `/v1/models` listing `models()` uses — calling both costs at
        most one round trip, since this reuses (and, on a miss, populates) the
        same cache. `None` when `model` isn't in any page, or on any probe
        failure `models()` itself would swallow into an empty set.
        """
        if model not in self.models():
            return None
        return self._models_cache.get(model) if self._models_cache else None

    def _reset_caches(self) -> None:
        self._models_cache = None
        self._cache_time = 0.0

    def _invalidate_models_cache_on_404(self, e: errors.APIError) -> None:
        if e.status == 404:
            self._reset_caches()

    def _content_source(self, block: ImageBlock | DocumentBlock) -> dict[str, Any]:
        if block["source"] == "url":
            return {"type": "url", "url": block["url"]}
        return {
            "type": "base64",
            "media_type": block["media_type"],
            "data": block["data"],
        }

    def _tool_result_content(
        self, content: str | list[TextBlock | ImageBlock]
    ) -> str | list[dict[str, Any]]:
        if isinstance(content, str):
            return content
        return [
            {"type": "text", "text": b["text"]}
            if b["type"] == "text"
            else {"type": "image", "source": self._content_source(b)}
            for b in content
        ]

    def _serialize(self, messages: list[Message]) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        dropped_thinking = 0
        for message in messages:
            content: list[dict[str, Any]] = []
            dropped_before = dropped_thinking
            for block in message["content"]:
                if block["type"] == "image":
                    content.append(
                        {"type": "image", "source": self._content_source(block)}
                    )
                elif block["type"] == "document":
                    content.append(
                        {"type": "document", "source": self._content_source(block)}
                    )
                elif block["type"] == "tool_result":
                    entry: dict[str, Any] = {
                        "type": "tool_result",
                        "tool_use_id": block["tool_use_id"],
                        "content": self._tool_result_content(block["content"]),
                    }
                    if block.get("is_error"):
                        entry["is_error"] = True
                    content.append(entry)
                elif block["type"] == "thinking" and not block.get("signature"):
                    dropped_thinking += 1
                else:
                    content.append(dict(block))
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
        _validate_headers("claude", extra_headers)
        url = self._MESSAGES_URL
        resolved = _resolve_headers(
            "claude", url, self._provider_headers, extra_headers
        )
        req = _new_request(
            url,
            json.dumps(payload).encode(),
            {"anthropic-version": "2023-06-01", "content-type": "application/json"},
            extra_headers,
            resolved,
            auth=(
                "x-api-key",
                "",
                lambda: _request_key(
                    "claude", url, self._key_source, "ANTHROPIC_API_KEY"
                ),
            ),
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
        try:
            data, latency_ms = _request_json(
                "claude",
                *self._build_request(
                    model, messages, system, tools, config, stream=False
                ),
            )
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
        def start() -> _CancellableStream:
            return _stream_request(
                "claude",
                *self._build_request(
                    model, messages, system, tools, config, stream=True
                ),
                frames=_iter_sse,
                handler=self._stream_handler,
                terminal="message_stop",
            )

        return _ErrorHookedStream("claude", start, self._invalidate_models_cache_on_404)

    def _stream_handler(
        self, timer: _StreamTimer
    ) -> Callable[[dict[str, Any]], Iterator[StreamEvent]]:
        blocks: dict[int, dict[str, Any]] = {}
        text_parts: dict[int, list[str]] = {}
        thinking_parts: dict[int, list[str]] = {}
        signature_parts: dict[int, list[str]] = {}
        json_buffers: dict[int, list[str]] = {}
        surfaced: set[int] = set()
        stream_usage: dict[str, Any] = {}
        stop_reason = ""

        def flush(index: int) -> None:
            block = blocks[index]
            if index in text_parts:
                block["text"] = "".join(text_parts.pop(index))
            if index in thinking_parts:
                block["thinking"] = "".join(thinking_parts.pop(index))
            if index in signature_parts:
                block["signature"] = "".join(signature_parts.pop(index))
            if json_buffers.get(index):
                args, truncated = _loads_tool_input("".join(json_buffers.pop(index)))
                block["input"] = args
                if truncated:
                    block["truncated"] = True

        def handle(event: dict[str, Any]) -> Iterator[StreamEvent]:
            nonlocal stream_usage, stop_reason
            etype = event.get("type")
            if etype == "message_start":
                stream_usage = dict(event["message"].get("usage") or {})
            elif etype == "content_block_start":
                index = event["index"]
                block = blocks[index] = dict(event["content_block"])
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
