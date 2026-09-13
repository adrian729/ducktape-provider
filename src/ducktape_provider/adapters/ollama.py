"""Adapter for a locally running Ollama server's chat API."""

import http.client
import json
import logging
import os
import time
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from .. import errors
from ..adapter import Adapter, _merge_config, _validate_headers
from ..streaming import (
    _PROBE_ERRORS,
    _iter_ndjson,
    _read_json,
    _request_json,
    _shape_checked,
    _stream_request,
    _StreamTimer,
)
from ..types import (
    Block,
    Message,
    Response,
    StopReason,
    StreamEvent,
    ToolDef,
    ToolUseBlock,
)

logger = logging.getLogger(__name__)


def _stream_error_status(message: str) -> int:
    """Best-effort HTTP status for an error chunk, which unlike Ollama's HTTP
    errors carries no status code of its own.

    Ollama answers these failures with 400 (input exceeds the context length),
    404 (model not found), or 500 (anything else, e.g. a crashed runner) when they
    happen before streaming starts, so the stream maps to the same classes.
    """
    lowered = message.lower()
    if any(m in lowered for m in errors._CONTEXT_OVERFLOW_MARKERS):
        return 400
    if "not found" in lowered:
        return 404
    return 500


def _tool_use_block(index: int, call: dict[str, Any]) -> ToolUseBlock:
    fn = call.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {
        "type": "tool_use",
        "id": f"call_{index}",
        "name": fn.get("name", ""),
        "input": args,
    }


class OllamaLocalAdapter(Adapter):
    _CHAT_TIMEOUT = 300
    _MODELS_TTL = 60
    _RESERVED_CONFIG = frozenset({"model", "messages", "stream"})

    def __init__(self):
        self._models_cache: set[str] | None = None
        self._cache_time = 0.0

    def _base_url(self) -> str:
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        return host

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self._base_url()}/api/version", timeout=0.5):
                return True
        except (OSError, ValueError, http.client.HTTPException):
            return False

    def models(self) -> set[str]:
        now = time.monotonic()
        if self._models_cache is not None and now - self._cache_time < self._MODELS_TTL:
            return set(self._models_cache)
        try:
            with urllib.request.urlopen(
                f"{self._base_url()}/api/tags", timeout=0.5
            ) as resp:
                data = _read_json(resp, "ollama")
            model_ids = {m["name"] for m in data.get("models", [])}
        except _PROBE_ERRORS:
            return set()
        self._models_cache = model_ids
        self._cache_time = now
        return set(model_ids)

    def _serialize(
        self, messages: list[Message], system: str | None
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        if system:
            serialized.append({"role": "system", "content": system})
        for message in messages:
            for b in message["content"]:
                if b["type"] == "document":
                    logger.warning(
                        "ollama-local adapter does not support document blocks — "
                        "dropping one from the request"
                    )
            content_blocks = [b for b in message["content"] if b["type"] != "document"]
            results = [b for b in content_blocks if b["type"] == "tool_result"]
            for block in results:
                content = block["content"]
                if block.get("is_error"):
                    content = f"ERROR: {content}"
                serialized.append(
                    {
                        "role": "tool",
                        "content": content,
                        "tool_name": block["name"],
                        "tool_use_id": block["tool_use_id"],
                    }
                )
            uses = [b for b in content_blocks if b["type"] == "tool_use"]
            images = []
            for b in content_blocks:
                if b["type"] == "image":
                    if b["source"] == "url":
                        raise errors.UnsupportedBlockError(
                            "ollama-local does not support URL image blocks (only base64) — "
                            "convert to a Base64ImageBlock before calling"
                        )
                    images.append(b["data"])
            if not (uses or images) and len(results) == len(content_blocks):
                continue
            text = "\n".join(b["text"] for b in content_blocks if b["type"] == "text")
            thinking = "\n".join(
                b["thinking"] for b in content_blocks if b["type"] == "thinking"
            )
            entry: dict[str, Any] = {"role": message["role"], "content": text}
            if thinking:
                entry["thinking"] = thinking
            if uses:
                entry["tool_calls"] = [
                    {"function": {"name": b["name"], "arguments": b["input"]}}
                    for b in uses
                ]
            if images:
                entry["images"] = images
            serialized.append(entry)
        return serialized

    def _serialize_tools(self, tools: list[ToolDef]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    def _deserialize(
        self,
        data: dict[str, Any],
        latency_ms: float,
        ttft_ms: float | None = None,
        blocks: list[Block] | None = None,
    ) -> Response:
        """`blocks`, when given, is the content already assembled in streamed order."""
        if blocks is None:
            message = data.get("message") or {}
            blocks = []
            if thinking := message.get("thinking"):
                blocks.append({"type": "thinking", "thinking": thinking})
            if content := message.get("content"):
                blocks.append({"type": "text", "text": content})
            for i, call in enumerate(message.get("tool_calls") or []):
                blocks.append(_tool_use_block(i, call))
        has_tool_use = any(b["type"] == "tool_use" for b in blocks)
        raw_reason = data.get("done_reason") or ""
        if raw_reason == "length":
            stop_reason: StopReason = "max_tokens"
        elif has_tool_use:
            stop_reason = "tool_use"
        elif raw_reason == "stop":
            stop_reason = "end_turn"
        else:
            stop_reason = "other"
        response: Response = {
            "content": blocks,
            "stop_reason": stop_reason,
            "raw_stop_reason": raw_reason,
            "usage": {
                "input_tokens": data.get("prompt_eval_count") or 0,
                "output_tokens": data.get("eval_count") or 0,
            },
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
            "messages": self._serialize(messages, system),
            "stream": stream,
            "keep_alive": "5m",
        }
        if tools:
            payload["tools"] = self._serialize_tools(tools)
        timeout, extra_headers = _merge_config(
            "ollama", payload, config, self._RESERVED_CONFIG, self._CHAT_TIMEOUT
        )
        headers = {"Content-Type": "application/json"}
        headers.update(extra_headers)
        _validate_headers("ollama", headers)
        req = urllib.request.Request(
            f"{self._base_url()}/api/chat",
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
        data, latency_ms = _request_json("ollama", req, timeout)
        with _shape_checked("ollama"):
            return self._deserialize(data, latency_ms)

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
        yield from _stream_request(
            "ollama",
            req,
            timeout,
            frames=_iter_ndjson,
            handler=self._stream_handler,
            terminal="a done chunk",
        )

    def _stream_handler(
        self, timer: _StreamTimer
    ) -> Callable[[dict[str, Any]], Iterator[StreamEvent]]:
        """Synthesizes a block lifecycle from Ollama's flat chunks: a new block index
        is allocated whenever the kind of streamed content changes, and each tool
        call is emitted as its own complete block as soon as its chunk arrives. The
        final content holds one block per streamed index, in that order."""
        blocks: list[Block] = []
        open_kind: str | None = None
        tool_calls: list[dict[str, Any]] = []

        def handle(chunk: dict[str, Any]) -> Iterator[StreamEvent]:
            nonlocal open_kind
            if "error" in chunk:
                message = str(chunk["error"])
                errors.raise_for_vendor_error(
                    "ollama",
                    message,
                    status=_stream_error_status(message),
                    body=json.dumps(chunk),
                )
            message = chunk.get("message") or {}
            for kind, delta in (
                ("thinking", message.get("thinking")),
                ("text", message.get("content")),
            ):
                if not delta:
                    continue
                if not isinstance(delta, str):
                    raise TypeError(f"{kind} must be a string")
                if open_kind != kind:
                    if open_kind is not None:
                        yield {"type": "block_stop", "index": len(blocks) - 1}
                    open_kind = kind
                    if kind == "thinking":
                        blocks.append({"type": "thinking", "thinking": ""})
                    else:
                        blocks.append({"type": "text", "text": ""})
                index = len(blocks) - 1
                block = blocks[index]
                if block["type"] == "thinking":
                    block["thinking"] += delta
                    yield {"type": "thinking_delta", "index": index, "thinking": delta}
                elif block["type"] == "text":
                    block["text"] += delta
                    yield {"type": "text_delta", "index": index, "text": delta}

            # Recent Ollama versions send tool calls in a done:false chunk, older
            # ones in the final chunk, so collect them from every chunk.
            for call in message.get("tool_calls") or []:
                if open_kind is not None:
                    yield {"type": "block_stop", "index": len(blocks) - 1}
                    open_kind = None
                block = _tool_use_block(len(tool_calls), call)
                index = len(blocks)
                blocks.append(block)
                tool_calls.append(call)
                yield {
                    "type": "tool_use_start",
                    "index": index,
                    "id": block["id"],
                    "name": block["name"],
                }
                # Re-encoded from the parsed input rather than passed through, so
                # missing or unparseable arguments stream as the same {} the final
                # block carries.
                yield {
                    "type": "tool_use_delta",
                    "index": index,
                    "partial_json": json.dumps(block["input"]),
                }
                yield {"type": "block_stop", "index": index}

            if chunk.get("done"):
                if open_kind is not None:
                    yield {"type": "block_stop", "index": len(blocks) - 1}
                data = dict(chunk)
                data["message"] = {
                    **message,
                    "content": "".join(
                        b["text"] for b in blocks if b["type"] == "text"
                    ),
                    "thinking": "".join(
                        b["thinking"] for b in blocks if b["type"] == "thinking"
                    ),
                    "tool_calls": tool_calls,
                }
                yield {
                    "type": "message_stop",
                    "response": self._deserialize(
                        data, timer.latency_ms(), timer.ttft_ms(), blocks
                    ),
                }

        return handle
