"""Adapter for a locally running Ollama server's chat API."""

import http.client
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

from .. import errors
from ..adapter import Adapter
from ..streaming import _iter_ndjson
from ..types import Block, Message, Response, StopReason, StreamEvent, ToolDef

logger = logging.getLogger(__name__)


class OllamaLocalAdapter(Adapter):
    _CHAT_TIMEOUT = 300

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
        try:
            with urllib.request.urlopen(
                f"{self._base_url()}/api/tags", timeout=0.5
            ) as resp:
                data = json.load(resp)
            return {m["name"] for m in data.get("models", [])}
        except (
            OSError,
            ValueError,
            AttributeError,
            KeyError,
            TypeError,
            http.client.HTTPException,
        ):
            return set()

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

    def _deserialize(self, data: dict[str, Any], latency_ms: float) -> Response:
        message = data.get("message", {})
        blocks: list[Block] = []
        if thinking := message.get("thinking"):
            blocks.append({"type": "thinking", "thinking": thinking})
        if content := message.get("content"):
            blocks.append({"type": "text", "text": content})
        for i, call in enumerate(message.get("tool_calls") or []):
            fn = call.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    args = {}
            if not isinstance(args, dict):
                args = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": f"call_{i}",
                    "name": fn.get("name", ""),
                    "input": args,
                }
            )
        has_tool_use = any(b["type"] == "tool_use" for b in blocks)
        raw_reason = data.get("done_reason", "")
        if raw_reason == "length":
            stop_reason: StopReason = "max_tokens"
        elif has_tool_use:
            stop_reason = "tool_use"
        elif raw_reason == "stop":
            stop_reason = "end_turn"
        else:
            stop_reason = "other"
        return {
            "content": blocks,
            "stop_reason": stop_reason,
            "raw_stop_reason": raw_reason,
            "usage": {
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
            "raw": data,
            "latency_ms": latency_ms,
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
            "messages": self._serialize(messages, system),
            "stream": stream,
            "keep_alive": "5m",
        }
        if tools:
            payload["tools"] = self._serialize_tools(tools)
        payload.update(config or {})
        timeout = payload.pop("timeout", self._CHAT_TIMEOUT)
        headers = {"Content-Type": "application/json"}
        headers.update(payload.pop("headers", {}))
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
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            errors.raise_for_http_error("ollama", e)
        except TimeoutError as e:
            errors.raise_for_connection_error("ollama", e)
        except urllib.error.URLError as e:
            errors.raise_for_connection_error("ollama", e)
        return self._deserialize(data, (time.monotonic() - start) * 1000)

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        start = time.monotonic()
        req, timeout = self._build_request(
            model, messages, system, tools, config, stream=True
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                yield from self._stream_events(resp, start)
        except urllib.error.HTTPError as e:
            errors.raise_for_http_error("ollama", e)
        except TimeoutError as e:
            errors.raise_for_connection_error("ollama", e)
        except urllib.error.URLError as e:
            errors.raise_for_connection_error("ollama", e)

    def _stream_events(
        self, resp: http.client.HTTPResponse, start: float
    ) -> Iterator[StreamEvent]:
        thinking_index: int | None = None
        text_index: int | None = None
        thinking = ""
        text = ""
        last: dict[str, Any] = {}

        for chunk in _iter_ndjson(resp):
            last = chunk
            message = chunk.get("message", {})
            if delta := message.get("thinking"):
                if thinking_index is None:
                    thinking_index = 0
                thinking += delta
                yield {
                    "type": "thinking_delta",
                    "index": thinking_index,
                    "thinking": delta,
                }
            if delta := message.get("content"):
                if text_index is None:
                    if thinking_index is not None:
                        yield {"type": "block_stop", "index": thinking_index}
                    text_index = 1 if thinking_index is not None else 0
                text += delta
                yield {"type": "text_delta", "index": text_index, "text": delta}
            if chunk.get("done"):
                break

        if text_index is not None:
            yield {"type": "block_stop", "index": text_index}
        elif thinking_index is not None:
            yield {"type": "block_stop", "index": thinking_index}

        next_index = sum(x is not None for x in (thinking_index, text_index))
        tool_calls = last.get("message", {}).get("tool_calls") or []
        for i, call in enumerate(tool_calls):
            index = next_index + i
            fn = call.get("function", {})
            args = fn.get("arguments")
            yield {
                "type": "tool_use_start",
                "index": index,
                "id": f"call_{i}",
                "name": fn.get("name", ""),
            }
            yield {
                "type": "tool_use_delta",
                "index": index,
                "partial_json": args if isinstance(args, str) else json.dumps(args),
            }
            yield {"type": "block_stop", "index": index}

        data = dict(last)
        data["message"] = {
            **last.get("message", {}),
            "content": text,
            "thinking": thinking,
        }
        yield {
            "type": "message_stop",
            "response": self._deserialize(data, (time.monotonic() - start) * 1000),
        }
