"""Adapter for Anthropic's Claude Messages API."""

import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterator
from typing import Any

from .. import errors
from ..adapter import Adapter
from ..streaming import _iter_sse
from ..types import Block, Message, Response, StopReason, StreamEvent, ToolDef


class ClaudeAdapter(Adapter):
    _MODELS_URL = "https://api.anthropic.com/v1/models"
    _MESSAGES_URL = "https://api.anthropic.com/v1/messages"
    _MODELS_TTL = 60
    _CHAT_TIMEOUT = 120
    _MAX_TOKENS = 4096

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
                    data = json.load(resp)
                model_ids.update(m["id"] for m in data.get("data", []))
                next_after_id = data.get("last_id")
                if not data.get("has_more") or not next_after_id:
                    break
                if next_after_id == after_id:
                    break
                after_id = next_after_id
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
        serialized: list[dict[str, Any]] = []
        for message in messages:
            content: list[dict[str, Any]] = []
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
                else:
                    content.append(dict(block))
            serialized.append({"role": message["role"], "content": content})
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

    def _deserialize(self, data: dict[str, Any]) -> Response:
        blocks: list[Block] = data.get("content", [])

        raw_reason = data.get("stop_reason", "")
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
        else:
            stop_reason = "other"

        usage = data.get("usage", {})
        return {
            "content": blocks,
            "stop_reason": stop_reason,
            "raw_stop_reason": raw_reason,
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
            "messages": self._serialize(messages),
            "max_tokens": self._MAX_TOKENS,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = self._serialize_tools(tools)
        payload.update(config or {})
        timeout = payload.pop("timeout", self._CHAT_TIMEOUT)
        headers = {
            "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        headers.update(payload.pop("headers", {}))
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
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            errors.raise_for_http_error("claude", e)
        except TimeoutError as e:
            errors.raise_for_connection_error("claude", e)
        except urllib.error.URLError as e:
            errors.raise_for_connection_error("claude", e)
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
            errors.raise_for_http_error("claude", e)
        except TimeoutError as e:
            errors.raise_for_connection_error("claude", e)
        except urllib.error.URLError as e:
            errors.raise_for_connection_error("claude", e)

    def _stream_events(self, resp: http.client.HTTPResponse) -> Iterator[StreamEvent]:
        blocks: list[dict[str, Any]] = []
        json_buffers: dict[int, str] = {}
        message: dict[str, Any] = {}
        stop_reason = ""
        output_tokens = 0

        for event in _iter_sse(resp):
            etype = event.get("type")
            if etype == "message_start":
                message = event["message"]
            elif etype == "content_block_start":
                index = event["index"]
                while len(blocks) <= index:
                    blocks.append({})
                blocks[index] = dict(event["content_block"])
                if blocks[index]["type"] == "tool_use":
                    json_buffers[index] = ""
                    yield {
                        "type": "tool_use_start",
                        "index": index,
                        "id": blocks[index]["id"],
                        "name": blocks[index]["name"],
                    }
            elif etype == "content_block_delta":
                index = event["index"]
                delta = event["delta"]
                dtype = delta.get("type")
                if dtype == "text_delta":
                    blocks[index]["text"] = (
                        blocks[index].get("text", "") + delta["text"]
                    )
                    yield {"type": "text_delta", "index": index, "text": delta["text"]}
                elif dtype == "thinking_delta":
                    blocks[index]["thinking"] = (
                        blocks[index].get("thinking", "") + delta["thinking"]
                    )
                    yield {
                        "type": "thinking_delta",
                        "index": index,
                        "thinking": delta["thinking"],
                    }
                elif dtype == "signature_delta":
                    blocks[index]["signature"] = (
                        blocks[index].get("signature", "") + delta["signature"]
                    )
                elif dtype == "input_json_delta":
                    json_buffers[index] += delta["partial_json"]
                    yield {
                        "type": "tool_use_delta",
                        "index": index,
                        "partial_json": delta["partial_json"],
                    }
            elif etype == "content_block_stop":
                index = event["index"]
                if index in json_buffers:
                    try:
                        blocks[index]["input"] = json.loads(json_buffers[index] or "{}")
                    except json.JSONDecodeError:
                        blocks[index]["input"] = {}
                yield {"type": "block_stop", "index": index}
            elif etype == "message_delta":
                stop_reason = event["delta"].get("stop_reason") or stop_reason
                output_tokens = event.get("usage", {}).get(
                    "output_tokens", output_tokens
                )
            elif etype == "message_stop":
                data = {
                    "content": blocks,
                    "stop_reason": stop_reason,
                    "usage": {
                        "input_tokens": message.get("usage", {}).get("input_tokens", 0),
                        "output_tokens": output_tokens,
                    },
                }
                yield {"type": "message_stop", "response": self._deserialize(data)}
