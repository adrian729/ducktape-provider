import http.client
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Literal, NotRequired, TypedDict


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


type Base64Str = str
type MimeType = str
type ToolCallId = str


class ImageBlock(TypedDict):
    type: Literal["image"]
    media_type: MimeType
    data: Base64Str


class ToolUseBlock(TypedDict):
    type: Literal["tool_use"]
    id: ToolCallId
    name: str
    input: dict[str, Any]


class ToolResultBlock(TypedDict):
    type: Literal["tool_result"]
    tool_use_id: ToolCallId
    name: str
    content: str
    is_error: NotRequired[bool]


class ThinkingBlock(TypedDict):
    type: Literal["thinking"]
    thinking: str
    signature: NotRequired[str]


Block = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: list[Block]


type JsonSchema = dict[str, Any]


class ToolDef(TypedDict):
    name: str
    description: str
    parameters: JsonSchema


StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "content_filter",
    "refusal",
    "other",
]


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class Response(TypedDict):
    content: list[Block]
    stop_reason: StopReason
    raw_stop_reason: str
    usage: Usage
    raw: dict[str, Any]


class TextDeltaEvent(TypedDict):
    type: Literal["text_delta"]
    index: int
    text: str


class ThinkingDeltaEvent(TypedDict):
    type: Literal["thinking_delta"]
    index: int
    thinking: str


class ToolUseStartEvent(TypedDict):
    type: Literal["tool_use_start"]
    index: int
    id: ToolCallId
    name: str


class ToolUseDeltaEvent(TypedDict):
    type: Literal["tool_use_delta"]
    index: int
    partial_json: str


class BlockStopEvent(TypedDict):
    type: Literal["block_stop"]
    index: int


class MessageStopEvent(TypedDict):
    type: Literal["message_stop"]
    response: Response


StreamEvent = (
    TextDeltaEvent
    | ThinkingDeltaEvent
    | ToolUseStartEvent
    | ToolUseDeltaEvent
    | BlockStopEvent
    | MessageStopEvent
)


def _iter_sse(resp: http.client.HTTPResponse) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    for raw in resp:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
        elif not line and data_lines:
            payload = "\n".join(data_lines)
            data_lines = []
            if payload != "[DONE]":
                yield json.loads(payload)
    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            yield json.loads(payload)


def _iter_ndjson(resp: http.client.HTTPResponse) -> Iterator[dict[str, Any]]:
    for raw in resp:
        line = raw.strip()
        if line:
            yield json.loads(line)


class Adapter(ABC):
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def models(self) -> set[str]:
        """Unprefixed ids servable now; empty when unreachable."""
        ...

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response: ...

    @abstractmethod
    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]: ...


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
        """text/tool_use/thinking blocks already match Claude's shape; only image and tool_result differ."""
        serialized: list[dict[str, Any]] = []
        for message in messages:
            content: list[dict[str, Any]] = []
            for block in message["content"]:
                if block["type"] == "image":
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
        req = urllib.request.Request(
            self._MESSAGES_URL,
            data=json.dumps(payload).encode(),
            headers={
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY", ""),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
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
            raise RuntimeError(f"claude chat failed: {e.code} {body}") from e
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
            raise RuntimeError(f"claude chat failed: {e.code} {body}") from e

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
            results = [b for b in message["content"] if b["type"] == "tool_result"]
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
            uses = [b for b in message["content"] if b["type"] == "tool_use"]
            images = [b["data"] for b in message["content"] if b["type"] == "image"]
            if not (uses or images) and len(results) == len(message["content"]):
                continue
            text = "\n".join(
                b["text"] for b in message["content"] if b["type"] == "text"
            )
            thinking = "\n".join(
                b["thinking"] for b in message["content"] if b["type"] == "thinking"
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

    def _deserialize(self, data: dict[str, Any]) -> Response:
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
        req = urllib.request.Request(
            f"{self._base_url()}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
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
            raise RuntimeError(f"ollama chat failed: {e.code} {body}") from e
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
            raise RuntimeError(f"ollama chat failed: {e.code} {body}") from e

    def _stream_events(self, resp: http.client.HTTPResponse) -> Iterator[StreamEvent]:
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
        yield {"type": "message_stop", "response": self._deserialize(data)}


class Config(TypedDict, total=False):
    timeout: float
    providers: dict[str, dict[str, Any]]


class Provider:
    def __init__(
        self,
        adapters: dict[str, Adapter] | None = None,
        timeout: float | None = None,
    ):
        self._adapters: dict[str, Adapter] = (
            adapters
            if adapters is not None
            else {
                "claude": ClaudeAdapter(),
                "openai": OpenAIAdapter(),
                "ollama-local": OllamaLocalAdapter(),
            }
        )
        self._config: dict[str, Any] = {}
        if timeout is not None:
            self._config["timeout"] = timeout

    def providers(self) -> dict[str, bool]:
        """Configured providers, available or not."""
        return {
            name: adapter.is_available() for name, adapter in self._adapters.items()
        }

    def models(self) -> dict[str, list[str]]:
        """Usable model ids by provider; nothing from unreachable vendors."""
        return {
            name: sorted(adapter.models())
            for name, adapter in self._adapters.items()
            if adapter.is_available()
        }

    def _merge_config(self, provider: str, config: Config | None) -> dict[str, Any]:
        """Merges self._config, config, config["providers"][provider] in order — each overrides the previous."""
        config = config or {}
        merged = dict(self._config)
        merged.update({k: v for k, v in config.items() if k != "providers"})
        merged.update(config.get("providers", {}).get(provider, {}))
        return merged

    def chat(
        self,
        provider: str,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: Config | None = None,
    ) -> Response:
        return self._adapters[provider].chat(
            model, messages, system, tools, self._merge_config(provider, config)
        )

    def stream_chat(
        self,
        provider: str,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: Config | None = None,
    ) -> Iterator[StreamEvent]:
        return self._adapters[provider].stream_chat(
            model, messages, system, tools, self._merge_config(provider, config)
        )
