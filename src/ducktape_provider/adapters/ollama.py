"""Adapter for a locally running Ollama server's chat API."""

import copy
import http.client
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any, TypedDict, cast

from .. import errors
from ..adapter import (
    Adapter,
    _BadConfiguredHeader,
    _key_url_allowed,
    _merge_config,
    _new_request,
    _resolve_headers,
    _Secret,
    _system_text,
    _validate_headers,
    _warn_headers_refused,
)
from ..streaming import (
    _PROBE_ERRORS,
    _CancellableStream,
    _clear_tracebacks,
    _ErrorHookedStream,
    _iter_ndjson,
    _loads_tool_input,
    _read_json,
    _request_json,
    _shape_checked,
    _stream_request,
    _StreamTimer,
)
from ..types import (
    Block,
    Capabilities,
    EmbedResponse,
    EmbedUsage,
    ImageBlock,
    Message,
    ModelInfo,
    Response,
    StopReason,
    StreamEvent,
    SystemBlock,
    TextBlock,
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
    truncated = False
    if isinstance(args, str):
        args, truncated = _loads_tool_input(args)
    elif not isinstance(args, dict):
        args = {}
    block: ToolUseBlock = {
        "type": "tool_use",
        "id": f"call_{index}",
        "name": fn.get("name", ""),
        "input": args,
    }
    if truncated:
        block["truncated"] = True
    return block


class _ShowFacts(TypedDict):
    """The `/api/show` facts both `model_info()` and `capabilities()` project."""

    context_window: int | None
    capabilities: list[str] | None
    vision: bool | None


class OllamaLocalAdapter(Adapter):
    _CHAT_TIMEOUT = 300
    _EMBED_TIMEOUT = 300
    _MODELS_TTL = 60
    _RESERVED_CONFIG = frozenset({"model", "messages", "stream"})
    _EMBED_RESERVED_CONFIG = frozenset({"model", "input"})
    _provider_headers: tuple[_Secret, ...] = ()
    _provider_name: str | None = None
    _warned_transport = False

    def __init__(self):
        """A subclass overriding `_build_request`, `models()` or
        `is_available()` must send any configured Provider headers
        (`self._provider_headers`) itself."""
        self._models_cache: set[str] | None = None
        self._cache_time = 0.0
        self._show_cache: dict[str, tuple[_ShowFacts | None, float]] = {}

    def _base_url(self) -> str:
        host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
        if not host.startswith(("http://", "https://")):
            host = f"http://{host}"
        return host.rstrip("/")

    def is_available(self) -> bool:
        url = f"{self._base_url()}/api/version"
        if self._provider_headers and not _key_url_allowed(url):
            _warn_headers_refused(self, logger, "ollama-local")
            return False
        try:
            headers = _resolve_headers("ollama-local", url, self._provider_headers, ())
        except _BadConfiguredHeader:
            return False
        stop = sys.exception()
        try:
            with urllib.request.urlopen(_new_request(url, resolved=headers), timeout=3):
                return True
        except urllib.error.HTTPError as e:
            e.close()
            return False
        except (OSError, ValueError, http.client.HTTPException):
            return False
        except BaseException as e:
            _clear_tracebacks(e, stop)
            raise

    def models(self) -> set[str]:
        now = time.monotonic()
        if self._models_cache is not None and now - self._cache_time < self._MODELS_TTL:
            return set(self._models_cache)
        url = f"{self._base_url()}/api/tags"
        if self._provider_headers and not _key_url_allowed(url):
            _warn_headers_refused(self, logger, "ollama-local")
            return set()
        try:
            headers = _resolve_headers("ollama-local", url, self._provider_headers, ())
        except _BadConfiguredHeader:
            return set()
        stop = sys.exception()
        try:
            with urllib.request.urlopen(
                _new_request(url, resolved=headers), timeout=3
            ) as resp:
                data = _read_json(resp, "ollama", operation="models")
            model_ids = {m["name"] for m in data.get("models", [])}
        except urllib.error.HTTPError as e:
            e.close()
            return set()
        except _PROBE_ERRORS:
            return set()
        except BaseException as e:
            _clear_tracebacks(e, stop)
            raise
        self._models_cache = model_ids
        self._cache_time = now
        return set(model_ids)

    def embed_models(self) -> set[str]:
        return self.models()

    def _reset_caches(self) -> None:
        self._models_cache = None
        self._cache_time = 0.0
        self._show_cache = {}

    def _invalidate_model_capabilities(self, model: str) -> None:
        self._show_cache.pop(model, None)

    def _invalidate_models_cache_on_404(self, e: errors.APIError) -> None:
        if e.status == 404:
            self._reset_caches()

    def _show(self, model: str, operation: str = "capabilities") -> _ShowFacts | None:
        """`/api/show` facts for `model`, cached; `None` for a vendor 404."""
        now = time.monotonic()
        cached = self._show_cache.get(model)
        if cached is not None and now - cached[1] < self._MODELS_TTL:
            return cached[0]
        url = f"{self._base_url()}/api/show"
        headers = _resolve_headers("ollama-local", url, self._provider_headers, ())
        try:
            data, _ = _request_json(
                "ollama",
                _new_request(
                    url,
                    json.dumps({"model": model}).encode(),
                    {"Content-Type": "application/json"},
                    resolved=headers,
                ),
                3,
                operation=operation,
            )
        except errors.APIError as e:
            if e.status != 404:
                raise
            facts: _ShowFacts | None = None
        else:
            with _shape_checked("ollama", operation=operation):
                info = data.get("model_info") or {}
                if not isinstance(info, dict):
                    info = {}
                context_window = next(
                    (
                        v
                        for k, v in info.items()
                        if k.endswith(".context_length")
                        and isinstance(v, int)
                        and not isinstance(v, bool)
                    ),
                    None,
                )
                caps = data.get("capabilities")
                if isinstance(caps, list):
                    vision: bool | None = "vision" in caps
                elif any(key.startswith("clip.") or ".vision." in key for key in info):
                    vision = True
                else:
                    vision = None
                facts = cast(
                    _ShowFacts,
                    {
                        "context_window": context_window,
                        "capabilities": caps if isinstance(caps, list) else None,
                        "vision": vision,
                    },
                )
        for stale in [
            key
            for key, (_, stamp) in self._show_cache.items()
            if now - stamp >= self._MODELS_TTL
        ]:
            del self._show_cache[stale]
        self._show_cache[model] = (facts, now)
        return facts

    def capabilities(self, model: str) -> Capabilities | None:
        facts = self._show(model)
        if facts is None:
            return None
        caps = facts["capabilities"]
        if not isinstance(caps, list):
            return {
                "tools": None,
                "vision": facts["vision"],
                "pdf_input": None,
                "thinking": None,
            }
        return {
            "tools": "tools" in caps,
            "vision": "vision" in caps,
            "pdf_input": None,
            "thinking": "thinking" in caps,
            "raw": {"capabilities": copy.deepcopy(caps)},
        }

    def model_info(self, model: str) -> ModelInfo | None:
        """The context window from Ollama's own GGUF metadata for `model` — its
        maximum *supported* context, not necessarily what a given request
        actually gets: `num_ctx` can override the effective window smaller or
        larger per call. No max-output-tokens equivalent exists in Ollama's
        API, so that field is always `None`. Reads the shared `/api/show` cache
        that `capabilities()` uses. `None` on any probe failure, as before.
        """
        url = f"{self._base_url()}/api/show"
        if self._provider_headers and not _key_url_allowed(url):
            _warn_headers_refused(self, logger, "ollama-local")
            return None
        try:
            facts = self._show(model, "model_info")
        except errors.APIError:
            return None
        except _PROBE_ERRORS:
            return None
        if facts is None:
            return None
        return {"context_window": facts["context_window"], "max_output_tokens": None}

    def _tool_result_content(self, content: str | list[TextBlock | ImageBlock]) -> str:
        if isinstance(content, str):
            return content
        if any(b["type"] == "image" for b in content):
            raise errors.UnsupportedBlockError(
                "ollama-local does not support image content in tool results — "
                "convert to text or drop it before calling"
            )
        return "\n".join(b["text"] for b in content if b["type"] == "text")

    def _serialize(
        self, messages: list[Message], system: str | list[SystemBlock] | None
    ) -> list[dict[str, Any]]:
        serialized: list[dict[str, Any]] = []
        if system:
            serialized.append({"role": "system", "content": _system_text(system)})
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
                content = self._tool_result_content(block["content"])
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
        system: str | list[SystemBlock] | None,
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
        _validate_headers("ollama", extra_headers)
        url = f"{self._base_url()}/api/chat"
        resolved = _resolve_headers(
            "ollama-local", url, self._provider_headers, extra_headers
        )
        req = _new_request(
            url,
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
            extra_headers,
            resolved,
        )
        return req, timeout

    def _build_embed_request(
        self, model: str, input: list[str], config: dict[str, Any] | None
    ) -> tuple[urllib.request.Request, float | None]:
        payload: dict[str, Any] = {"model": model, "input": input}
        timeout, extra_headers = _merge_config(
            "ollama",
            payload,
            config,
            self._EMBED_RESERVED_CONFIG,
            self._EMBED_TIMEOUT,
            operation="embed",
        )
        _validate_headers("ollama", extra_headers)
        url = f"{self._base_url()}/api/embed"
        resolved = _resolve_headers(
            "ollama-local", url, self._provider_headers, extra_headers
        )
        req = _new_request(
            url,
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
            extra_headers,
            resolved,
        )
        return req, timeout

    def embed(
        self, model: str, input: list[str], config: dict[str, Any] | None = None
    ) -> EmbedResponse:
        try:
            data, latency_ms = _request_json(
                "ollama",
                *self._build_embed_request(model, input, config),
                operation="embed",
            )
            with _shape_checked("ollama", operation="embed"):
                if not isinstance(data, dict):
                    raise TypeError("embedding response is not an object")
                vectors = data["embeddings"]
                if len(vectors) != len(input):
                    raise errors.MalformedResponseError(
                        f"ollama embed failed: expected {len(input)} embeddings, "
                        f"got {len(vectors)}"
                    )
                raw: dict[str, Any] = {}
                if data.get("model") is not None:
                    raw["model"] = data["model"]
                if data.get("total_duration") is not None:
                    raw["total_duration"] = data["total_duration"]
                if data.get("load_duration") is not None:
                    raw["load_duration"] = data["load_duration"]
                usage: EmbedUsage = {"input_tokens": data.get("prompt_eval_count") or 0}
                return {
                    "embeddings": [list(v) for v in vectors],
                    "usage": usage,
                    "raw": raw,
                    "latency_ms": latency_ms,
                }
        except errors.APIError as e:
            self._invalidate_models_cache_on_404(e)
            raise

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | list[SystemBlock] | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response:
        try:
            data, latency_ms = _request_json(
                "ollama",
                *self._build_request(
                    model, messages, system, tools, config, stream=False
                ),
            )
            with _shape_checked("ollama"):
                return self._deserialize(data, latency_ms)
        except errors.APIError as e:
            self._invalidate_models_cache_on_404(e)
            raise

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | list[SystemBlock] | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        def start() -> _CancellableStream:
            return _stream_request(
                "ollama",
                *self._build_request(
                    model, messages, system, tools, config, stream=True
                ),
                frames=_iter_ndjson,
                handler=self._stream_handler,
                terminal="a done chunk",
            )

        return _ErrorHookedStream("ollama", start, self._invalidate_models_cache_on_404)

    def _stream_handler(
        self, timer: _StreamTimer
    ) -> Callable[[dict[str, Any]], Iterator[StreamEvent]]:
        """Synthesizes a block lifecycle from Ollama's flat chunks: a new block index
        is allocated whenever the kind of streamed content changes, and each tool
        call is emitted as its own complete block as soon as its chunk arrives. The
        final content holds one block per streamed index, in that order."""
        blocks: list[Block] = []
        parts: dict[int, list[str]] = {}
        open_kind: str | None = None
        tool_calls: list[dict[str, Any]] = []

        def close_open_block() -> int:
            index = len(blocks) - 1
            block = blocks[index]
            joined = "".join(parts.pop(index, ()))
            if block["type"] == "thinking":
                block["thinking"] = joined
            elif block["type"] == "text":
                block["text"] = joined
            return index

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
                        yield {"type": "block_stop", "index": close_open_block()}
                    open_kind = kind
                    if kind == "thinking":
                        blocks.append({"type": "thinking", "thinking": ""})
                    else:
                        blocks.append({"type": "text", "text": ""})
                    parts[len(blocks) - 1] = []
                index = len(blocks) - 1
                parts[index].append(delta)
                if blocks[index]["type"] == "thinking":
                    yield {"type": "thinking_delta", "index": index, "thinking": delta}
                elif blocks[index]["type"] == "text":
                    yield {"type": "text_delta", "index": index, "text": delta}

            for call in message.get("tool_calls") or []:
                if open_kind is not None:
                    yield {"type": "block_stop", "index": close_open_block()}
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
                yield {
                    "type": "tool_use_delta",
                    "index": index,
                    "partial_json": json.dumps(block["input"]),
                }
                yield {"type": "block_stop", "index": index}

            if chunk.get("done"):
                if open_kind is not None:
                    yield {"type": "block_stop", "index": close_open_block()}
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
