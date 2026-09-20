"""End-to-end adapter tests against a real HTTP server on 127.0.0.1.

Unlike the urlopen fakes, these go through urllib and http.client for real:
chunked transfer decoding, readline() over chunk boundaries, Content-Length
bodies, and HTTPError construction. The server binds an ephemeral port, so
nothing ever reaches a vendor or a locally running Ollama daemon.
"""

import json
import os
import unittest
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import Mock, patch

from http_test_utils import (
    LocalServer,
    RecordedRequest,
    Reply,
    final_response,
    split_every,
)

from ducktape_provider import (
    Adapter,
    ClaudeAdapter,
    Message,
    OllamaLocalAdapter,
    OpenAIAdapter,
    RateLimitError,
    StreamEvent,
)

MESSAGES: list[Message] = [
    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
]

SPLIT = 7


def sse(*events: dict[str, Any]) -> bytes:
    return b"".join(
        f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events
    )


def ndjson(*chunks: dict[str, Any]) -> bytes:
    return b"".join(json.dumps(c).encode() + b"\n" for c in chunks)


@contextmanager
def serving(adapter: Adapter, *replies: Reply) -> Iterator[LocalServer]:
    """Runs a LocalServer and points `adapter` at it instead of the vendor."""
    with ExitStack() as stack:
        server = stack.enter_context(LocalServer(*replies))
        env = {"no_proxy": "*", "NO_PROXY": "*", "OLLAMA_HOST": server.url}
        stack.enter_context(patch.dict(os.environ, env))
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
            os.environ.pop(var, None)
        if isinstance(adapter, ClaudeAdapter):
            stack.enter_context(
                patch.object(adapter, "_MESSAGES_URL", f"{server.url}/v1/messages")
            )
        elif isinstance(adapter, OpenAIAdapter):
            stack.enter_context(
                patch.object(adapter, "_RESPONSES_URL", f"{server.url}/v1/responses")
            )
            stack.enter_context(
                patch.object(adapter, "_EMBEDDINGS_URL", f"{server.url}/v1/embeddings")
            )
        if isinstance(adapter, (ClaudeAdapter, OpenAIAdapter)):
            stack.enter_context(
                patch.object(adapter, "_MODELS_URL", f"{server.url}/v1/models")
            )
        yield server


def header(request: RecordedRequest, name: str) -> str | None:
    """A received header, matched case-insensitively."""
    return next(
        (v for k, v in request.headers.items() if k.lower() == name.lower()), None
    )


CLAUDE_BODY = {
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}
CLAUDE_STREAM = sse(
    {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
    {"type": "ping"},
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hel"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "lo"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '{"a": '},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": "1}"},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {"output_tokens": 7},
    },
    {"type": "message_stop"},
)

OPENAI_BODY = {
    "status": "completed",
    "output": [
        {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [
                {"type": "summary_text", "text": "a"},
                {"type": "summary_text", "text": "b"},
            ],
        },
        {
            "id": "msg_1",
            "type": "message",
            "content": [
                {"type": "output_text", "text": "one"},
                {"type": "output_text", "text": "two"},
            ],
        },
    ],
    "usage": {"input_tokens": 3, "output_tokens": 4},
}
OPENAI_STREAM = sse(
    {"type": "response.created", "response": {}},
    {"type": "response.output_item.added", "item": {"id": "rs_1", "type": "reasoning"}},
    {
        "type": "response.reasoning_summary_text.delta",
        "item_id": "rs_1",
        "summary_index": 0,
        "delta": "a",
    },
    {
        "type": "response.reasoning_summary_text.delta",
        "item_id": "rs_1",
        "summary_index": 1,
        "delta": "b",
    },
    {"type": "response.output_item.done", "item": {"id": "rs_1", "type": "reasoning"}},
    {"type": "response.output_item.added", "item": {"id": "msg_1", "type": "message"}},
    *(
        event
        for i, text in enumerate(["one", "two"])
        for event in (
            {
                "type": "response.content_part.added",
                "item_id": "msg_1",
                "content_index": i,
                "part": {"type": "output_text", "text": ""},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "content_index": i,
                "delta": text,
            },
        )
    ),
    {"type": "response.output_item.done", "item": {"id": "msg_1", "type": "message"}},
    {"type": "response.completed", "response": OPENAI_BODY},
)

OLLAMA_BODY = {
    "message": {"content": "hello"},
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 3,
    "eval_count": 5,
}
OLLAMA_STREAM = ndjson(
    {"message": {"role": "assistant", "content": "", "thinking": "hm"}, "done": False},
    {"message": {"role": "assistant", "content": "hel"}, "done": False},
    {"message": {"role": "assistant", "content": "lo"}, "done": False},
    {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
        },
        "done": False,
    },
    {
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 3,
        "eval_count": 5,
    },
)


class BufferedChatTests(unittest.TestCase):
    def test_every_adapter_reads_length_delimited_and_chunked_bodies(self):
        cases: list[tuple[Adapter, dict[str, Any], str, str]] = [
            (ClaudeAdapter(api_key="test"), CLAUDE_BODY, "/v1/messages", "hello"),
            (OpenAIAdapter(api_key="test"), OPENAI_BODY, "/v1/responses", "one"),
            (OllamaLocalAdapter(), OLLAMA_BODY, "/api/chat", "hello"),
        ]
        for adapter, body, path, first_text in cases:
            payload = json.dumps(body).encode()
            for chunked in (False, True):
                with (
                    self.subTest(type(adapter).__name__, chunked=chunked),
                    serving(
                        adapter,
                        Reply(split_every(payload, SPLIT), chunked=chunked),
                    ) as server,
                ):
                    response = adapter.chat("m", MESSAGES)
                    self.assertIn(
                        {"type": "text", "text": first_text}, response["content"]
                    )
                    [request] = server.requests
                    self.assertEqual(request.path, path)
                    self.assertIs(request.body["stream"], False)

    def test_http_error_keeps_status_retry_after_and_body(self):
        adapter = OllamaLocalAdapter()
        reply = Reply(
            [b'{"error": "slow down"}'],
            status=429,
            headers={"Retry-After": "7"},
            chunked=False,
        )
        with serving(adapter, reply), self.assertRaises(RateLimitError) as ctx:
            adapter.chat("m", MESSAGES)
        self.assertEqual(ctx.exception.status, 429)
        self.assertEqual(ctx.exception.retry_after, 7.0)
        self.assertEqual(ctx.exception.body, '{"error": "slow down"}')


class StreamingChatTests(unittest.TestCase):
    def _stream(self, adapter: Adapter, body: bytes) -> list[StreamEvent]:
        with serving(adapter, Reply(split_every(body, SPLIT))) as server:
            events = list(adapter.stream_chat("m", MESSAGES))
        [request] = server.requests
        self.assertIs(request.body["stream"], True)
        return events

    def test_claude_events_split_across_http_chunks(self):
        events = self._stream(ClaudeAdapter(api_key="test"), CLAUDE_STREAM)
        self.assertEqual(
            [(e["type"], e.get("index")) for e in events],
            [
                ("text_delta", 0),
                ("text_delta", 0),
                ("block_stop", 0),
                ("tool_use_start", 1),
                ("tool_use_delta", 1),
                ("tool_use_delta", 1),
                ("block_stop", 1),
                ("message_stop", None),
            ],
        )
        response = final_response(events)
        self.assertEqual(
            response["content"],
            [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "t1", "name": "f", "input": {"a": 1}},
            ],
        )
        self.assertIn("ttft_ms", response)

    def test_openai_multi_part_message_matches_buffered_content(self):
        adapter = OpenAIAdapter(api_key="test")
        events = self._stream(adapter, OPENAI_STREAM)
        self.assertEqual(
            events[:-1],
            [
                {"type": "thinking_delta", "index": 0, "thinking": "a"},
                {"type": "thinking_delta", "index": 0, "thinking": "\nb"},
                {"type": "block_stop", "index": 0},
                {"type": "text_delta", "index": 1, "text": "one"},
                {"type": "text_delta", "index": 2, "text": "two"},
                {"type": "block_stop", "index": 1},
                {"type": "block_stop", "index": 2},
            ],
        )
        with serving(adapter, Reply([json.dumps(OPENAI_BODY).encode()])):
            buffered = adapter.chat("m", MESSAGES)
        response = final_response(events)
        self.assertEqual(response["content"], buffered["content"])
        self.assertIn("ttft_ms", response)

    def test_ollama_chunks_split_across_http_chunks(self):
        events = self._stream(OllamaLocalAdapter(), OLLAMA_STREAM)
        response = final_response(events)
        self.assertEqual(
            response["content"],
            [
                {"type": "thinking", "thinking": "hm"},
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "call_0", "name": "f", "input": {"a": 1}},
            ],
        )
        self.assertEqual(response["usage"], {"input_tokens": 3, "output_tokens": 5})

    def test_ollama_ttft_when_whole_reply_is_the_done_chunk(self):
        cases = {
            "text": {"content": "hello"},
            "tool call": {"tool_calls": [{"function": {"name": "f"}}]},
        }
        for label, message in cases.items():
            with self.subTest(label):
                body = ndjson({"message": message, "done": True, "done_reason": "stop"})
                response = final_response(self._stream(OllamaLocalAdapter(), body))
                self.assertIsInstance(response.get("ttft_ms"), float)


def body(data: dict[str, Any]) -> Reply:
    return Reply([json.dumps(data).encode()], chunked=False)


KEYED_CASES: list[
    tuple[type[ClaudeAdapter] | type[OpenAIAdapter], str, str, bytes, dict[str, Any]]
] = [
    (ClaudeAdapter, "x-api-key", "", CLAUDE_STREAM, CLAUDE_BODY),
    (OpenAIAdapter, "authorization", "Bearer ", OPENAI_STREAM, OPENAI_BODY),
]


class ApiKeyWireTests(unittest.TestCase):
    """What actually goes out on the wire for an explicit key."""

    def test_explicit_key_reaches_chat_stream_and_models_instead_of_env(self):
        for cls, name, prefix, stream, chat_body in KEYED_CASES:
            for source in ("sk-explicit", lambda: "sk-explicit"):
                adapter = cls(api_key=source)
                replies = (
                    body(chat_body),
                    Reply([stream]),
                    body({"data": [{"id": "gpt-x"}]}),
                )
                with (
                    self.subTest(cls.__name__, source=type(source).__name__),
                    serving(adapter, *replies) as server,
                    patch.dict(
                        os.environ,
                        {"ANTHROPIC_API_KEY": "env", "OPENAI_API_KEY": "env"},
                    ),
                ):
                    adapter.chat("m", MESSAGES)
                    list(adapter.stream_chat("m", MESSAGES))
                    adapter.models()
                    self.assertEqual(
                        [header(r, name) for r in server.requests],
                        [f"{prefix}sk-explicit"] * 3,
                    )

    def test_function_source_is_called_once_per_request(self):
        for cls, name, prefix, _, chat_body in KEYED_CASES:
            source = Mock(side_effect=["k1", "k2"])
            adapter = cls(api_key=source)
            with (
                self.subTest(cls.__name__),
                serving(adapter, body(chat_body), body(chat_body)) as server,
            ):
                adapter.chat("m", MESSAGES)
                adapter.chat("m", MESSAGES)
            self.assertEqual(
                [header(r, name) for r in server.requests],
                [f"{prefix}k1", f"{prefix}k2"],
            )

    def test_claude_models_pagination_calls_source_once(self):
        source = Mock(return_value="k")
        adapter = ClaudeAdapter(api_key=source)
        pages = (
            body({"data": [{"id": "a"}], "has_more": True, "last_id": "a"}),
            body({"data": [{"id": "b"}], "has_more": False}),
        )
        with serving(adapter, *pages) as server:
            self.assertEqual(adapter.models(), {"a", "b"})
        source.assert_called_once_with()
        self.assertEqual([header(r, "x-api-key") for r in server.requests], ["k", "k"])

    def test_source_exception_propagates_out_of_models(self):
        for cls, *_ in KEYED_CASES:
            adapter = cls(api_key=Mock(side_effect=ConnectionError("vault down")))
            with (
                self.subTest(cls.__name__),
                serving(adapter) as server,
                self.assertRaises(ConnectionError),
            ):
                adapter.models()
            self.assertEqual(server.requests, [])


class RedirectTests(unittest.TestCase):
    """A redirect, even to another host, never carries the key or config headers."""

    def test_no_header_follows_a_redirect(self):
        cases: list[tuple[Adapter, dict[str, Any]]] = [
            (ClaudeAdapter(api_key="sk-secret"), CLAUDE_BODY),
            (OpenAIAdapter(api_key="sk-secret"), OPENAI_BODY),
            (OllamaLocalAdapter(), OLLAMA_BODY),
        ]
        config = {"headers": {"X-Custom": "config-value"}}
        for adapter, final_body in cases:
            with (
                self.subTest(type(adapter).__name__),
                LocalServer(body(final_body)) as target,
                serving(
                    adapter,
                    Reply(
                        status=302,
                        headers={"Location": f"{target.url}/elsewhere"},
                        chunked=False,
                    ),
                ) as origin,
            ):
                adapter.chat("m", MESSAGES, config=config)
            [first] = origin.requests
            [second] = target.requests
            self.assertEqual(header(first, "x-custom"), "config-value")
            self.assertEqual(second.path, "/elsewhere")
            for name in ("x-api-key", "authorization", "x-custom", "anthropic-version"):
                self.assertIsNone(header(second, name), name)


if __name__ == "__main__":
    unittest.main()
