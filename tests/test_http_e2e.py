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
from unittest.mock import patch

from http_test_utils import LocalServer, Reply, final_response, split_every

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

# Small enough that most SSE lines and NDJSON chunks straddle several HTTP chunks.
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
        # A proxy from the environment must not intercept loopback requests.
        env = {"no_proxy": "*", "NO_PROXY": "*", "OLLAMA_HOST": server.url}
        stack.enter_context(patch.dict(os.environ, env))
        if isinstance(adapter, ClaudeAdapter):
            stack.enter_context(
                patch.object(adapter, "_MESSAGES_URL", f"{server.url}/v1/messages")
            )
        elif isinstance(adapter, OpenAIAdapter):
            stack.enter_context(
                patch.object(adapter, "_RESPONSES_URL", f"{server.url}/v1/responses")
            )
        yield server


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
            (ClaudeAdapter(), CLAUDE_BODY, "/v1/messages", "hello"),
            (OpenAIAdapter(), OPENAI_BODY, "/v1/responses", "one"),
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
        events = self._stream(ClaudeAdapter(), CLAUDE_STREAM)
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
        adapter = OpenAIAdapter()
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


if __name__ == "__main__":
    unittest.main()
