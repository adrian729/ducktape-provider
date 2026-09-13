import json
import unittest
from collections.abc import Generator
from typing import cast
from unittest.mock import patch

from http_test_utils import (
    FakeStreamResponse,
    buffered_response,
    ndjson_lines,
    sse_lines,
)

from ducktape_provider import (
    Adapter,
    APIError,
    ClaudeAdapter,
    MalformedResponseError,
    Message,
    OllamaLocalAdapter,
    OpenAIAdapter,
    Provider,
    StreamEvent,
    ToolDef,
    streaming,
)

TEXT_MESSAGES: list[Message] = [
    {"role": "user", "content": [{"type": "text", "text": "hi there"}]}
]
TOOL_MESSAGES: list[Message] = [
    {"role": "user", "content": [{"type": "text", "text": "weather in NYC?"}]}
]
WEATHER_TOOL: list[ToolDef] = [
    {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
]

CLAUDE_TEXT_DATA = {
    "content": [{"type": "text", "text": "hello"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}
CLAUDE_TOOL_DATA = {
    "content": [
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "NYC"},
        }
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 10, "output_tokens": 8},
}
CLAUDE_STREAM_EVENTS = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 5}}},
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hello"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 3},
    },
    {"type": "message_stop"},
]

OPENAI_TEXT_DATA = {
    "status": "completed",
    "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "hello"}]}
    ],
    "usage": {"input_tokens": 5, "output_tokens": 3},
}
OPENAI_TOOL_DATA = {
    "status": "completed",
    "output": [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "NYC"}',
        }
    ],
    "usage": {"input_tokens": 10, "output_tokens": 8},
}
OPENAI_STREAM_EVENTS = [
    {
        "type": "response.output_item.added",
        "item": {"id": "msg_1", "type": "message"},
    },
    {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "delta": "hello",
    },
    {
        "type": "response.output_item.done",
        "item": {"id": "msg_1", "type": "message"},
    },
    {"type": "response.completed", "response": OPENAI_TEXT_DATA},
]

OLLAMA_TEXT_DATA = {
    "message": {"content": "hello"},
    "done_reason": "stop",
    "prompt_eval_count": 5,
    "eval_count": 3,
}
OLLAMA_TOOL_DATA = {
    "message": {
        "content": "",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}
        ],
    },
    "done_reason": "stop",
    "prompt_eval_count": 10,
    "eval_count": 8,
}
OLLAMA_STREAM_CHUNKS = [
    {"message": {"role": "assistant", "content": "hello"}, "done": False},
    {
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 3,
    },
]


class _ConformanceBase(unittest.TestCase):
    def _assert_valid(self, response):
        self.assertEqual(response["stop_reason"], "end_turn")
        text_blocks = [b for b in response["content"] if b["type"] == "text"]
        self.assertGreaterEqual(len(text_blocks), 1)
        self.assertIsInstance(response["usage"]["input_tokens"], int)
        self.assertIsInstance(response["usage"]["output_tokens"], int)


class PlainTextChatTests(_ConformanceBase):
    @patch("urllib.request.urlopen")
    def test_claude(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(CLAUDE_TEXT_DATA).encode()
        )
        provider = Provider(adapters={"claude": ClaudeAdapter()})
        response = provider.chat("claude-x", TEXT_MESSAGES, provider="claude")
        self._assert_valid(response)

    @patch("urllib.request.urlopen")
    def test_openai(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(OPENAI_TEXT_DATA).encode()
        )
        provider = Provider(adapters={"openai": OpenAIAdapter()})
        response = provider.chat("gpt-x", TEXT_MESSAGES, provider="openai")
        self._assert_valid(response)

    @patch("urllib.request.urlopen")
    def test_ollama(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(OLLAMA_TEXT_DATA).encode()
        )
        provider = Provider(adapters={"ollama-local": OllamaLocalAdapter()})
        response = provider.chat("llama3", TEXT_MESSAGES, provider="ollama-local")
        self._assert_valid(response)


class ToolUseRoundTripTests(_ConformanceBase):
    def _assert_tool_use(self, response):
        self.assertEqual(response["stop_reason"], "tool_use")
        tool_blocks = [b for b in response["content"] if b["type"] == "tool_use"]
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0]["name"], "get_weather")
        self.assertEqual(tool_blocks[0]["input"], {"city": "NYC"})

    @patch("urllib.request.urlopen")
    def test_claude(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(CLAUDE_TOOL_DATA).encode()
        )
        provider = Provider(adapters={"claude": ClaudeAdapter()})
        response = provider.chat(
            "claude-x", TOOL_MESSAGES, tools=WEATHER_TOOL, provider="claude"
        )
        self._assert_tool_use(response)

    @patch("urllib.request.urlopen")
    def test_openai(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(OPENAI_TOOL_DATA).encode()
        )
        provider = Provider(adapters={"openai": OpenAIAdapter()})
        response = provider.chat(
            "gpt-x", TOOL_MESSAGES, tools=WEATHER_TOOL, provider="openai"
        )
        self._assert_tool_use(response)

    @patch("urllib.request.urlopen")
    def test_ollama(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(OLLAMA_TOOL_DATA).encode()
        )
        provider = Provider(adapters={"ollama-local": OllamaLocalAdapter()})
        response = provider.chat(
            "llama3", TOOL_MESSAGES, tools=WEATHER_TOOL, provider="ollama-local"
        )
        self._assert_tool_use(response)


class StreamingChatTests(_ConformanceBase):
    def _assert_stream(self, events):
        self.assertTrue(any(e["type"] == "text_delta" for e in events))
        final = [e for e in events if e["type"] == "message_stop"]
        self.assertEqual(len(final), 1)
        self._assert_valid(final[0]["response"])
        self.assertIsInstance(final[0]["response"]["ttft_ms"], float)
        self.assertLessEqual(
            final[0]["response"]["ttft_ms"], final[0]["response"]["latency_ms"]
        )

    @patch("urllib.request.urlopen")
    def test_claude(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*CLAUDE_STREAM_EVENTS))
        provider = Provider(adapters={"claude": ClaudeAdapter()})
        events = list(
            provider.stream_chat("claude-x", TEXT_MESSAGES, provider="claude")
        )
        self._assert_stream(events)

    @patch("urllib.request.urlopen")
    def test_openai(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*OPENAI_STREAM_EVENTS))
        provider = Provider(adapters={"openai": OpenAIAdapter()})
        events = list(provider.stream_chat("gpt-x", TEXT_MESSAGES, provider="openai"))
        self._assert_stream(events)

    @patch("urllib.request.urlopen")
    def test_ollama(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(
            ndjson_lines(*OLLAMA_STREAM_CHUNKS)
        )
        provider = Provider(adapters={"ollama-local": OllamaLocalAdapter()})
        events = list(
            provider.stream_chat("llama3", TEXT_MESSAGES, provider="ollama-local")
        )
        self._assert_stream(events)


ADAPTERS = {
    "claude": (ClaudeAdapter, "ANTHROPIC_API_KEY"),
    "openai": (OpenAIAdapter, "OPENAI_API_KEY"),
    "ollama-local": (OllamaLocalAdapter, None),
}


def _calls(adapter, config):
    return {
        "chat": lambda: adapter.chat("m", TEXT_MESSAGES, config=config),
        "stream_chat": lambda: list(
            adapter.stream_chat("m", TEXT_MESSAGES, config=config)
        ),
    }


class RequestValidationTests(unittest.TestCase):
    """Client-side mistakes surface as ValueError before any request, on every
    backend, and never quote a header value (which may be a credential)."""

    @patch("urllib.request.urlopen")
    def test_invalid_config_raises_value_error_not_api_error(self, mock_urlopen):
        configs = {
            "negative timeout": {"timeout": -1},
            "zero timeout": {"timeout": 0},
            "string timeout": {"timeout": "5"},
            "bool timeout": {"timeout": True},
            "infinite timeout": {"timeout": float("inf")},
            "NaN timeout": {"timeout": float("nan")},
            # Finite, but overflows the socket layer's clock inside urlopen.
            "huge timeout": {"timeout": 1e300},
            "huge int timeout": {"timeout": 10**400},
            "newline in header": {"headers": {"x-token": "SECRET\ninjected: 1"}},
            "trailing CRLF": {"headers": {"x-token": "SECRET\r\n"}},
            "non-latin-1 header": {"headers": {"x-token": "SECRET\u2603"}},
            "bad header name": {"headers": {"x token:": "SECRET"}},
        }
        for name, (cls, _) in ADAPTERS.items():
            for label, config in configs.items():
                for call_name, call in _calls(cls(), config).items():
                    with self.subTest(name, label=label, call=call_name):
                        with self.assertRaises(ValueError) as ctx:
                            call()
                        self.assertNotIsInstance(ctx.exception, APIError)
                        self.assertNotIn("SECRET", str(ctx.exception))
                        self.assertIsNone(ctx.exception.__context__)
        mock_urlopen.assert_not_called()

    def test_timeout_none_means_no_timeout_and_absent_means_default(self):
        for name, (cls, _) in ADAPTERS.items():
            adapter = cls()
            for config, expected in (
                ({"timeout": None}, None),
                ({}, cls._CHAT_TIMEOUT),
                ({"timeout": 2.5}, 2.5),
            ):
                with self.subTest(name, config=config):
                    _, timeout = adapter._build_request(
                        "m", TEXT_MESSAGES, None, None, config, stream=False
                    )
                    self.assertEqual(timeout, expected)

    @patch("urllib.request.urlopen")
    def test_timeout_none_reaches_urlopen(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(OLLAMA_TEXT_DATA).encode()
        )
        OllamaLocalAdapter().chat("m", TEXT_MESSAGES, config={"timeout": None})
        self.assertIsNone(mock_urlopen.call_args.kwargs["timeout"])


STREAM_FIXTURES: dict[str, tuple[Adapter, list[bytes]]] = {
    "claude": (ClaudeAdapter(), sse_lines(*CLAUDE_STREAM_EVENTS)),
    "openai": (OpenAIAdapter(), sse_lines(*OPENAI_STREAM_EVENTS)),
    "ollama-local": (OllamaLocalAdapter(), ndjson_lines(*OLLAMA_STREAM_CHUNKS)),
}


class ConsumerExceptionTests(unittest.TestCase):
    """An exception the consumer throws into a stream is the consumer's, so it
    propagates unchanged rather than being relabeled as a vendor failure."""

    def test_thrown_exceptions_propagate_unchanged(self):
        for name, (adapter, lines) in STREAM_FIXTURES.items():
            for thrown in (ConnectionResetError("consumer"), KeyError("consumer")):
                with (
                    self.subTest(name, thrown=type(thrown).__name__),
                    patch("urllib.request.urlopen") as mock_urlopen,
                ):
                    mock_urlopen.return_value = FakeStreamResponse(lines)
                    # Concrete adapters return generators; the Adapter ABC only
                    # promises an Iterator, which has no throw().
                    stream = cast(
                        Generator[StreamEvent],
                        adapter.stream_chat("m", TEXT_MESSAGES),
                    )
                    next(stream)
                    with self.assertRaises(type(thrown)) as ctx:
                        stream.throw(thrown)
                    self.assertIs(ctx.exception, thrown)


class MalformedBodyTests(unittest.TestCase):
    """Unparseable bodies raise MalformedResponseError on every backend and path,
    never a raw RecursionError or an unbounded read."""

    def test_deeply_nested_json(self):
        deep = b"[" * 200_000 + b"]" * 200_000
        for name, (adapter, _) in STREAM_FIXTURES.items():
            with (
                self.subTest(name, path="chat"),
                patch("urllib.request.urlopen", return_value=buffered_response(deep)),
                self.assertRaises(MalformedResponseError),
            ):
                adapter.chat("m", TEXT_MESSAGES)
            with (
                self.subTest(name, path="stream_chat"),
                patch(
                    "urllib.request.urlopen",
                    return_value=FakeStreamResponse(
                        [b"data: " + deep + b"\n\n" if name != "ollama-local" else deep]
                    ),
                ),
                self.assertRaises(MalformedResponseError),
            ):
                list(adapter.stream_chat("m", TEXT_MESSAGES))

    def test_oversized_chat_body(self):
        for name, (adapter, _) in STREAM_FIXTURES.items():
            body = buffered_response(json.dumps({"pad": "x" * 1000}).encode())
            with (
                self.subTest(name),
                patch("urllib.request.urlopen", return_value=body),
                patch.object(streaming, "_MAX_EVENT_BYTES", 100),
                self.assertRaises(MalformedResponseError) as ctx,
            ):
                adapter.chat("m", TEXT_MESSAGES)
            self.assertIn("exceeds", str(ctx.exception))


class TruncatedStreamTests(unittest.TestCase):
    """A stream cut off before its terminal event must raise on every backend,
    never end with a message_stop built from partial data."""

    def _assert_raises(self, provider_name, adapter, model, lines):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = FakeStreamResponse(lines)
            provider = Provider(adapters={provider_name: adapter})
            with self.assertRaises(APIError):
                list(provider.stream_chat(model, TEXT_MESSAGES, provider=provider_name))

    def test_claude(self):
        self._assert_raises(
            "claude", ClaudeAdapter(), "claude-x", sse_lines(*CLAUDE_STREAM_EVENTS[:-1])
        )

    def test_openai(self):
        self._assert_raises(
            "openai", OpenAIAdapter(), "gpt-x", sse_lines(*OPENAI_STREAM_EVENTS[:-1])
        )

    def test_ollama(self):
        self._assert_raises(
            "ollama-local",
            OllamaLocalAdapter(),
            "llama3",
            ndjson_lines(*OLLAMA_STREAM_CHUNKS[:-1]),
        )


if __name__ == "__main__":
    unittest.main()
