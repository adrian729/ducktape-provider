"""Tests for OllamaLocalAdapter: _serialize/_deserialize plus chat() and stream_chat()
with urllib.request.urlopen mocked out. No real network call is ever made."""

import json
import time
import unittest
from typing import Any
from unittest.mock import patch

from http_test_utils import (
    FakeStreamResponse,
    buffered_response,
    final_response,
    http_error,
    ndjson_lines,
)

from ducktape_provider import (
    APIError,
    ContextOverflowError,
    MalformedResponseError,
    Message,
    OllamaLocalAdapter,
    ServerError,
    ToolDef,
    UnsupportedBlockError,
)
from ducktape_provider.adapters import ollama as ollama_module
from ducktape_provider.streaming import _StreamTimer

MESSAGES: list[Message] = [
    {"role": "user", "content": [{"type": "text", "text": "weather in NYC?"}]}
]

FINAL_DATA: dict[str, Any] = {
    "message": {
        "content": "Let me check",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}
        ],
    },
    "done_reason": "stop",
    "prompt_eval_count": 7,
    "eval_count": 5,
}

FINAL_CHUNK_STATS = {
    "done": True,
    "done_reason": "stop",
    "prompt_eval_count": 7,
    "eval_count": 5,
}
WEATHER_CALL = {"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}

STREAM_CHUNKS = [
    {"message": {"role": "assistant", "content": "Let "}, "done": False},
    {"message": {"role": "assistant", "content": "me check"}, "done": False},
    {
        "message": {"role": "assistant", "content": "", "tool_calls": [WEATHER_CALL]},
        **FINAL_CHUNK_STATS,
    },
]

STREAM_CHUNKS_EARLY_TOOL_CALLS = [
    {"message": {"role": "assistant", "content": "Let "}, "done": False},
    {"message": {"role": "assistant", "content": "me check"}, "done": False},
    {
        "message": {"role": "assistant", "content": "", "tool_calls": [WEATHER_CALL]},
        "done": False,
    },
    {"message": {"role": "assistant", "content": ""}, **FINAL_CHUNK_STATS},
]


class OllamaSerializeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    def test_serialize_maps_tool_result_and_tool_use(self):
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {
                        "type": "tool_use",
                        "id": "call_0",
                        "name": "get_weather",
                        "input": {"city": "NYC"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_0",
                        "name": "get_weather",
                        "content": "sunny",
                    }
                ],
            },
        ]
        serialized = self.adapter._serialize(messages, system="be terse")
        self.assertEqual(
            serialized,
            [
                {"role": "system", "content": "be terse"},
                {
                    "role": "assistant",
                    "content": "checking",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "NYC"},
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": "sunny",
                    "tool_name": "get_weather",
                    "tool_use_id": "call_0",
                },
            ],
        )

    def test_serialize_tools(self):
        tools: list[ToolDef] = [
            {"name": "get_weather", "description": "...", "parameters": {"a": 1}}
        ]
        self.assertEqual(
            self.adapter._serialize_tools(tools),
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "...",
                        "parameters": {"a": 1},
                    },
                }
            ],
        )


class OllamaDeserializeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    def test_plain_text_response(self):
        data = {
            "message": {"content": "hi there"},
            "done_reason": "stop",
            "prompt_eval_count": 4,
            "eval_count": 6,
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(response["content"], [{"type": "text", "text": "hi there"}])
        self.assertEqual(response["stop_reason"], "end_turn")
        self.assertEqual(response["usage"], {"input_tokens": 4, "output_tokens": 6})

    def test_tool_call_arguments_parsed_from_json_string(self):
        data = {
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "search", "arguments": '{"q": "cats"}'}}
                ],
            },
            "done_reason": "stop",
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(
            response["content"],
            [
                {
                    "type": "tool_use",
                    "id": "call_0",
                    "name": "search",
                    "input": {"q": "cats"},
                }
            ],
        )
        self.assertEqual(response["stop_reason"], "tool_use")

    def test_length_stop_reason_maps_to_max_tokens(self):
        data = {"message": {"content": "x"}, "done_reason": "length"}
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(response["stop_reason"], "max_tokens")


class OllamaChatHTTPTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    @patch("urllib.request.urlopen")
    def test_chat_returns_deserialized_response(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(json.dumps(FINAL_DATA).encode())

        response = self.adapter.chat("llama3", MESSAGES)

        self.assertEqual(
            response["content"],
            [
                {"type": "text", "text": "Let me check"},
                {
                    "type": "tool_use",
                    "id": "call_0",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
            ],
        )
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(response["usage"], {"input_tokens": 7, "output_tokens": 5})
        self.assertIsInstance(response["latency_ms"], float)
        self.assertGreaterEqual(response["latency_ms"], 0)
        request = mock_urlopen.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/api/chat"))
        sent = json.loads(request.data)
        self.assertEqual(sent["stream"], False)

    @patch("urllib.request.urlopen")
    def test_chat_raises_server_error_on_500(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            "http://127.0.0.1:11434/api/chat", 500, b"boom"
        )

        with self.assertRaises(ServerError) as ctx:
            self.adapter.chat("llama3", MESSAGES)

        self.assertIn("ollama", str(ctx.exception))
        self.assertIn("500", str(ctx.exception))


class OllamaStreamChatHTTPTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    @patch("urllib.request.urlopen")
    def test_stream_chat_yields_expected_event_sequence(self, mock_urlopen):
        for name, chunks in (
            ("tool calls in final chunk", STREAM_CHUNKS),
            ("tool calls in earlier chunk", STREAM_CHUNKS_EARLY_TOOL_CALLS),
        ):
            with self.subTest(name):
                mock_urlopen.return_value = FakeStreamResponse(ndjson_lines(*chunks))

                events = list(self.adapter.stream_chat("llama3", MESSAGES))

                self.assertEqual(
                    [e["type"] for e in events],
                    [
                        "text_delta",
                        "text_delta",
                        "block_stop",
                        "tool_use_start",
                        "tool_use_delta",
                        "block_stop",
                        "message_stop",
                    ],
                )
                self.assertEqual(
                    [e.get("index") for e in events[:-1]], [0, 0, 0, 1, 1, 1]
                )
                self.assertEqual(
                    events[4],
                    {
                        "type": "tool_use_delta",
                        "index": 1,
                        "partial_json": json.dumps({"city": "NYC"}),
                    },
                )
                final = final_response(events)
                buffered = self.adapter._deserialize(FINAL_DATA, 0.0)
                self.assertEqual(final["content"], buffered["content"])
                self.assertEqual(final["stop_reason"], buffered["stop_reason"])
                self.assertEqual(final["usage"], buffered["usage"])
                self.assertIsInstance(final["latency_ms"], float)

    @patch("urllib.request.urlopen")
    def test_stream_chat_raises_server_error_on_500(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            "http://127.0.0.1:11434/api/chat", 500, b"boom"
        )

        with self.assertRaises(ServerError) as ctx:
            list(self.adapter.stream_chat("llama3", MESSAGES))

        self.assertIn("ollama", str(ctx.exception))
        self.assertIn("500", str(ctx.exception))


class OllamaHeaderOverrideTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    def test_config_headers_override_default(self):
        req, _ = self.adapter._build_request(
            "llama3",
            MESSAGES,
            None,
            None,
            {"headers": {"Content-Type": "application/x-custom"}},
            stream=False,
        )
        self.assertEqual(req.get_header("Content-type"), "application/x-custom")

    @patch("urllib.request.urlopen")
    def test_no_header_follows_a_redirect(self, mock_urlopen):
        config = {"headers": {"X-Token": "t"}}
        calls = {
            "chat": (
                lambda: self.adapter.chat("llama3", MESSAGES, config=config),
                lambda: buffered_response(json.dumps(FINAL_DATA).encode()),
            ),
            "stream_chat": (
                lambda: list(
                    self.adapter.stream_chat("llama3", MESSAGES, config=config)
                ),
                lambda: FakeStreamResponse(
                    ndjson_lines({"message": {"content": "hi"}, "done": True})
                ),
            ),
            "models": (
                self.adapter.models,
                lambda: buffered_response(b'{"models": [{"name": "llama3"}]}'),
            ),
        }
        with patch.dict("os.environ", {"OLLAMA_HOST": "http://127.0.0.1:9"}):
            for label, (call, reply) in calls.items():
                with self.subTest(label):
                    mock_urlopen.return_value = reply()
                    call()
                    req = mock_urlopen.call_args.args[0]
                    self.assertEqual(req.headers, {})
                    if label != "models":
                        self.assertEqual(req.get_header("X-token"), "t")


class OllamaUnsupportedBlockTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    @patch("urllib.request.urlopen")
    def test_url_image_block_raises_before_any_network_request(self, mock_urlopen):
        mock_urlopen.side_effect = AssertionError("urlopen should never be called")
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": "url", "url": "https://x/img.png"}
                ],
            }
        ]
        with self.assertRaises(UnsupportedBlockError):
            self.adapter.chat("llama3", messages)
        mock_urlopen.assert_not_called()

    def test_document_block_logs_warning_and_is_dropped(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see attached"},
                    {
                        "type": "document",
                        "source": "base64",
                        "media_type": "application/pdf",
                        "data": "abc",
                    },
                ],
            }
        ]
        with self.assertLogs(ollama_module.logger, level="WARNING"):
            serialized = self.adapter._serialize(messages, system=None)
        self.assertEqual(serialized, [{"role": "user", "content": "see attached"}])

    def test_document_only_message_produces_no_entry(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": "base64",
                        "media_type": "application/pdf",
                        "data": "abc",
                    }
                ],
            }
        ]
        with self.assertLogs(ollama_module.logger, level="WARNING"):
            serialized = self.adapter._serialize(messages, system=None)
        self.assertEqual(serialized, [])


class OllamaStreamContentTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    @patch("urllib.request.urlopen")
    def test_content_kind_change_allocates_new_block_index(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(
            ndjson_lines(
                {"message": {"thinking": "plan"}, "done": False},
                {"message": {"content": "draft"}, "done": False},
                {"message": {"thinking": "reconsider"}, "done": False},
                {"message": {"content": ""}, **FINAL_CHUNK_STATS},
            )
        )

        events = list(self.adapter.stream_chat("llama3", MESSAGES))

        self.assertEqual(
            [(e["type"], e.get("index")) for e in events],
            [
                ("thinking_delta", 0),
                ("block_stop", 0),
                ("text_delta", 1),
                ("block_stop", 1),
                ("thinking_delta", 2),
                ("block_stop", 2),
                ("message_stop", None),
            ],
        )
        final = final_response(events)
        self.assertEqual(
            final["content"],
            [
                {"type": "thinking", "thinking": "plan"},
                {"type": "text", "text": "draft"},
                {"type": "thinking", "thinking": "reconsider"},
            ],
        )
        self.assertEqual(final["raw"]["message"]["thinking"], "planreconsider")

    @patch("urllib.request.urlopen")
    def test_text_around_tool_call_keeps_stream_order(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(
            ndjson_lines(
                {"message": {"content": "a"}, "done": False},
                {"message": {"tool_calls": [WEATHER_CALL]}, "done": False},
                {"message": {"content": "b"}, **FINAL_CHUNK_STATS},
            )
        )

        events = list(self.adapter.stream_chat("llama3", MESSAGES))

        self.assertEqual(
            [(e["type"], e.get("index")) for e in events],
            [
                ("text_delta", 0),
                ("block_stop", 0),
                ("tool_use_start", 1),
                ("tool_use_delta", 1),
                ("block_stop", 1),
                ("text_delta", 2),
                ("block_stop", 2),
                ("message_stop", None),
            ],
        )
        final = final_response(events)
        self.assertEqual(
            final["content"],
            [
                {"type": "text", "text": "a"},
                {
                    "type": "tool_use",
                    "id": "call_0",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
                {"type": "text", "text": "b"},
            ],
        )
        self.assertEqual(final["stop_reason"], "tool_use")

    @patch("urllib.request.urlopen")
    def test_ordinary_stream_matches_buffered_content(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(
            ndjson_lines(
                {"message": {"thinking": "plan"}, "done": False},
                {"message": {"content": "Let me check"}, "done": False},
                {"message": {"tool_calls": [WEATHER_CALL]}, **FINAL_CHUNK_STATS},
            )
        )
        streamed = final_response(list(self.adapter.stream_chat("llama3", MESSAGES)))
        buffered = self.adapter._deserialize(
            {**FINAL_DATA, "message": {**FINAL_DATA["message"], "thinking": "plan"}},
            0.0,
        )
        self.assertEqual(streamed["content"], buffered["content"])

    @patch("urllib.request.urlopen")
    def test_tool_call_arguments_stream_as_the_json_of_the_final_input(
        self, mock_urlopen
    ):
        cases = {
            "missing": ({"name": "f"}, {}),
            "null": ({"name": "f", "arguments": None}, {}),
            "unparseable string": ({"name": "f", "arguments": "{oops"}, {}),
            "non-object string": ({"name": "f", "arguments": "[1]"}, {}),
            "json string": ({"name": "f", "arguments": '{"a": 1}'}, {"a": 1}),
            "object": ({"name": "f", "arguments": {"a": 1}}, {"a": 1}),
        }
        for label, (function, expected_input) in cases.items():
            with self.subTest(label):
                mock_urlopen.return_value = FakeStreamResponse(
                    ndjson_lines(
                        {
                            "message": {"tool_calls": [{"function": function}]},
                            **FINAL_CHUNK_STATS,
                        }
                    )
                )
                events = list(self.adapter.stream_chat("llama3", MESSAGES))
                partial = "".join(
                    e["partial_json"] for e in events if e["type"] == "tool_use_delta"
                )
                self.assertEqual(json.loads(partial), expected_input)
                [block] = final_response(events)["content"]
                self.assertEqual(block, {**block, "input": expected_input})

    @patch("urllib.request.urlopen")
    def test_multiple_tool_calls_across_chunks_get_sequential_ids(self, mock_urlopen):
        other_call = {"function": {"name": "get_time", "arguments": {"tz": "EST"}}}
        mock_urlopen.return_value = FakeStreamResponse(
            ndjson_lines(
                {"message": {"tool_calls": [WEATHER_CALL]}, "done": False},
                {"message": {"tool_calls": [other_call]}, **FINAL_CHUNK_STATS},
            )
        )

        events = list(self.adapter.stream_chat("llama3", MESSAGES))

        starts = [e for e in events if e["type"] == "tool_use_start"]
        self.assertEqual(
            [(e["index"], e["id"]) for e in starts], [(0, "call_0"), (1, "call_1")]
        )
        final = final_response(events)
        self.assertEqual(
            [b["id"] for b in final["content"] if b["type"] == "tool_use"],
            ["call_0", "call_1"],
        )


class OllamaNullUsageTests(unittest.TestCase):
    def test_null_counts_become_zero(self):
        data = {
            "message": {"content": "x"},
            "done_reason": None,
            "prompt_eval_count": None,
            "eval_count": None,
        }
        response = OllamaLocalAdapter()._deserialize(data, 0.0)
        self.assertEqual(response["usage"], {"input_tokens": 0, "output_tokens": 0})


class OllamaStreamErrorTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    def _stream(self, lines, error=None):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = FakeStreamResponse(lines, error)
            return list(self.adapter.stream_chat("llama3", MESSAGES))

    def test_error_chunk_raises_server_error_like_http_500(self):
        lines = ndjson_lines(STREAM_CHUNKS[0], {"error": "model runner crashed"})
        with self.assertRaises(ServerError) as ctx:
            self._stream(lines)
        self.assertEqual(ctx.exception.status, 500)
        self.assertIn("model runner crashed", str(ctx.exception))

    def test_not_found_error_chunk_raises_plain_api_error_like_http_404(self):
        lines = ndjson_lines({"error": "model 'nope' not found"})
        with self.assertRaises(APIError) as ctx:
            self._stream(lines)
        self.assertIs(type(ctx.exception), APIError)
        self.assertEqual(ctx.exception.status, 404)

    def test_wrong_shape_chunks_raise_malformed_api_error(self):
        cases = {
            "non-object chunk": [[1]],
            "non-object message": [{"message": "hi", "done": False}],
            "non-string content": [{"message": {"content": ["x"]}, "done": False}],
            "non-object tool call": [{"message": {"tool_calls": ["f"]}, "done": False}],
        }
        for label, chunks in cases.items():
            with self.subTest(label), self.assertRaises(MalformedResponseError) as ctx:
                self._stream(ndjson_lines(*chunks))
            self.assertIs(type(ctx.exception), MalformedResponseError)
            self.assertIn("malformed", str(ctx.exception))

    def test_context_overflow_error_chunk_raises_context_overflow_error(self):
        lines = ndjson_lines({"error": "input exceeds maximum context length"})
        with self.assertRaises(ContextOverflowError):
            self._stream(lines)

    def test_stream_ending_without_done_raises(self):
        with self.assertRaises(APIError) as ctx:
            self._stream(ndjson_lines(*STREAM_CHUNKS[:2]))
        self.assertIn("done", str(ctx.exception))

    def test_malformed_line_raises_api_error(self):
        with self.assertRaises(MalformedResponseError) as ctx:
            self._stream([*ndjson_lines(STREAM_CHUNKS[0]), b"{truncated\n"])
        self.assertIn("malformed", str(ctx.exception))

    def test_connection_reset_mid_stream_raises_api_error(self):
        with self.assertRaises(APIError):
            self._stream(ndjson_lines(STREAM_CHUNKS[0]), ConnectionResetError())


class OllamaLatencyTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    @patch("urllib.request.urlopen")
    def test_chat_latency_spans_request_to_body_read(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(json.dumps(FINAL_DATA).encode())
        with patch("time.monotonic", side_effect=[10.0, 12.0]):
            response = self.adapter.chat("llama3", MESSAGES)
        self.assertEqual(response["latency_ms"], 2000.0)
        self.assertNotIn("ttft_ms", response)

    @patch("urllib.request.urlopen")
    def test_stream_latency_and_ttft_use_wire_read_times(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(ndjson_lines(*STREAM_CHUNKS))
        with patch("time.monotonic", side_effect=[100.0, 100.25, 100.5, 101.0]):
            events = list(self.adapter.stream_chat("llama3", MESSAGES))
        final = final_response(events)
        self.assertEqual(final["ttft_ms"], 250.0)
        self.assertEqual(final["latency_ms"], 1000.0)

    @patch("urllib.request.urlopen")
    def test_ttft_is_set_when_first_content_arrives_in_the_done_chunk(
        self, mock_urlopen
    ):
        cases = {
            "tool call only in done chunk": [
                {"message": {"content": ""}, "done": False},
                {"message": {"tool_calls": [WEATHER_CALL]}, **FINAL_CHUNK_STATS},
            ],
            "whole reply in one chunk": [
                {"message": {"content": "hello"}, **FINAL_CHUNK_STATS},
            ],
        }
        for label, chunks in cases.items():
            with self.subTest(label):
                mock_urlopen.return_value = FakeStreamResponse(ndjson_lines(*chunks))
                clock = [100.0 + i * 0.5 for i in range(len(chunks) + 1)]
                with patch("time.monotonic", side_effect=clock):
                    events = list(self.adapter.stream_chat("llama3", MESSAGES))
                final = final_response(events)
                self.assertEqual(final["ttft_ms"], len(chunks) * 500.0)
                self.assertEqual(final["latency_ms"], final["ttft_ms"])

    @patch("urllib.request.urlopen")
    def test_stream_without_content_omits_ttft(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(
            ndjson_lines({"message": {"content": ""}, **FINAL_CHUNK_STATS})
        )
        events = list(self.adapter.stream_chat("llama3", MESSAGES))
        self.assertNotIn("ttft_ms", final_response(events))


class OllamaReservedConfigTests(unittest.TestCase):
    @patch("urllib.request.urlopen", side_effect=AssertionError("request sent"))
    def test_reserved_config_keys_are_rejected_before_any_request(self, mock_urlopen):
        adapter = OllamaLocalAdapter()
        for key in ("stream", "model", "messages"):
            with self.subTest(key=key), self.assertRaises(ValueError) as ctx:
                adapter.chat("llama3", MESSAGES, config={key: True})
            self.assertIn(key, str(ctx.exception))
        mock_urlopen.assert_not_called()


class OllamaInvalidToolArgsTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    @patch("urllib.request.urlopen")
    def test_unparseable_arguments_string_falls_back_to_empty_input_in_chat(
        self, mock_urlopen
    ):
        data = {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "f", "arguments": "{oops"}}],
            },
            "done_reason": "stop",
        }
        mock_urlopen.return_value = buffered_response(json.dumps(data).encode())
        response = self.adapter.chat("llama3", MESSAGES)
        self.assertEqual(
            response["content"],
            [{"type": "tool_use", "id": "call_0", "name": "f", "input": {}}],
        )

    @patch("urllib.request.urlopen")
    def test_deeply_nested_arguments_fall_back_to_empty_input_in_chat(
        self, mock_urlopen
    ):
        deep = "[" * 200_000 + "]" * 200_000
        data = {
            "message": {
                "content": "",
                "tool_calls": [{"function": {"name": "f", "arguments": deep}}],
            },
            "done_reason": "stop",
        }
        mock_urlopen.return_value = buffered_response(json.dumps(data).encode())
        response = self.adapter.chat("llama3", MESSAGES)
        self.assertEqual(
            response["content"],
            [{"type": "tool_use", "id": "call_0", "name": "f", "input": {}}],
        )

    @patch("urllib.request.urlopen")
    def test_deeply_nested_arguments_fall_back_to_empty_input_in_stream(
        self, mock_urlopen
    ):
        deep = "[" * 200_000 + "]" * 200_000
        chunk = {
            "message": {"tool_calls": [{"function": {"name": "f", "arguments": deep}}]},
            **FINAL_CHUNK_STATS,
        }
        mock_urlopen.return_value = FakeStreamResponse(ndjson_lines(chunk))
        response = final_response(list(self.adapter.stream_chat("llama3", MESSAGES)))
        self.assertEqual(
            response["content"],
            [{"type": "tool_use", "id": "call_0", "name": "f", "input": {}}],
        )


class OllamaStreamAccumulationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    def test_long_streams_accumulate_in_linear_time(self):
        n, delta = 4000, "x" * 16 * 1024
        for field, block_type in [("thinking", "thinking"), ("content", "text")]:
            with self.subTest(field):
                handle = self.adapter._stream_handler(_StreamTimer())
                chunk = {"message": {"role": "assistant", field: delta}, "done": False}
                start = time.perf_counter()
                for _ in range(n):
                    list(handle(chunk))
                response = final_response(
                    list(
                        handle(
                            {
                                "message": {"role": "assistant", "content": ""},
                                "done": True,
                            }
                        )
                    )
                )
                self.assertLess(time.perf_counter() - start, 1.0)
                [block] = response["content"]
                self.assertEqual(block["type"], block_type)

    def test_accumulated_text_is_correct(self):
        n = 1000
        handle = self.adapter._stream_handler(_StreamTimer())
        for _ in range(n):
            list(
                handle(
                    {"message": {"role": "assistant", "content": "ab"}, "done": False}
                )
            )
        response = final_response(
            list(
                handle({"message": {"role": "assistant", "content": ""}, "done": True})
            )
        )
        self.assertEqual(response["content"], [{"type": "text", "text": "ab" * n}])


class OllamaModelsCacheInvalidationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()
        self.adapter._models_cache = {"llama-old"}
        self.adapter._cache_time = time.monotonic()

    @patch("urllib.request.urlopen")
    def test_chat_404_invalidates_models_cache(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            "http://127.0.0.1:11434/api/chat", 404, b"not found"
        )
        with self.assertRaises(APIError):
            self.adapter.chat("llama3", MESSAGES)
        self.assertIsNone(self.adapter._models_cache)

    @patch("urllib.request.urlopen")
    def test_stream_chat_404_invalidates_models_cache(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            "http://127.0.0.1:11434/api/chat", 404, b"not found"
        )
        with self.assertRaises(APIError):
            list(self.adapter.stream_chat("llama3", MESSAGES))
        self.assertIsNone(self.adapter._models_cache)

    @patch("urllib.request.urlopen")
    def test_non_404_error_leaves_models_cache_untouched(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            "http://127.0.0.1:11434/api/chat", 500, b"boom"
        )
        with self.assertRaises(APIError):
            self.adapter.chat("llama3", MESSAGES)
        self.assertEqual(self.adapter._models_cache, {"llama-old"})


if __name__ == "__main__":
    unittest.main()
