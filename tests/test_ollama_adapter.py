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
    ollama_embed_response,
    request_body,
)
from test_provider_api_keys import clean_env, pointed_at, secret_headers

from ducktape_provider import (
    APIError,
    ContextOverflowError,
    MalformedResponseError,
    Message,
    OllamaLocalAdapter,
    Provider,
    RequestTimeoutError,
    ServerError,
    SystemBlock,
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

    def test_tool_result_with_list_content_joins_text_sub_blocks(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_0",
                        "name": "search",
                        "content": [
                            {"type": "text", "text": "first"},
                            {"type": "text", "text": "second"},
                        ],
                    }
                ],
            }
        ]
        serialized = self.adapter._serialize(messages, system=None)
        self.assertEqual(serialized[0]["content"], "first\nsecond")

    def test_tool_result_is_error_with_list_content_prefixes_joined_text(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_0",
                        "name": "search",
                        "content": [{"type": "text", "text": "failed"}],
                        "is_error": True,
                    }
                ],
            }
        ]
        serialized = self.adapter._serialize(messages, system=None)
        self.assertEqual(serialized[0]["content"], "ERROR: failed")

    def test_tool_result_is_error_with_str_content_still_prefixes(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_0",
                        "name": "get_weather",
                        "content": "sunny",
                        "is_error": True,
                    }
                ],
            }
        ]
        serialized = self.adapter._serialize(messages, system=None)
        self.assertEqual(serialized[0]["content"], "ERROR: sunny")

    def test_tool_result_empty_list_content_becomes_empty_string(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_0",
                        "name": "search",
                        "content": [],
                    }
                ],
            }
        ]
        serialized = self.adapter._serialize(messages, system=None)
        self.assertEqual(serialized[0]["content"], "")

    def test_system_blocks_flatten_to_system_message(self):
        system: list[SystemBlock] = [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
        ]
        serialized = self.adapter._serialize(MESSAGES, system)
        self.assertEqual(serialized[0], {"role": "system", "content": "a\nb"})

    def test_cache_control_is_not_serialized(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
        self.assertNotIn("cache_control", self.adapter._serialize(messages, None)[0])


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
    def test_base_url_strips_trailing_slash(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(json.dumps(FINAL_DATA).encode())

        with patch.dict("os.environ", {"OLLAMA_HOST": "http://127.0.0.1:11434/"}):
            self.adapter.chat("llama3", MESSAGES)

        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")

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

    def test_image_in_tool_result_content_raises_before_any_network_request(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = AssertionError("urlopen should never be called")
            messages: list[Message] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_0",
                            "name": "screenshot",
                            "content": [
                                {"type": "text", "text": "here"},
                                {
                                    "type": "image",
                                    "source": "base64",
                                    "media_type": "image/png",
                                    "data": "abc",
                                },
                            ],
                        }
                    ],
                }
            ]
            with self.assertRaises(UnsupportedBlockError):
                self.adapter.chat("llama3", messages)
            mock_urlopen.assert_not_called()


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
            "missing": ({"name": "f"}, {}, False),
            "null": ({"name": "f", "arguments": None}, {}, False),
            "unparseable string": ({"name": "f", "arguments": "{oops"}, {}, True),
            "non-object string": ({"name": "f", "arguments": "[1]"}, {}, True),
            "json string": ({"name": "f", "arguments": '{"a": 1}'}, {"a": 1}, False),
            "object": ({"name": "f", "arguments": {"a": 1}}, {"a": 1}, False),
        }
        for label, (function, expected_input, expected_truncated) in cases.items():
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
                self.assertEqual(block.get("truncated", False), expected_truncated)

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
            events = list(
                self.adapter.stream_chat("llama3", MESSAGES, config={"timeout": None})
            )
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
                    events = list(
                        self.adapter.stream_chat(
                            "llama3", MESSAGES, config={"timeout": None}
                        )
                    )
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
            [
                {
                    "type": "tool_use",
                    "id": "call_0",
                    "name": "f",
                    "input": {},
                    "truncated": True,
                }
            ],
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
            [
                {
                    "type": "tool_use",
                    "id": "call_0",
                    "name": "f",
                    "input": {},
                    "truncated": True,
                }
            ],
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
            [
                {
                    "type": "tool_use",
                    "id": "call_0",
                    "name": "f",
                    "input": {},
                    "truncated": True,
                }
            ],
        )

    @patch("urllib.request.urlopen")
    def test_legitimately_empty_arguments_are_not_marked_truncated_in_chat(
        self, mock_urlopen
    ):
        for arguments in ("", "{}"):
            with self.subTest(arguments=arguments):
                data = {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "f", "arguments": arguments}}
                        ],
                    },
                    "done_reason": "stop",
                }
                mock_urlopen.return_value = buffered_response(json.dumps(data).encode())
                response = self.adapter.chat("llama3", MESSAGES)
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


class OllamaModelInfoTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()
        self.enterContext(clean_env(OLLAMA_HOST="http://127.0.0.1:9"))

    @patch("urllib.request.urlopen")
    def test_reads_context_window_from_a_context_length_key(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps({"model_info": {"llama.context_length": 8192}}).encode()
        )
        self.assertEqual(
            self.adapter.model_info("llama3"),
            {"context_window": 8192, "max_output_tokens": None},
        )

    @patch("urllib.request.urlopen")
    def test_sends_the_model_name_in_the_request_body(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps({"model_info": {}}).encode()
        )
        self.adapter.model_info("llama3")
        request = mock_urlopen.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/api/show"))
        self.assertEqual(json.loads(request.data), {"model": "llama3"})

    @patch("urllib.request.urlopen")
    def test_no_matching_key_returns_none_context_window(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps({"model_info": {"some.other_field": 1}}).encode()
        )
        self.assertEqual(
            self.adapter.model_info("llama3"),
            {"context_window": None, "max_output_tokens": None},
        )

    @patch("urllib.request.urlopen")
    def test_a_bool_valued_context_length_key_is_not_mistaken_for_an_int(
        self, mock_urlopen
    ):
        mock_urlopen.return_value = buffered_response(
            json.dumps({"model_info": {"llama.context_length": True}}).encode()
        )
        self.assertEqual(
            self.adapter.model_info("llama3"),
            {"context_window": None, "max_output_tokens": None},
        )

    @patch("urllib.request.urlopen")
    def test_missing_model_info_returns_none_context_window(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(json.dumps({}).encode())
        self.assertEqual(
            self.adapter.model_info("llama3"),
            {"context_window": None, "max_output_tokens": None},
        )

    @patch("urllib.request.urlopen")
    def test_max_output_tokens_is_always_none(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps({"model_info": {"llama.context_length": 4096}}).encode()
        )
        info = self.adapter.model_info("llama3")
        assert info is not None
        self.assertIsNone(info["max_output_tokens"])

    @patch("urllib.request.urlopen")
    def test_404_returns_none_not_raised(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            "http://127.0.0.1:9/api/show", 404, b"not found"
        )
        self.assertIsNone(self.adapter.model_info("llama3"))

    @patch("urllib.request.urlopen")
    def test_connection_error_returns_none_not_raised(self, mock_urlopen):
        mock_urlopen.side_effect = ConnectionRefusedError()
        self.assertIsNone(self.adapter.model_info("llama3"))

    @patch("urllib.request.urlopen")
    def test_timeout_returns_none_not_raised(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError()
        self.assertIsNone(self.adapter.model_info("llama3"))

    def test_transport_refusal_gate_returns_none_with_one_warning(self):
        provider = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()},
            config=secret_headers("ollama-local", "t"),
        )
        adapter = provider._adapters["ollama-local"]
        with (
            pointed_at(provider, "ollama-local", "http://gateway.example"),
            patch("urllib.request.urlopen", side_effect=AssertionError("no request")),
            self.assertLogs("ducktape_provider", "WARNING") as logs,
        ):
            self.assertIsNone(adapter.model_info("llama3"))
            self.assertEqual(adapter.models(), set())
            self.assertFalse(adapter.is_available())
        self.assertEqual(len(logs.records), 1)
        self.assertIn("ollama-local", logs.records[0].getMessage())


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


class TestOllamaEmbedHTTP(unittest.TestCase):
    """Ollama embed HTTP contract."""

    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    @patch("urllib.request.urlopen")
    def test_posts_to_api_embed_with_list_input(self, mock_urlopen):
        """POST /api/embed with list input."""
        vectors = [[0.1, 0.2, 0.3]]
        payload = ollama_embed_response(
            "nomic-embed-text:latest", vectors, prompt_eval_count=5
        )
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        resp = self.adapter.embed("nomic-embed-text:latest", ["hello"])
        self.assertEqual(resp["embeddings"], vectors)
        req = mock_urlopen.call_args.args[0]
        self.assertTrue(req.full_url.endswith("/api/embed"))
        body = request_body(req)
        self.assertEqual(body["model"], "nomic-embed-text:latest")
        self.assertEqual(body["input"], ["hello"])

    @patch("urllib.request.urlopen")
    def test_batch_input_sends_list(self, mock_urlopen):
        """Batch input is sent as a list."""
        vectors = [[0.1], [0.2], [0.3]]
        payload = ollama_embed_response("nomic-embed-text:latest", vectors)
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        resp = self.adapter.embed("nomic-embed-text:latest", ["a", "b", "c"])
        self.assertEqual(resp["embeddings"], vectors)
        body = request_body(mock_urlopen.call_args.args[0])
        self.assertEqual(body["input"], ["a", "b", "c"])

    @patch("urllib.request.urlopen")
    def test_optional_fields_only_when_present(self, mock_urlopen):
        """Optional fields reach body only when present."""
        vectors = [[0.1]]
        payload = ollama_embed_response("nomic-embed-text:latest", vectors)
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        self.adapter.embed("nomic-embed-text:latest", ["hi"])
        body = request_body(mock_urlopen.call_args.args[0])
        self.assertNotIn("keep_alive", body)
        self.assertNotIn("truncate", body)
        self.assertNotIn("dimensions", body)
        self.assertNotIn("options", body)
        cases = [
            ({"keep_alive": "10m"}, "keep_alive", "10m"),
            ({"truncate": False}, "truncate", False),
            ({"dimensions": 512}, "dimensions", 512),
            ({"options": {"num_ctx": 2048}}, "options", {"num_ctx": 2048}),
        ]
        for config, key, expected in cases:
            with self.subTest(key=key):
                mock_urlopen.return_value = buffered_response(
                    json.dumps(payload).encode()
                )
                self.adapter.embed("nomic-embed-text:latest", ["hi"], config=config)
                body = request_body(mock_urlopen.call_args.args[0])
                self.assertEqual(body[key], expected)

    @patch("urllib.request.urlopen")
    def test_returns_ordered_vectors(self, mock_urlopen):
        """Returned vectors match input order."""
        vectors = [[0.1, 0.2], [0.3, 0.4]]
        payload = ollama_embed_response(
            "nomic-embed-text:latest", vectors, prompt_eval_count=2
        )
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        resp = self.adapter.embed("nomic-embed-text:latest", ["first", "second"])
        self.assertEqual(resp["embeddings"], vectors)

    @patch("urllib.request.urlopen")
    def test_count_mismatch_raises(self, mock_urlopen):
        """Count mismatch raises MalformedResponseError."""
        for vectors in [[[0.1]], [[0.1], [0.2], [0.3]]]:
            payload = ollama_embed_response("nomic-embed-text:latest", vectors)
            mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
            with (
                self.subTest(vectors=vectors),
                self.assertRaises(MalformedResponseError),
            ):
                self.adapter.embed("nomic-embed-text:latest", ["a", "b"])

    @patch("urllib.request.urlopen")
    def test_prompt_eval_count_maps_to_usage_and_raw(self, mock_urlopen):
        """prompt_eval_count maps to usage and raw."""
        vectors = [[0.1, 0.2]]
        payload = ollama_embed_response(
            "nomic-embed-text:latest", vectors, prompt_eval_count=9
        )
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        resp = self.adapter.embed("nomic-embed-text:latest", ["hi"])
        self.assertEqual(resp["usage"], {"input_tokens": 9})
        self.assertEqual(resp["raw"]["total_duration"], 1000)
        self.assertEqual(resp["raw"]["load_duration"], 500)
        self.assertIsInstance(resp["latency_ms"], float)
        self.assertGreaterEqual(resp["latency_ms"], 0)

    @patch("urllib.request.urlopen")
    def test_404_clears_cache(self, mock_urlopen):
        """404 raises APIError and clears cache."""
        self.adapter._models_cache = {"llama-old"}
        self.adapter._cache_time = time.monotonic()
        mock_urlopen.side_effect = http_error(
            "http://127.0.0.1:11434/api/embed", 404, b"not found"
        )
        with self.assertRaises(APIError) as ctx:
            self.adapter.embed("nomic-embed-text:latest", ["hi"])
        self.assertEqual(ctx.exception.status, 404)
        self.assertIsNone(self.adapter._models_cache)

    @patch("urllib.request.urlopen")
    def test_malformed_body_raises(self, mock_urlopen):
        """Malformed body raises MalformedResponseError."""
        for body in [
            b"not json",
            json.dumps({"model": "x"}).encode(),
            json.dumps({"embeddings": "bad"}).encode(),
        ]:
            mock_urlopen.return_value = buffered_response(body)
            with self.subTest(body=body), self.assertRaises(MalformedResponseError):
                self.adapter.embed("nomic-embed-text:latest", ["hi"])

    def test_refusal_raises_value_error_without_request_or_warning(self):
        """Provider headers with disallowed host raise ValueError."""
        provider = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()},
            config=secret_headers("ollama-local", "t"),
        )
        adapter = provider._adapters["ollama-local"]
        with (
            pointed_at(provider, "ollama-local", "http://gateway.example"),
            patch(
                "urllib.request.urlopen", side_effect=AssertionError("no request")
            ) as mock_urlopen,
        ):
            with self.assertRaises(ValueError):
                adapter.embed("nomic-embed-text:latest", ["hi"])
            mock_urlopen.assert_not_called()

    def test_refusal_logs_no_warning(self):
        """Embed refusal logs no warning."""
        provider = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()},
            config=secret_headers("ollama-local", "t"),
        )
        adapter = provider._adapters["ollama-local"]
        with (
            pointed_at(provider, "ollama-local", "http://gateway.example"),
            patch(
                "urllib.request.urlopen", side_effect=AssertionError("no request")
            ) as mock_urlopen,
            patch.object(ollama_module.logger, "warning") as mock_warn,
        ):
            with self.assertRaises(ValueError):
                adapter.embed("nomic-embed-text:latest", ["hi"])
            mock_warn.assert_not_called()
            mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen", side_effect=AssertionError("request sent"))
    def test_reserved_key_clash_raises(self, mock_urlopen):
        """Reserved key clash raises ValueError mentioning embed()."""
        with self.assertRaises(ValueError) as ctx:
            self.adapter.embed("nomic-embed-text:latest", ["hi"], config={"model": "x"})
        self.assertIn("embed()", str(ctx.exception))
        mock_urlopen.assert_not_called()
        with self.assertRaises(ValueError) as ctx:
            self.adapter.embed(
                "nomic-embed-text:latest", ["hi"], config={"input": ["x"]}
            )
        self.assertIn("embed()", str(ctx.exception))


class TestOllamaEmbedModels(unittest.TestCase):
    """Ollama embed_models delegation."""

    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    @patch("urllib.request.urlopen")
    def test_embed_models_equals_models_with_latest_tag(self, mock_urlopen):
        """embed_models equals models including :latest."""
        payload = {"models": [{"name": "nomic-embed-text:latest"}, {"name": "llama3"}]}
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        models = self.adapter.models()
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        self.adapter._models_cache = None
        self.adapter._cache_time = 0.0
        embed_models = self.adapter.embed_models()
        self.assertEqual(models, {"nomic-embed-text:latest", "llama3"})
        self.assertEqual(embed_models, models)

    @patch("urllib.request.urlopen")
    def test_embed_models_uses_same_cache_as_models(self, mock_urlopen):
        """embed_models returns same set as models."""
        payload = {
            "models": [
                {"name": "nomic-embed-text:latest"},
                {"name": "mxbai-embed-large"},
            ]
        }
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        self.assertEqual(
            self.adapter.embed_models(),
            {"nomic-embed-text:latest", "mxbai-embed-large"},
        )
        self.assertEqual(
            self.adapter.models(), {"nomic-embed-text:latest", "mxbai-embed-large"}
        )
        mock_urlopen.assert_called_once()

    def test_embed_models_inherits_probe_gate(self):
        """embed_models inherits probe gate."""
        provider = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()},
            config=secret_headers("ollama-local", "t"),
        )
        adapter = provider._adapters["ollama-local"]
        with (
            pointed_at(provider, "ollama-local", "http://gateway.example"),
            patch("urllib.request.urlopen", side_effect=AssertionError("no request")),
            self.assertLogs("ducktape_provider", "WARNING") as logs,
        ):
            self.assertEqual(adapter.embed_models(), set())
        self.assertEqual(len(logs.records), 1)
        self.assertIn("ollama-local", logs.records[0].getMessage())
        provider2 = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()},
            config=secret_headers("ollama-local", "t"),
        )
        adapter2 = provider2._adapters["ollama-local"]
        with (
            pointed_at(provider2, "ollama-local", "http://gateway.example"),
            patch("urllib.request.urlopen", side_effect=AssertionError("no request")),
            self.assertLogs("ducktape_provider", "WARNING") as logs2,
        ):
            self.assertEqual(adapter2.models(), set())
        self.assertEqual(len(logs2.records), 1)


class TestOllamaCapabilities(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()
        self.enterContext(clean_env(OLLAMA_HOST="http://127.0.0.1:9"))

    @patch("urllib.request.urlopen")
    def test_maps_completion_tools_vision_embedding(self, mock_urlopen):
        """capabilities with tools vision maps correctly and raw preserves list."""
        payload = {
            "model_info": {"llama.context_length": 8192},
            "capabilities": ["completion", "tools", "vision", "embedding"],
        }
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        caps = self.adapter.capabilities("llama3")
        assert caps is not None
        self.assertEqual(
            caps,
            {
                "tools": True,
                "vision": True,
                "pdf_input": None,
                "thinking": False,
                "raw": {"capabilities": ["completion", "tools", "vision", "embedding"]},
            },
        )
        assert "raw" in caps
        self.assertEqual(
            caps["raw"],
            {"capabilities": ["completion", "tools", "vision", "embedding"]},
        )
        self.assertEqual(set(caps["raw"].keys()), {"capabilities"})

    @patch("urllib.request.urlopen")
    def test_diffusion_image_not_mapped_to_vision(self, mock_urlopen):
        """diffusion image capability never maps to vision."""
        payload = {"capabilities": ["completion", "image"]}
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        caps = self.adapter.capabilities("flux")
        assert caps is not None
        self.assertEqual(caps["tools"], False)
        self.assertEqual(caps["vision"], False)
        self.assertEqual(caps["thinking"], False)
        self.assertIsNone(caps["pdf_input"])
        assert "raw" in caps
        self.assertEqual(caps["raw"]["capabilities"], ["completion", "image"])
        self.assertIn("image", caps["raw"]["capabilities"])

    @patch("urllib.request.urlopen")
    def test_empty_capabilities_list(self, mock_urlopen):
        """empty list maps all present flags to False."""
        payload = {"capabilities": []}
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        caps = self.adapter.capabilities("m")
        assert caps is not None
        self.assertEqual(caps["tools"], False)
        self.assertEqual(caps["vision"], False)
        self.assertEqual(caps["thinking"], False)
        self.assertIsNone(caps["pdf_input"])
        assert "raw" in caps
        self.assertEqual(caps["raw"], {"capabilities": []})

    @patch("urllib.request.urlopen")
    def test_missing_and_null_and_non_list_capabilities(self, mock_urlopen):
        """missing null non-list capabilities all yield None and no raw."""
        cases: list[tuple[str, Any]] = [
            ("absent", {}),
            ("null", {"capabilities": None}),
            ("string", {"capabilities": "tools"}),
            ("dict", {"capabilities": {}}),
        ]
        for label, payload in cases:
            with self.subTest(label):
                self.adapter._show_cache.clear()
                mock_urlopen.return_value = buffered_response(
                    json.dumps(payload).encode()
                )
                caps = self.adapter.capabilities("m")
                assert caps is not None
                self.assertIsNone(caps["tools"], label)
                self.assertIsNone(caps["vision"], label)
                self.assertIsNone(caps["pdf_input"], label)
                self.assertIsNone(caps["thinking"], label)
                self.assertNotIn("raw", caps, label)
                self.assertEqual(
                    set(caps.keys()),
                    {"tools", "vision", "pdf_input", "thinking"},
                    label,
                )

    @patch("urllib.request.urlopen")
    def test_request_body_and_headers_and_path(self, mock_urlopen):
        """POST /api/show with model json and json content type."""
        payload = {"capabilities": ["tools"]}
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        self.adapter.capabilities("llama3")
        req = mock_urlopen.call_args.args[0]
        self.assertTrue(req.full_url.endswith("/api/show"))
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(json.loads(req.data), {"model": "llama3"})
        self.assertEqual(req.get_header("Content-type"), "application/json")

    @patch("urllib.request.urlopen")
    def test_404_is_cached_negative_and_ttl_refetch(self, mock_urlopen):
        """404 yields None cached and refetched after TTL."""
        mock_urlopen.side_effect = http_error(
            "http://127.0.0.1:9/api/show", 404, b"not found"
        )
        with patch("time.monotonic", return_value=1000.0):
            self.assertIsNone(self.adapter.capabilities("missing"))
            self.assertEqual(mock_urlopen.call_count, 1)
            self.assertIsNone(self.adapter.capabilities("missing"))
            self.assertEqual(mock_urlopen.call_count, 1)
            self.assertIn("missing", self.adapter._show_cache)
            self.assertIsNone(self.adapter._show_cache["missing"][0])
        payload = {"capabilities": ["tools"]}
        mock_urlopen.side_effect = None
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        with patch("time.monotonic", return_value=1000.0 + 61):
            caps = self.adapter.capabilities("missing")
            assert caps is not None
            self.assertEqual(caps["tools"], True)
            self.assertEqual(mock_urlopen.call_count, 2)

    @patch("urllib.request.urlopen")
    def test_probe_failures_raise_through_capabilities_and_none_through_model_info(
        self, mock_urlopen
    ):
        """connection timeout 500 raise through capabilities and None through model_info."""
        for label, expected in [
            ("connection", APIError),
            ("timeout", RequestTimeoutError),
            ("500", ServerError),
        ]:
            with self.subTest(label):
                self.adapter._show_cache.clear()
                if label == "connection":
                    mock_urlopen.side_effect = ConnectionRefusedError()
                elif label == "timeout":
                    mock_urlopen.side_effect = TimeoutError()
                else:
                    mock_urlopen.side_effect = http_error(
                        "http://127.0.0.1:9/api/show", 500, b"boom"
                    )
                with self.assertRaises(expected):
                    self.adapter.capabilities("llama3")
                self.adapter._show_cache.clear()
                if label == "connection":
                    mock_urlopen.side_effect = ConnectionRefusedError()
                elif label == "timeout":
                    mock_urlopen.side_effect = TimeoutError()
                else:
                    mock_urlopen.side_effect = http_error(
                        "http://127.0.0.1:9/api/show", 500, b"boom"
                    )
                self.assertIsNone(self.adapter.model_info("llama3"))
                self.assertEqual(self.adapter._show_cache, {})

    def test_gate_refusal_raises_value_error_without_request(self):
        """gate refusal through capabilities raises ValueError with zero requests."""
        provider = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()},
            config=secret_headers("ollama-local", "t"),
        )
        adapter = provider._adapters["ollama-local"]
        with (
            pointed_at(provider, "ollama-local", "http://gateway.example"),
            patch(
                "urllib.request.urlopen", side_effect=AssertionError("no request")
            ) as mock_urlopen,
        ):
            with self.assertRaises(ValueError):
                adapter.capabilities("llama3")
            mock_urlopen.assert_not_called()
        with (
            pointed_at(provider, "ollama-local", "http://gateway.example"),
            patch(
                "urllib.request.urlopen", side_effect=AssertionError("no request")
            ) as mock_urlopen,
            patch.object(ollama_module.logger, "warning") as mock_warn,
        ):
            with self.assertRaises(ValueError):
                adapter.capabilities("llama3")
            mock_warn.assert_not_called()
            mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_shared_cache_between_model_info_and_capabilities(self, mock_urlopen):
        """model_info then capabilities then model_info share one show fetch."""
        payload = {
            "model_info": {"llama.context_length": 8192},
            "capabilities": ["tools", "vision"],
        }
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        with patch("time.monotonic", return_value=2000.0):
            info1 = self.adapter.model_info("llama3")
            caps = self.adapter.capabilities("llama3")
            info2 = self.adapter.model_info("llama3")
        self.assertEqual(info1, {"context_window": 8192, "max_output_tokens": None})
        assert caps is not None
        self.assertEqual(caps["tools"], True)
        self.assertEqual(caps["vision"], True)
        self.assertEqual(info2, {"context_window": 8192, "max_output_tokens": None})
        self.assertEqual(mock_urlopen.call_count, 1)
        req = mock_urlopen.call_args.args[0]
        self.assertTrue(req.full_url.endswith("/api/show"))

    @patch("urllib.request.urlopen")
    def test_show_cache_purges_expired_entries_on_write(self, mock_urlopen):
        """expired entries are purged on write."""
        self.adapter._show_cache = {
            "old-one": (
                {"context_window": 100, "capabilities": ["tools"], "vision": False},
                0.0,
            ),
            "old-two": (None, 0.0),
        }
        payload = {"capabilities": ["vision"]}
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        with patch("time.monotonic", return_value=61.0):
            caps = self.adapter.capabilities("new-model")
        assert caps is not None
        self.assertEqual(caps["vision"], True)
        self.assertNotIn("old-one", self.adapter._show_cache)
        self.assertNotIn("old-two", self.adapter._show_cache)
        self.assertIn("new-model", self.adapter._show_cache)

    @patch("urllib.request.urlopen")
    def test_raw_carries_only_vendor_capabilities(self, mock_urlopen):
        """raw carries only vendor capabilities list."""
        cases = [
            ["completion", "tools", "vision", "embedding"],
            ["completion", "image"],
            [],
        ]
        for caps_list in cases:
            with self.subTest(caps_list=caps_list):
                self.adapter._show_cache.clear()
                payload = {"capabilities": caps_list}
                mock_urlopen.return_value = buffered_response(
                    json.dumps(payload).encode()
                )
                caps = self.adapter.capabilities("m")
                assert caps is not None
                assert "raw" in caps
                raw = caps["raw"]
                self.assertEqual(raw, {"capabilities": caps_list})
                self.assertEqual(set(raw.keys()), {"capabilities"})

    @patch("urllib.request.urlopen")
    def test_malformed_body_raises_then_model_info_swallows(self, mock_urlopen):
        """Malformed /api/show raises through capabilities, None through model_info."""
        mock_urlopen.return_value = buffered_response(b"not json")
        with self.assertRaises(MalformedResponseError):
            self.adapter.capabilities("llama3")
        mock_urlopen.return_value = buffered_response(b"not json")
        self.assertIsNone(self.adapter.model_info("llama3"))

    @patch("urllib.request.urlopen")
    def test_normal_ttl_expiry_refetches(self, mock_urlopen):
        """A cached capabilities result refetches past the 60s boundary."""
        payload = {"model_info": {"llama.context_length": 8192}, "capabilities": []}
        base = 1000.0
        with patch("ducktape_provider.adapters.ollama.time.monotonic") as mock_time:
            mock_time.return_value = base
            mock_urlopen.side_effect = lambda *a, **k: buffered_response(
                json.dumps(payload).encode()
            )
            self.adapter.capabilities("llama3")
            self.assertEqual(mock_urlopen.call_count, 1)
            mock_time.return_value = base + 59
            self.adapter.capabilities("llama3")
            self.assertEqual(mock_urlopen.call_count, 1)
            mock_time.return_value = base + 61
            self.adapter.capabilities("llama3")
            self.assertEqual(mock_urlopen.call_count, 2)

    @patch("urllib.request.urlopen")
    def test_mutating_raw_does_not_poison_cache(self, mock_urlopen):
        """Mutating a returned raw list leaves the cached entry intact."""
        payload = {"capabilities": ["tools", "vision"]}
        mock_urlopen.side_effect = lambda *a, **k: buffered_response(
            json.dumps(payload).encode()
        )
        caps = self.adapter.capabilities("llama3")
        assert caps is not None
        assert "raw" in caps
        caps["raw"]["capabilities"].append("mutated")
        caps2 = self.adapter.capabilities("llama3")
        assert caps2 is not None
        assert "raw" in caps2
        self.assertEqual(caps2["raw"]["capabilities"], ["tools", "vision"])

    @patch("urllib.request.urlopen")
    def test_zero_context_window_preserved(self, mock_urlopen):
        """A zero context_length is preserved, not coerced to None."""
        mock_urlopen.return_value = buffered_response(
            json.dumps({"model_info": {"llama.context_length": 0}}).encode()
        )
        self.assertEqual(
            self.adapter.model_info("llama3"),
            {"context_window": 0, "max_output_tokens": None},
        )

    @patch("urllib.request.urlopen")
    def test_legacy_server_projector_key_implies_vision(self, mock_urlopen):
        """Without a capabilities list, a vision projector key implies vision."""
        payload = {
            "model_info": {"llama.context_length": 4096, "clip.vision.block_count": 1}
        }
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        caps = self.adapter.capabilities("llava")
        assert caps is not None
        self.assertEqual(
            caps,
            {"tools": None, "vision": True, "pdf_input": None, "thinking": None},
        )

    @patch("urllib.request.urlopen")
    def test_legacy_server_without_projector_stays_unknown(self, mock_urlopen):
        """Without a capabilities list or projector keys, vision stays unknown."""
        payload = {"model_info": {"llama.context_length": 4096}}
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        caps = self.adapter.capabilities("llama3")
        assert caps is not None
        self.assertIsNone(caps["vision"])

    def test_invalidate_model_capabilities_drops_only_that_model(self):
        """The per-model hook drops one entry and keeps the rest."""
        self.adapter._show_cache = {
            "a": (
                {"context_window": 1, "capabilities": ["tools"], "vision": False},
                1.0,
            ),
            "b": ({"context_window": 2, "capabilities": None, "vision": None}, 1.0),
        }
        self.adapter._invalidate_model_capabilities("a")
        self.assertNotIn("a", self.adapter._show_cache)
        self.assertIn("b", self.adapter._show_cache)


class TestOllamaCompaction(unittest.TestCase):
    def test_supports_compaction_is_false(self):
        self.assertFalse(OllamaLocalAdapter().supports_compaction())

    def test_compaction_block_raises_unsupported(self):
        messages: list[Message] = [
            {"role": "user", "content": [{"type": "compaction", "content": "sum"}]}
        ]
        with self.assertRaises(UnsupportedBlockError):
            OllamaLocalAdapter()._serialize(messages, None)


if __name__ == "__main__":
    unittest.main()
