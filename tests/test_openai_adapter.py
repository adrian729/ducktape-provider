"""Tests for OpenAIAdapter: _serialize/_deserialize plus chat() and stream_chat()
with urllib.request.urlopen mocked out. No real network call is ever made."""

import json
import unittest
from unittest.mock import patch

from http_test_utils import FakeStreamResponse, buffered_response, http_error, sse_lines

from ducktape_provider import APIError, AuthError, OpenAIAdapter, ServerError
from ducktape_provider.adapters import openai as openai_module

MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "weather in NYC?"}]}]

FINAL_DATA = {
    "status": "completed",
    "output": [
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Let me check"}],
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "NYC"}',
        },
    ],
    "usage": {"input_tokens": 12, "output_tokens": 9},
}

STREAM_EVENTS = [
    {
        "type": "response.output_item.added",
        "item": {"id": "msg_1", "type": "message"},
    },
    {
        "type": "response.output_text.delta",
        "item_id": "msg_1",
        "delta": "Let me check",
    },
    {
        "type": "response.output_item.done",
        "item": {"id": "msg_1", "type": "message"},
    },
    {
        "type": "response.output_item.added",
        "item": {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_weather",
        },
    },
    {
        "type": "response.function_call_arguments.delta",
        "item_id": "fc_1",
        "delta": '{"city": "NYC"}',
    },
    {
        "type": "response.output_item.done",
        "item": {"id": "fc_1", "type": "function_call"},
    },
    {"type": "response.completed", "response": FINAL_DATA},
]


class OpenAISerializeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter()

    def test_serialize_interleaves_function_call_and_output(self):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "checking"},
                    {
                        "type": "tool_use",
                        "id": "call_1",
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
                        "tool_use_id": "call_1",
                        "name": "get_weather",
                        "content": "sunny",
                        "is_error": False,
                    }
                ],
            },
        ]
        serialized = self.adapter._serialize(messages)
        self.assertEqual(
            serialized,
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": "checking"}],
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny",
                },
            ],
        )

    def test_serialize_tools(self):
        tools = [{"name": "get_weather", "description": "...", "parameters": {"a": 1}}]
        self.assertEqual(
            self.adapter._serialize_tools(tools),
            [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "...",
                    "parameters": {"a": 1},
                }
            ],
        )


class OpenAIDeserializeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter()

    def test_text_message_completed(self):
        data = {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2},
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(response["content"], [{"type": "text", "text": "hi"}])
        self.assertEqual(response["stop_reason"], "end_turn")

    def test_function_call_sets_tool_use_stop_reason(self):
        data = {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                }
            ],
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(
            response["content"],
            [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                }
            ],
        )
        self.assertEqual(response["stop_reason"], "tool_use")

    def test_incomplete_max_output_tokens(self):
        data = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(response["stop_reason"], "max_tokens")
        self.assertEqual(response["raw_stop_reason"], "max_output_tokens")

    def test_maps_cache_usage_fields_when_present(self):
        data = {
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "input_tokens_details": {
                    "cached_tokens": 100,
                    "cache_write_tokens": 20,
                },
            },
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(
            response["usage"],
            {
                "input_tokens": 3,
                "output_tokens": 5,
                "cache_read_tokens": 100,
                "cache_write_tokens": 20,
            },
        )

    def test_maps_cache_read_only(self):
        data = {
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "input_tokens_details": {"cached_tokens": 100},
            },
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(
            response["usage"],
            {"input_tokens": 3, "output_tokens": 5, "cache_read_tokens": 100},
        )
        self.assertNotIn("cache_write_tokens", response["usage"])

    def test_maps_cache_write_only(self):
        data = {
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "input_tokens_details": {"cache_write_tokens": 20},
            },
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(
            response["usage"],
            {"input_tokens": 3, "output_tokens": 5, "cache_write_tokens": 20},
        )
        self.assertNotIn("cache_read_tokens", response["usage"])

    def test_omits_cache_usage_fields_when_absent(self):
        data = {
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertNotIn("cache_read_tokens", response["usage"])
        self.assertNotIn("cache_write_tokens", response["usage"])

    def test_reasoning_item_produces_thinking_block(self):
        data = {
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "thinking hard"}],
                }
            ],
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(
            response["content"],
            [{"type": "thinking", "thinking": "thinking hard"}],
        )

    def test_reasoning_item_joins_multiple_summary_parts(self):
        data = {
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {"type": "summary_text", "text": "step one"},
                        {"type": "summary_text", "text": "step two"},
                    ],
                }
            ],
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(
            response["content"],
            [{"type": "thinking", "thinking": "step one\nstep two"}],
        )

    def test_reasoning_item_with_empty_summary_produces_no_thinking_block(self):
        data = {
            "status": "completed",
            "output": [{"type": "reasoning", "summary": []}],
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(response["content"], [])

    def test_reasoning_item_with_absent_summary_produces_no_thinking_block(self):
        data = {
            "status": "completed",
            "output": [{"type": "reasoning"}],
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(response["content"], [])


class OpenAIChatHTTPTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter()

    @patch("urllib.request.urlopen")
    def test_chat_returns_deserialized_response(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(json.dumps(FINAL_DATA).encode())

        response = self.adapter.chat("gpt-x", MESSAGES)

        self.assertEqual(
            response["content"],
            [
                {"type": "text", "text": "Let me check"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
            ],
        )
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(response["usage"], {"input_tokens": 12, "output_tokens": 9})
        self.assertIsInstance(response["latency_ms"], float)
        self.assertGreaterEqual(response["latency_ms"], 0)
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, OpenAIAdapter._RESPONSES_URL)
        sent = json.loads(request.data)
        self.assertEqual(sent["stream"], False)

    @patch("urllib.request.urlopen")
    def test_chat_raises_auth_error_on_401(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            OpenAIAdapter._RESPONSES_URL, 401, b"unauthorized"
        )

        with self.assertRaises(AuthError) as ctx:
            self.adapter.chat("gpt-x", MESSAGES)

        self.assertIn("openai", str(ctx.exception))
        self.assertIn("401", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_chat_raises_api_error_on_failed_status(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(
                {"status": "failed", "error": {"message": "boom"}, "output": []}
            ).encode()
        )

        with self.assertRaises(APIError):
            self.adapter.chat("gpt-x", MESSAGES)


class OpenAIStreamChatHTTPTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter()

    @patch("urllib.request.urlopen")
    def test_stream_chat_yields_expected_event_sequence(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*STREAM_EVENTS))

        events = list(self.adapter.stream_chat("gpt-x", MESSAGES))

        self.assertEqual(
            [e["type"] for e in events],
            [
                "text_delta",
                "block_stop",
                "tool_use_start",
                "tool_use_delta",
                "block_stop",
                "message_stop",
            ],
        )
        final = events[-1]["response"]
        buffered = self.adapter._deserialize(FINAL_DATA, 0.0)
        self.assertEqual(final["content"], buffered["content"])
        self.assertEqual(final["stop_reason"], buffered["stop_reason"])
        self.assertEqual(final["usage"], buffered["usage"])
        self.assertIsInstance(final["latency_ms"], float)
        self.assertGreaterEqual(final["latency_ms"], 0)

    @patch("urllib.request.urlopen")
    def test_stream_chat_reasoning_summary_deltas_produce_thinking_block(
        self, mock_urlopen
    ):
        events_in = [
            {
                "type": "response.output_item.added",
                "item": {"id": "rs_1", "type": "reasoning"},
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "summary_index": 0,
                "delta": "thinking ",
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "summary_index": 0,
                "delta": "hard",
            },
            {
                "type": "response.output_item.done",
                "item": {"id": "rs_1", "type": "reasoning"},
            },
            {"type": "response.completed", "response": FINAL_DATA},
        ]
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*events_in))

        events = list(self.adapter.stream_chat("gpt-x", MESSAGES))

        self.assertEqual(
            [e["type"] for e in events],
            ["thinking_delta", "thinking_delta", "block_stop", "message_stop"],
        )
        self.assertEqual(events[0]["index"], 0)
        self.assertEqual(events[0]["thinking"], "thinking ")
        self.assertEqual(events[1]["thinking"], "hard")
        self.assertEqual(events[2]["index"], 0)

    @patch("urllib.request.urlopen")
    def test_stream_chat_reasoning_item_without_deltas_emits_no_block_stop(
        self, mock_urlopen
    ):
        events_in = [
            {
                "type": "response.output_item.added",
                "item": {"id": "rs_1", "type": "reasoning"},
            },
            {
                "type": "response.output_item.done",
                "item": {"id": "rs_1", "type": "reasoning"},
            },
            {"type": "response.completed", "response": FINAL_DATA},
        ]
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*events_in))

        events = list(self.adapter.stream_chat("gpt-x", MESSAGES))

        self.assertEqual([e["type"] for e in events], ["message_stop"])

    @patch("urllib.request.urlopen")
    def test_stream_chat_raises_server_error_on_500(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            OpenAIAdapter._RESPONSES_URL, 500, b"boom"
        )

        with self.assertRaises(ServerError) as ctx:
            list(self.adapter.stream_chat("gpt-x", MESSAGES))

        self.assertIn("openai", str(ctx.exception))
        self.assertIn("500", str(ctx.exception))


class OpenAIHeaderOverrideTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter()

    def test_config_headers_override_default(self):
        req, _ = self.adapter._build_request(
            "gpt-x",
            MESSAGES,
            None,
            None,
            {"headers": {"Authorization": "Bearer overridden"}},
            stream=False,
        )
        self.assertEqual(req.get_header("Authorization"), "Bearer overridden")


class OpenAIBlockShapeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter()

    def test_url_image_block_passes_url_through_as_is(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": "url", "url": "https://x/img.png"}
                ],
            }
        ]
        serialized = self.adapter._serialize(messages)
        self.assertEqual(
            serialized[0]["content"],
            [{"type": "input_image", "image_url": "https://x/img.png"}],
        )

    def test_document_block_is_dropped_and_logs_warning(self):
        messages = [
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
        with self.assertLogs(openai_module.logger, level="WARNING"):
            serialized = self.adapter._serialize(messages)
        self.assertEqual(
            serialized,
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "see attached"}],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
