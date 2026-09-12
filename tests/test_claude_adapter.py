"""Tests for ClaudeAdapter: _serialize/_deserialize plus chat() and stream_chat()
with urllib.request.urlopen mocked out. No real network call is ever made."""

import json
import unittest
from unittest.mock import patch

from http_test_utils import FakeStreamResponse, buffered_response, http_error, sse_lines

from ducktape_provider import ClaudeAdapter, RateLimitError, ServerError

MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "weather in NYC?"}]}]

FINAL_DATA = {
    "content": [
        {"type": "text", "text": "Let me check"},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {"city": "NYC"},
        },
    ],
    "stop_reason": "tool_use",
    "usage": {"input_tokens": 10, "output_tokens": 8},
}

STREAM_EVENTS = [
    {"type": "message_start", "message": {"usage": {"input_tokens": 10}}},
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Let me check"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "get_weather",
            "input": {},
        },
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "input_json_delta", "partial_json": '{"city": "NYC"}'},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "tool_use"},
        "usage": {"output_tokens": 8},
    },
    {"type": "message_stop"},
]


class ClaudeSerializeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    def test_serialize_passes_through_image_and_tool_result(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": "base64",
                        "media_type": "image/png",
                        "data": "abc",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "name": "get_weather",
                        "content": "sunny",
                        "is_error": True,
                    },
                ],
            }
        ]
        serialized = self.adapter._serialize(messages)
        self.assertEqual(
            serialized,
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "sunny",
                            "is_error": True,
                        },
                    ],
                }
            ],
        )

    def test_serialize_tools(self):
        tools = [{"name": "get_weather", "description": "...", "parameters": {"a": 1}}]
        self.assertEqual(
            self.adapter._serialize_tools(tools),
            [{"name": "get_weather", "description": "...", "input_schema": {"a": 1}}],
        )


class ClaudeDeserializeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    def test_maps_known_stop_reason(self):
        data = {
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(response["stop_reason"], "end_turn")
        self.assertEqual(response["raw_stop_reason"], "end_turn")
        self.assertEqual(response["usage"], {"input_tokens": 3, "output_tokens": 5})
        self.assertEqual(response["content"], data["content"])
        self.assertIs(response["raw"], data)

    def test_unknown_stop_reason_falls_back_to_other(self):
        data = {"content": [], "stop_reason": "weird", "usage": {}}
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(response["stop_reason"], "other")
        self.assertEqual(response["raw_stop_reason"], "weird")
        self.assertEqual(response["usage"], {"input_tokens": 0, "output_tokens": 0})

    def test_maps_pause_turn_stop_reason(self):
        data = {"content": [], "stop_reason": "pause_turn", "usage": {}}
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(response["stop_reason"], "pause_turn")
        self.assertEqual(response["raw_stop_reason"], "pause_turn")

    def test_maps_cache_usage_fields_when_present(self):
        data = {
            "content": [],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 50,
                "output_tokens": 5,
                "cache_read_input_tokens": 100000,
                "cache_creation_input_tokens": 0,
            },
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(
            response["usage"],
            {
                "input_tokens": 100050,
                "output_tokens": 5,
                "cache_read_tokens": 100000,
                "cache_write_tokens": 0,
            },
        )

    def test_maps_cache_read_only(self):
        data = {
            "content": [],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "cache_read_input_tokens": 100,
            },
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(
            response["usage"],
            {"input_tokens": 103, "output_tokens": 5, "cache_read_tokens": 100},
        )
        self.assertNotIn("cache_write_tokens", response["usage"])

    def test_maps_cache_write_only(self):
        data = {
            "content": [],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "cache_creation_input_tokens": 20,
            },
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertEqual(
            response["usage"],
            {"input_tokens": 23, "output_tokens": 5, "cache_write_tokens": 20},
        )
        self.assertNotIn("cache_read_tokens", response["usage"])

    def test_omits_cache_usage_fields_when_absent(self):
        data = {
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }
        response = self.adapter._deserialize(data, 0.0)
        self.assertNotIn("cache_read_tokens", response["usage"])
        self.assertNotIn("cache_write_tokens", response["usage"])


class ClaudeChatHTTPTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    @patch("urllib.request.urlopen")
    def test_chat_returns_deserialized_response(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(json.dumps(FINAL_DATA).encode())

        response = self.adapter.chat("claude-x", MESSAGES)

        self.assertEqual(response["content"], FINAL_DATA["content"])
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(response["usage"], {"input_tokens": 10, "output_tokens": 8})
        self.assertIsInstance(response["latency_ms"], float)
        self.assertGreaterEqual(response["latency_ms"], 0)
        # request actually went through urlopen, never a real socket
        mock_urlopen.assert_called_once()
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, ClaudeAdapter._MESSAGES_URL)
        sent = json.loads(request.data)
        self.assertEqual(sent["stream"], False)

    @patch("urllib.request.urlopen")
    def test_chat_raises_rate_limit_error_on_429(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            ClaudeAdapter._MESSAGES_URL, 429, b"rate limited"
        )

        with self.assertRaises(RateLimitError) as ctx:
            self.adapter.chat("claude-x", MESSAGES)

        self.assertIn("claude", str(ctx.exception))
        self.assertIn("429", str(ctx.exception))


class ClaudeStreamChatHTTPTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    @patch("urllib.request.urlopen")
    def test_stream_chat_yields_expected_event_sequence(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*STREAM_EVENTS))

        events = list(self.adapter.stream_chat("claude-x", MESSAGES))

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
    def test_stream_chat_carries_cache_fields_in_message_stop(self, mock_urlopen):
        events_with_cache = [
            {
                "type": "message_start",
                "message": {
                    "usage": {
                        "input_tokens": 10,
                        "cache_read_input_tokens": 100,
                        "cache_creation_input_tokens": 20,
                    }
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "hi"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 8},
            },
            {"type": "message_stop"},
        ]
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*events_with_cache))

        events = list(self.adapter.stream_chat("claude-x", MESSAGES))

        final = events[-1]["response"]
        self.assertEqual(
            final["usage"],
            {
                "input_tokens": 130,
                "output_tokens": 8,
                "cache_read_tokens": 100,
                "cache_write_tokens": 20,
            },
        )

    @patch("urllib.request.urlopen")
    def test_stream_chat_raises_server_error_on_500(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(ClaudeAdapter._MESSAGES_URL, 500, b"boom")

        with self.assertRaises(ServerError) as ctx:
            list(self.adapter.stream_chat("claude-x", MESSAGES))

        self.assertIn("claude", str(ctx.exception))
        self.assertIn("500", str(ctx.exception))


class ClaudeHeaderOverrideTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    def test_config_headers_override_default(self):
        req, _ = self.adapter._build_request(
            "claude-x",
            MESSAGES,
            None,
            None,
            {"headers": {"anthropic-version": "2099-01-01"}},
            stream=False,
        )
        self.assertEqual(req.get_header("Anthropic-version"), "2099-01-01")


class ClaudeBlockShapeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    def test_url_image_block_serializes_to_nested_url_source(self):
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
            [{"type": "image", "source": {"type": "url", "url": "https://x/img.png"}}],
        )

    def test_base64_document_block_serializes_to_nested_base64_source(self):
        messages = [
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
        serialized = self.adapter._serialize(messages)
        self.assertEqual(
            serialized[0]["content"],
            [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "abc",
                    },
                }
            ],
        )

    def test_url_document_block_serializes_to_nested_url_source(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "document", "source": "url", "url": "https://x/doc.pdf"}
                ],
            }
        ]
        serialized = self.adapter._serialize(messages)
        self.assertEqual(
            serialized[0]["content"],
            [
                {
                    "type": "document",
                    "source": {"type": "url", "url": "https://x/doc.pdf"},
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
