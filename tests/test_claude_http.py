"""HTTP-path tests for ClaudeAdapter: chat() and stream_chat() with
urllib.request.urlopen mocked out. No real network call is ever made."""

import json
import unittest
from unittest.mock import patch

from http_test_utils import FakeStreamResponse, buffered_response, http_error, sse_lines

from ducktape_provider import ClaudeAdapter

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
                    {"type": "image", "media_type": "image/png", "data": "abc"},
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
        # request actually went through urlopen, never a real socket
        mock_urlopen.assert_called_once()
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, ClaudeAdapter._MESSAGES_URL)
        sent = json.loads(request.data)
        self.assertEqual(sent["stream"], False)

    @patch("urllib.request.urlopen")
    def test_chat_raises_runtime_error_on_http_error(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(ClaudeAdapter._MESSAGES_URL, 429, b"rate limited")

        with self.assertRaises(RuntimeError) as ctx:
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
        buffered = self.adapter._deserialize(FINAL_DATA)
        self.assertEqual(final["content"], buffered["content"])
        self.assertEqual(final["stop_reason"], buffered["stop_reason"])
        self.assertEqual(final["usage"], buffered["usage"])

    @patch("urllib.request.urlopen")
    def test_stream_chat_raises_runtime_error_on_http_error(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(ClaudeAdapter._MESSAGES_URL, 500, b"boom")

        with self.assertRaises(RuntimeError) as ctx:
            list(self.adapter.stream_chat("claude-x", MESSAGES))

        self.assertIn("claude", str(ctx.exception))
        self.assertIn("500", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
