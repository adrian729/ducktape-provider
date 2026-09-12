"""HTTP-path tests for OpenAIAdapter: chat() and stream_chat() with
urllib.request.urlopen mocked out. No real network call is ever made."""

import json
import unittest
from unittest.mock import patch

from http_test_utils import FakeStreamResponse, buffered_response, http_error, sse_lines

from ducktape_provider import OpenAIAdapter

MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "weather in NYC?"}]}]

FINAL_DATA = {
    "status": "completed",
    "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "Let me check"}]},
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
        "item": {"id": "fc_1", "type": "function_call", "call_id": "call_1", "name": "get_weather"},
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
        request = mock_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, OpenAIAdapter._RESPONSES_URL)
        sent = json.loads(request.data)
        self.assertEqual(sent["stream"], False)

    @patch("urllib.request.urlopen")
    def test_chat_raises_runtime_error_on_http_error(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(OpenAIAdapter._RESPONSES_URL, 401, b"unauthorized")

        with self.assertRaises(RuntimeError) as ctx:
            self.adapter.chat("gpt-x", MESSAGES)

        self.assertIn("openai", str(ctx.exception))
        self.assertIn("401", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_chat_raises_runtime_error_on_failed_status(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps({"status": "failed", "error": {"message": "boom"}, "output": []}).encode()
        )

        with self.assertRaises(RuntimeError):
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
        buffered = self.adapter._deserialize(FINAL_DATA)
        self.assertEqual(final["content"], buffered["content"])
        self.assertEqual(final["stop_reason"], buffered["stop_reason"])
        self.assertEqual(final["usage"], buffered["usage"])

    @patch("urllib.request.urlopen")
    def test_stream_chat_raises_runtime_error_on_http_error(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(OpenAIAdapter._RESPONSES_URL, 500, b"boom")

        with self.assertRaises(RuntimeError) as ctx:
            list(self.adapter.stream_chat("gpt-x", MESSAGES))

        self.assertIn("openai", str(ctx.exception))
        self.assertIn("500", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
