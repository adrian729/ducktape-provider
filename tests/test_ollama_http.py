"""HTTP-path tests for OllamaLocalAdapter: chat() and stream_chat() with
urllib.request.urlopen mocked out. No real network call is ever made."""

import json
import unittest
from unittest.mock import patch

from http_test_utils import (
    FakeStreamResponse,
    buffered_response,
    http_error,
    ndjson_lines,
)

from ducktape_provider import OllamaLocalAdapter

MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "weather in NYC?"}]}]

FINAL_DATA = {
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

STREAM_CHUNKS = [
    {"message": {"role": "assistant", "content": "Let "}, "done": False},
    {"message": {"role": "assistant", "content": "me check"}, "done": False},
    {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "get_weather", "arguments": {"city": "NYC"}}}
            ],
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 7,
        "eval_count": 5,
    },
]


class OllamaSerializeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    def test_serialize_maps_tool_result_and_tool_use(self):
        messages = [
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
        tools = [{"name": "get_weather", "description": "...", "parameters": {"a": 1}}]
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
        request = mock_urlopen.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/api/chat"))
        sent = json.loads(request.data)
        self.assertEqual(sent["stream"], False)

    @patch("urllib.request.urlopen")
    def test_chat_raises_runtime_error_on_http_error(self, mock_urlopen):
        mock_urlopen.side_effect = http_error("http://127.0.0.1:11434/api/chat", 500, b"boom")

        with self.assertRaises(RuntimeError) as ctx:
            self.adapter.chat("llama3", MESSAGES)

        self.assertIn("ollama", str(ctx.exception))
        self.assertIn("500", str(ctx.exception))


class OllamaStreamChatHTTPTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    @patch("urllib.request.urlopen")
    def test_stream_chat_yields_expected_event_sequence(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(ndjson_lines(*STREAM_CHUNKS))

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
        final = events[-1]["response"]
        buffered = self.adapter._deserialize(FINAL_DATA)
        self.assertEqual(final["content"], buffered["content"])
        self.assertEqual(final["stop_reason"], buffered["stop_reason"])
        self.assertEqual(final["usage"], buffered["usage"])

    @patch("urllib.request.urlopen")
    def test_stream_chat_raises_runtime_error_on_http_error(self, mock_urlopen):
        mock_urlopen.side_effect = http_error("http://127.0.0.1:11434/api/chat", 500, b"boom")

        with self.assertRaises(RuntimeError) as ctx:
            list(self.adapter.stream_chat("llama3", MESSAGES))

        self.assertIn("ollama", str(ctx.exception))
        self.assertIn("500", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
