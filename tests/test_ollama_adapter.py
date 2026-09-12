"""Tests for OllamaLocalAdapter: _serialize/_deserialize plus chat() and stream_chat()
with urllib.request.urlopen mocked out. No real network call is ever made."""

import json
import unittest
from unittest.mock import patch

from http_test_utils import (
    FakeStreamResponse,
    buffered_response,
    http_error,
    ndjson_lines,
)

from ducktape_provider import OllamaLocalAdapter, ServerError, UnsupportedBlockError
from ducktape_provider.adapters import ollama as ollama_module

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
        response = self.adapter._deserialize(data)
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
        response = self.adapter._deserialize(data)
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
        response = self.adapter._deserialize(data)
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


class OllamaUnsupportedBlockTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OllamaLocalAdapter()

    @patch("urllib.request.urlopen")
    def test_url_image_block_raises_before_any_network_request(self, mock_urlopen):
        mock_urlopen.side_effect = AssertionError("urlopen should never be called")
        messages = [
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
        with self.assertLogs(ollama_module.logger, level="WARNING"):
            serialized = self.adapter._serialize(messages, system=None)
        self.assertEqual(serialized, [{"role": "user", "content": "see attached"}])

    def test_document_only_message_produces_no_entry(self):
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
        with self.assertLogs(ollama_module.logger, level="WARNING"):
            serialized = self.adapter._serialize(messages, system=None)
        self.assertEqual(serialized, [])


if __name__ == "__main__":
    unittest.main()
