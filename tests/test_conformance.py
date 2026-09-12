import json
import unittest
from unittest.mock import patch

from http_test_utils import (
    FakeStreamResponse,
    buffered_response,
    ndjson_lines,
    sse_lines,
)

from ducktape_provider import ClaudeAdapter, OllamaLocalAdapter, OpenAIAdapter, Provider

TEXT_MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "hi there"}]}]
TOOL_MESSAGES = [
    {"role": "user", "content": [{"type": "text", "text": "weather in NYC?"}]}
]
WEATHER_TOOL = [
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
        response = provider.chat("claude", "claude-x", TEXT_MESSAGES)
        self._assert_valid(response)

    @patch("urllib.request.urlopen")
    def test_openai(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(OPENAI_TEXT_DATA).encode()
        )
        provider = Provider(adapters={"openai": OpenAIAdapter()})
        response = provider.chat("openai", "gpt-x", TEXT_MESSAGES)
        self._assert_valid(response)

    @patch("urllib.request.urlopen")
    def test_ollama(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(OLLAMA_TEXT_DATA).encode()
        )
        provider = Provider(adapters={"ollama-local": OllamaLocalAdapter()})
        response = provider.chat("ollama-local", "llama3", TEXT_MESSAGES)
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
            "claude", "claude-x", TOOL_MESSAGES, tools=WEATHER_TOOL
        )
        self._assert_tool_use(response)

    @patch("urllib.request.urlopen")
    def test_openai(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(OPENAI_TOOL_DATA).encode()
        )
        provider = Provider(adapters={"openai": OpenAIAdapter()})
        response = provider.chat("openai", "gpt-x", TOOL_MESSAGES, tools=WEATHER_TOOL)
        self._assert_tool_use(response)

    @patch("urllib.request.urlopen")
    def test_ollama(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(OLLAMA_TOOL_DATA).encode()
        )
        provider = Provider(adapters={"ollama-local": OllamaLocalAdapter()})
        response = provider.chat(
            "ollama-local", "llama3", TOOL_MESSAGES, tools=WEATHER_TOOL
        )
        self._assert_tool_use(response)


class StreamingChatTests(_ConformanceBase):
    def _assert_stream(self, events):
        self.assertTrue(any(e["type"] == "text_delta" for e in events))
        final = [e for e in events if e["type"] == "message_stop"]
        self.assertEqual(len(final), 1)
        self._assert_valid(final[0]["response"])

    @patch("urllib.request.urlopen")
    def test_claude(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*CLAUDE_STREAM_EVENTS))
        provider = Provider(adapters={"claude": ClaudeAdapter()})
        events = list(provider.stream_chat("claude", "claude-x", TEXT_MESSAGES))
        self._assert_stream(events)

    @patch("urllib.request.urlopen")
    def test_openai(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*OPENAI_STREAM_EVENTS))
        provider = Provider(adapters={"openai": OpenAIAdapter()})
        events = list(provider.stream_chat("openai", "gpt-x", TEXT_MESSAGES))
        self._assert_stream(events)

    @patch("urllib.request.urlopen")
    def test_ollama(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(
            ndjson_lines(*OLLAMA_STREAM_CHUNKS)
        )
        provider = Provider(adapters={"ollama-local": OllamaLocalAdapter()})
        events = list(provider.stream_chat("ollama-local", "llama3", TEXT_MESSAGES))
        self._assert_stream(events)


if __name__ == "__main__":
    unittest.main()
