"""Tests for ClaudeAdapter: _serialize/_deserialize plus chat() and stream_chat()
with urllib.request.urlopen mocked out. No real network call is ever made."""

import http.client
import json
import time
import unittest
from collections.abc import Generator
from typing import Any, cast
from unittest.mock import patch

from http_test_utils import (
    FakeStreamResponse,
    buffered_response,
    final_response,
    http_error,
    request_body,
    sse_lines,
)

from ducktape_provider import (
    APIError,
    ClaudeAdapter,
    ContextOverflowError,
    MalformedResponseError,
    Message,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    StreamEvent,
    ThinkingBlock,
    ToolDef,
)
from ducktape_provider.adapters import claude as claude_module
from ducktape_provider.streaming import _StreamTimer

MESSAGES: list[Message] = [
    {"role": "user", "content": [{"type": "text", "text": "weather in NYC?"}]}
]

FINAL_DATA: dict[str, Any] = {
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
        messages: list[Message] = [
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
        tools: list[ToolDef] = [
            {"name": "get_weather", "description": "...", "parameters": {"a": 1}}
        ]
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
        final = final_response(events)
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

        final = final_response(events)
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
        messages: list[Message] = [
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
        messages: list[Message] = [
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


def _text_stream(*middle: dict) -> list[dict]:
    """A minimal valid Claude stream with `middle` spliced in before message_stop."""
    return [
        {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
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
        *middle,
        {"type": "message_stop"},
    ]


class ClaudeThinkingSerializeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    def test_signed_thinking_block_passes_through(self):
        block: ThinkingBlock = {
            "type": "thinking",
            "thinking": "hmm",
            "signature": "sig",
        }
        serialized = self.adapter._serialize(
            [{"role": "assistant", "content": [block]}]
        )
        self.assertEqual(serialized[0]["content"], [block])

    def test_unsigned_thinking_block_is_dropped_and_logs_warning(self):
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "from another vendor"},
                    {"type": "text", "text": "answer"},
                ],
            }
        ]
        with self.assertLogs(claude_module.logger, level="WARNING"):
            serialized = self.adapter._serialize(messages)
        self.assertEqual(serialized[0]["content"], [{"type": "text", "text": "answer"}])

    def test_drop_warning_fires_once_per_request_with_count(self):
        turn: Message = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "a"},
                {"type": "thinking", "thinking": "b"},
                {"type": "text", "text": "answer"},
            ],
        }
        user: Message = {"role": "user", "content": [{"type": "text", "text": "more"}]}
        with self.assertLogs(claude_module.logger, level="WARNING") as logs:
            self.adapter._serialize([user, turn, user, turn, user])
        self.assertEqual(len(logs.records), 1)
        self.assertIn("dropping 4", logs.output[0])

    def test_message_left_empty_by_drop_is_omitted(self):
        messages: list[Message] = [
            {"role": "user", "content": [{"type": "text", "text": "q"}]},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "t"}]},
            {"role": "user", "content": [{"type": "text", "text": "again"}]},
        ]
        with self.assertLogs(claude_module.logger, level="WARNING"):
            serialized = self.adapter._serialize(messages)
        # Claude merges the now-consecutive user turns itself.
        self.assertEqual([m["role"] for m in serialized], ["user", "user"])
        self.assertTrue(all(m["content"] for m in serialized))

    def test_message_already_empty_is_left_alone(self):
        serialized = self.adapter._serialize([{"role": "user", "content": []}])
        self.assertEqual(serialized, [{"role": "user", "content": []}])


class ClaudeNullUsageTests(unittest.TestCase):
    def test_null_usage_fields_count_as_zero_and_omit_cache_keys(self):
        data = {
            "content": [],
            "stop_reason": None,
            "usage": {
                "input_tokens": 3,
                "output_tokens": None,
                "cache_read_input_tokens": None,
                "cache_creation_input_tokens": None,
            },
        }
        response = ClaudeAdapter()._deserialize(data, 0.0)
        self.assertEqual(response["usage"], {"input_tokens": 3, "output_tokens": 0})
        self.assertEqual(response["raw_stop_reason"], "")


class ClaudeStreamContentTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    @patch("urllib.request.urlopen")
    def test_thinking_and_signature_accumulate_into_final_block(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(
            sse_lines(
                {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "let me "},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "think"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "EqQB"},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 4},
                },
                {"type": "message_stop"},
            )
        )

        events = list(self.adapter.stream_chat("claude-x", MESSAGES))

        self.assertEqual(
            [e["type"] for e in events],
            ["thinking_delta", "thinking_delta", "block_stop", "message_stop"],
        )
        self.assertEqual(
            final_response(events)["content"],
            [{"type": "thinking", "thinking": "let me think", "signature": "EqQB"}],
        )

    @patch("urllib.request.urlopen")
    def test_server_tool_use_input_is_parsed_and_usage_merged(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(
            sse_lines(
                {
                    "type": "message_start",
                    "message": {
                        "usage": {
                            "input_tokens": 2679,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "output_tokens": 3,
                        }
                    },
                },
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "server_tool_use",
                        "id": "srvtoolu_1",
                        "name": "web_search",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"query'},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '": "nyc"}'},
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_1",
                        "content": [],
                    },
                },
                {"type": "content_block_stop", "index": 1},
                {
                    "type": "content_block_start",
                    "index": 2,
                    "content_block": {"type": "text", "text": ""},
                },
                {
                    "type": "content_block_delta",
                    "index": 2,
                    "delta": {"type": "text_delta", "text": "Sunny"},
                },
                {"type": "content_block_stop", "index": 2},
                {"type": "ping"},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {
                        "input_tokens": 10682,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 100,
                        "output_tokens": 510,
                        "server_tool_use": {"web_search_requests": 1},
                    },
                },
                {"type": "message_stop"},
            )
        )

        events = list(self.adapter.stream_chat("claude-x", MESSAGES))

        # server-executed blocks surface no events at all, not even a lone block_stop
        self.assertEqual(
            [(e["type"], e.get("index")) for e in events],
            [("text_delta", 2), ("block_stop", 2), ("message_stop", None)],
        )
        final = final_response(events)
        self.assertEqual(
            final["content"][0],
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"query": "nyc"},
            },
        )
        self.assertEqual(final["content"][2], {"type": "text", "text": "Sunny"})
        self.assertEqual(
            final["usage"],
            {
                "input_tokens": 10782,
                "output_tokens": 510,
                "cache_read_tokens": 100,
                "cache_write_tokens": 0,
            },
        )

    @patch("urllib.request.urlopen")
    def test_message_delta_without_usage_or_with_nulls_keeps_prior_counts(
        self, mock_urlopen
    ):
        mock_urlopen.return_value = FakeStreamResponse(
            sse_lines(
                *_text_stream(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 7, "input_tokens": None},
                    },
                    {"type": "message_delta", "delta": {"stop_reason": None}},
                )
            )
        )

        events = list(self.adapter.stream_chat("claude-x", MESSAGES))

        final = final_response(events)
        self.assertEqual(final["usage"], {"input_tokens": 1, "output_tokens": 7})
        self.assertEqual(final["stop_reason"], "end_turn")


class ClaudeStreamFlushOnMessageStopTests(unittest.TestCase):
    """A block still open at message_stop (no content_block_stop for it) must
    still contribute its buffered deltas to the final response."""

    def setUp(self):
        self.adapter = ClaudeAdapter()

    @patch("urllib.request.urlopen")
    def test_text_without_content_block_stop_is_kept(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(
            sse_lines(
                {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
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
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 4},
                },
                {"type": "message_stop"},
            )
        )
        response = final_response(list(self.adapter.stream_chat("claude-x", MESSAGES)))
        self.assertEqual(
            response["content"], [{"type": "text", "text": "Let me check"}]
        )

    @patch("urllib.request.urlopen")
    def test_thinking_and_signature_without_content_block_stop_are_kept(
        self, mock_urlopen
    ):
        mock_urlopen.return_value = FakeStreamResponse(
            sse_lines(
                {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "thinking",
                        "thinking": "",
                        "signature": "",
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "thinking_delta", "thinking": "let me think"},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "EqQB"},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 4},
                },
                {"type": "message_stop"},
            )
        )
        response = final_response(list(self.adapter.stream_chat("claude-x", MESSAGES)))
        self.assertEqual(
            response["content"],
            [{"type": "thinking", "thinking": "let me think", "signature": "EqQB"}],
        )

    @patch("urllib.request.urlopen")
    def test_tool_use_input_without_content_block_stop_is_kept(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(
            sse_lines(
                {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"city": "NYC"}',
                    },
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "tool_use"},
                    "usage": {"output_tokens": 4},
                },
                {"type": "message_stop"},
            )
        )
        response = final_response(list(self.adapter.stream_chat("claude-x", MESSAGES)))
        self.assertEqual(
            response["content"],
            [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                }
            ],
        )


class ClaudeStreamErrorTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    def _stream(self, lines, error=None):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = FakeStreamResponse(lines, error)
            return list(self.adapter.stream_chat("claude-x", MESSAGES))

    def test_overloaded_error_event_raises_server_error(self):
        lines = sse_lines(
            _text_stream()[0],
            {
                "type": "error",
                "error": {"type": "overloaded_error", "message": "Overloaded"},
            },
        )
        with self.assertRaises(ServerError) as ctx:
            self._stream(lines)
        self.assertEqual(ctx.exception.status, 529)
        self.assertIn("Overloaded", str(ctx.exception))

    def test_rate_limit_error_event_raises_rate_limit_error(self):
        lines = sse_lines(
            {"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}
        )
        with self.assertRaises(RateLimitError):
            self._stream(lines)

    def test_context_overflow_error_event_raises_context_overflow_error(self):
        lines = sse_lines(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "prompt is too long: 300000 tokens",
                },
            }
        )
        with self.assertRaises(ContextOverflowError):
            self._stream(lines)

    def test_error_event_maps_every_documented_type(self):
        for error_type, status in claude_module._ERROR_TYPE_STATUS.items():
            lines = sse_lines(
                {"type": "error", "error": {"type": error_type, "message": "x"}}
            )
            with (
                self.subTest(error_type=error_type),
                self.assertRaises(APIError) as ctx,
            ):
                self._stream(lines)
            self.assertEqual(ctx.exception.status, status)

    def test_wrong_shape_events_raise_malformed_api_error(self):
        start = _text_stream()[0]
        cases = {
            "delta before block start": [
                start,
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "x"},
                },
            ],
            "missing index": [
                start,
                {"type": "content_block_start", "content_block": {"type": "text"}},
            ],
            "non-object event": [[1]],
            "non-object error": [{"type": "error", "error": "boom"}],
            "stop for unknown block": [
                start,
                {"type": "content_block_stop", "index": 3},
            ],
        }
        for label, events in cases.items():
            with self.subTest(label), self.assertRaises(MalformedResponseError) as ctx:
                self._stream(sse_lines(*events))
            self.assertIs(type(ctx.exception), MalformedResponseError)
            self.assertIn("malformed", str(ctx.exception))

    def test_exception_thrown_in_by_consumer_is_not_relabeled(self):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = FakeStreamResponse(sse_lines(*_text_stream()))
            # The concrete adapter returns a generator; the Adapter ABC only
            # promises an Iterator, which has no throw().
            stream = cast(
                Generator[StreamEvent], self.adapter.stream_chat("claude-x", MESSAGES)
            )
            next(stream)
            with self.assertRaises(KeyError):
                stream.throw(KeyError("consumer bug"))

    def test_stream_ending_without_message_stop_raises(self):
        with self.assertRaises(APIError) as ctx:
            self._stream(sse_lines(*_text_stream()[:-1]))
        self.assertIn("message_stop", str(ctx.exception))

    def test_malformed_sse_line_raises_api_error(self):
        lines = [*sse_lines(_text_stream()[0]), b"data: {not json\n", b"\n"]
        with self.assertRaises(MalformedResponseError) as ctx:
            self._stream(lines)
        self.assertIn("malformed", str(ctx.exception))

    def test_connection_reset_mid_stream_raises_api_error(self):
        with self.assertRaises(APIError) as ctx:
            self._stream(sse_lines(*_text_stream()[:3]), ConnectionResetError())
        self.assertIsInstance(ctx.exception.__cause__, ConnectionResetError)

    def test_incomplete_read_mid_stream_raises_api_error(self):
        with self.assertRaises(APIError):
            self._stream(
                sse_lines(*_text_stream()[:3]), http.client.IncompleteRead(b"")
            )

    def test_read_timeout_mid_stream_raises_request_timeout_error(self):
        with self.assertRaises(RequestTimeoutError):
            self._stream(sse_lines(*_text_stream()[:3]), TimeoutError())

    @patch("urllib.request.urlopen")
    def test_buffered_chat_with_malformed_body_raises_api_error(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(b"<html>bad gateway</html>")
        with self.assertRaises(APIError):
            self.adapter.chat("claude-x", MESSAGES)


class ClaudeLatencyTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    @patch("urllib.request.urlopen")
    def test_chat_latency_spans_request_to_body_read(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(json.dumps(FINAL_DATA).encode())
        with patch("time.monotonic", side_effect=[10.0, 10.5]):
            response = self.adapter.chat("claude-x", MESSAGES)
        self.assertEqual(response["latency_ms"], 500.0)
        self.assertNotIn("ttft_ms", response)

    @patch("urllib.request.urlopen")
    def test_stream_latency_and_ttft_use_wire_read_times(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*STREAM_EVENTS))
        # one clock read at request start, then one per SSE event parsed
        clock = [100.0 + i * 0.125 for i in range(len(STREAM_EVENTS) + 1)]
        with patch("time.monotonic", side_effect=clock):
            events = list(self.adapter.stream_chat("claude-x", MESSAGES))
        final = final_response(events)
        # first text_delta is the 3rd SSE event, message_stop the 9th
        self.assertEqual(final["ttft_ms"], 375.0)
        self.assertEqual(final["latency_ms"], 1125.0)


class ClaudeRequestValidationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    @patch("urllib.request.urlopen")
    def test_api_key_with_line_break_raises_value_error_without_the_key(
        self, mock_urlopen
    ):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-SECRET123\r\n"}):
            for call in (
                lambda: self.adapter.chat("claude-x", MESSAGES),
                lambda: list(self.adapter.stream_chat("claude-x", MESSAGES)),
            ):
                with self.assertRaises(ValueError) as ctx:
                    call()
                self.assertNotIsInstance(ctx.exception, APIError)
                self.assertIn("x-api-key", str(ctx.exception))
                self.assertNotIn("SECRET", str(ctx.exception))
                self.assertIsNone(ctx.exception.__cause__)
                self.assertIsNone(ctx.exception.__context__)
        mock_urlopen.assert_not_called()


class ClaudeReservedConfigTests(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_reserved_config_keys_are_rejected_before_any_request(self, mock_urlopen):
        adapter = ClaudeAdapter()
        for key in ("stream", "model", "messages"):
            with self.subTest(key=key), self.assertRaises(ValueError) as ctx:
                adapter.chat("claude-x", MESSAGES, config={key: True})
            self.assertIn(key, str(ctx.exception))
        with self.assertRaises(ValueError):
            list(adapter.stream_chat("claude-x", MESSAGES, config={"stream": False}))
        mock_urlopen.assert_not_called()

    def test_timeout_and_headers_are_not_sent_in_body(self):
        req, timeout = ClaudeAdapter()._build_request(
            "claude-x",
            MESSAGES,
            None,
            None,
            {"timeout": 5, "headers": {"x-extra": "1"}, "max_tokens": 10},
            stream=False,
        )
        sent = request_body(req)
        self.assertEqual(timeout, 5)
        self.assertNotIn("timeout", sent)
        self.assertNotIn("headers", sent)
        self.assertEqual(sent["max_tokens"], 10)
        self.assertEqual(req.get_header("X-extra"), "1")


class ClaudeStreamAccumulationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    def test_long_streams_accumulate_in_linear_time(self):
        # Few large deltas, so quadratic accumulation copies tens of GiB (~45 s
        # measured at this size) while linear takes ~30 ms: a wide margin either way.
        n, delta = 4000, "x" * 16 * 1024
        cases = [
            ("thinking", "thinking_delta", "thinking"),
            ("thinking", "signature_delta", "signature"),
            ("text", "text_delta", "text"),
            ("tool_use", "input_json_delta", "partial_json"),
        ]
        for block_type, delta_type, field in cases:
            with self.subTest(field):
                handle = self.adapter._stream_handler(_StreamTimer())
                list(handle({"type": "message_start", "message": {"usage": {}}}))
                block: dict[str, Any] = {"type": block_type}
                if block_type == "tool_use":
                    block |= {"id": "t1", "name": "f", "input": {}}
                list(
                    handle(
                        {
                            "type": "content_block_start",
                            "index": 0,
                            "content_block": block,
                        }
                    )
                )
                event = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": delta_type, field: delta},
                }
                start = time.perf_counter()
                for _ in range(n):
                    list(handle(event))
                list(handle({"type": "content_block_stop", "index": 0}))
                self.assertLess(time.perf_counter() - start, 1.0)

    def test_accumulated_text_and_thinking_are_correct(self):
        n = 1000
        handle = self.adapter._stream_handler(_StreamTimer())
        list(handle({"type": "message_start", "message": {"usage": {}}}))
        list(
            handle(
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                }
            )
        )
        for _ in range(n):
            list(
                handle(
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "ab"},
                    }
                )
            )
        list(handle({"type": "content_block_stop", "index": 0}))
        list(
            handle(
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {},
                }
            )
        )
        response = final_response(list(handle({"type": "message_stop"})))
        self.assertEqual(response["content"], [{"type": "text", "text": "ab" * n}])


class ClaudeInvalidToolArgsTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    @patch("urllib.request.urlopen")
    def test_deeply_nested_partial_json_falls_back_to_empty_input(self, mock_urlopen):
        deep = "[" * 200_000 + "]" * 200_000
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "t1",
                    "name": "f",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": deep},
            },
            {"type": "content_block_stop", "index": 0},
            {"type": "message_stop"},
        ]
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*events))
        response = final_response(list(self.adapter.stream_chat("claude-x", MESSAGES)))
        self.assertEqual(
            response["content"][0],
            {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
        )

    @patch("urllib.request.urlopen")
    def test_non_object_json_falls_back_to_empty_input(self, mock_urlopen):
        for partial_json in ("[1,2,3]", "1", "null"):
            with self.subTest(partial_json=partial_json):
                events = [
                    {
                        "type": "message_start",
                        "message": {"usage": {"input_tokens": 1}},
                    },
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "f",
                            "input": {},
                        },
                    },
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": partial_json,
                        },
                    },
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_stop"},
                ]
                mock_urlopen.return_value = FakeStreamResponse(sse_lines(*events))
                response = final_response(
                    list(self.adapter.stream_chat("claude-x", MESSAGES))
                )
                self.assertEqual(
                    response["content"][0],
                    {"type": "tool_use", "id": "t1", "name": "f", "input": {}},
                )


class ClaudeModelsCacheInvalidationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()
        self.adapter._models_cache = {"claude-old"}
        self.adapter._cache_time = time.monotonic()

    @patch("urllib.request.urlopen")
    def test_chat_404_invalidates_models_cache(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            ClaudeAdapter._MESSAGES_URL, 404, b"no such model"
        )
        with self.assertRaises(APIError):
            self.adapter.chat("claude-x", MESSAGES)
        self.assertIsNone(self.adapter._models_cache)

    @patch("urllib.request.urlopen")
    def test_stream_chat_404_invalidates_models_cache(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            ClaudeAdapter._MESSAGES_URL, 404, b"no such model"
        )
        with self.assertRaises(APIError):
            list(self.adapter.stream_chat("claude-x", MESSAGES))
        self.assertIsNone(self.adapter._models_cache)

    @patch("urllib.request.urlopen")
    def test_non_404_error_leaves_models_cache_untouched(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(ClaudeAdapter._MESSAGES_URL, 500, b"boom")
        with self.assertRaises(APIError):
            self.adapter.chat("claude-x", MESSAGES)
        self.assertEqual(self.adapter._models_cache, {"claude-old"})


if __name__ == "__main__":
    unittest.main()
