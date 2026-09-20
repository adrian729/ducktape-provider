"""Tests for ClaudeAdapter: _serialize/_deserialize plus chat() and stream_chat()
with urllib.request.urlopen mocked out. No real network call is ever made."""

import http.client
import json
import os
import time
import unittest
from collections.abc import Generator
from typing import Any, cast
from unittest.mock import Mock, patch

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
    AuthError,
    CacheControl,
    ClaudeAdapter,
    ContextOverflowError,
    MalformedResponseError,
    Message,
    ModelInfo,
    Provider,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    StreamEvent,
    SystemBlock,
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
        self.adapter = ClaudeAdapter(api_key="test")

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

    def test_serialize_preserves_cache_control_on_blocks(self):
        cache: CacheControl = {"type": "ephemeral"}
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hi", "cache_control": cache},
                    {
                        "type": "image",
                        "source": "base64",
                        "media_type": "image/png",
                        "data": "abc",
                        "cache_control": cache,
                    },
                    {
                        "type": "document",
                        "source": "url",
                        "url": "https://example.test/doc",
                        "cache_control": cache,
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "name": "n",
                        "content": "ok",
                        "cache_control": cache,
                    },
                ],
            }
        ]
        serialized = self.adapter._serialize(messages)
        for block in serialized[0]["content"]:
            self.assertEqual(block["cache_control"], cache)

    def test_serialize_tools_preserves_cache_control(self):
        cache: CacheControl = {"type": "ephemeral", "ttl": "1h"}
        tools: list[ToolDef] = [
            {
                "name": "get_weather",
                "description": "...",
                "parameters": {"a": 1},
                "cache_control": cache,
            }
        ]
        self.assertEqual(
            self.adapter._serialize_tools(tools)[0]["cache_control"], cache
        )

    def test_system_blocks_pass_through(self):
        system: list[SystemBlock] = [
            {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
        ]
        req, _ = self.adapter._build_request("m", [], system, None, None, False)
        self.assertEqual(request_body(req)["system"], system)

    def test_system_string_passes_through_unchanged(self):
        req, _ = self.adapter._build_request("m", [], "plain", None, None, False)
        self.assertEqual(request_body(req)["system"], "plain")

    def test_serialize_omits_cache_control_when_absent(self):
        messages: list[Message] = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]
        self.assertNotIn(
            "cache_control", self.adapter._serialize(messages)[0]["content"][0]
        )

    def test_serialize_tools_omits_cache_control_when_absent(self):
        tools: list[ToolDef] = [{"name": "t", "description": "d", "parameters": {}}]
        self.assertNotIn("cache_control", self.adapter._serialize_tools(tools)[0])

    def test_serialize_copies_cache_control(self):
        cache: CacheControl = {"type": "ephemeral"}
        messages: list[Message] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi", "cache_control": cache}],
            }
        ]
        serialized = self.adapter._serialize(messages)
        cache["ttl"] = "1h"
        self.assertEqual(
            serialized[0]["content"][0]["cache_control"], {"type": "ephemeral"}
        )

    def test_system_blocks_are_copied(self):
        cache: CacheControl = {"type": "ephemeral"}
        system: list[SystemBlock] = [
            {"type": "text", "text": "a", "cache_control": cache}
        ]
        req, _ = self.adapter._build_request("m", [], system, None, None, False)
        cache["ttl"] = "1h"
        self.assertEqual(
            request_body(req)["system"],
            [{"type": "text", "text": "a", "cache_control": {"type": "ephemeral"}}],
        )

    def test_cache_control_ttl_preserved(self):
        cache: CacheControl = {"type": "ephemeral", "ttl": "1h"}
        messages: list[Message] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi", "cache_control": cache}],
            }
        ]
        self.assertEqual(
            self.adapter._serialize(messages)[0]["content"][0]["cache_control"], cache
        )


class ClaudeDeserializeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter(api_key="test")

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
        usage = response["usage"]
        assert usage is not None
        self.assertEqual(
            usage, {"input_tokens": 103, "output_tokens": 5, "cache_read_tokens": 100}
        )
        self.assertNotIn("cache_write_tokens", usage)

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
        usage = response["usage"]
        assert usage is not None
        self.assertEqual(
            usage, {"input_tokens": 23, "output_tokens": 5, "cache_write_tokens": 20}
        )
        self.assertNotIn("cache_read_tokens", usage)

    def test_omits_cache_usage_fields_when_absent(self):
        data = {
            "content": [],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }
        response = self.adapter._deserialize(data, 0.0)
        usage = response["usage"]
        assert usage is not None
        self.assertNotIn("cache_read_tokens", usage)
        self.assertNotIn("cache_write_tokens", usage)


class ClaudeChatHTTPTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter(api_key="test")

    @patch("urllib.request.urlopen")
    def test_chat_returns_deserialized_response(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(json.dumps(FINAL_DATA).encode())

        response = self.adapter.chat("claude-x", MESSAGES)

        self.assertEqual(response["content"], FINAL_DATA["content"])
        self.assertEqual(response["stop_reason"], "tool_use")
        self.assertEqual(response["usage"], {"input_tokens": 10, "output_tokens": 8})
        self.assertIsInstance(response["latency_ms"], float)
        self.assertGreaterEqual(response["latency_ms"], 0)
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
        self.adapter = ClaudeAdapter(api_key="test")

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
        self.adapter = ClaudeAdapter(api_key="test")

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

    @patch("urllib.request.urlopen")
    def test_no_header_follows_a_redirect(self, mock_urlopen):
        calls = {
            "chat": (
                lambda: self.adapter.chat("claude-x", MESSAGES),
                lambda: buffered_response(json.dumps(FINAL_DATA).encode()),
            ),
            "stream_chat": (
                lambda: list(self.adapter.stream_chat("claude-x", MESSAGES)),
                lambda: FakeStreamResponse(sse_lines(*STREAM_EVENTS)),
            ),
            "models": (
                self.adapter.models,
                lambda: buffered_response(b'{"data": [{"id": "claude-x"}]}'),
            ),
        }
        for label, (call, reply) in calls.items():
            with self.subTest(label):
                mock_urlopen.return_value = reply()
                call()
                req = mock_urlopen.call_args.args[0]
                self.assertEqual(req.headers, {})
                self.assertEqual(req.get_header("X-api-key"), "test")
                self.assertEqual(req.get_header("Anthropic-version"), "2023-06-01")

    @patch("urllib.request.urlopen")
    def test_config_auth_header_wins_and_key_source_is_not_called(self, mock_urlopen):
        source = Mock(return_value="from-source")
        adapter = ClaudeAdapter(api_key=source)
        for name in ("x-api-key", "X-API-KEY"):
            with self.subTest(name):
                mock_urlopen.return_value = buffered_response(
                    json.dumps(FINAL_DATA).encode()
                )
                adapter.chat("claude-x", MESSAGES, config={"headers": {name: "cfg"}})
                req = mock_urlopen.call_args.args[0]
                self.assertEqual(req.get_header("X-api-key"), "cfg")
        source.assert_not_called()


class ClaudeBlockShapeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter(api_key="test")

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

    def test_tool_result_with_list_content_serializes_text_and_image_blocks(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
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
        serialized = self.adapter._serialize(messages)
        self.assertEqual(
            serialized[0]["content"],
            [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": [
                        {"type": "text", "text": "here"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "abc",
                            },
                        },
                    ],
                }
            ],
        )

    def test_tool_result_with_empty_list_content_sends_an_empty_array(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "name": "noop",
                        "content": [],
                    }
                ],
            }
        ]
        serialized = self.adapter._serialize(messages)
        self.assertEqual(serialized[0]["content"][0]["content"], [])

    def test_tool_result_is_error_with_list_content_still_sets_is_error(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "name": "screenshot",
                        "content": [{"type": "text", "text": "failed"}],
                        "is_error": True,
                    }
                ],
            }
        ]
        serialized = self.adapter._serialize(messages)
        self.assertEqual(serialized[0]["content"][0]["is_error"], True)
        self.assertEqual(
            serialized[0]["content"][0]["content"], [{"type": "text", "text": "failed"}]
        )

    def test_tool_result_plain_str_content_still_passes_through(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "name": "get_weather",
                        "content": "sunny",
                    }
                ],
            }
        ]
        serialized = self.adapter._serialize(messages)
        self.assertEqual(serialized[0]["content"][0]["content"], "sunny")


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
        self.adapter = ClaudeAdapter(api_key="test")

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
        response = ClaudeAdapter(api_key="test")._deserialize(data, 0.0)
        self.assertEqual(response["usage"], {"input_tokens": 3, "output_tokens": 0})
        self.assertEqual(response["raw_stop_reason"], "")


class ClaudeStreamContentTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter(api_key="test")

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
        self.adapter = ClaudeAdapter(api_key="test")

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
        self.adapter = ClaudeAdapter(api_key="test")

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
        self.adapter = ClaudeAdapter(api_key="test")

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
        clock = [100.0 + i * 0.125 for i in range(len(STREAM_EVENTS) + 1)]
        with patch("time.monotonic", side_effect=clock):
            events = list(
                self.adapter.stream_chat("claude-x", MESSAGES, config={"timeout": None})
            )
        final = final_response(events)
        self.assertEqual(final["ttft_ms"], 375.0)
        self.assertEqual(final["latency_ms"], 1125.0)


class ClaudeRequestValidationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    @patch("urllib.request.urlopen", side_effect=AssertionError("request sent"))
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
                self.assertIn("claude API key", str(ctx.exception))
                self.assertIn("trailing newline", str(ctx.exception))
                self.assertNotIn("SECRET", str(ctx.exception))
                self.assertIsNone(ctx.exception.__cause__)
                self.assertIsNone(ctx.exception.__context__)
        mock_urlopen.assert_not_called()


class ClaudeReservedConfigTests(unittest.TestCase):
    @patch("urllib.request.urlopen", side_effect=AssertionError("request sent"))
    def test_reserved_config_keys_are_rejected_before_any_request(self, mock_urlopen):
        adapter = ClaudeAdapter(api_key="test")
        for key in ("stream", "model", "messages"):
            with self.subTest(key=key), self.assertRaises(ValueError) as ctx:
                adapter.chat("claude-x", MESSAGES, config={key: True})
            self.assertIn(key, str(ctx.exception))
        with self.assertRaises(ValueError):
            list(adapter.stream_chat("claude-x", MESSAGES, config={"stream": False}))
        mock_urlopen.assert_not_called()

    def test_timeout_and_headers_are_not_sent_in_body(self):
        req, timeout = ClaudeAdapter(api_key="test")._build_request(
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
        self.adapter = ClaudeAdapter(api_key="test")

    def test_long_streams_accumulate_in_linear_time(self):
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
        self.adapter = ClaudeAdapter(api_key="test")

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
            {
                "type": "tool_use",
                "id": "t1",
                "name": "f",
                "input": {},
                "truncated": True,
            },
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
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "f",
                        "input": {},
                        "truncated": True,
                    },
                )

    @patch("urllib.request.urlopen")
    def test_legitimately_empty_json_is_not_marked_truncated(self, mock_urlopen):
        for partial_json in ("", "{}"):
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


class ClaudeModelInfoTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter(api_key="test")

    @patch("urllib.request.urlopen")
    def test_reads_context_window_and_max_output_from_the_models_listing(
        self, mock_urlopen
    ):
        mock_urlopen.return_value = buffered_response(
            json.dumps(
                {
                    "data": [
                        {
                            "id": "claude-x",
                            "max_input_tokens": 200000,
                            "max_tokens": 8192,
                        }
                    ]
                }
            ).encode()
        )
        self.assertEqual(
            self.adapter.model_info("claude-x"),
            {"context_window": 200000, "max_output_tokens": 8192},
        )

    @patch("urllib.request.urlopen")
    def test_null_fields_become_none_not_zero_or_missing(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(
                {
                    "data": [
                        {
                            "id": "claude-x",
                            "max_input_tokens": None,
                            "max_tokens": None,
                        }
                    ]
                }
            ).encode()
        )
        self.assertEqual(
            self.adapter.model_info("claude-x"),
            {"context_window": None, "max_output_tokens": None},
        )

    @patch("urllib.request.urlopen")
    def test_model_not_in_any_page_returns_none(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps({"data": [{"id": "claude-other"}]}).encode()
        )
        self.assertIsNone(self.adapter.model_info("claude-x"))

    @patch("urllib.request.urlopen")
    def test_reuses_the_same_cache_as_models(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(
                {"data": [{"id": "claude-x", "max_input_tokens": 200000}]}
            ).encode()
        )
        self.assertEqual(self.adapter.models(), {"claude-x"})
        self.assertEqual(
            self.adapter.model_info("claude-x"),
            {"context_window": 200000, "max_output_tokens": None},
        )
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_model_info_alone_populates_the_cache_in_one_call(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(
                {"data": [{"id": "claude-x", "max_input_tokens": 200000}]}
            ).encode()
        )
        self.assertEqual(
            self.adapter.model_info("claude-x"),
            {"context_window": 200000, "max_output_tokens": None},
        )
        self.assertEqual(self.adapter.models(), {"claude-x"})
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_probe_failure_returns_none_like_models_returns_empty(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(ClaudeAdapter._MODELS_URL, 500, b"boom")
        self.assertIsNone(self.adapter.model_info("claude-x"))

    @patch("urllib.request.urlopen")
    def test_a_stale_warm_cache_does_not_survive_a_later_failed_probe(
        self, mock_urlopen
    ):
        self.adapter._models_cache = {
            "claude-x": {"context_window": 200000, "max_output_tokens": 8192}
        }
        self.adapter._cache_time = 0.0
        mock_urlopen.side_effect = http_error(ClaudeAdapter._MODELS_URL, 500, b"boom")
        self.assertEqual(self.adapter.models(), set())
        self.assertIsNone(self.adapter.model_info("claude-x"))


STALE_MODEL_INFO: dict[str, ModelInfo] = {
    "claude-old": {"context_window": None, "max_output_tokens": None}
}


class ClaudeModelsCacheInvalidationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter(api_key="test")
        self.adapter._models_cache = dict(STALE_MODEL_INFO)
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
        self.assertEqual(self.adapter._models_cache, STALE_MODEL_INFO)


class TestClaudeCapabilities(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter(api_key="test")

    def _paged_capabilities_payloads(self):
        """Two pages where second page omits some capability keys."""
        caps_a = {
            "image_input": {"supported": True},
            "pdf_input": {"supported": False},
            "thinking": {"supported": True},
            "citations": {"supported": True},
            "batch": {"supported": False},
        }
        caps_b = {
            "image_input": {"supported": False},
            "thinking": {"supported": False},
            "effort": {"supported": True},
        }
        page1 = {
            "data": [
                {
                    "id": "claude-a",
                    "max_input_tokens": 200000,
                    "max_tokens": 8192,
                    "capabilities": caps_a,
                }
            ],
            "has_more": True,
            "last_id": "claude-a",
        }
        page2 = {
            "data": [
                {
                    "id": "claude-b",
                    "max_input_tokens": 100000,
                    "max_tokens": 4096,
                    "capabilities": caps_b,
                }
            ],
            "has_more": False,
            "last_id": "claude-b",
        }
        return caps_a, caps_b, page1, page2

    def _paged_urlopen(self, page1, page2):
        """Mock urlopen returning page1 then page2 based on after_id."""

        def urlopen(req, timeout=None):
            url = req.full_url
            if "after_id=claude-a" in url:
                return buffered_response(json.dumps(page2).encode())
            return buffered_response(json.dumps(page1).encode())

        return urlopen

    @patch("urllib.request.urlopen")
    def test_paged_capabilities_projection_and_raw(self, mock_urlopen):
        """Paged list projects capabilities and keeps raw equal to vendor map."""
        caps_a, caps_b, page1, page2 = self._paged_capabilities_payloads()
        mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
        caps = self.adapter.capabilities("claude-a")
        assert caps is not None
        self.assertEqual(caps["vision"], True)
        self.assertEqual(caps["pdf_input"], False)
        self.assertEqual(caps["thinking"], True)
        self.assertIsNone(caps["tools"])
        assert "raw" in caps
        self.assertEqual(caps["raw"], caps_a)
        self.assertEqual(set(caps["raw"].keys()), set(caps_a.keys()))
        caps_b_result = self.adapter.capabilities("claude-b")
        assert caps_b_result is not None
        self.assertEqual(caps_b_result["vision"], False)
        self.assertEqual(caps_b_result["thinking"], False)
        self.assertIsNone(caps_b_result["pdf_input"])
        self.assertIsNone(caps_b_result["tools"])
        assert "raw" in caps_b_result
        self.assertEqual(caps_b_result["raw"], caps_b)
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("urllib.request.urlopen")
    def test_capabilities_warms_model_info_and_reverse(self, mock_urlopen):
        """Warm cache from one serves the other with no extra request."""
        _, _, page1, page2 = self._paged_capabilities_payloads()
        mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
        caps = self.adapter.capabilities("claude-a")
        assert caps is not None
        self.assertEqual(mock_urlopen.call_count, 2)
        info = self.adapter.model_info("claude-a")
        self.assertEqual(info, {"context_window": 200000, "max_output_tokens": 8192})
        self.assertEqual(mock_urlopen.call_count, 2)
        info_b = self.adapter.model_info("claude-b")
        self.assertEqual(info_b, {"context_window": 100000, "max_output_tokens": 4096})
        self.assertEqual(mock_urlopen.call_count, 2)
        adapter2 = ClaudeAdapter(api_key="test")
        mock_urlopen.reset_mock()
        mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
        self.assertEqual(adapter2.models(), {"claude-a", "claude-b"})
        self.assertEqual(mock_urlopen.call_count, 2)
        caps2 = adapter2.capabilities("claude-a")
        assert caps2 is not None
        self.assertEqual(caps2["vision"], True)
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("urllib.request.urlopen")
    def test_absent_id_falls_back_to_single_model(self, mock_urlopen):
        """An id absent from the listing resolves via the single-model endpoint."""
        _, _, page1, page2 = self._paged_capabilities_payloads()

        def urlopen(req, timeout=None):
            if "/models/" in req.full_url:
                return buffered_response(
                    json.dumps(
                        {
                            "id": "alias-model",
                            "capabilities": {"image_input": {"supported": True}},
                        }
                    ).encode()
                )
            if "after_id=claude-a" in req.full_url:
                return buffered_response(json.dumps(page2).encode())
            return buffered_response(json.dumps(page1).encode())

        mock_urlopen.side_effect = urlopen
        caps = self.adapter.capabilities("alias-model")
        assert caps is not None
        self.assertEqual(caps["vision"], True)
        self.assertEqual(
            len(
                [
                    c
                    for c in mock_urlopen.call_args_list
                    if "/models/" in c.args[0].full_url
                ]
            ),
            1,
        )
        mock_urlopen.reset_mock()
        caps2 = self.adapter.capabilities("alias-model")
        assert caps2 is not None
        self.assertEqual(caps2["vision"], True)
        self.assertEqual(mock_urlopen.call_count, 0)

    @patch("urllib.request.urlopen")
    def test_single_model_404_is_none_and_negative_cached(self, mock_urlopen):
        """A single-model 404 yields None and is cached under the TTL."""
        _, _, page1, page2 = self._paged_capabilities_payloads()

        def urlopen(req, timeout=None):
            if "/models/" in req.full_url:
                raise http_error(req.full_url, 404, b"not found")
            if "after_id=claude-a" in req.full_url:
                return buffered_response(json.dumps(page2).encode())
            return buffered_response(json.dumps(page1).encode())

        mock_urlopen.side_effect = urlopen
        self.assertIsNone(self.adapter.capabilities("missing-model"))
        mock_urlopen.reset_mock()
        self.assertIsNone(self.adapter.capabilities("missing-model"))
        self.assertEqual(mock_urlopen.call_count, 0)

    @patch("urllib.request.urlopen")
    def test_chat_404_clears_caches_and_next_refetches(self, mock_urlopen):
        """Chat 404 clears models caches and next call refetches."""
        _, _, page1, page2 = self._paged_capabilities_payloads()
        mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
        self.adapter.capabilities("claude-a")
        self.assertIsNotNone(self.adapter._models_cache)
        self.assertIsNotNone(self.adapter._models_raw)
        self.assertNotEqual(self.adapter._cache_time, 0.0)
        mock_urlopen.side_effect = http_error(
            ClaudeAdapter._MESSAGES_URL, 404, b"no such model"
        )
        with self.assertRaises(APIError) as ctx:
            self.adapter.chat("claude-a", MESSAGES)
        self.assertEqual(ctx.exception.status, 404)
        self.assertIsNone(self.adapter._models_cache)
        self.assertIsNone(self.adapter._models_raw)
        self.assertEqual(self.adapter._cache_time, 0.0)
        mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
        caps = self.adapter.capabilities("claude-a")
        assert caps is not None
        self.assertEqual(mock_urlopen.call_count, 5)
        adapter2 = ClaudeAdapter(api_key="test")
        mock_urlopen.reset_mock()
        mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
        adapter2.models()
        mock_urlopen.side_effect = http_error(
            ClaudeAdapter._MESSAGES_URL, 404, b"no such model"
        )
        with self.assertRaises(APIError):
            adapter2.chat("claude-a", MESSAGES)
        self.assertIsNone(adapter2._models_cache)
        self.assertIsNone(adapter2._models_raw)
        self.assertEqual(adapter2._cache_time, 0.0)
        mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
        self.assertEqual(adapter2.models(), {"claude-a", "claude-b"})
        self.assertEqual(mock_urlopen.call_count, 5)

    def test_missing_api_key_raises_auth_error_while_models_empty(self):
        """Missing key raises AuthError from capabilities and empty set from models."""
        with patch.dict(os.environ, {}, clear=False):
            for key in list(os.environ):
                if key.lower() == "anthropic_api_key":
                    del os.environ[key]
            adapter = ClaudeAdapter()
            provider = Provider(adapters={"claude": adapter})
            with patch(
                "urllib.request.urlopen", side_effect=AssertionError("no request")
            ) as mock_urlopen:
                with self.assertRaises(AuthError):
                    provider.capabilities("claude-a", provider="claude")
                mock_urlopen.assert_not_called()
            self.assertEqual(adapter.models(), set())
            self.assertEqual(provider.models(), {})

    @patch("urllib.request.urlopen")
    def test_401_from_listing_raises_auth_error(self, mock_urlopen):
        """401 from listing raises AuthError from capabilities."""
        mock_urlopen.side_effect = http_error(
            ClaudeAdapter._MODELS_URL, 401, b"unauthorized"
        )
        with self.assertRaises(AuthError) as ctx:
            self.adapter.capabilities("claude-a")
        self.assertEqual(ctx.exception.status, 401)
        with patch.dict(os.environ, {}, clear=False):
            for key in list(os.environ):
                if key.lower() == "anthropic_api_key":
                    del os.environ[key]
            with patch(
                "urllib.request.urlopen",
                side_effect=http_error(ClaudeAdapter._MODELS_URL, 401, b"unauthorized"),
            ):
                provider = Provider(adapters={"claude": ClaudeAdapter()})
                with self.assertRaises(AuthError):
                    provider.capabilities("claude-a", provider="claude")

    @patch("urllib.request.urlopen")
    def test_warm_cache_bidirectional_no_extra_request(self, mock_urlopen):
        """Warm cache filled by either caller serves the other."""
        _, _, page1, page2 = self._paged_capabilities_payloads()
        mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
        self.assertEqual(self.adapter.models(), {"claude-a", "claude-b"})
        self.assertEqual(mock_urlopen.call_count, 2)
        caps = self.adapter.capabilities("claude-b")
        assert caps is not None
        self.assertEqual(caps["vision"], False)
        self.assertEqual(mock_urlopen.call_count, 2)
        adapter2 = ClaudeAdapter(api_key="test")
        mock_urlopen.reset_mock()
        mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
        adapter2.capabilities("claude-a")
        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(adapter2.models(), {"claude-a", "claude-b"})
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("urllib.request.urlopen")
    def test_zero_tokens_preserved(self, mock_urlopen):
        """Zero max tokens are preserved not coerced to None."""
        payload = {
            "data": [
                {
                    "id": "claude-zero",
                    "max_input_tokens": 0,
                    "max_tokens": 0,
                    "capabilities": {"image_input": {"supported": True}},
                }
            ],
            "has_more": False,
        }
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        info = self.adapter.model_info("claude-zero")
        self.assertEqual(info, {"context_window": 0, "max_output_tokens": 0})
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        adapter2 = ClaudeAdapter(api_key="test")
        with patch(
            "urllib.request.urlopen",
            return_value=buffered_response(json.dumps(payload).encode()),
        ):
            caps = adapter2.capabilities("claude-zero")
            assert caps is not None
            self.assertEqual(caps["vision"], True)
        self.assertEqual(self.adapter.models(), {"claude-zero"})
        info2 = self.adapter.model_info("claude-zero")
        self.assertEqual(info2, {"context_window": 0, "max_output_tokens": 0})

    @patch("urllib.request.urlopen")
    def test_cache_poisoning_via_raw_isolated(self, mock_urlopen):
        """Mutating returned raw does not poison cache or model_info."""
        caps_a, _caps_b, page1, page2 = self._paged_capabilities_payloads()
        mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
        caps = self.adapter.capabilities("claude-a")
        assert caps is not None
        assert "raw" in caps
        caps["raw"]["image_input"]["supported"] = False
        caps["raw"]["new_key"] = {"supported": True}
        caps2 = self.adapter.capabilities("claude-a")
        assert caps2 is not None
        assert "raw" in caps2
        self.assertEqual(caps2["raw"], caps_a)
        self.assertNotIn("new_key", caps2["raw"])
        self.assertEqual(caps2["vision"], True)
        assert self.adapter._models_raw is not None
        self.assertEqual(self.adapter._models_raw["claude-a"]["capabilities"], caps_a)
        info = self.adapter.model_info("claude-a")
        self.assertEqual(info, {"context_window": 200000, "max_output_tokens": 8192})

    @patch("urllib.request.urlopen")
    def test_raw_equals_vendor_capability_map(self, mock_urlopen):
        """Raw equals vendor capability map with no extra keys."""
        caps_a, caps_b, page1, page2 = self._paged_capabilities_payloads()
        mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
        caps = self.adapter.capabilities("claude-a")
        assert caps is not None
        self.assertEqual(caps["raw"], caps_a)
        self.assertEqual(
            set(caps.keys()), {"tools", "vision", "pdf_input", "thinking", "raw"}
        )
        self.assertIsNone(caps["tools"])
        caps_b_result = self.adapter.capabilities("claude-b")
        assert caps_b_result is not None
        self.assertEqual(caps_b_result["raw"], caps_b)
        self.assertIsNone(caps_b_result["tools"])

    @patch("urllib.request.urlopen")
    def test_ttl_expiry_refetches(self, mock_urlopen):
        """Expired cache refetches on the 60s boundary."""
        _, _, page1, page2 = self._paged_capabilities_payloads()
        base = 1000.0
        with patch("ducktape_provider.adapters.claude.time.monotonic") as mock_time:
            mock_time.return_value = base
            mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
            self.adapter.capabilities("claude-a")
            self.assertEqual(mock_urlopen.call_count, 2)
            mock_time.return_value = base + 59
            self.adapter.capabilities("claude-a")
            self.assertEqual(mock_urlopen.call_count, 2)
            mock_time.return_value = base + 61
            self.adapter.capabilities("claude-a")
            self.assertEqual(mock_urlopen.call_count, 4)

    @patch("urllib.request.urlopen")
    def test_configured_auth_header_covers_missing_key(self, mock_urlopen):
        """A configured x-api-key header serves capabilities with no key source."""
        _, _, page1, page2 = self._paged_capabilities_payloads()
        mock_urlopen.side_effect = self._paged_urlopen(page1, page2)
        with patch.dict(os.environ, {}, clear=False):
            for key in list(os.environ):
                if key.lower() == "anthropic_api_key":
                    del os.environ[key]
            provider = Provider(
                adapters={"claude": ClaudeAdapter()},
                config={"providers": {"claude": {"headers": {"x-api-key": "t"}}}},
            )
            caps = provider.capabilities("claude-a", provider="claude")
            assert caps is not None
            self.assertEqual(caps["vision"], True)
            req = mock_urlopen.call_args_list[0].args[0]
            self.assertEqual(req.get_header("X-api-key"), "t")

    @patch("urllib.request.urlopen")
    def test_missing_data_key_is_malformed_for_capabilities(self, mock_urlopen):
        """A listing without a data key raises for capabilities, empty for models."""
        body = json.dumps({"has_more": False}).encode()
        mock_urlopen.return_value = buffered_response(body)
        with self.assertRaises(MalformedResponseError):
            self.adapter.capabilities("claude-a")
        mock_urlopen.return_value = buffered_response(body)
        self.assertEqual(self.adapter.models(), set())

    @patch("urllib.request.urlopen")
    def test_non_bool_supported_raises_malformed(self, mock_urlopen):
        """A non-bool supported value for a mapped key is malformed."""
        payload = {
            "data": [
                {
                    "id": "claude-x",
                    "capabilities": {"image_input": {"supported": "yes"}},
                }
            ],
            "has_more": False,
        }
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        with self.assertRaises(MalformedResponseError):
            self.adapter.capabilities("claude-x")

    @patch("urllib.request.urlopen")
    def test_duplicate_last_id_terminates_pagination(self, mock_urlopen):
        """A repeated last_id terminates the page loop instead of looping."""
        page = {
            "data": [
                {
                    "id": "claude-a",
                    "capabilities": {"image_input": {"supported": True}},
                }
            ],
            "has_more": True,
            "last_id": "claude-a",
        }
        mock_urlopen.side_effect = lambda *a, **k: buffered_response(
            json.dumps(page).encode()
        )
        caps = self.adapter.capabilities("claude-a")
        assert caps is not None
        self.assertEqual(caps["vision"], True)
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("urllib.request.urlopen")
    def test_fresh_per_id_cache_skips_list_fetch(self, mock_urlopen):
        """A fresh per-id entry is served without touching the list cache."""
        _, _, page1, page2 = self._paged_capabilities_payloads()
        alias = {"id": "alias", "capabilities": {"image_input": {"supported": True}}}

        def urlopen(req, timeout=None):
            if "/models/" in req.full_url:
                return buffered_response(json.dumps(alias).encode())
            if "after_id=claude-a" in req.full_url:
                return buffered_response(json.dumps(page2).encode())
            return buffered_response(json.dumps(page1).encode())

        mock_urlopen.side_effect = urlopen
        base = 1000.0
        with patch("ducktape_provider.adapters.claude.time.monotonic") as mock_time:
            mock_time.return_value = base
            self.adapter.capabilities("claude-a")
            mock_time.return_value = base + 30
            self.adapter.capabilities("alias")
            mock_urlopen.reset_mock()
            mock_time.return_value = base + 70
            caps = self.adapter.capabilities("alias")
            assert caps is not None
            self.assertEqual(caps["vision"], True)
            self.assertEqual(mock_urlopen.call_count, 0)

    @patch("urllib.request.urlopen")
    def test_per_id_cache_ttl_expiry_refetches(self, mock_urlopen):
        """An expired per-id entry refetches the list and the model."""
        _, _, page1, page2 = self._paged_capabilities_payloads()
        alias = {"id": "alias", "capabilities": {"image_input": {"supported": True}}}

        def urlopen(req, timeout=None):
            if "/models/" in req.full_url:
                return buffered_response(json.dumps(alias).encode())
            if "after_id=claude-a" in req.full_url:
                return buffered_response(json.dumps(page2).encode())
            return buffered_response(json.dumps(page1).encode())

        mock_urlopen.side_effect = urlopen
        base = 1000.0
        with patch("ducktape_provider.adapters.claude.time.monotonic") as mock_time:
            mock_time.return_value = base
            self.adapter.capabilities("alias")
            mock_urlopen.reset_mock()
            mock_time.return_value = base + 61
            caps = self.adapter.capabilities("alias")
            assert caps is not None
            self.assertEqual(caps["vision"], True)
            self.assertEqual(mock_urlopen.call_count, 3)

    @patch("urllib.request.urlopen")
    def test_single_model_path_is_percent_encoded(self, mock_urlopen):
        """The single-model path percent-encodes the model id."""
        _, _, page1, page2 = self._paged_capabilities_payloads()
        seen: list[str] = []

        def urlopen(req, timeout=None):
            seen.append(req.full_url)
            if "/models/" in req.full_url:
                return buffered_response(
                    json.dumps({"id": "x", "capabilities": {}}).encode()
                )
            if "after_id=claude-a" in req.full_url:
                return buffered_response(json.dumps(page2).encode())
            return buffered_response(json.dumps(page1).encode())

        mock_urlopen.side_effect = urlopen
        self.adapter.capabilities("foo/bar baz")
        single = [url for url in seen if "/models/" in url]
        self.assertEqual(len(single), 1)
        self.assertIn("foo%2Fbar%20baz", single[0])

    def test_invalidate_model_capabilities_drops_only_that_model(self):
        """The per-model hook drops the model from every projection."""
        self.adapter._models_cache = {
            "a": {"context_window": 1, "max_output_tokens": 2},
            "b": {"context_window": 3, "max_output_tokens": 4},
        }
        self.adapter._models_raw = {"a": {"id": "a"}, "b": {"id": "b"}}
        self.adapter._model_raw_cache = {
            "a": ({"id": "a"}, 1.0),
            "b": ({"id": "b"}, 1.0),
        }
        self.adapter._invalidate_model_capabilities("a")
        self.assertIn("a", self.adapter._models_cache)
        self.assertIn("b", self.adapter._models_cache)
        self.assertNotIn("a", self.adapter._models_raw)
        self.assertIn("b", self.adapter._models_raw)
        self.assertNotIn("a", self.adapter._model_raw_cache)
        self.assertIn("b", self.adapter._model_raw_cache)


if __name__ == "__main__":
    unittest.main()
