"""Tests for OpenAIAdapter: _serialize/_deserialize plus chat() and stream_chat()
with urllib.request.urlopen mocked out. No real network call is ever made."""

import json
import os
import time
import unittest
import urllib.error
from typing import Any
from unittest.mock import Mock, patch

from http_test_utils import (
    FakeStreamResponse,
    buffered_response,
    embed_response,
    final_response,
    http_error,
    request_body,
    sse_lines,
)

from ducktape_provider import (
    APIError,
    AuthError,
    ContextOverflowError,
    MalformedResponseError,
    Message,
    OpenAIAdapter,
    RateLimitError,
    ServerError,
    SystemBlock,
    ToolDef,
)
from ducktape_provider.adapters import openai as openai_module

MESSAGES: list[Message] = [
    {"role": "user", "content": [{"type": "text", "text": "weather in NYC?"}]}
]

FINAL_DATA: dict[str, Any] = {
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
        self.adapter = OpenAIAdapter(api_key="test")

    def test_serialize_interleaves_function_call_and_output(self):
        messages: list[Message] = [
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
        tools: list[ToolDef] = [
            {"name": "get_weather", "description": "...", "parameters": {"a": 1}}
        ]
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

    def test_system_blocks_flatten_to_instructions(self):
        system: list[SystemBlock] = [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b", "cache_control": {"type": "ephemeral"}},
        ]
        req, _ = self.adapter._build_request("m", MESSAGES, system, None, None, False)
        self.assertEqual(request_body(req)["instructions"], "a\nb")

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
        block = self.adapter._serialize(messages)[0]["content"][0]
        self.assertNotIn("cache_control", block)


class OpenAIDeserializeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="test")

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
        usage = response["usage"]
        assert usage is not None
        self.assertEqual(
            usage, {"input_tokens": 3, "output_tokens": 5, "cache_read_tokens": 100}
        )
        self.assertNotIn("cache_write_tokens", usage)

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
        usage = response["usage"]
        assert usage is not None
        self.assertEqual(
            usage, {"input_tokens": 3, "output_tokens": 5, "cache_write_tokens": 20}
        )
        self.assertNotIn("cache_read_tokens", usage)

    def test_omits_cache_usage_fields_when_absent(self):
        data = {
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }
        response = self.adapter._deserialize(data, 0.0)
        usage = response["usage"]
        assert usage is not None
        self.assertNotIn("cache_read_tokens", usage)
        self.assertNotIn("cache_write_tokens", usage)

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
        self.adapter = OpenAIAdapter(api_key="test")

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
        self.adapter = OpenAIAdapter(api_key="test")

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
        final = final_response(events)
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
        self.assertEqual(
            events[:3],
            [
                {"type": "thinking_delta", "index": 0, "thinking": "thinking "},
                {"type": "thinking_delta", "index": 0, "thinking": "hard"},
                {"type": "block_stop", "index": 0},
            ],
        )

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
        self.adapter = OpenAIAdapter(api_key="test")

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

    @patch("urllib.request.urlopen")
    def test_no_header_follows_a_redirect(self, mock_urlopen):
        calls = {
            "chat": (
                lambda: self.adapter.chat("gpt-x", MESSAGES),
                lambda: buffered_response(json.dumps(FINAL_DATA).encode()),
            ),
            "stream_chat": (
                lambda: list(self.adapter.stream_chat("gpt-x", MESSAGES)),
                lambda: FakeStreamResponse(sse_lines(*STREAM_EVENTS)),
            ),
            "models": (
                self.adapter.models,
                lambda: buffered_response(b'{"data": [{"id": "gpt-x"}]}'),
            ),
        }
        for label, (call, reply) in calls.items():
            with self.subTest(label):
                mock_urlopen.return_value = reply()
                call()
                req = mock_urlopen.call_args.args[0]
                self.assertEqual(req.headers, {})
                self.assertEqual(req.get_header("Authorization"), "Bearer test")

    @patch("urllib.request.urlopen")
    def test_config_auth_header_wins_and_key_source_is_not_called(self, mock_urlopen):
        source = Mock(return_value="from-source")
        adapter = OpenAIAdapter(api_key=source)
        for name in ("authorization", "AUTHORIZATION"):
            with self.subTest(name):
                mock_urlopen.return_value = buffered_response(
                    json.dumps(FINAL_DATA).encode()
                )
                adapter.chat(
                    "gpt-x", MESSAGES, config={"headers": {name: "Bearer cfg"}}
                )
                req = mock_urlopen.call_args.args[0]
                self.assertEqual(req.get_header("Authorization"), "Bearer cfg")
        source.assert_not_called()


class OpenAIBlockShapeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="test")

    def test_url_image_block_passes_url_through_as_is(self):
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
            [{"type": "input_image", "image_url": "https://x/img.png"}],
        )

    def test_document_block_is_dropped_and_logs_warning(self):
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

    def test_tool_result_with_list_content_serializes_text_and_image_blocks(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "name": "screenshot",
                        "content": [
                            {"type": "text", "text": "here"},
                            {
                                "type": "image",
                                "source": "url",
                                "url": "https://x/img.png",
                            },
                        ],
                    }
                ],
            }
        ]
        serialized = self.adapter._serialize(messages)
        self.assertEqual(
            serialized,
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [
                        {"type": "input_text", "text": "here"},
                        {"type": "input_image", "image_url": "https://x/img.png"},
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
                        "tool_use_id": "call_1",
                        "name": "noop",
                        "content": [],
                    }
                ],
            }
        ]
        serialized = self.adapter._serialize(messages)
        self.assertEqual(serialized[0]["output"], [])

    def test_tool_result_is_error_with_list_content_prepends_error_text(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "name": "screenshot",
                        "content": [{"type": "text", "text": "failed"}],
                        "is_error": True,
                    }
                ],
            }
        ]
        serialized = self.adapter._serialize(messages)
        self.assertEqual(
            serialized[0]["output"],
            [
                {"type": "input_text", "text": "ERROR:"},
                {"type": "input_text", "text": "failed"},
            ],
        )

    def test_tool_result_is_error_with_str_content_still_prefixes_error(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "name": "get_weather",
                        "content": "sunny",
                        "is_error": True,
                    }
                ],
            }
        ]
        serialized = self.adapter._serialize(messages)
        self.assertEqual(serialized[0]["output"], "ERROR: sunny")

    def test_tool_result_plain_str_content_still_passes_through(self):
        messages: list[Message] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "name": "get_weather",
                        "content": "sunny",
                    }
                ],
            }
        ]
        serialized = self.adapter._serialize(messages)
        self.assertEqual(serialized[0]["output"], "sunny")


REASONING_ITEM = {
    "id": "rs_1",
    "type": "reasoning",
    "summary": [
        {"type": "summary_text", "text": "step one"},
        {"type": "summary_text", "text": "step two"},
    ],
}


class OpenAIThinkingSerializeTests(unittest.TestCase):
    def test_thinking_block_is_dropped_and_logs_warning(self):
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "text", "text": "answer"},
                ],
            }
        ]
        with self.assertLogs(openai_module.logger, level="WARNING") as logs:
            serialized = OpenAIAdapter(api_key="test")._serialize(messages * 3)[:1]
        self.assertEqual(len(logs.records), 1)
        self.assertIn("dropping 3", logs.output[0])
        self.assertEqual(
            serialized,
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": "answer"}],
                }
            ],
        )


class OpenAINullUsageTests(unittest.TestCase):
    def test_null_usage_fields_count_as_zero_and_omit_cache_keys(self):
        data = {
            "status": "completed",
            "incomplete_details": None,
            "output": [],
            "usage": {
                "input_tokens": None,
                "output_tokens": 4,
                "input_tokens_details": {
                    "cached_tokens": None,
                    "cache_write_tokens": None,
                },
            },
        }
        response = OpenAIAdapter(api_key="test")._deserialize(data, 0.0)
        self.assertEqual(response["usage"], {"input_tokens": 0, "output_tokens": 4})


class OpenAIStreamContentTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="test")

    @patch("urllib.request.urlopen")
    def test_stream_chat_carries_cache_fields_in_message_stop(self, mock_urlopen):
        completed = {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
            ],
            "usage": {
                "input_tokens": 130,
                "output_tokens": 8,
                "input_tokens_details": {
                    "cached_tokens": 100,
                    "cache_write_tokens": 20,
                },
            },
        }
        mock_urlopen.return_value = FakeStreamResponse(
            sse_lines(
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_1",
                    "delta": "hi",
                },
                {"type": "response.completed", "response": completed},
            )
        )

        events = list(self.adapter.stream_chat("gpt-x", MESSAGES))

        self.assertEqual(
            final_response(events)["usage"],
            {
                "input_tokens": 130,
                "output_tokens": 8,
                "cache_read_tokens": 100,
                "cache_write_tokens": 20,
            },
        )

    @patch("urllib.request.urlopen")
    def test_multi_part_summary_joins_like_buffered_path(self, mock_urlopen):
        completed = {"status": "completed", "output": [REASONING_ITEM]}
        mock_urlopen.return_value = FakeStreamResponse(
            sse_lines(
                {
                    "type": "response.output_item.added",
                    "item": {"id": "rs_1", "type": "reasoning"},
                },
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": "rs_1",
                    "summary_index": 0,
                    "delta": "step ",
                },
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": "rs_1",
                    "summary_index": 0,
                    "delta": "one",
                },
                {
                    "type": "response.reasoning_summary_text.delta",
                    "item_id": "rs_1",
                    "summary_index": 1,
                    "delta": "step two",
                },
                {
                    "type": "response.output_item.done",
                    "item": {"id": "rs_1", "type": "reasoning"},
                },
                {"type": "response.completed", "response": completed},
            )
        )

        events = list(self.adapter.stream_chat("gpt-x", MESSAGES))

        streamed = "".join(
            e["thinking"] for e in events if e["type"] == "thinking_delta"
        )
        buffered = self.adapter._deserialize(completed, 0.0)["content"]
        self.assertEqual(streamed, "step one\nstep two")
        self.assertEqual(buffered, [{"type": "thinking", "thinking": streamed}])
        self.assertEqual(final_response(events)["content"], buffered)

    @patch("urllib.request.urlopen")
    def test_each_message_content_part_streams_under_its_own_index(self, mock_urlopen):
        completed = {
            "status": "completed",
            "output": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "one"},
                        {"type": "output_text", "text": ""},
                        {"type": "refusal", "refusal": "no"},
                    ],
                },
                {
                    "id": "fc_1",
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "f",
                    "arguments": "{}",
                },
            ],
        }

        def part_added(content_index: int, part_type: str) -> dict[str, Any]:
            return {
                "type": "response.content_part.added",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": content_index,
                "part": {"type": part_type},
            }

        mock_urlopen.return_value = FakeStreamResponse(
            sse_lines(
                {
                    "type": "response.output_item.added",
                    "item": {"id": "msg_1", "type": "message"},
                },
                part_added(0, "output_text"),
                {
                    "type": "response.output_text.delta",
                    "item_id": "msg_1",
                    "content_index": 0,
                    "delta": "one",
                },
                part_added(1, "output_text"),
                part_added(2, "refusal"),
                {
                    "type": "response.refusal.delta",
                    "item_id": "msg_1",
                    "content_index": 2,
                    "delta": "no",
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
                        "name": "f",
                    },
                },
                {
                    "type": "response.output_item.done",
                    "item": {"id": "fc_1", "type": "function_call"},
                },
                {"type": "response.completed", "response": completed},
            )
        )

        events = list(self.adapter.stream_chat("gpt-x", MESSAGES))

        self.assertEqual(
            [(e["type"], e.get("index")) for e in events],
            [
                ("text_delta", 0),
                ("text_delta", 2),
                ("block_stop", 0),
                ("block_stop", 2),
                ("tool_use_start", 3),
                ("block_stop", 3),
                ("message_stop", None),
            ],
        )
        content = final_response(events)["content"]
        self.assertEqual(content, self.adapter._deserialize(completed, 0.0)["content"])
        for event in events:
            if event["type"] == "text_delta":
                self.assertEqual(
                    content[event["index"]], {"type": "text", "text": event["text"]}
                )
        self.assertEqual(content[3]["type"], "tool_use")

    @patch("urllib.request.urlopen")
    def test_parts_without_content_part_added_still_get_separate_indices(
        self, mock_urlopen
    ):
        completed = {
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "one"},
                        {"type": "output_text", "text": "two"},
                    ],
                }
            ],
        }
        mock_urlopen.return_value = FakeStreamResponse(
            sse_lines(
                *(
                    {
                        "type": "response.output_text.delta",
                        "item_id": "msg_1",
                        "content_index": i,
                        "delta": text,
                    }
                    for i, text in enumerate(["one", "two"])
                ),
                {
                    "type": "response.output_item.done",
                    "item": {"id": "msg_1", "type": "message"},
                },
                {"type": "response.completed", "response": completed},
            )
        )

        events = list(self.adapter.stream_chat("gpt-x", MESSAGES))

        self.assertEqual(
            events[:-1],
            [
                {"type": "text_delta", "index": 0, "text": "one"},
                {"type": "text_delta", "index": 1, "text": "two"},
                {"type": "block_stop", "index": 0},
                {"type": "block_stop", "index": 1},
            ],
        )
        self.assertEqual(
            final_response(events)["content"],
            [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}],
        )

    @patch("urllib.request.urlopen")
    def test_refusal_deltas_stream_as_text(self, mock_urlopen):
        completed = {
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}
            ],
        }
        mock_urlopen.return_value = FakeStreamResponse(
            sse_lines(
                {"type": "response.refusal.delta", "item_id": "msg_1", "delta": "no"},
                {"type": "response.completed", "response": completed},
            )
        )

        events = list(self.adapter.stream_chat("gpt-x", MESSAGES))

        self.assertEqual(events[0], {"type": "text_delta", "index": 0, "text": "no"})
        self.assertEqual(final_response(events)["stop_reason"], "refusal")

    @patch("urllib.request.urlopen")
    def test_incomplete_response_ends_stream_with_max_tokens(self, mock_urlopen):
        incomplete = {
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [],
        }
        mock_urlopen.return_value = FakeStreamResponse(
            sse_lines({"type": "response.incomplete", "response": incomplete})
        )

        events = list(self.adapter.stream_chat("gpt-x", MESSAGES))

        self.assertEqual([e["type"] for e in events], ["message_stop"])
        self.assertEqual(final_response(events)["stop_reason"], "max_tokens")


class OpenAIStreamErrorTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="test")

    def _stream(self, lines, error=None):
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value = FakeStreamResponse(lines, error)
            return list(self.adapter.stream_chat("gpt-x", MESSAGES))

    def test_error_event_raises_mapped_error(self):
        lines = sse_lines(
            {
                "type": "error",
                "code": "rate_limit_exceeded",
                "message": "slow down",
                "param": None,
                "sequence_number": 1,
            }
        )
        with self.assertRaises(RateLimitError) as ctx:
            self._stream(lines)
        self.assertEqual(ctx.exception.status, 429)
        self.assertIn("slow down", str(ctx.exception))

    def test_error_event_with_fields_nested_under_error(self):
        lines = sse_lines(
            {
                "type": "error",
                "error": {
                    "type": "rate_limit_error",
                    "code": "rate_limit_exceeded",
                    "message": "slow down",
                },
            }
        )
        with self.assertRaises(RateLimitError) as ctx:
            self._stream(lines)
        self.assertEqual(ctx.exception.status, 429)
        self.assertIn("slow down", str(ctx.exception))

    def test_quota_and_overload_errors_match_http_classes(self):
        cases = [
            ({"type": "insufficient_quota", "code": None}, RateLimitError, 429),
            ({"code": "insufficient_quota"}, RateLimitError, 429),
            ({"code": "organization_spend_limit_exceeded"}, RateLimitError, 429),
            ({"code": "server_is_overloaded"}, ServerError, 503),
            ({"code": "invalid_prompt"}, APIError, None),
        ]
        for error, expected, status in cases:
            error = {**error, "message": "nope"}
            failed = {"status": "failed", "error": error, "output": []}
            for label, lines in (
                ("error event", sse_lines({"type": "error", "error": error})),
                (
                    "response.failed",
                    sse_lines({"type": "response.failed", "response": failed}),
                ),
            ):
                with self.subTest(error=error, event=label):
                    with self.assertRaises(APIError) as ctx:
                        self._stream(lines)
                    self.assertIs(type(ctx.exception), expected)
                    self.assertEqual(ctx.exception.status, status)

    def test_wrong_shape_events_raise_malformed_api_error(self):
        cases = {
            "non-object event": [[1]],
            "delta without item_id": [
                {"type": "response.output_text.delta", "delta": "x"}
            ],
            "item without id": [{"type": "response.output_item.done", "item": {}}],
            "non-object response": [{"type": "response.completed", "response": [1]}],
        }
        for label, events in cases.items():
            with self.subTest(label), self.assertRaises(MalformedResponseError) as ctx:
                self._stream(sse_lines(*events))
            self.assertIs(type(ctx.exception), MalformedResponseError)
            self.assertIn("malformed", str(ctx.exception))

    @patch("urllib.request.urlopen")
    def test_buffered_wrong_shape_body_raises_malformed_api_error(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(b"[1, 2]")
        with self.assertRaises(MalformedResponseError) as ctx:
            self.adapter.chat("gpt-x", MESSAGES)
        self.assertIn("malformed", str(ctx.exception))

    def test_error_event_with_unmapped_code_raises_api_error(self):
        lines = sse_lines(
            {"type": "error", "code": None, "message": "weird", "param": None}
        )
        with self.assertRaises(APIError) as ctx:
            self._stream(lines)
        self.assertIs(type(ctx.exception), APIError)

    def test_response_failed_raises_server_error(self):
        failed = {
            "status": "failed",
            "error": {"code": "server_error", "message": "boom"},
            "output": [],
        }
        with self.assertRaises(ServerError):
            self._stream(sse_lines({"type": "response.failed", "response": failed}))

    def test_stream_ending_without_terminal_event_raises(self):
        with self.assertRaises(APIError) as ctx:
            self._stream(sse_lines(*STREAM_EVENTS[:-1]))
        self.assertIn("response.completed", str(ctx.exception))

    def test_malformed_sse_line_raises_api_error(self):
        with self.assertRaises(MalformedResponseError) as ctx:
            self._stream([b"data: {oops\n", b"\n"])
        self.assertIn("malformed", str(ctx.exception))

    def test_connection_reset_mid_stream_raises_api_error(self):
        with self.assertRaises(APIError):
            self._stream(sse_lines(*STREAM_EVENTS[:2]), ConnectionResetError())

    @patch("urllib.request.urlopen")
    def test_buffered_failed_status_maps_error_code(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(
                {
                    "status": "failed",
                    "error": {
                        "code": "context_length_exceeded",
                        "message": "too long",
                    },
                    "output": [],
                }
            ).encode()
        )
        with self.assertRaises(ContextOverflowError):
            self.adapter.chat("gpt-x", MESSAGES)


class OpenAILatencyTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="test")

    @patch("urllib.request.urlopen")
    def test_chat_latency_spans_request_to_body_read(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(json.dumps(FINAL_DATA).encode())
        with patch("time.monotonic", side_effect=[10.0, 10.75]):
            response = self.adapter.chat("gpt-x", MESSAGES)
        self.assertEqual(response["latency_ms"], 750.0)
        self.assertNotIn("ttft_ms", response)

    @patch("urllib.request.urlopen")
    def test_stream_latency_and_ttft_use_wire_read_times(self, mock_urlopen):
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*STREAM_EVENTS))
        clock = [100.0 + i * 0.25 for i in range(len(STREAM_EVENTS) + 1)]
        with patch("time.monotonic", side_effect=clock):
            events = list(
                self.adapter.stream_chat("gpt-x", MESSAGES, config={"timeout": None})
            )
        final = final_response(events)
        self.assertEqual(final["ttft_ms"], 500.0)
        self.assertEqual(final["latency_ms"], 1750.0)


class OpenAIReservedConfigTests(unittest.TestCase):
    @patch("urllib.request.urlopen", side_effect=AssertionError("request sent"))
    def test_reserved_config_keys_are_rejected_before_any_request(self, mock_urlopen):
        adapter = OpenAIAdapter(api_key="test")
        for key in ("stream", "model", "input"):
            with self.subTest(key=key), self.assertRaises(ValueError) as ctx:
                adapter.chat("gpt-x", MESSAGES, config={key: True})
            self.assertIn(key, str(ctx.exception))
        mock_urlopen.assert_not_called()

    def test_non_reserved_defaults_can_be_overridden(self):
        req, _ = OpenAIAdapter(api_key="test")._build_request(
            "gpt-x", MESSAGES, None, None, {"store": True}, stream=False
        )
        self.assertIs(request_body(req)["store"], True)


class OpenAIInvalidToolArgsTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="test")

    def test_deeply_nested_arguments_fall_back_to_empty_input_in_chat(self):
        deep = "[" * 200_000 + "]" * 200_000
        data = {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": deep,
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
        final = {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": deep,
                }
            ],
        }
        events = [
            {
                "type": "response.output_item.added",
                "item": {"id": "fc_1", "type": "function_call"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": deep,
            },
            {"type": "response.completed", "response": final},
        ]
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*events))
        response = final_response(list(self.adapter.stream_chat("gpt-x", MESSAGES)))
        self.assertEqual(
            response["content"],
            [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "get_weather",
                    "input": {},
                    "truncated": True,
                }
            ],
        )

    def test_non_object_arguments_fall_back_to_empty_input_in_chat(self):
        for arguments in ("[1,2,3]", "1", "null"):
            with self.subTest(arguments=arguments):
                data = {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "get_weather",
                            "arguments": arguments,
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
                            "input": {},
                            "truncated": True,
                        }
                    ],
                )

    @patch("urllib.request.urlopen")
    def test_non_object_arguments_fall_back_to_empty_input_in_stream(
        self, mock_urlopen
    ):
        for arguments in ("[1,2,3]", "1", "null"):
            with self.subTest(arguments=arguments):
                final = {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "get_weather",
                            "arguments": arguments,
                        }
                    ],
                }
                events = [
                    {
                        "type": "response.output_item.added",
                        "item": {"id": "fc_1", "type": "function_call"},
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_1",
                        "delta": arguments,
                    },
                    {"type": "response.completed", "response": final},
                ]
                mock_urlopen.return_value = FakeStreamResponse(sse_lines(*events))
                response = final_response(
                    list(self.adapter.stream_chat("gpt-x", MESSAGES))
                )
                self.assertEqual(
                    response["content"],
                    [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "get_weather",
                            "input": {},
                            "truncated": True,
                        }
                    ],
                )

    def test_legitimately_empty_arguments_are_not_marked_truncated_in_chat(self):
        for arguments in ("", "{}"):
            with self.subTest(arguments=arguments):
                data = {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "get_weather",
                            "arguments": arguments,
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
                            "input": {},
                        }
                    ],
                )

    @patch("urllib.request.urlopen")
    def test_legitimately_empty_arguments_are_not_marked_truncated_in_stream(
        self, mock_urlopen
    ):
        for arguments in ("", "{}"):
            with self.subTest(arguments=arguments):
                final = {
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_1",
                            "name": "get_weather",
                            "arguments": arguments,
                        }
                    ],
                }
                events = [
                    {
                        "type": "response.output_item.added",
                        "item": {"id": "fc_1", "type": "function_call"},
                    },
                    {
                        "type": "response.function_call_arguments.delta",
                        "item_id": "fc_1",
                        "delta": arguments,
                    },
                    {"type": "response.completed", "response": final},
                ]
                mock_urlopen.return_value = FakeStreamResponse(sse_lines(*events))
                response = final_response(
                    list(self.adapter.stream_chat("gpt-x", MESSAGES))
                )
                self.assertEqual(
                    response["content"],
                    [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "get_weather",
                            "input": {},
                        }
                    ],
                )


class OpenAIModelInfoTests(unittest.TestCase):
    """OpenAI's models list/get endpoints expose no context-window or
    max-output field at all, so `model_info` is the unmodified `Adapter`
    default: always `None`, and never touches the network."""

    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="test")

    @patch("urllib.request.urlopen", side_effect=AssertionError("request sent"))
    def test_always_returns_none_without_any_request(self, mock_urlopen):
        for model in ("gpt-x", "gpt-nonexistent", ""):
            with self.subTest(model=model):
                self.assertIsNone(self.adapter.model_info(model))
        mock_urlopen.assert_not_called()


class OpenAIModelsCacheInvalidationTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="test")
        self.adapter._models_cache = {"gpt-old"}
        self.adapter._embed_models_cache = {"text-embedding-old"}
        self.adapter._cache_time = time.monotonic()

    @patch("urllib.request.urlopen")
    def test_chat_404_invalidates_models_cache(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            OpenAIAdapter._RESPONSES_URL, 404, b"no such model"
        )
        with self.assertRaises(APIError):
            self.adapter.chat("gpt-x", MESSAGES)
        self.assertIsNone(self.adapter._models_cache)
        self.assertIsNone(self.adapter._embed_models_cache)

    @patch("urllib.request.urlopen")
    def test_stream_chat_404_invalidates_models_cache(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            OpenAIAdapter._RESPONSES_URL, 404, b"no such model"
        )
        with self.assertRaises(APIError):
            list(self.adapter.stream_chat("gpt-x", MESSAGES))
        self.assertIsNone(self.adapter._models_cache)
        self.assertIsNone(self.adapter._embed_models_cache)

    @patch("urllib.request.urlopen")
    def test_non_404_error_leaves_models_cache_untouched(self, mock_urlopen):
        mock_urlopen.side_effect = http_error(
            OpenAIAdapter._RESPONSES_URL, 500, b"boom"
        )
        with self.assertRaises(APIError):
            self.adapter.chat("gpt-x", MESSAGES)
        self.assertEqual(self.adapter._models_cache, {"gpt-old"})
        self.assertEqual(self.adapter._embed_models_cache, {"text-embedding-old"})


class TestOpenAIEmbedHTTP(unittest.TestCase):
    """Embed HTTP behaviour."""

    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="test")

    @patch("urllib.request.urlopen")
    def test_single_input_happy_path(self, mock_urlopen):
        """Single input returns correctly shaped response."""
        model = "text-embedding-3-small"
        vectors = [[0.1, 0.2, 0.3]]
        payload = embed_response(model, vectors, prompt_tokens=7)
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        resp = self.adapter.embed(model, ["hello"])
        self.assertEqual(resp["embeddings"], vectors)
        self.assertEqual(resp["usage"], {"input_tokens": 7})
        self.assertEqual(resp["raw"]["model"], model)
        self.assertNotIn("model", resp)
        self.assertIsInstance(resp["latency_ms"], float)
        self.assertGreaterEqual(resp["latency_ms"], 0)
        req = mock_urlopen.call_args.args[0]
        self.assertEqual(req.full_url, OpenAIAdapter._EMBEDDINGS_URL)
        body = request_body(req)
        self.assertEqual(body["input"], ["hello"])
        self.assertEqual(body["model"], model)

    @patch("urllib.request.urlopen")
    def test_batch_happy_path(self, mock_urlopen):
        """Batch input returns one vector per input in order."""
        model = "text-embedding-3-small"
        vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        payload = embed_response(model, vectors, prompt_tokens=10)
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        resp = self.adapter.embed(model, ["a", "b", "c"])
        self.assertEqual(resp["embeddings"], vectors)
        req = mock_urlopen.call_args.args[0]
        self.assertEqual(request_body(req)["input"], ["a", "b", "c"])

    @patch("urllib.request.urlopen")
    def test_dimensions_passthrough_unvalidated(self, mock_urlopen):
        """Dimensions config reaches body unvalidated."""
        model = "text-embedding-3-small"
        vectors = [[0.1, 0.2]]
        payload = embed_response(model, vectors, prompt_tokens=1)
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        self.adapter.embed(model, ["hi"], config={"dimensions": 512})
        body = request_body(mock_urlopen.call_args.args[0])
        self.assertEqual(body["dimensions"], 512)

    @patch("urllib.request.urlopen")
    def test_unsorted_indices_are_ordered(self, mock_urlopen):
        """Unsorted data indices yield correctly ordered vectors."""
        model = "text-embedding-3-small"
        payload = {
            "object": "list",
            "data": [
                {"object": "embedding", "embedding": [0.3, 0.4], "index": 1},
                {"object": "embedding", "embedding": [0.1, 0.2], "index": 0},
            ],
            "model": model,
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        resp = self.adapter.embed(model, ["first", "second"])
        self.assertEqual(resp["embeddings"], [[0.1, 0.2], [0.3, 0.4]])

    @patch("urllib.request.urlopen")
    def test_count_mismatch_raises(self, mock_urlopen):
        """Count mismatch raises MalformedResponseError."""
        model = "text-embedding-3-small"
        for vectors in ([[0.1]], [[0.1], [0.2], [0.3]]):
            payload = embed_response(model, vectors, prompt_tokens=1)
            mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
            with (
                self.subTest(vectors=vectors),
                self.assertRaises(MalformedResponseError),
            ):
                self.adapter.embed(model, ["a", "b"])

    @patch("urllib.request.urlopen")
    def test_duplicate_and_missing_index_raises(self, mock_urlopen):
        """Duplicate or missing index raises MalformedResponseError."""
        model = "text-embedding-3-small"
        cases = [
            [
                {"object": "embedding", "embedding": [0.1], "index": 0},
                {"object": "embedding", "embedding": [0.2], "index": 0},
            ],
            [
                {"object": "embedding", "embedding": [0.1], "index": 0},
                {"object": "embedding", "embedding": [0.2], "index": 2},
            ],
        ]
        for data in cases:
            payload = {"object": "list", "data": data, "model": model}
            mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
            with self.subTest(data=data), self.assertRaises(MalformedResponseError):
                self.adapter.embed(model, ["a", "b"])

    @patch("urllib.request.urlopen")
    def test_absent_usage_is_none(self, mock_urlopen):
        """Absent usage yields None."""
        model = "text-embedding-3-small"
        payload = embed_response(model, [[0.1, 0.2]])
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        resp = self.adapter.embed(model, ["hi"])
        self.assertIsNone(resp["usage"])

    @patch("urllib.request.urlopen")
    def test_empty_usage_object_is_zero_tokens(self, mock_urlopen):
        """A present but empty usage object is zero tokens, not None."""
        model = "text-embedding-3-small"
        payload = embed_response(model, [[0.1, 0.2]])
        payload["usage"] = {}
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        resp = self.adapter.embed(model, ["hi"])
        self.assertEqual(resp["usage"], {"input_tokens": 0})
        self.assertNotIn("total_tokens", resp["raw"])

    @patch("urllib.request.urlopen")
    def test_usage_without_total_tokens_omits_raw_key(self, mock_urlopen):
        """A usage object lacking total_tokens leaves it out of raw."""
        model = "text-embedding-3-small"
        payload = embed_response(model, [[0.1, 0.2]])
        payload["usage"] = {"prompt_tokens": 5}
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        resp = self.adapter.embed(model, ["hi"])
        self.assertEqual(resp["usage"], {"input_tokens": 5})
        self.assertNotIn("total_tokens", resp["raw"])

    @patch("urllib.request.urlopen")
    def test_context_overflow_maps_with_embed_label(self, mock_urlopen):
        """A context-overflow 400 maps to ContextOverflowError with embed label."""
        mock_urlopen.side_effect = http_error(
            OpenAIAdapter._EMBEDDINGS_URL, 400, b"context_length_exceeded"
        )
        with self.assertRaises(ContextOverflowError) as ctx:
            self.adapter.embed("text-embedding-3-small", ["hi"])
        self.assertIn("embed", str(ctx.exception))
        self.assertNotIn("chat", str(ctx.exception))

    @patch("urllib.request.urlopen", side_effect=AssertionError("request sent"))
    def test_reserved_key_clash_raises_before_request(self, mock_urlopen):
        """Reserved key clash raises ValueError before request."""
        with self.assertRaises(ValueError) as ctx:
            self.adapter.embed("text-embedding-3-small", ["hi"], config={"model": "x"})
        self.assertIn("embed()", str(ctx.exception))
        mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen", side_effect=AssertionError("request sent"))
    def test_encoding_format_base64_raises_before_request(self, mock_urlopen):
        """Base64 encoding format raises ValueError before request."""
        with self.assertRaises(ValueError) as ctx:
            self.adapter.embed(
                "text-embedding-3-small", ["hi"], config={"encoding_format": "base64"}
            )
        self.assertIn("base64", str(ctx.exception).lower())
        mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_404_invalidates_both_caches(self, mock_urlopen):
        """404 clears both listing caches."""
        self.adapter._models_cache = {"gpt-old"}
        self.adapter._embed_models_cache = {"text-embedding-old"}
        self.adapter._cache_time = time.monotonic()
        mock_urlopen.side_effect = http_error(
            OpenAIAdapter._EMBEDDINGS_URL, 404, b"nope"
        )
        with self.assertRaises(APIError) as ctx:
            self.adapter.embed("text-embedding-3-small", ["hi"])
        self.assertEqual(ctx.exception.status, 404)
        self.assertIsNone(self.adapter._models_cache)
        self.assertIsNone(self.adapter._embed_models_cache)

    @patch("urllib.request.urlopen")
    def test_401_429_500_raise_with_embed_label(self, mock_urlopen):
        """401/429/500 map to correct errors with embed in message."""
        cases = [
            (401, AuthError),
            (429, RateLimitError),
            (500, ServerError),
        ]
        for code, exc_type in cases:
            with self.subTest(code=code):
                mock_urlopen.side_effect = http_error(
                    OpenAIAdapter._EMBEDDINGS_URL, code, b"boom"
                )
                with self.assertRaises(exc_type) as ctx:
                    self.adapter.embed("text-embedding-3-small", ["hi"])
                self.assertIn("embed", str(ctx.exception))
                self.assertNotIn("chat", str(ctx.exception))
                self.assertEqual(ctx.exception.status, code)

    @patch("urllib.request.urlopen")
    def test_malformed_body_raises(self, mock_urlopen):
        """Malformed body raises MalformedResponseError."""
        for body in [b"not json", json.dumps({"object": "list"}).encode()]:
            mock_urlopen.return_value = buffered_response(body)
            with self.subTest(body=body), self.assertRaises(MalformedResponseError):
                self.adapter.embed("text-embedding-3-small", ["hi"])

    @patch("urllib.request.urlopen")
    def test_header_override_reaches_request(self, mock_urlopen):
        """Call-level header override reaches request."""
        model = "text-embedding-3-small"
        payload = embed_response(model, [[0.1]], prompt_tokens=1)
        mock_urlopen.return_value = buffered_response(json.dumps(payload).encode())
        self.adapter.embed(model, ["hi"], config={"headers": {"X-Custom": "yes"}})
        req = mock_urlopen.call_args.args[0]
        self.assertEqual(req.get_header("X-custom"), "yes")


class TestOpenAIEmbedModels(unittest.TestCase):
    """Embed models listing."""

    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="test")

    @patch("urllib.request.urlopen")
    def test_embed_models_and_models_share_single_fetch(self, mock_urlopen):
        """Models and embed_models are projections of one fetch."""
        data = {
            "data": [
                {"id": "gpt-4"},
                {"id": "gpt-4o"},
                {"id": "gpt-image-1"},
                {"id": "text-embedding-3-small"},
                {"id": "text-embedding-ada-002"},
                {"id": "whisper-1"},
                {"id": "tts-1"},
            ]
        }
        mock_urlopen.return_value = buffered_response(json.dumps(data).encode())
        chat_ids = self.adapter.models()
        embed_ids = self.adapter.embed_models()
        self.assertEqual(chat_ids, {"gpt-4", "gpt-4o"})
        self.assertEqual(
            embed_ids, {"text-embedding-3-small", "text-embedding-ada-002"}
        )
        mock_urlopen.assert_called_once()

    @patch("urllib.request.urlopen")
    def test_one_fetch_serves_both_within_ttl_and_refetches_after(self, mock_urlopen):
        """One GET serves both within TTL, second after expiry."""
        data = {"data": [{"id": "gpt-4"}, {"id": "text-embedding-3-small"}]}
        mock_urlopen.side_effect = lambda *a, **k: buffered_response(
            json.dumps(data).encode()
        )
        base = 1000.0
        with patch("ducktape_provider.adapters.openai.time.monotonic") as mock_time:
            mock_time.return_value = base
            self.adapter.models()
            self.adapter.embed_models()
            self.assertEqual(mock_urlopen.call_count, 1)
            mock_time.return_value = base + 10
            self.adapter.models()
            self.adapter.embed_models()
            self.assertEqual(mock_urlopen.call_count, 1)
            mock_time.return_value = base + 61
            self.adapter.models()
            self.assertEqual(mock_urlopen.call_count, 2)
            mock_time.return_value = base + 62
            self.adapter.embed_models()
            self.assertEqual(mock_urlopen.call_count, 2)

    @patch("urllib.request.urlopen")
    def test_failed_refresh_returns_empty_not_stale(self, mock_urlopen):
        """An expired cache whose refresh fails yields empty, not stale ids."""
        data = {"data": [{"id": "gpt-4"}, {"id": "text-embedding-3-small"}]}
        mock_urlopen.return_value = buffered_response(json.dumps(data).encode())
        base = 1000.0
        with patch("ducktape_provider.adapters.openai.time.monotonic") as mock_time:
            mock_time.return_value = base
            self.assertEqual(self.adapter.models(), {"gpt-4"})
            self.assertEqual(self.adapter.embed_models(), {"text-embedding-3-small"})
            mock_urlopen.side_effect = http_error(
                OpenAIAdapter._MODELS_URL, 500, b"boom"
            )
            mock_time.return_value = base + 61
            self.assertEqual(self.adapter.models(), set())
            self.assertEqual(self.adapter.embed_models(), set())

    @patch("urllib.request.urlopen")
    def test_probe_http_error_swallowed(self, mock_urlopen):
        """HTTP probe error returns empty set."""
        mock_urlopen.side_effect = http_error(OpenAIAdapter._MODELS_URL, 500, b"boom")
        self.assertEqual(self.adapter.models(), set())
        self.assertEqual(self.adapter.embed_models(), set())

    @patch("urllib.request.urlopen")
    def test_probe_connection_error_swallowed(self, mock_urlopen):
        """Connection probe error returns empty set."""
        mock_urlopen.side_effect = urllib.error.URLError("down")
        self.assertEqual(self.adapter.models(), set())
        self.assertEqual(self.adapter.embed_models(), set())

    def test_missing_api_key_returns_empty_without_request(self):
        """Missing key returns empty set with no request."""
        with patch.dict(os.environ, {}, clear=False):
            for key in list(os.environ):
                if key.lower() == "openai_api_key":
                    del os.environ[key]
            adapter = OpenAIAdapter()
            with patch(
                "urllib.request.urlopen", side_effect=AssertionError("request sent")
            ) as m:
                self.assertEqual(adapter.embed_models(), set())
                self.assertEqual(adapter.models(), set())
                m.assert_not_called()

    def test_copied_adapter_starts_with_empty_caches(self):
        """Copied adapter starts with both caches empty."""
        from ducktape_provider import Provider

        original = OpenAIAdapter()
        original._models_cache = {"gpt-old"}
        original._embed_models_cache = {"text-embedding-old"}
        original._cache_time = time.monotonic()
        provider = Provider(
            adapters={"openai": original}, api_keys={"openai": "sk-new"}
        )
        copied = provider._adapters["openai"]
        self.assertIsNot(copied, original)
        assert isinstance(copied, OpenAIAdapter)
        self.assertIsNone(copied._models_cache)
        self.assertIsNone(copied._embed_models_cache)
        self.assertEqual(copied._cache_time, 0.0)
        self.assertEqual(original._models_cache, {"gpt-old"})
        self.assertEqual(original._embed_models_cache, {"text-embedding-old"})

        original2 = OpenAIAdapter(api_key="test")
        original2._models_cache = {"gpt-old"}
        original2._embed_models_cache = {"text-embedding-old"}
        original2._cache_time = time.monotonic()
        provider2 = Provider(
            adapters={"openai": original2},
            config={"providers": {"openai": {"headers": {"X-Custom": "v"}}}},
        )
        copied2 = provider2._adapters["openai"]
        assert isinstance(copied2, OpenAIAdapter)
        self.assertIsNone(copied2._models_cache)
        self.assertIsNone(copied2._embed_models_cache)
        self.assertEqual(copied2._cache_time, 0.0)


class TestOpenAICompaction(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter(api_key="test")

    def test_auto_compaction_payload(self):
        req, _ = self.adapter._build_request(
            "m", MESSAGES, None, None, {"_compaction": {"threshold": 50000}}, False
        )
        self.assertEqual(
            request_body(req)["context_management"],
            [{"type": "compaction", "compact_threshold": 50000}],
        )

    def test_auto_compaction_omits_unset_threshold(self):
        req, _ = self.adapter._build_request(
            "m", MESSAGES, None, None, {"_compaction": {}}, False
        )
        self.assertEqual(
            request_body(req)["context_management"], [{"type": "compaction"}]
        )

    def test_pause_raises_before_request(self):
        with patch(
            "urllib.request.urlopen", side_effect=AssertionError("request sent")
        ) as mock:
            with self.assertRaises(ValueError):
                self.adapter._build_request(
                    "m", MESSAGES, None, None, {"_compaction": {"pause": True}}, False
                )
            mock.assert_not_called()

    def test_instructions_in_auto_raises(self):
        with (
            patch(
                "urllib.request.urlopen", side_effect=AssertionError("request sent")
            ) as mock,
            self.assertRaises(ValueError),
        ):
            self.adapter._build_request(
                "m",
                MESSAGES,
                None,
                None,
                {"_compaction": {"instructions": "s"}},
                False,
            )
        mock.assert_not_called()

    def test_compaction_block_serialized(self):
        messages: list[Message] = [
            {
                "role": "assistant",
                "content": [
                    {"type": "compaction", "content": None, "encrypted_content": "e"}
                ],
            }
        ]
        self.assertEqual(
            self.adapter._serialize(messages),
            [{"type": "compaction", "encrypted_content": "e"}],
        )

    @patch("urllib.request.urlopen")
    def test_compact_wraps_opaque_item(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(
                {
                    "object": "response.compaction",
                    "output": [{"type": "compaction", "encrypted_content": "e"}],
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                }
            ).encode()
        )
        result = self.adapter.compact("m", MESSAGES)
        body = request_body(mock_urlopen.call_args.args[0])
        self.assertIn("input", body)
        self.assertEqual(
            result["block"],
            {"type": "compaction", "content": None, "encrypted_content": "e"},
        )
        self.assertEqual(result["usage"], {"input_tokens": 3, "output_tokens": 1})

    @patch("urllib.request.urlopen")
    def test_stream_compaction_item(self, mock_urlopen):
        events = [
            {
                "type": "response.output_item.added",
                "item": {"id": "c1", "type": "compaction", "encrypted_content": "e"},
            },
            {
                "type": "response.output_item.done",
                "item": {"id": "c1", "type": "compaction"},
            },
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [{"type": "compaction", "encrypted_content": "e"}],
                    "usage": {},
                },
            },
        ]
        mock_urlopen.return_value = FakeStreamResponse(sse_lines(*events))
        out = list(self.adapter.stream_chat("m", MESSAGES))
        self.assertIn("compaction_delta", [e["type"] for e in out])
        final = final_response(out)
        self.assertNotEqual(final["stop_reason"], "compaction")
        self.assertEqual(
            final["content"],
            [{"type": "compaction", "content": None, "encrypted_content": "e"}],
        )

    def test_compaction_item_kept_in_response_content(self):
        data: dict[str, Any] = {
            "status": "completed",
            "output": [
                {"type": "compaction", "encrypted_content": "e"},
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "hi"}],
                },
            ],
            "usage": {},
        }
        response = self.adapter._deserialize(data, 1.0)
        self.assertEqual(
            response["content"],
            [
                {"type": "compaction", "content": None, "encrypted_content": "e"},
                {"type": "text", "text": "hi"},
            ],
        )

    def test_compaction_item_without_encrypted_content(self):
        response = self.adapter._deserialize(
            {"status": "completed", "output": [{"type": "compaction"}], "usage": {}},
            1.0,
        )
        self.assertEqual(response["content"], [{"type": "compaction", "content": None}])

    def test_raw_context_management_entries_merged(self):
        config = {
            "_compaction": {"threshold": 50000},
            "context_management": [{"type": "clear_tool_uses"}],
        }
        req, _ = self.adapter._build_request("m", MESSAGES, None, None, config, False)
        self.assertEqual(
            request_body(req)["context_management"],
            [
                {"type": "compaction", "compact_threshold": 50000},
                {"type": "clear_tool_uses"},
            ],
        )

    def test_raw_compaction_entry_overridden_with_warning(self):
        config = {
            "_compaction": {},
            "context_management": [
                {"type": "compaction", "compact_threshold": 90000},
                {"type": "clear_tool_uses"},
            ],
        }
        with self.assertLogs(openai_module.logger, "WARNING") as logs:
            req, _ = self.adapter._build_request(
                "m", MESSAGES, None, None, config, False
            )
        self.assertEqual(
            request_body(req)["context_management"],
            [{"type": "compaction"}, {"type": "clear_tool_uses"}],
        )
        self.assertTrue(
            any("overrides a raw compaction entry" in line for line in logs.output)
        )

    def test_raw_context_management_of_wrong_shape_warns(self):
        config = {"_compaction": {}, "context_management": {"edits": []}}
        with self.assertLogs(openai_module.logger, "WARNING") as logs:
            req, _ = self.adapter._build_request(
                "m", MESSAGES, None, None, config, False
            )
        self.assertEqual(
            request_body(req)["context_management"], [{"type": "compaction"}]
        )
        self.assertTrue(
            any("not a list of entries" in line for line in logs.output),
        )

    @patch("urllib.request.urlopen")
    def test_compact_warns_when_dropping_raw_context_management(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(
                {"output": [{"type": "compaction", "encrypted_content": "e"}]}
            ).encode()
        )
        with self.assertLogs(openai_module.logger, "WARNING") as logs:
            self.adapter.compact(
                "m", MESSAGES, config={"context_management": [{"type": "compaction"}]}
            )
        self.assertNotIn(
            "context_management", request_body(mock_urlopen.call_args.args[0])
        )
        self.assertTrue(
            any(
                "compact() overrides raw context_management" in line
                for line in logs.output
            )
        )

    @patch("urllib.request.urlopen")
    def test_compact_usage_includes_cache_tokens(self, mock_urlopen):
        mock_urlopen.return_value = buffered_response(
            json.dumps(
                {
                    "output": [{"type": "compaction", "encrypted_content": "e"}],
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 2,
                        "input_tokens_details": {"cached_tokens": 7},
                    },
                }
            ).encode()
        )
        result = self.adapter.compact("m", MESSAGES)
        self.assertEqual(
            result["usage"],
            {"input_tokens": 10, "output_tokens": 2, "cache_read_tokens": 7},
        )

    def test_supports_compaction(self):
        self.assertTrue(self.adapter.supports_compaction())


if __name__ == "__main__":
    unittest.main()
