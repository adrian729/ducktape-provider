"""Tests for OpenAIAdapter: _serialize/_deserialize plus chat() and stream_chat()
with urllib.request.urlopen mocked out. No real network call is ever made."""

import json
import unittest
from typing import Any
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
    AuthError,
    ContextOverflowError,
    MalformedResponseError,
    Message,
    OpenAIAdapter,
    RateLimitError,
    ServerError,
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
        self.adapter = OpenAIAdapter()

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
            serialized = OpenAIAdapter()._serialize(messages * 3)[:1]
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
        response = OpenAIAdapter()._deserialize(data, 0.0)
        self.assertEqual(response["usage"], {"input_tokens": 0, "output_tokens": 4})


class OpenAIStreamContentTests(unittest.TestCase):
    def setUp(self):
        self.adapter = OpenAIAdapter()

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
                # An empty part gets no deltas but still occupies a final block.
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
        # every streamed index addresses the final block its deltas built
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
        self.adapter = OpenAIAdapter()

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
        self.adapter = OpenAIAdapter()

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
            events = list(self.adapter.stream_chat("gpt-x", MESSAGES))
        final = final_response(events)
        # first text delta is the 2nd SSE event, response.completed the 7th
        self.assertEqual(final["ttft_ms"], 500.0)
        self.assertEqual(final["latency_ms"], 1750.0)


class OpenAIReservedConfigTests(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_reserved_config_keys_are_rejected_before_any_request(self, mock_urlopen):
        adapter = OpenAIAdapter()
        for key in ("stream", "model", "input"):
            with self.subTest(key=key), self.assertRaises(ValueError) as ctx:
                adapter.chat("gpt-x", MESSAGES, config={key: True})
            self.assertIn(key, str(ctx.exception))
        mock_urlopen.assert_not_called()

    def test_non_reserved_defaults_can_be_overridden(self):
        req, _ = OpenAIAdapter()._build_request(
            "gpt-x", MESSAGES, None, None, {"store": True}, stream=False
        )
        self.assertIs(request_body(req)["store"], True)


if __name__ == "__main__":
    unittest.main()
