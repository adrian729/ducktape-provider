"""Offline, fixture-based tests for the stream-parsing helpers and each
adapter's _deserialize(). No network access — everything runs on canned
dicts, matching pre-recorded shapes from each vendor's API.
"""

import unittest

from ducktape_provider import (
    ClaudeAdapter,
    OllamaLocalAdapter,
    OpenAIAdapter,
    _iter_ndjson,
    _iter_sse,
)


def _lines(*lines: str) -> list[bytes]:
    return [f"{line}\n".encode() for line in lines]


class IterSSETests(unittest.TestCase):
    def test_parses_single_event(self):
        lines = _lines('data: {"type": "message_start"}', "")
        self.assertEqual(list(_iter_sse(lines)), [{"type": "message_start"}])

    def test_multi_line_data_is_joined(self):
        lines = _lines('data: {"type":', 'data: "message_start"}', "")
        self.assertEqual(list(_iter_sse(lines)), [{"type": "message_start"}])

    def test_skips_done_sentinel(self):
        lines = _lines("data: [DONE]", "")
        self.assertEqual(list(_iter_sse(lines)), [])

    def test_trailing_event_without_final_blank_line(self):
        lines = _lines('data: {"type": "ping"}')
        self.assertEqual(list(_iter_sse(lines)), [{"type": "ping"}])


class IterNDJSONTests(unittest.TestCase):
    def test_parses_each_line(self):
        lines = _lines('{"done": false}', '{"done": true}')
        self.assertEqual(list(_iter_ndjson(lines)), [{"done": False}, {"done": True}])

    def test_skips_blank_lines(self):
        lines = _lines('{"a": 1}', "", '{"b": 2}')
        self.assertEqual(list(_iter_ndjson(lines)), [{"a": 1}, {"b": 2}])


class ClaudeDeserializeTests(unittest.TestCase):
    def setUp(self):
        self.adapter = ClaudeAdapter()

    def test_maps_known_stop_reason(self):
        data = {
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }
        response = self.adapter._deserialize(data)
        self.assertEqual(response["stop_reason"], "end_turn")
        self.assertEqual(response["raw_stop_reason"], "end_turn")
        self.assertEqual(response["usage"], {"input_tokens": 3, "output_tokens": 5})
        self.assertEqual(response["content"], data["content"])
        self.assertIs(response["raw"], data)

    def test_unknown_stop_reason_falls_back_to_other(self):
        data = {"content": [], "stop_reason": "weird", "usage": {}}
        response = self.adapter._deserialize(data)
        self.assertEqual(response["stop_reason"], "other")
        self.assertEqual(response["raw_stop_reason"], "weird")
        self.assertEqual(response["usage"], {"input_tokens": 0, "output_tokens": 0})


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
        response = self.adapter._deserialize(data)
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
        response = self.adapter._deserialize(data)
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
        response = self.adapter._deserialize(data)
        self.assertEqual(response["stop_reason"], "max_tokens")
        self.assertEqual(response["raw_stop_reason"], "max_output_tokens")


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


if __name__ == "__main__":
    unittest.main()
