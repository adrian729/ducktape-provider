"""Tests for the low-level SSE/NDJSON line parsers, independent of any adapter."""

import unittest

from ducktape_provider.streaming import _iter_ndjson, _iter_sse


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


if __name__ == "__main__":
    unittest.main()
