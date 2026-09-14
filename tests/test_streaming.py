"""Tests for the shared wire plumbing (SSE/NDJSON framing, size caps, timing, the
request driver), independent of any adapter."""

import io
import json
import unittest
import urllib.request
from collections.abc import Callable, Generator, Iterator
from typing import Any
from unittest.mock import patch

from http_test_utils import FakeStreamResponse, final_response

from ducktape_provider import (
    APIError,
    MalformedResponseError,
    Response,
    StreamEvent,
    streaming,
)
from ducktape_provider.streaming import (
    _iter_ndjson,
    _iter_sse,
    _read_json,
    _shape_checked,
    _stream_request,
    _StreamTimer,
)


def _lines(*lines: str) -> FakeStreamResponse:
    return FakeStreamResponse([f"{line}\n".encode() for line in lines])


class IterSSETests(unittest.TestCase):
    def test_parses_single_event(self):
        lines = _lines('data: {"type": "message_start"}', "")
        self.assertEqual(list(_iter_sse(lines, "test")), [{"type": "message_start"}])

    def test_multi_line_data_is_joined(self):
        lines = _lines('data: {"type":', 'data: "message_start"}', "")
        self.assertEqual(list(_iter_sse(lines, "test")), [{"type": "message_start"}])

    def test_skips_done_sentinel(self):
        lines = _lines("data: [DONE]", "")
        self.assertEqual(list(_iter_sse(lines, "test")), [])

    def test_trailing_event_without_final_blank_line(self):
        lines = _lines('data: {"type": "ping"}')
        self.assertEqual(list(_iter_sse(lines, "test")), [{"type": "ping"}])


class IterNDJSONTests(unittest.TestCase):
    def test_parses_each_line(self):
        lines = _lines('{"done": false}', '{"done": true}')
        self.assertEqual(
            list(_iter_ndjson(lines, "test")), [{"done": False}, {"done": True}]
        )

    def test_skips_blank_lines(self):
        lines = _lines('{"a": 1}', "", '{"b": 2}')
        self.assertEqual(list(_iter_ndjson(lines, "test")), [{"a": 1}, {"b": 2}])


class OversizeAndMalformedTests(unittest.TestCase):
    def test_oversize_line_raises_malformed_before_buffering_it_whole(self):
        resp = FakeStreamResponse([b"data: " + b"x" * 100 + b"\n"])
        with (
            patch.object(streaming, "_MAX_EVENT_BYTES", 32),
            self.assertRaises(MalformedResponseError) as ctx,
        ):
            list(_iter_sse(resp, "test"))
        self.assertIn("malformed", str(ctx.exception))
        # only limit + 1 bytes were read from the endless line
        self.assertEqual(len(resp._lines[0]), 100 + 7 - 33)

    def test_oversize_multi_line_event_raises_malformed(self):
        lines = _lines(*["data: " + "x" * 20] * 3, "")
        with (
            patch.object(streaming, "_MAX_EVENT_BYTES", 64),
            self.assertRaises(MalformedResponseError),
        ):
            list(_iter_sse(lines, "test"))

    def test_oversize_ndjson_line_raises_malformed(self):
        with (
            patch.object(streaming, "_MAX_EVENT_BYTES", 8),
            self.assertRaises(MalformedResponseError),
        ):
            list(_iter_ndjson(_lines('{"a": "long value"}'), "test"))

    def test_invalid_utf8_raises_malformed(self):
        resp = FakeStreamResponse([b"data: \xff\n", b"\n"])
        with self.assertRaises(MalformedResponseError) as ctx:
            list(_iter_sse(resp, "test"))
        self.assertIn("malformed", str(ctx.exception))

    def test_shape_checked_converts_shape_errors_only(self):
        for exc in (KeyError("index"), IndexError(), TypeError(), AttributeError()):
            with (
                self.subTest(exc=exc),
                self.assertRaises(MalformedResponseError) as ctx,
                _shape_checked("test"),
            ):
                raise exc
            self.assertIs(ctx.exception.__cause__, exc)
        with self.assertRaises(RuntimeError), _shape_checked("test"):
            raise RuntimeError()

    def test_deeply_nested_json_is_malformed_not_recursion_error(self):
        deep = "[" * 200_000 + "]" * 200_000
        with self.assertRaises(MalformedResponseError):
            list(_iter_ndjson(_lines(deep), "test"))
        with self.assertRaises(MalformedResponseError):
            _read_json(io.BytesIO(deep.encode()), "test")


class ReadJsonTests(unittest.TestCase):
    def test_reads_body_arriving_in_short_pieces(self):
        body = json.dumps({"text": "x" * 200_000}).encode()
        self.assertEqual(_read_json(_Trickle(body, 1000), "test"), json.loads(body))

    def test_body_over_the_cap_raises_after_reading_at_most_cap_plus_one(self):
        resp = io.BytesIO(b'"' + b"x" * 100 + b'"')
        with (
            patch.object(streaming, "_MAX_EVENT_BYTES", 32),
            self.assertRaises(MalformedResponseError) as ctx,
        ):
            _read_json(resp, "test")
        self.assertIn("exceeds", str(ctx.exception))
        self.assertEqual(resp.tell(), 33)

    def test_body_exactly_at_the_cap_is_accepted(self):
        body = b'"' + b"x" * 30 + b'"'
        with patch.object(streaming, "_MAX_EVENT_BYTES", len(body)):
            self.assertEqual(_read_json(io.BytesIO(body), "test"), "x" * 30)


class _Trickle(io.BytesIO):
    """Returns at most `step` bytes per read, like a socket delivering in pieces."""

    def __init__(self, data: bytes, step: int):
        super().__init__(data)
        self._step = step

    def read(self, size: int | None = -1, /) -> bytes:
        if size is None or size < 0 or size > self._step:
            size = self._step
        return super().read(size)


class StreamTimerTests(unittest.TestCase):
    def test_timings_use_read_time_of_the_producing_event(self):
        with patch("time.monotonic", side_effect=[10.0, 10.5, 11.0, 12.0]):
            timer = _StreamTimer()
            reads = timer.track(["ping", "delta", "stop"])
            next(reads)
            timer.observe({"type": "block_stop", "index": 0})
            next(reads)
            timer.observe({"type": "text_delta", "index": 0, "text": "hi"})
            next(reads)
            timer.observe({"type": "text_delta", "index": 0, "text": "later"})
        self.assertEqual(timer.ttft_ms(), 1000.0)
        self.assertEqual(timer.latency_ms(), 2000.0)

    def test_ttft_is_none_without_content_events(self):
        with patch("time.monotonic", side_effect=[1.0, 2.0]):
            timer = _StreamTimer()
            list(timer.track(["stop"]))
        self.assertIsNone(timer.ttft_ms())
        self.assertEqual(timer.latency_ms(), 1000.0)


type _Handler = Callable[[Any], Iterator[StreamEvent]]


def _text_handler(timer: _StreamTimer) -> _Handler:
    """Emits each frame's "text" as a delta and a message_stop on "done", built
    after the delta so it reads the timer's state the way adapters do."""

    def handle(frame: Any) -> Iterator[StreamEvent]:
        if "text" in frame:
            yield {"type": "text_delta", "index": 0, "text": frame["text"]}
        if frame.get("done"):
            response: Response = {
                "content": [],
                "stop_reason": "end_turn",
                "raw_stop_reason": "",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "raw": {},
                "latency_ms": timer.latency_ms(),
            }
            if (ttft := timer.ttft_ms()) is not None:
                response["ttft_ms"] = ttft
            yield {"type": "message_stop", "response": response}

    return handle


def _stream(resp: FakeStreamResponse) -> Generator[StreamEvent]:
    req = urllib.request.Request("http://127.0.0.1:9/unused")
    with patch("urllib.request.urlopen", return_value=resp):
        yield from _stream_request(
            "test",
            req,
            None,
            frames=_iter_ndjson,
            handler=_text_handler,
            terminal="the end",
        )


def _ndjson(*frames: object, error: BaseException | None = None) -> FakeStreamResponse:
    return FakeStreamResponse([f"{json.dumps(f)}\n".encode() for f in frames], error)


class StreamRequestTests(unittest.TestCase):
    """The request driver every built-in adapter streams through."""

    def test_message_stop_sees_content_from_its_own_frame(self):
        with patch("time.monotonic", side_effect=[5.0, 5.5, 6.0]):
            events = list(_stream(_ndjson({}, {"text": "hi", "done": True})))
        response = final_response(events)
        self.assertEqual(response.get("ttft_ms"), 1000.0)
        self.assertEqual(response["latency_ms"], 1000.0)

    def test_stream_without_terminal_event_is_truncated_not_malformed(self):
        with self.assertRaises(APIError) as ctx:
            list(_stream(_ndjson({"text": "hi"})))
        self.assertIs(type(ctx.exception), APIError)
        self.assertIn("the end", str(ctx.exception))

    def test_shape_error_in_handler_is_malformed(self):
        with self.assertRaises(MalformedResponseError):
            list(_stream(_ndjson([1])))

    def test_read_failure_is_a_connection_error_but_thrown_one_is_not(self):
        with self.assertRaises(APIError) as ctx:
            list(_stream(_ndjson({"text": "a"}, error=ConnectionResetError())))
        self.assertIsInstance(ctx.exception.__cause__, ConnectionResetError)

        thrown = ConnectionResetError("consumer")
        consuming = _stream(_ndjson({"text": "a"}, {"text": "b"}))
        next(consuming)
        with self.assertRaises(ConnectionResetError) as thrown_ctx:
            consuming.throw(thrown)
        self.assertIs(thrown_ctx.exception, thrown)


if __name__ == "__main__":
    unittest.main()
