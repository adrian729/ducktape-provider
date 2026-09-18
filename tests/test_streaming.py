"""Tests for the shared wire plumbing (SSE/NDJSON framing, size caps, timing, the
request driver), independent of any adapter."""

import contextlib
import io
import itertools
import json
import socket
import threading
import time
import unittest
import urllib.request
from collections.abc import Callable, Generator, Iterator
from types import SimpleNamespace
from typing import Any, Self, cast
from unittest.mock import patch

from http_test_utils import FakeStreamResponse, final_response

from ducktape_provider import (
    APIError,
    ClaudeAdapter,
    MalformedResponseError,
    Message,
    OllamaLocalAdapter,
    OpenAIAdapter,
    RequestTimeoutError,
    Response,
    StreamEvent,
    streaming,
)
from ducktape_provider.streaming import (
    _ErrorHookedStream,
    _iter_ndjson,
    _iter_sse,
    _loads_tool_input,
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


class LoadsToolInputTests(unittest.TestCase):
    """`_loads_tool_input` reports a parse failure, or a non-dict result,
    distinctly from a legitimate no-argument call."""

    def test_empty_payload_is_legitimately_no_arguments(self):
        self.assertEqual(_loads_tool_input(""), ({}, False))

    def test_empty_object_parses_to_no_arguments(self):
        self.assertEqual(_loads_tool_input("{}"), ({}, False))

    def test_valid_object_parses_through(self):
        self.assertEqual(_loads_tool_input('{"a": 1}'), ({"a": 1}, False))

    def test_unparseable_json_is_truncated(self):
        self.assertEqual(_loads_tool_input("{oops"), ({}, True))

    def test_non_object_json_is_truncated(self):
        for payload in ("[1, 2]", "1", "null", '"str"'):
            with self.subTest(payload=payload):
                self.assertEqual(_loads_tool_input(payload), ({}, True))

    def test_deeply_nested_json_is_truncated_not_a_recursion_error(self):
        deep = "[" * 200_000 + "]" * 200_000
        self.assertEqual(_loads_tool_input(deep), ({}, True))


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


def _stream(
    resp: FakeStreamResponse, timeout: float | None = None
) -> Generator[StreamEvent]:
    req = urllib.request.Request("http://127.0.0.1:9/unused")
    with patch("urllib.request.urlopen", return_value=resp):
        yield from _stream_request(
            "test",
            req,
            timeout,
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


class _TrackedStreamResponse(FakeStreamResponse):
    """A FakeStreamResponse that remembers whether its context manager exited,
    so a test can assert the socket was released."""

    def __init__(self, lines: list[bytes], error: BaseException | None = None):
        super().__init__(lines, error)
        self.closed = False

    def __exit__(self, *exc_info: object) -> bool:
        self.closed = True
        return super().__exit__(*exc_info)


class StreamDeadlineTests(unittest.TestCase):
    """`timeout` bounds the whole stream, not just each individual read."""

    def test_cumulative_frame_time_past_deadline_raises_timeout(self):
        """Each frame's own read is well within any per-op socket timeout (mocked,
        not actually slept), but three of them in a row exceed the 2.5s total
        budget before the terminal frame ever arrives."""
        clock = itertools.count(0.0, 1.0)
        with (
            patch("time.monotonic", side_effect=lambda: next(clock)),
            self.assertRaises(RequestTimeoutError) as ctx,
        ):
            list(
                _stream(
                    _ndjson({"text": "a"}, {"text": "b"}, {"text": "c", "done": True}),
                    timeout=2.5,
                )
            )
        self.assertIn("test", str(ctx.exception))

    def test_none_timeout_disables_the_deadline_check(self):
        clock = itertools.count(0.0, 1.0)
        with patch("time.monotonic", side_effect=lambda: next(clock)):
            events = list(
                _stream(
                    _ndjson({"text": "a"}, {"text": "b", "done": True}), timeout=None
                )
            )
        self.assertEqual(events[-1]["type"], "message_stop")

    def test_closing_a_suspended_generator_closes_the_response(self):
        resp = _TrackedStreamResponse([f"{json.dumps({'text': 'a'})}\n".encode()])
        gen = _stream(resp)
        next(gen)
        gen.close()
        self.assertTrue(resp.closed)


WAIT_SECONDS = 5.0


class _FakeSocket:
    """Stands in for the socket the stream reaches through `resp.fp.raw._sock`:
    `shutdown` records the call and unblocks whatever read is waiting on it, the way
    a real socket shutdown unblocks a blocked recv."""

    def __init__(self) -> None:
        self.shutdowns: list[int] = []
        self.down = threading.Event()

    def shutdown(self, how: int) -> None:
        self.shutdowns.append(how)
        self.down.set()


class _BlockingResponse:
    """A urlopen() stand-in that blocks in readline() once its queued lines run out,
    so only a shutdown of its socket ends the read — the shape a cancel() from another
    thread has to break out of. Falls back to EOF after WAIT_SECONDS so a broken test
    fails instead of hanging."""

    def __init__(self, frames: list[object]):
        self._lines = [f"{json.dumps(frame)}\n".encode() for frame in frames]
        self.sock = _FakeSocket()
        self.fp = SimpleNamespace(raw=SimpleNamespace(_sock=self.sock))
        self.blocked = threading.Event()
        self.closed = False

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        self.closed = True
        return False

    def readline(self, limit: int = -1) -> bytes:
        if self.sock.down.is_set():
            raise ConnectionResetError("socket shut down")
        if self._lines:
            return self._lines.pop(0)
        self.blocked.set()
        if self.sock.down.wait(WAIT_SECONDS):
            raise ConnectionResetError("socket shut down")
        return b""


class CancellableStreamTests(unittest.TestCase):
    """`cancel()` on the object `_stream_request` returns, at every point of a
    stream's life and with a socket that may not be reachable at all."""

    def _open(self, resp: Any, timeout: float | None = None) -> Any:
        """Starts the patch outside a `with`, since the generator is lazy: `urlopen`
        runs on the first `next()`, long after a `with` block here would have exited."""
        patcher = patch("urllib.request.urlopen", return_value=resp)
        patcher.start()
        self.addCleanup(patcher.stop)
        return _stream_request(
            "test",
            urllib.request.Request("http://127.0.0.1:9/unused"),
            timeout,
            frames=_iter_ndjson,
            handler=_text_handler,
            terminal="the end",
        )

    def test_cancel_from_another_thread_ends_a_blocked_read(self):
        resp = _BlockingResponse([{"text": "a"}])
        stream = self._open(resp)
        self.assertEqual(next(stream)["text"], "a")
        failures: list[APIError] = []

        def drain() -> None:
            try:
                list(stream)
            except APIError as exc:
                failures.append(exc)

        reader = threading.Thread(target=drain)
        reader.start()
        self.addCleanup(reader.join, WAIT_SECONDS)
        self.assertTrue(resp.blocked.wait(WAIT_SECONDS))
        stream.cancel()
        reader.join(WAIT_SECONDS)
        self.assertFalse(reader.is_alive())
        self.assertEqual(resp.sock.shutdowns, [socket.SHUT_RDWR])
        self.assertIsInstance(failures[0], APIError)
        self.assertTrue(resp.closed)

    def test_cancel_before_the_stream_starts_prevents_the_request(self):
        resp = _BlockingResponse([{"text": "a"}])
        stream = self._open(resp)
        stream.cancel()
        with self.assertRaises(APIError):
            list(stream)
        self.assertEqual(resp.sock.shutdowns, [])
        self.assertFalse(resp.closed)

    def test_cancel_before_the_stream_starts_never_calls_urlopen(self):
        resp = _BlockingResponse([{"text": "a"}])
        stream = self._open(resp)
        stream.cancel()
        with self.assertRaises(APIError):
            next(stream)
        cast(Any, urllib.request.urlopen).assert_not_called()

    def test_cancel_after_a_finished_stream_is_a_no_op(self):
        resp = _BlockingResponse([{"text": "a"}, {"text": "b", "done": True}])
        stream = self._open(resp)
        events = list(stream)
        self.assertEqual(events[-1]["type"], "message_stop")
        stream.cancel()
        self.assertEqual(resp.sock.shutdowns, [])

    def test_cancel_without_a_reachable_socket_is_a_no_op_mid_stream(self):
        resp = _ndjson({"text": "a"}, {"text": "b", "done": True})
        stream = self._open(resp)
        self.assertEqual(next(stream)["text"], "a")
        stream.cancel()
        self.assertEqual(list(stream)[-1]["type"], "message_stop")

    def test_close_on_the_wrapper_closes_the_response(self):
        resp = _TrackedStreamResponse([f"{json.dumps({'text': 'a'})}\n".encode()])
        stream = self._open(resp)
        next(stream)
        stream.close()
        self.assertTrue(resp.closed)

    def test_close_before_the_stream_starts_never_opens_a_connection(self):
        resp = _TrackedStreamResponse([f"{json.dumps({'text': 'a'})}\n".encode()])
        stream = self._open(resp)
        stream.close()
        self.assertFalse(resp.closed)


class ErrorHookedStreamTests(unittest.TestCase):
    """`_ErrorHookedStream`'s own state machine, isolated from any adapter or
    socket: what a real generator does when its body raises on the first
    `next()`, what `throw()` does before the body ever ran, and what the
    error-eviction hook does and doesn't see."""

    def test_a_second_use_after_start_raises_gets_stopiteration_not_a_race_error(self):
        def start() -> Any:
            raise ValueError("refused")

        stream = _ErrorHookedStream("test", start, lambda e: None)
        with self.assertRaises(ValueError):
            next(stream)
        with self.assertRaises(StopIteration):
            next(stream)

    def test_throw_before_any_use_raises_directly_without_calling_start(self):
        started = []

        def start() -> Any:
            started.append(True)
            raise AssertionError(
                "start() must not run for throw() on an untouched stream"
            )

        class Boom(Exception):
            pass

        stream = _ErrorHookedStream("test", start, lambda e: None)
        with self.assertRaises(Boom):
            stream.throw(Boom("x"))
        self.assertEqual(started, [])
        with self.assertRaises(StopIteration):
            next(stream)

    def test_cancel_before_start_never_reaches_the_error_hook(self):
        def start() -> Any:
            raise AssertionError("start() must not run once cancelled before start")

        hooked: list[APIError] = []
        stream = _ErrorHookedStream("test", start, hooked.append)
        stream.cancel()
        with self.assertRaises(APIError):
            next(stream)
        self.assertEqual(hooked, [])


class _SilentServer:
    """A raw TCP listener on 127.0.0.1 that sends response headers and one chunk, then
    holds the connection open sending nothing more — so a client's next read really
    blocks on the wire, which no fake readline() can stand in for."""

    def __init__(self, first_chunk: bytes):
        self._first_chunk = first_chunk
        self._stop = threading.Event()
        self._conn: socket.socket | None = None
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.settimeout(WAIT_SECONDS)
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self._thread = threading.Thread(target=self._serve)

    @property
    def url(self) -> str:
        host, port = self._listener.getsockname()[:2]
        return f"http://{host!s}:{port}/stream"

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._stop.set()
        self._thread.join(WAIT_SECONDS)
        if self._conn is not None:
            self._conn.close()
        self._listener.close()

    def _serve(self) -> None:
        try:
            conn, _ = self._listener.accept()
        except OSError:
            return
        self._conn = conn
        with contextlib.suppress(OSError):
            conn.recv(65536)
            conn.sendall(
                b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
                b"%x\r\n%s\r\n" % (len(self._first_chunk), self._first_chunk)
            )
        self._stop.wait(WAIT_SECONDS)


class RealSocketCancelTests(unittest.TestCase):
    """The socket-shutdown path over a real TCP connection: that a genuinely blocked
    read is what `cancel()` breaks is a race property no mock can establish."""

    def test_cancel_ends_a_real_blocked_readline(self):
        with _SilentServer(f"{json.dumps({'text': 'a'})}\n".encode()) as server:
            stream = _stream_request(
                "test",
                urllib.request.Request(server.url),
                None,
                frames=_iter_ndjson,
                handler=_text_handler,
                terminal="the end",
            )
            self.assertEqual(
                next(stream), {"type": "text_delta", "index": 0, "text": "a"}
            )
            self.assertIsNotNone(stream._sock)
            failures: list[APIError] = []

            def drain() -> None:
                try:
                    list(stream)
                except APIError as exc:
                    failures.append(exc)

            reader = threading.Thread(target=drain)
            reader.start()
            self.addCleanup(reader.join, WAIT_SECONDS)
            time.sleep(0.1)
            self.assertTrue(reader.is_alive())
            stream.cancel()
            reader.join(WAIT_SECONDS)
            self.assertFalse(reader.is_alive())
            self.assertIsInstance(failures[0], APIError)
            self.assertIsNone(stream._sock)


def _sse_bytes(*events: object) -> bytes:
    """Each event ends with its own blank line, so `_iter_sse` flushes it
    immediately — a real, open-but-silent connection never gives it the
    "line iterator exhausted" fallback flush that a closed one would."""
    return b"".join(f"data: {json.dumps(e)}\n\n".encode() for e in events)


_ADAPTER_CANCEL_CASES = {
    "claude": (
        lambda: ClaudeAdapter(api_key="test"),
        "_MESSAGES_URL",
        _sse_bytes(
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
        ),
    ),
    "openai": (
        lambda: OpenAIAdapter(api_key="test"),
        "_RESPONSES_URL",
        _sse_bytes(
            {
                "type": "response.output_item.added",
                "item": {"id": "msg_1", "type": "message"},
            },
            {
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "delta": "hi",
            },
        ),
    ),
    "ollama-local": (
        lambda: OllamaLocalAdapter(),
        None,
        (
            json.dumps(
                {"message": {"role": "assistant", "content": "hi"}, "done": False}
            )
            + "\n"
        ).encode(),
    ),
}


class AdapterCancelTests(unittest.TestCase):
    """`cancel()` must reach the real socket through every built-in adapter's
    actual `stream_chat()` return value, not just through `_stream_request`
    directly — a first version of this feature had `cancel()` fully wired at
    that lower layer while every adapter's `stream_chat` hid it behind a
    `yield from`, so the feature was completely inert end to end. Only a real
    blocked socket, driven through the public adapter method, proves this;
    a mock can't."""

    def test_cancel_through_stream_chat_ends_a_real_blocked_read(self):
        messages: list[Message] = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]
        for name, (
            make_adapter,
            url_attr,
            first_chunk,
        ) in _ADAPTER_CANCEL_CASES.items():
            with self.subTest(name), _SilentServer(first_chunk) as server:
                adapter = make_adapter()
                if url_attr is not None:
                    setattr(adapter, url_attr, server.url)
                    env = {}
                else:
                    env = {"OLLAMA_HOST": server.url}
                with patch.dict("os.environ", env):
                    stream = cast(Any, adapter.stream_chat("m", messages))
                    next(stream)
                    failures: list[APIError] = []

                    def drain(stream=stream, failures=failures) -> None:
                        try:
                            list(stream)
                        except APIError as exc:
                            failures.append(exc)

                    reader = threading.Thread(target=drain)
                    reader.start()
                    self.addCleanup(reader.join, WAIT_SECONDS)
                    time.sleep(0.1)
                    self.assertTrue(reader.is_alive())
                    stream.cancel()
                    reader.join(WAIT_SECONDS)
                    self.assertFalse(reader.is_alive())
                    self.assertEqual(len(failures), 1)


def _listener_that_flags_a_connection() -> tuple[
    socket.socket, str, threading.Event, threading.Thread
]:
    connected = threading.Event()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.settimeout(WAIT_SECONDS)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    host, port = listener.getsockname()[:2]

    def accept_once() -> None:
        try:
            conn, _ = listener.accept()
            connected.set()
            conn.close()
        except OSError:
            pass

    acceptor = threading.Thread(target=accept_once, daemon=True)
    acceptor.start()
    return listener, f"http://{host!s}:{port}", connected, acceptor


class AdapterStreamLazinessTests(unittest.TestCase):
    """`stream_chat()` must never do real work — resolving headers, building
    the request — until the first `next()`/`send()`/`throw()`: exactly like a
    bare generator defers its whole body until then. A first fix for the
    cancel feature broke this by building the request eagerly inside
    `stream_chat()` itself, which made a transport-refusal `ValueError`
    surface from the wrong place instead of from the first `next()`."""

    def test_cancel_before_first_next_raises_apierror_and_never_connects(self):
        text_messages: list[Message] = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]
        listener, url, connected, acceptor = _listener_that_flags_a_connection()
        try:
            with patch.dict("os.environ", {"OLLAMA_HOST": url}):
                adapter = OllamaLocalAdapter()
                stream = cast(Any, adapter.stream_chat("m", text_messages))
                stream.cancel()
                with self.assertRaises(APIError):
                    list(stream)
                connected.wait(0.3)
                self.assertFalse(connected.is_set())
        finally:
            listener.close()
            acceptor.join(WAIT_SECONDS)

    def test_close_before_first_next_never_connects_and_ends_cleanly(self):
        text_messages: list[Message] = [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]}
        ]
        listener, url, connected, acceptor = _listener_that_flags_a_connection()
        try:
            with patch.dict("os.environ", {"OLLAMA_HOST": url}):
                adapter = OllamaLocalAdapter()
                stream = cast(Any, adapter.stream_chat("m", text_messages))
                stream.close()
                with self.assertRaises(StopIteration):
                    next(stream)
                connected.wait(0.3)
                self.assertFalse(connected.is_set())
        finally:
            listener.close()
            acceptor.join(WAIT_SECONDS)


if __name__ == "__main__":
    unittest.main()
