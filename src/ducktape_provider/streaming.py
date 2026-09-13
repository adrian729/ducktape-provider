"""HTTP plumbing the built-in adapters share: request/stream drivers, SSE and NDJSON
framing, size limits, stream timing, and mapping wire failures onto `errors`."""

import contextlib
import http.client
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from typing import Any, Protocol

from . import errors
from .types import StreamEvent

_FIRST_CONTENT_EVENTS = frozenset({"text_delta", "thinking_delta", "tool_use_start"})

# Far above any legitimate event or body (a whole long response is well under
# this), but bounds what a misbehaving server can make us buffer.
_MAX_EVENT_BYTES = 16 * 1024 * 1024

# Read a buffered body in pieces so the cap never costs a 16 MiB allocation up front.
_READ_CHUNK_BYTES = 64 * 1024

# Errors that only arise from a wire event having an unexpected shape (a missing
# key, a list where an object belongs, a delta for a block that never started).
_SHAPE_ERRORS = (KeyError, IndexError, TypeError, AttributeError)

# Everything a best-effort probe such as models() treats as "vendor unreachable".
# ValueError covers an unusable base URL (e.g. a malformed OLLAMA_HOST).
_PROBE_ERRORS = (
    OSError,
    ValueError,
    http.client.HTTPException,
    errors.MalformedResponseError,
    *_SHAPE_ERRORS,
)

_END = object()


class _LineReader(Protocol):
    def readline(self, limit: int = -1, /) -> bytes: ...


class _Reader(Protocol):
    def read(self, amt: int, /) -> bytes: ...


def _iter_lines(resp: _LineReader, vendor: str) -> Iterator[bytes]:
    # readline() with a limit, rather than iterating the response, so an endless
    # line is rejected before it is buffered in full.
    while line := resp.readline(_MAX_EVENT_BYTES + 1):
        if len(line) > _MAX_EVENT_BYTES:
            errors.raise_for_malformed_response(
                vendor, ValueError(f"line exceeds {_MAX_EVENT_BYTES} bytes")
            )
        yield line


def _loads(payload: str | bytes, vendor: str) -> Any:
    try:
        return json.loads(payload)
    except (ValueError, RecursionError) as e:
        # RecursionError: nesting deep enough to exhaust the parser's stack.
        errors.raise_for_malformed_response(vendor, e)


def _iter_sse(resp: _LineReader, vendor: str) -> Iterator[Any]:
    data_lines: list[str] = []
    size = 0
    for raw in _iter_lines(resp, vendor):
        try:
            line = raw.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as e:
            errors.raise_for_malformed_response(vendor, e)
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
            size += len(raw)
            if size > _MAX_EVENT_BYTES:
                errors.raise_for_malformed_response(
                    vendor, ValueError(f"event exceeds {_MAX_EVENT_BYTES} bytes")
                )
        elif not line and data_lines:
            payload = "\n".join(data_lines)
            data_lines = []
            size = 0
            if payload != "[DONE]":
                yield _loads(payload, vendor)
    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            yield _loads(payload, vendor)


def _iter_ndjson(resp: _LineReader, vendor: str) -> Iterator[Any]:
    for raw in _iter_lines(resp, vendor):
        line = raw.strip()
        if line:
            yield _loads(line, vendor)


def _read_json(resp: _Reader, vendor: str) -> Any:
    chunks: list[bytes] = []
    size = 0
    while chunk := resp.read(min(_READ_CHUNK_BYTES, _MAX_EVENT_BYTES + 1 - size)):
        size += len(chunk)
        if size > _MAX_EVENT_BYTES:
            errors.raise_for_malformed_response(
                vendor, ValueError(f"body exceeds {_MAX_EVENT_BYTES} bytes")
            )
        chunks.append(chunk)
    return _loads(b"".join(chunks), vendor)


class _StreamTimer:
    """Timestamps for one streamed request, taken when each wire event is parsed.

    Starts right before the request is sent. Every normalized event is stamped with
    the time the wire event that produced it was read, so time the caller spends
    handling an already-yielded event is not counted. The socket is still read
    lazily, though: a caller that is slow to pull the *next* event delays that
    read, and therefore every later timestamp.
    """

    def __init__(self) -> None:
        self._start = time.monotonic()
        self._last_read = self._start
        self._first_content: float | None = None

    def track[T](self, items: Iterable[T]) -> Iterator[T]:
        for item in items:
            self._last_read = time.monotonic()
            yield item

    def observe(self, event: StreamEvent) -> None:
        if self._first_content is None and event["type"] in _FIRST_CONTENT_EVENTS:
            self._first_content = self._last_read

    def latency_ms(self) -> float:
        return (self._last_read - self._start) * 1000

    def ttft_ms(self) -> float | None:
        if self._first_content is None:
            return None
        return (self._first_content - self._start) * 1000


@contextlib.contextmanager
def _shape_checked(vendor: str) -> Iterator[None]:
    """Reports a wire payload of unexpected shape as a malformed response.

    Wrap only the parsing of a payload, never a `yield` to the consumer: an
    exception the consumer throws into the generator must not be relabeled as the
    vendor's fault.
    """
    try:
        yield
    except _SHAPE_ERRORS as e:
        errors.raise_for_malformed_response(vendor, e)


@contextlib.contextmanager
def _transport_errors(vendor: str) -> Iterator[None]:
    """Maps an HTTP error status or a failed connection/read onto `APIError`.

    Same rule as `_shape_checked`: never wrap a `yield` to the consumer.
    """
    try:
        yield
    except urllib.error.HTTPError as e:
        errors.raise_for_http_error(vendor, e)
    except (OSError, http.client.HTTPException) as e:
        errors.raise_for_connection_error(vendor, e)


def _request_json(
    vendor: str, req: urllib.request.Request, timeout: float | None
) -> tuple[Any, float]:
    """Sends `req` and returns its parsed JSON body with the latency in ms."""
    start = time.monotonic()
    with (
        _transport_errors(vendor),
        urllib.request.urlopen(req, timeout=timeout) as resp,
    ):
        data = _read_json(resp, vendor)
    return data, (time.monotonic() - start) * 1000


type _FrameHandler = Callable[[Any], Iterable[StreamEvent]]


def _stream_request(
    vendor: str,
    req: urllib.request.Request,
    timeout: float | None,
    *,
    frames: Callable[[_LineReader, str], Iterator[Any]],
    handler: Callable[[_StreamTimer], _FrameHandler],
    terminal: str,
) -> Iterator[StreamEvent]:
    """Sends `req` and yields the normalized events for each wire frame.

    `handler(timer)` returns the adapter's per-frame handler, which holds the
    stream's state and ends by producing a `message_stop` for the `terminal` wire
    event; a stream that ends before that is truncated.
    """
    timer = _StreamTimer()
    handle = handler(timer)
    with _transport_errors(vendor):
        resp = urllib.request.urlopen(req, timeout=timeout)
    with resp:
        wire = timer.track(frames(resp, vendor))
        while True:
            with _transport_errors(vendor):
                frame = next(wire, _END)
            if frame is _END:
                errors.raise_for_truncated_stream(vendor, terminal)
            out: list[StreamEvent] = []
            with _shape_checked(vendor):
                for event in handle(frame):
                    # Observed as it is produced, so a message_stop built from the
                    # same frame already accounts for content that preceded it.
                    timer.observe(event)
                    out.append(event)
            yield from out
            if out and out[-1]["type"] == "message_stop":
                return
