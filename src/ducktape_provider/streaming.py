"""HTTP plumbing the built-in adapters share: request/stream drivers, SSE and NDJSON
framing, size limits, stream timing, and mapping wire failures onto `errors`."""

import contextlib
import http.client
import json
import socket
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from collections.abc import Callable, Generator, Iterable, Iterator
from types import TracebackType
from typing import Any, Protocol

from . import errors
from .types import StreamEvent

_FIRST_CONTENT_EVENTS = frozenset({"text_delta", "thinking_delta", "tool_use_start"})

_MAX_EVENT_BYTES = 16 * 1024 * 1024

_READ_CHUNK_BYTES = 64 * 1024

_SHAPE_ERRORS = (KeyError, IndexError, TypeError, AttributeError)

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
    while line := resp.readline(_MAX_EVENT_BYTES + 1):
        if len(line) > _MAX_EVENT_BYTES:
            errors.raise_for_malformed_response(
                vendor, ValueError(f"line exceeds {_MAX_EVENT_BYTES} bytes")
            )
        yield line


def _loads(payload: str | bytes, vendor: str, *, operation: str = "chat") -> Any:
    try:
        return json.loads(payload)
    except (ValueError, RecursionError) as e:
        errors.raise_for_malformed_response(vendor, e, operation=operation)


def _loads_tool_input(payload: str) -> tuple[dict[str, Any], bool]:
    """Returns (args, truncated). `truncated` is only True when `payload` was
    non-empty and failed to parse, or parsed to something other than a dict —
    both signs of a `max_tokens` cutoff mid-arguments, not a legitimate
    no-argument call."""
    if not payload:
        return {}, False
    try:
        parsed = json.loads(payload)
    except (ValueError, RecursionError):
        return {}, True
    if isinstance(parsed, dict):
        return parsed, False
    return {}, True


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


def _read_json(resp: _Reader, vendor: str, *, operation: str = "chat") -> Any:
    chunks: list[bytes] = []
    size = 0
    while chunk := resp.read(min(_READ_CHUNK_BYTES, _MAX_EVENT_BYTES + 1 - size)):
        size += len(chunk)
        if size > _MAX_EVENT_BYTES:
            errors.raise_for_malformed_response(
                vendor,
                ValueError(f"body exceeds {_MAX_EVENT_BYTES} bytes"),
                operation=operation,
            )
        chunks.append(chunk)
    return _loads(b"".join(chunks), vendor, operation=operation)


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
def _shape_checked(vendor: str, *, operation: str = "chat") -> Iterator[None]:
    """Reports a wire payload of unexpected shape as a malformed response.

    Wrap only the parsing of a payload, never a `yield` to the consumer: an
    exception the consumer throws into the generator must not be relabeled as the
    vendor's fault.
    """
    try:
        yield
    except _SHAPE_ERRORS as e:
        errors.raise_for_malformed_response(vendor, e, operation=operation)


def _clear_tracebacks(exc: BaseException, stop: BaseException | None = None) -> None:
    """Swaps the traceback of `exc`, and of every exception chained to it, for its
    text — stopping at (and leaving untouched) `stop` and anything chained from it.

    urllib's `do_open` and http.client's frames on those tracebacks hold the
    request's header dict, API key included, where anything that inspects frame
    locals (debuggers, error reporters) would find it. The note keeps each
    file/line/source for debugging, without the locals. `stop` is the exception
    already being handled when the caller's guarded step started (captured via
    `sys.exception()` before it): a user's own `except` block that constructs a
    `Provider` or makes a call is not gutted of its own traceback.
    """
    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending:
        current = pending.pop()
        if current is None or current is stop or id(current) in seen:
            continue
        seen.add(id(current))
        if current.__traceback__ is not None:
            try:
                with contextlib.suppress(TypeError):
                    current.add_note(
                        "".join(traceback.format_tb(current.__traceback__))
                    )
            finally:
                current.__traceback__ = None
        pending += (current.__cause__, current.__context__)


class _transport_errors:
    """Maps an HTTP error status or a failed connection/read onto `APIError`.

    Same rule as `_shape_checked`: never wrap a `yield` to the consumer. A class
    rather than `@contextlib.contextmanager`, whose generator machinery would
    keep its own references to the original traceback.
    """

    def __init__(self, vendor: str, *, operation: str = "chat") -> None:
        self._vendor = vendor
        self._operation = operation

    def __enter__(self) -> None:
        self._stop = sys.exception()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del tb
        if exc is None or isinstance(exc, errors.DucktapeError):
            return
        _clear_tracebacks(exc, self._stop)
        if isinstance(exc, urllib.error.HTTPError):
            errors.raise_for_http_error(self._vendor, exc, operation=self._operation)
        if isinstance(exc, (OSError, http.client.HTTPException)):
            errors.raise_for_connection_error(
                self._vendor, exc, operation=self._operation
            )


def _request_json(
    vendor: str,
    req: urllib.request.Request,
    timeout: float | None,
    *,
    operation: str = "chat",
) -> tuple[Any, float]:
    """Sends `req` and returns its parsed JSON body with the latency in ms.

    `req` holds the revealed API key in its unredirected headers; it is dropped
    as soon as `urlopen` returns or raises, so neither a connection failure's nor
    a malformed-response's traceback carries it as a frame local.
    """
    with _transport_errors(vendor, operation=operation):
        start = time.monotonic()
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        finally:
            del req
        with resp:
            data = _read_json(resp, vendor, operation=operation)
    return data, (time.monotonic() - start) * 1000


type _FrameHandler = Callable[[Any], Iterable[StreamEvent]]


class _CancellableStream:
    """Wraps a wire-reading generator with a thread-safe `cancel()` that
    shuts down its live socket. Degrades to a no-op if CPython's internal
    socket handle ever moves."""

    __slots__ = ("_cancelled", "_gen", "_lock", "_sock")

    def __init__(self) -> None:
        self._gen: Generator[StreamEvent] | None = None
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._cancelled = False

    def __iter__(self) -> "_CancellableStream":
        return self

    def __next__(self) -> StreamEvent:
        if self._gen is None:
            raise RuntimeError("_CancellableStream used before its generator was set")
        return next(self._gen)

    def close(self) -> None:
        if self._gen is not None:
            self._gen.close()

    def throw(self, *args: Any, **kwargs: Any) -> StreamEvent:
        if self._gen is None:
            raise RuntimeError("_CancellableStream used before its generator was set")
        return self._gen.throw(*args, **kwargs)

    def send(self, value: None) -> StreamEvent:
        if self._gen is None:
            raise RuntimeError("_CancellableStream used before its generator was set")
        return self._gen.send(value)

    def _track(self, resp: Any) -> None:
        sock = getattr(getattr(resp, "fp", None), "raw", None)
        sock = getattr(sock, "_sock", None)
        with self._lock:
            if self._cancelled:
                do_shutdown = sock
            else:
                self._sock = sock
                do_shutdown = None
        if do_shutdown is not None:
            with contextlib.suppress(OSError, AttributeError):
                do_shutdown.shutdown(socket.SHUT_RDWR)

    def _untrack(self) -> None:
        with self._lock:
            self._sock = None

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            sock, self._sock = self._sock, None
        if sock is not None:
            with contextlib.suppress(OSError, AttributeError):
                sock.shutdown(socket.SHUT_RDWR)

    def _cancelled_before_send(self) -> bool:
        with self._lock:
            return self._cancelled


class _ErrorHookedStream:
    """Lazily calls `start`, forwards `close`/`throw`/`send`/`cancel` to the
    result, and calls `on_error` on any `APIError` that reaches the consumer."""

    __slots__ = (
        "_cancelled",
        "_closed",
        "_lock",
        "_on_error",
        "_start",
        "_starting",
        "_stream",
        "_vendor",
    )

    def __init__(
        self,
        vendor: str,
        start: Callable[[], "_CancellableStream"],
        on_error: Callable[[errors.APIError], None],
    ) -> None:
        self._vendor = vendor
        self._start = start
        self._stream: _CancellableStream | _NeverStarted | None = None
        self._on_error = on_error
        self._lock = threading.Lock()
        self._cancelled = False
        self._closed = False
        self._starting = False

    def _ensure(self) -> "_CancellableStream | _NeverStarted":
        if self._stream is not None:
            return self._stream
        with self._lock:
            if self._stream is not None:
                return self._stream
            if self._cancelled:
                self._stream = _NeverStarted(
                    errors.APIError(
                        f"{self._vendor} chat stream cancelled before it started"
                    )
                )
                return self._stream
            if self._closed:
                self._stream = _NeverStarted(None)
                return self._stream
            already_starting = self._starting
            self._starting = True
        if already_starting:
            raise RuntimeError(
                f"{self._vendor} stream consumed from two threads at once"
            )
        try:
            stream = self._start()
        except BaseException:
            with self._lock:
                self._stream = _NeverStarted(None)
            raise
        with self._lock:
            cancelled = self._cancelled
            closed = self._closed
            self._stream = stream
        if cancelled:
            stream.cancel()
        elif closed:
            stream.close()
        return stream

    def __iter__(self) -> "_ErrorHookedStream":
        return self

    def __next__(self) -> StreamEvent:
        stream = self._ensure()
        try:
            return next(stream)
        except errors.APIError as e:
            if not isinstance(stream, _NeverStarted):
                self._on_error(e)
            raise

    def close(self) -> None:
        with self._lock:
            self._closed = True
            stream = self._stream
        if stream is not None:
            stream.close()

    def throw(self, *args: Any, **kwargs: Any) -> StreamEvent:
        with self._lock:
            never_touched = (
                self._stream is None
                and not self._starting
                and not self._cancelled
                and not self._closed
            )
            if never_touched:
                self._stream = _NeverStarted(None)
        if never_touched:
            try:
                return _NeverStarted(None).throw(*args, **kwargs)
            except errors.APIError as e:
                self._on_error(e)
                raise
        stream = self._ensure()
        try:
            return stream.throw(*args, **kwargs)
        except errors.APIError as e:
            if not isinstance(stream, _NeverStarted):
                self._on_error(e)
            raise

    def send(self, value: None) -> StreamEvent:
        stream = self._ensure()
        try:
            return stream.send(value)
        except errors.APIError as e:
            if not isinstance(stream, _NeverStarted):
                self._on_error(e)
            raise

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            stream = self._stream
        if stream is not None:
            stream.cancel()


class _NeverStarted:
    """Stands in for a stream cancelled or closed before `start` ran.
    `error=None` means closed (raises `StopIteration`); otherwise cancelled
    (raises `error`)."""

    __slots__ = ("_error",)

    def __init__(self, error: "errors.APIError | None") -> None:
        self._error = error

    def __iter__(self) -> "_NeverStarted":
        return self

    def __next__(self) -> StreamEvent:
        if self._error is not None:
            raise self._error
        raise StopIteration

    def close(self) -> None:
        return None

    def throw(self, *args: Any, **kwargs: Any) -> StreamEvent:
        if kwargs:
            raise TypeError("generator.throw() takes no keyword arguments")
        if not args:
            raise TypeError(f"throw expected at least 1 argument, got {len(args)}")
        exc = args[0]
        if isinstance(exc, BaseException):
            raise exc
        if isinstance(exc, type) and issubclass(exc, BaseException):
            value = args[1] if len(args) > 1 else None
            if isinstance(value, BaseException):
                raise value
            raise exc(value) if value is not None else exc()
        raise TypeError(
            "exceptions must be classes or instances deriving from "
            f"BaseException, not {type(exc).__name__}"
        )

    def send(self, value: None) -> StreamEvent:
        if self._error is not None:
            raise self._error
        raise StopIteration

    def cancel(self) -> None:
        return None


def _stream_request(
    vendor: str,
    req: urllib.request.Request,
    timeout: float | None,
    *,
    frames: Callable[[_LineReader, str], Iterator[Any]],
    handler: Callable[[_StreamTimer], _FrameHandler],
    terminal: str,
) -> _CancellableStream:
    """Sends `req` and yields the normalized events for each wire frame; the returned
    object also has `cancel()` — see `_CancellableStream`.

    The generator is built after the wrapper so it can hold a reference to it; a
    generator body doesn't run until the first `next()`, by which point the wrapper
    is fully constructed.
    """
    stream = _CancellableStream()
    stream._gen = _stream_wire(
        vendor,
        req,
        timeout,
        frames=frames,
        handler=handler,
        terminal=terminal,
        tracker=stream,
    )
    return stream


def _stream_wire(
    vendor: str,
    req: urllib.request.Request,
    timeout: float | None,
    *,
    frames: Callable[[_LineReader, str], Iterator[Any]],
    handler: Callable[[_StreamTimer], _FrameHandler],
    terminal: str,
    tracker: _CancellableStream,
) -> Generator[StreamEvent]:
    """Sends `req` and yields the normalized events for each wire frame.

    `handler(timer)` returns the adapter's per-frame handler, which holds the
    stream's state and ends by producing a `message_stop` for the `terminal` wire
    event; a stream that ends before that is truncated.

    `req` holds the revealed API key in its unredirected headers; it is dropped
    as soon as `urlopen` returns or raises, so this generator's frame — which
    stays alive for the whole stream — never holds it.

    `timeout` bounds the whole stream, checked once per frame rather than left
    to the socket: a peer trickling one byte at a time forever would never trip
    the per-read socket timeout underneath, so this is what makes `timeout` a
    total-time ceiling for a stream rather than a per-op one. A single frame's
    own read past `timeout` is still bounded by that socket timeout, unchanged.
    """
    timer = _StreamTimer()
    deadline = None if timeout is None else time.monotonic() + timeout
    handle = handler(timer)
    if tracker._cancelled_before_send():
        raise errors.APIError(f"{vendor} chat stream cancelled before it started")
    with _transport_errors(vendor):
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        finally:
            del req
    tracker._track(resp)
    with resp:
        try:
            wire = timer.track(frames(resp, vendor))
            while True:
                if deadline is not None and time.monotonic() > deadline:
                    raise errors.RequestTimeoutError(f"{vendor} chat timed out")
                with _transport_errors(vendor):
                    frame = next(wire, _END)
                if frame is _END:
                    errors.raise_for_truncated_stream(vendor, terminal)
                out: list[StreamEvent] = []
                with _shape_checked(vendor):
                    for event in handle(frame):
                        timer.observe(event)
                        out.append(event)
                yield from out
                if out and out[-1]["type"] == "message_stop":
                    return
        finally:
            tracker._untrack()
