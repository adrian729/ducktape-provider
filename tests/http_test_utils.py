"""Shared HTTP fakes and fixtures for adapter tests. Nothing here reaches the network.

FakeStreamResponse/buffered_response stand in for what urlopen() returns, exposing
only what the adapters use: a context manager with readline(limit) (streaming) or
read(amt) (buffered). LocalServer is the real thing on 127.0.0.1, for end-to-end
tests through http.client's own response parsing.
"""

import io
import json
import threading
import urllib.error
from collections.abc import Sequence
from dataclasses import dataclass, field
from email.message import Message
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Self, cast
from urllib.request import Request

from ducktape_provider import Response, StreamEvent


def buffered_response(payload: bytes) -> io.BytesIO:
    """A urlopen() stand-in for the non-streaming path: readable and already a
    context manager via BytesIO's own __enter__/__exit__."""
    return io.BytesIO(payload)


class FakeStreamResponse:
    """A urlopen() stand-in for the streaming path: readline() returns the given
    lines as bytes, one per call (split when longer than the limit, as a real
    response would), then b"" at the end. With `error`, raises it once the lines
    run out, like a connection dropping mid-stream."""

    def __init__(self, lines: list[bytes], error: BaseException | None = None):
        self._lines = list(lines)
        self._error = error

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        return False

    def readline(self, limit: int = -1) -> bytes:
        if not self._lines:
            if self._error is not None:
                raise self._error
            return b""
        line = self._lines.pop(0)
        if 0 <= limit < len(line):
            self._lines.insert(0, line[limit:])
            line = line[:limit]
        return line


def sse_lines(*events: object) -> list[bytes]:
    lines: list[bytes] = []
    for event in events:
        lines.append(f"data: {json.dumps(event)}\n".encode())
        lines.append(b"\n")
    return lines


def ndjson_lines(*chunks: object) -> list[bytes]:
    return [f"{json.dumps(chunk)}\n".encode() for chunk in chunks]


def http_error(
    url: str,
    code: int,
    body: bytes = b"boom",
    headers: dict[str, str] | None = None,
) -> urllib.error.HTTPError:
    msg = Message()
    for key, value in (headers or {}).items():
        msg[key] = value
    return urllib.error.HTTPError(url, code, "error", msg, io.BytesIO(body))


def final_response(events: Sequence[StreamEvent]) -> Response:
    """The response on a stream's closing message_stop event."""
    last = events[-1]
    if last["type"] != "message_stop":
        raise AssertionError(f"stream did not end with message_stop: {last!r}")
    return last["response"]


def request_body(req: Request) -> Any:
    """The JSON body an adapter built into `req`."""
    return json.loads(cast(bytes, req.data))


def split_every(data: bytes, size: int) -> list[bytes]:
    """Cuts `data` into fixed-size pieces, ignoring line and event boundaries."""
    return [data[i : i + size] for i in range(0, len(data), size)]


@dataclass
class Reply:
    """One canned response. `parts` are sent as separate HTTP chunks when
    `chunked`, otherwise concatenated under a Content-Length."""

    parts: Sequence[bytes] = ()
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    chunked: bool = True


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: Any


class LocalServer:
    """A threaded HTTP/1.1 server on 127.0.0.1 with an ephemeral port that answers
    each request with the next queued Reply, and records what it received."""

    def __init__(self, *replies: Reply):
        self.replies = list(replies)
        self.requests: list[RecordedRequest] = []
        self._lock = threading.Lock()
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._handler_class())
        self._httpd.daemon_threads = False
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.01}
        )

    @property
    def url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return f"http://{host!s}:{port}"

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self._httpd.shutdown()
        self._thread.join()
        self._httpd.server_close()

    def _next_reply(self, request: RecordedRequest) -> Reply:
        with self._lock:
            self.requests.append(request)
            return self.replies.pop(0)

    def _handler_class(self) -> type[BaseHTTPRequestHandler]:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                self._respond(None)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                self._respond(json.loads(self.rfile.read(length)))

            def _respond(self, body: Any) -> None:
                reply = server._next_reply(
                    RecordedRequest(self.command, self.path, dict(self.headers), body)
                )
                self.send_response(reply.status)
                for key, value in reply.headers.items():
                    self.send_header(key, value)
                if reply.chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    for part in reply.parts:
                        if part:
                            self.wfile.write(b"%x\r\n%s\r\n" % (len(part), part))
                            self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    payload = b"".join(reply.parts)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                self.close_connection = True

        return Handler
