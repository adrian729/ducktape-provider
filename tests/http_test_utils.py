"""Shared fakes for mocking urllib.request.urlopen in HTTP-path tests.

Nothing here ever opens a socket: FakeStreamResponse/FakeBufferedResponse
stand in for what urlopen() returns, so adapters under test see the same
context-manager + iteration/read shape a real http.client.HTTPResponse has.
"""

import io
import urllib.error
from email.message import Message
from typing import Self


def buffered_response(payload: bytes) -> io.BytesIO:
    """A urlopen() stand-in for the non-streaming path: json.load()-able and
    already a context manager via BytesIO's own __enter__/__exit__."""
    return io.BytesIO(payload)


class FakeStreamResponse:
    """A urlopen() stand-in for the streaming path: iterating it yields the
    given lines as bytes, one per `for raw in resp` iteration."""

    def __init__(self, lines: list[bytes]):
        self._lines = lines

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info: object) -> bool:
        return False

    def __iter__(self):
        return iter(self._lines)


def sse_lines(*events: dict) -> list[bytes]:
    import json

    lines: list[bytes] = []
    for event in events:
        lines.append(f"data: {json.dumps(event)}\n".encode())
        lines.append(b"\n")
    return lines


def ndjson_lines(*chunks: dict) -> list[bytes]:
    import json

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
