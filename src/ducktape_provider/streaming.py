import http.client
import json
from collections.abc import Iterator
from typing import Any


def _iter_sse(resp: http.client.HTTPResponse) -> Iterator[dict[str, Any]]:
    data_lines: list[str] = []
    for raw in resp:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
        elif not line and data_lines:
            payload = "\n".join(data_lines)
            data_lines = []
            if payload != "[DONE]":
                yield json.loads(payload)
    if data_lines:
        payload = "\n".join(data_lines)
        if payload != "[DONE]":
            yield json.loads(payload)


def _iter_ndjson(resp: http.client.HTTPResponse) -> Iterator[dict[str, Any]]:
    for raw in resp:
        line = raw.strip()
        if line:
            yield json.loads(line)
