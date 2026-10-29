"""Replays every wire fixture through the native module and checks the golden exchange."""
import base64
import json

import pytest
from conftest import load, wire_fixtures

from ducktape_provider import _core

ENTROPY = bytes([128] * 32)


def clock(tick):
    return {"now_ms": 1_761_696_000_000 + tick * 100, "entropy": ENTROPY}


def normalize(step):
    step = json.loads(json.dumps(step))
    request = step.get("effect", {}).get("request")
    if request and isinstance(request.get("body"), str):
        try:
            request["body"] = json.loads(request["body"])
        except ValueError:
            pass
    return step


def replay(fixture):
    start = dict(fixture["start"])
    start["clock"] = clock(0)
    op, step = _core.start(start)
    http = iter(fixture["http"])
    queue = []
    out = []
    tick = 0
    for _ in range(10_000):
        effect = step["effect"]
        kind = effect["type"]
        if kind in ("done", "failed"):
            out.append({"step": step})
            return out
        if kind == "need_secret":
            ref = effect["secret_ref"]
            value = fixture.get("secrets", {}).get(ref)
            if value is not None:
                op.set_secret(ref, value.encode())
            data = {"type": "secret", "secret_ref": ref, "found": value is not None}
        elif kind == "need_http":
            h = next(http)
            if h.get("error"):
                data = {"type": "http_error", "error": h["error"]}
            else:
                chunks = h.get("chunks") or ([h["body"]] if h.get("body") is not None else [])
                queue = [{"type": "http_chunk", "bytes": base64.b64encode(c.encode()).decode()} for c in chunks]
                queue.append({"type": "http_end"})
                queue.reverse()
                data = {"type": "http_response", "status": h.get("status", 200), "headers": h.get("headers", {})}
        elif kind == "receive":
            data = queue.pop() if queue else {"type": "http_end"}
        elif kind == "need_sleep":
            data = {"type": "slept"}
        elif kind == "need_read":
            loc = effect["locator"]
            if loc["type"] == "file":
                value = fixture.get("files", {}).get(loc["path"])
            elif loc["type"] == "env":
                value = fixture.get("env", {}).get(loc["name"])
            else:
                value = fixture.get("keychain", {}).get(f"{loc['service']}|{loc['account']}")
            data = {"type": "read", "found": value is not None}
            if value is not None:
                data["bytes"] = base64.b64encode(value.encode()).decode()
        elif kind == "need_write":
            data = {"type": "written", "ok": True, "conflict": False}
        else:
            raise AssertionError(f"unsupported effect in replay: {kind}")
        out.append({"step": step, "input": data})
        tick += 1
        step = op.resume({"clock": clock(tick), "input": data})
    raise AssertionError("replay did not finish")


@pytest.mark.parametrize("path", wire_fixtures(), ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_wire_fixture(path):
    fixture = load(path)
    actual = replay(fixture)
    expected = fixture["exchange"]
    assert len(actual) == len(expected), json.dumps(actual[-1], indent=1)
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert normalize(a["step"]) == normalize(e["step"]), f"step {i}"
        assert a.get("input") == e.get("input"), f"input {i}"
