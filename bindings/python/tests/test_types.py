"""Round-trips the type samples through the Python dataclasses."""
import pytest
from conftest import load, type_fixtures

from ducktape_provider import _types
from ducktape_provider._types import _DEFS, decode, encode

SAMPLES = [p for p in type_fixtures() if load(p)["def"] in _DEFS]


@pytest.mark.parametrize("path", SAMPLES, ids=lambda p: p.stem)
def test_type_sample_round_trips(path):
    sample = load(path)
    typed = decode(_DEFS[sample["def"]], sample["value"])
    assert encode(typed) == sample["value"]


def test_unknown_capabilities_stay_unknown():
    caps = _types.Capabilities.from_dict({"source": "shipped", "tool_choice": None})
    assert caps.inputs is _types.UNKNOWN
    assert caps.tool_choice is None
    assert caps.to_dict() == {"source": "shipped", "tool_choice": None}


def test_response_views():
    r = _types.Response.from_dict(
        {
            "content": [
                {"type": "reasoning", "text": "think"},
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b"},
                {"type": "tool_call", "id": "c1", "name": "add", "arguments": {"a": 1}},
            ],
            "stop": {"reason": "tool_call"},
            "usage": None,
            "warnings": [],
            "origin": {"wire": "w", "vendor": "anthropic", "base_url": "u", "model": "m"},
            "latency_ms": 1.0,
        }
    )
    assert r.text == "ab"
    assert r.reasoning == "think"
    assert [c.name for c in r.tool_calls] == ["add"]
    assert r.tool_calls[0].type == "tool_call"
