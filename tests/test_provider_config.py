import os
import unittest
import urllib.request
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any
from unittest.mock import patch

from ducktape_provider import Adapter, Config, Provider
from ducktape_provider.adapters.ollama import OllamaLocalAdapter
from ducktape_provider.types import Message, Response, StreamEvent, ToolDef

MESSAGES: list[Message] = [
    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
]

FIXED_RESPONSE: Response = {
    "content": [{"type": "text", "text": "hi there"}],
    "stop_reason": "end_turn",
    "raw_stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 2},
    "raw": {},
    "latency_ms": 0.0,
}


class RecordingAdapter(Adapter):
    """Records the config each call receives, after Provider has resolved it."""

    def __init__(self) -> None:
        self.configs: list[dict[str, Any] | None] = []

    def is_available(self) -> bool:
        return True

    def models(self) -> set[str]:
        return set()

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response:
        self.configs.append(config)
        return FIXED_RESPONSE

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        self.configs.append(config)
        return iter(())


_OMITTED = object()


class StopBeforeSending(BaseException):
    """Aborts the request once captured; BaseException so no adapter wraps it."""


def resolve(
    config: Config | Mapping[str, Any] | None = None, timeout: Any = _OMITTED
) -> dict[str, Any] | None:
    """The config a "rec" adapter receives for one chat() call."""
    adapter = RecordingAdapter()
    if timeout is _OMITTED:
        provider = Provider(adapters={"rec": adapter})
    else:
        provider = Provider(adapters={"rec": adapter}, timeout=timeout)
    provider.chat("model", MESSAGES, config=config, provider="rec")
    [resolved] = adapter.configs
    return resolved


class TestProviderConfigResolution(unittest.TestCase):
    def test_later_layers_override_earlier_ones(self):
        config: Config = {
            "timeout": 20,
            "providers": {"rec": {"timeout": 30}, "other": {"timeout": 40}},
        }
        self.assertEqual(resolve(config, timeout=10), {"timeout": 30})
        self.assertEqual(resolve({"timeout": 20}, timeout=10), {"timeout": 20})
        self.assertEqual(resolve(timeout=10), {"timeout": 10})

    def test_vendor_fields_pass_through_without_providers_key(self):
        config = {"temperature": 0.2, "providers": {"rec": {"top_p": 0.9}}}
        self.assertEqual(resolve(config), {"temperature": 0.2, "top_p": 0.9})

    def test_headers_merge_key_by_key_with_later_layers_winning(self):
        config = {
            "headers": {"X-Global": "1", "X-Shared": "call"},
            "providers": {"rec": {"headers": {"X-Prov": "2", "X-Shared": "prov"}}},
        }
        resolved = resolve(config)
        assert resolved is not None
        self.assertEqual(
            resolved["headers"], {"X-Global": "1", "X-Prov": "2", "X-Shared": "prov"}
        )
        self.assertEqual(config["headers"], {"X-Global": "1", "X-Shared": "call"})

    def test_accepts_any_mapping_and_copies_headers(self):
        headers = {"X-Global": "1"}
        resolved = resolve(MappingProxyType({"headers": headers}))
        assert resolved is not None
        self.assertEqual(resolved["headers"], headers)
        self.assertIsNot(resolved["headers"], headers)

    def test_merged_headers_reach_the_request(self):
        sent: list[urllib.request.Request] = []

        def capture(req: urllib.request.Request, timeout: float | None = None):
            sent.append(req)
            raise StopBeforeSending

        provider = Provider(adapters={"ollama-local": OllamaLocalAdapter()})
        config: Config = {
            "headers": {"X-Global": "1"},
            "providers": {"ollama-local": {"headers": {"X-Prov": "2"}}},
        }
        with (
            patch.dict(os.environ, {"OLLAMA_HOST": "http://127.0.0.1:9"}),
            patch("urllib.request.urlopen", capture),
            self.assertRaises(StopBeforeSending),
        ):
            provider.chat("model", MESSAGES, config=config, provider="ollama-local")
        [req] = sent
        headers = {key.lower(): value for key, value in req.header_items()}
        self.assertEqual(headers["x-global"], "1")
        self.assertEqual(headers["x-prov"], "2")


class TestProviderTimeout(unittest.TestCase):
    def test_omitted_timeout_leaves_adapter_default(self):
        self.assertEqual(resolve(), {})

    def test_none_disables_timeouts_like_config_none(self):
        self.assertEqual(resolve(timeout=None), {"timeout": None})

    def test_valid_timeouts_are_accepted(self):
        for timeout in (1, 0.5, 300):
            with self.subTest(timeout=timeout):
                self.assertEqual(resolve(timeout=timeout), {"timeout": timeout})

    def test_invalid_timeouts_raise_at_construction(self):
        invalid: list[Any] = [0, -1, 0.0, float("nan"), float("inf"), 1e10, True, "5"]
        for timeout in invalid:
            with (
                self.subTest(timeout=timeout),
                self.assertRaisesRegex(
                    ValueError, "^Provider timeout must be a positive number of seconds"
                ),
            ):
                Provider(adapters={}, timeout=timeout)

    def test_none_timeout_reaches_urlopen(self):
        timeouts: list[float | None] = []

        def capture(req: urllib.request.Request, timeout: float | None = None):
            timeouts.append(timeout)
            raise StopBeforeSending

        provider = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()}, timeout=None
        )
        with (
            patch.dict(os.environ, {"OLLAMA_HOST": "http://127.0.0.1:9"}),
            patch("urllib.request.urlopen", capture),
            self.assertRaises(StopBeforeSending),
        ):
            provider.chat("model", MESSAGES, provider="ollama-local")
        self.assertEqual(timeouts, [None])


if __name__ == "__main__":
    unittest.main()
