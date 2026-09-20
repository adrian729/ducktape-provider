"""Tests for config layering: `Provider(config=...)`, `Provider(timeout=...)` and each
call's `config`, plus configured per-provider headers, which are secrets.

urlopen is mocked, or requests go to a LocalServer on 127.0.0.1 or a closed port
there; nothing reaches a vendor or a local Ollama daemon.
"""

import asyncio
import contextlib
import functools
import itertools
import json
import os
import pickle
import unittest
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NoReturn, cast
from unittest.mock import Mock, patch

from http_test_utils import (
    FakeStreamResponse,
    LocalServer,
    Reply,
    buffered_response,
    ollama_embed_response,
)
from test_provider_api_keys import (
    KEYED,
    OLLAMA_MODEL,
    SENTINEL,
    SameInstanceCopyAdapter,
    _sentinel_in_value,
    assert_no_sentinel_in_frames,
    clean_env,
    closed_loopback_url,
    drain,
    fake_urlopen,
    no_request,
    ollama_urlopen,
    pointed_at,
    raised,
    secret_headers,
)

from ducktape_provider import (
    Adapter,
    APIError,
    AuthError,
    ClaudeAdapter,
    Config,
    MalformedResponseError,
    OpenAIAdapter,
    Provider,
)
from ducktape_provider.adapters.ollama import OllamaLocalAdapter
from ducktape_provider.streaming import _clear_tracebacks
from ducktape_provider.types import (
    EmbedResponse,
    Message,
    Response,
    StreamEvent,
    ToolDef,
)

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

HEADER = "X-Proxy-Token"


class RecordingAdapter(Adapter):
    """Records the config each call receives, after Provider has resolved it."""

    def __init__(self) -> None:
        self.configs: list[dict[str, Any] | None] = []
        self.embed_configs: list[dict[str, Any] | None] = []

    def is_available(self) -> bool:
        return True

    def models(self) -> set[str]:
        return set()

    def embed_models(self) -> set[str]:
        """Model ids this adapter can embed with."""
        return {"rec-embed"}

    def embed(
        self,
        model: str,
        input: list[str],
        config: dict[str, Any] | None = None,
    ) -> EmbedResponse:
        """Embed texts."""
        self.embed_configs.append(config)
        return {
            "embeddings": [[0.1]],
            "usage": {"input_tokens": 1},
            "raw": {},
            "latency_ms": 0.0,
        }

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
    config: Config | Mapping[str, Any] | None = None,
    timeout: Any = _OMITTED,
    provider_config: Config | Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """The config a "rec" adapter receives for one chat() call."""
    adapter = RecordingAdapter()
    kwargs: dict[str, Any] = {"config": provider_config}
    if timeout is not _OMITTED:
        kwargs["timeout"] = timeout
    provider = Provider(
        adapters={"rec": adapter, "other": RecordingAdapter()}, **kwargs
    )
    provider.chat("model", MESSAGES, config=config, provider="rec")
    [resolved] = adapter.configs
    return resolved


def resolve_embed(
    config: Config | Mapping[str, Any] | None = None,
    timeout: Any = _OMITTED,
    provider_config: Config | Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """The config a "rec" adapter receives for one embed() call."""
    adapter = RecordingAdapter()
    kwargs: dict[str, Any] = {"config": provider_config}
    if timeout is not _OMITTED:
        kwargs["timeout"] = timeout
    provider = Provider(
        adapters={"rec": adapter, "other": RecordingAdapter()}, **kwargs
    )
    provider.embed("rec-embed", "hi", config=config, provider="rec")
    [resolved] = adapter.embed_configs
    return resolved


def adapter_of(provider: Provider, name: str) -> Any:
    return provider._adapters[name]


def sent_header(req: urllib.request.Request, name: str = HEADER) -> str | None:
    """A header on a built request, matched case-insensitively."""
    return {k.lower(): v for k, v in req.header_items()}.get(name.lower())


def vault_down() -> NoReturn:
    raise OSError("vault down")


async def consume_stream(provider: Provider, model: str, name: str) -> None:
    async for _ in provider.async_stream_chat(model, MESSAGES, provider=name):
        pass


class ExplodingItems(Mapping[str, Any]):
    """A mapping whose `items()` yields its first pair, then raises."""

    def __init__(self, pairs: dict[str, Any]) -> None:
        self._pairs = pairs

    def __getitem__(self, key: str) -> Any:
        return self._pairs[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def items(self) -> Any:
        yield next(iter(self._pairs.items()))
        raise RuntimeError("mapping broke")


class Impostor(str):
    def __hash__(self) -> int:
        return hash("ollama-local")

    def __eq__(self, other: object) -> bool:
        return True


def assert_no_secret_in_messages(
    test: unittest.TestCase, exc: BaseException, *secrets: str
) -> None:
    """Fails if `exc`'s message or notes, or any chained exception's, has a secret."""
    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        notes = getattr(current, "__notes__", [])
        texts = [str(current), *(n for n in notes if isinstance(n, str))]
        for secret in (SENTINEL, *secrets):
            for text in texts:
                test.assertNotIn(secret, text)
        pending += (current.__cause__, current.__context__)


@dataclass(frozen=True)
class Target:
    name: str
    cls: type[ClaudeAdapter] | type[OpenAIAdapter] | type[OllamaLocalAdapter]
    model: str
    host: str
    auth_header: str | None
    urlopen: Callable[..., Any]
    bad_stream: list[bytes]

    def provider(
        self,
        headers: Mapping[str, object] | None,
        key: str | Callable[[], str | None] | None = "k",
        **kwargs: Any,
    ) -> Provider:
        """A Provider with only this target, its key, and `headers` configured."""
        return Provider(
            adapters={self.name: self.cls()},
            api_keys={self.name: key} if self.auth_header and key is not None else None,
            config=None
            if headers is None
            else {"providers": {self.name: {"headers": headers}}},
            **kwargs,
        )

    def requests(self, adapter: Any) -> dict[str, Callable[[], object]]:
        return {
            "chat": functools.partial(adapter.chat, self.model, MESSAGES),
            "stream_chat": functools.partial(drain, adapter, self.model),
        }

    def probes(self, adapter: Any) -> dict[str, Callable[[], object]]:
        """The calls that send a request only to check the vendor."""
        probes = {"models": adapter.models}
        if self.auth_header is None:
            probes["is_available"] = adapter.is_available
        return probes


TARGETS = [
    *(
        Target(
            case.name,
            case.cls,
            case.model,
            case.cls._MODELS_URL.split("/")[2],
            case.header.lower(),
            fake_urlopen(case),
            [b"data: not json\n", b"\n"],
        )
        for case in KEYED
    ),
    Target(
        "ollama-local",
        OllamaLocalAdapter,
        OLLAMA_MODEL,
        "127.0.0.1",
        None,
        ollama_urlopen,
        [b"not json\n"],
    ),
]


def any_urlopen(req: Any, timeout: float | None = None) -> Any:
    """A urlopen stand-in answering whichever target `req` is for."""
    [target] = [t for t in TARGETS if t.host in req.full_url]
    return target.urlopen(req, timeout)


class TestProviderConfigResolution(unittest.TestCase):
    def test_later_layers_override_earlier_ones(self):
        config: Config = {
            "timeout": 20,
            "providers": {"rec": {"timeout": 30}, "other": {"timeout": 40}},
        }
        self.assertEqual(resolve(config, timeout=10), {"timeout": 30})
        self.assertEqual(resolve({"timeout": 20}, timeout=10), {"timeout": 20})
        self.assertEqual(resolve(timeout=10), {"timeout": 10})

    def test_four_layers_each_override_the_previous(self):
        provider_config = {
            "a": "provider",
            "b": "provider",
            "c": "provider",
            "d": "provider",
            "providers": {
                "rec": {"b": "provider rec", "c": "provider rec", "d": "provider rec"},
                "other": {"a": "provider other"},
            },
        }
        call_config = {
            "c": "call",
            "d": "call",
            "providers": {"rec": {"d": "call rec"}, "other": {"a": "call other"}},
        }
        self.assertEqual(
            resolve(call_config, provider_config=provider_config),
            {"a": "provider", "b": "provider rec", "c": "call", "d": "call rec"},
        )
        self.assertEqual(
            resolve(provider_config={"timeout": 5, "providers": {"rec": {"x": 1}}}),
            {"timeout": 5, "x": 1},
        )

    def test_provider_config_is_a_snapshot(self):
        stop = ["a"]
        providers: dict[str, dict[str, Any]] = {"rec": {"stop": stop}}
        provider_config: dict[str, Any] = {"temperature": 0.1, "providers": providers}
        adapter = RecordingAdapter()
        provider = Provider(adapters={"rec": adapter}, config=provider_config)
        stop.append("b")
        providers["rec"]["top_p"] = 1
        provider_config["temperature"] = 0.9
        provider.chat("model", MESSAGES, provider="rec")
        self.assertEqual(adapter.configs, [{"temperature": 0.1, "stop": ["a"]}])

    def test_timeout_and_config_timeout_together_raise(self):
        with self.assertRaisesRegex(ValueError, "timeout"):
            Provider(adapters={}, timeout=5, config={"timeout": 5})
        self.assertEqual(
            resolve(timeout=5, provider_config={"providers": {"rec": {"timeout": 6}}}),
            {"timeout": 6},
        )

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
                self.assertEqual(
                    resolve(provider_config={"timeout": timeout}), {"timeout": timeout}
                )

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
            for config in ({"timeout": timeout}, {"providers": {"r": {"timeout": 0}}}):
                with (
                    self.subTest(timeout=timeout, config=config),
                    self.assertRaises(ValueError),
                ):
                    Provider(adapters={"r": RecordingAdapter()}, config=config)

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


class ProviderConfigValidationTests(unittest.TestCase):
    """Bad config fails at construction, naming only providers and types: a header
    mapping written backwards would put its token where the name goes."""

    def setUp(self):
        self.enterContext(clean_env(OLLAMA_HOST="http://127.0.0.1:9"))

    def test_invalid_config_raises_without_secrets(self):
        def secret() -> dict[str, Any]:
            """Computed fresh on every call, never captured by a closure: a
            lambda that closed over a pre-built copy would carry SENTINEL in
            its own closure cell, which the frame scan below would (rightly)
            flag — as a property of this test's own code, not the library's.
            """
            return secret_headers("ollama-local")["providers"]["ollama-local"]

        def with_secret(extra: dict[str, Any]) -> dict[str, Any]:
            return {
                **extra,
                "providers": {"ollama-local": secret(), **extra.get("providers", {})},
            }

        failures: dict[str, tuple[type[Exception], Callable[[], object]]] = {
            "config not a mapping": (TypeError, lambda: Provider(config=[SENTINEL])),  # ty: ignore[invalid-argument-type]
            "config a str": (TypeError, lambda: Provider(config=SENTINEL)),  # ty: ignore[invalid-argument-type]
            "providers not a mapping": (
                TypeError,
                lambda: Provider(config={"providers": [SENTINEL]}),
            ),
            "provider entry not a mapping": (
                TypeError,
                lambda: Provider(config={"providers": {"ollama-local": SENTINEL}}),
            ),
            "headers not a mapping": (
                TypeError,
                lambda: Provider(
                    config={"providers": {"ollama-local": {"headers": [SENTINEL]}}}
                ),
            ),
            "non-str provider name": (
                TypeError,
                lambda: Provider(config={"providers": {42: secret()}}),
            ),
            "str subclass provider name": (
                TypeError,
                lambda: Provider(config={"providers": {Impostor("x"): secret()}}),
            ),
            "non-str header name": (
                TypeError,
                lambda: Provider(
                    config={"providers": {"ollama-local": {"headers": {42: SENTINEL}}}}
                ),
            ),
            "str subclass header name": (
                TypeError,
                lambda: Provider(config=secret_headers("ollama-local") | {}),
            )
            if False
            else (
                TypeError,
                lambda: Provider(
                    config={
                        "providers": {
                            "ollama-local": {"headers": {Impostor(SENTINEL): "v"}}
                        }
                    }
                ),
            ),
            "unregistered provider": (
                ValueError,
                lambda: Provider(config=with_secret({"providers": {"nope": {}}})),
            ),
            "top-level headers": (
                ValueError,
                lambda: Provider(config={"headers": {HEADER: SENTINEL}}),
            ),
            "top-level key-like key": (
                ValueError,
                lambda: Provider(config=with_secret({"api_key": "x"})),
            ),
            "per-provider key-like key": (
                ValueError,
                lambda: Provider(
                    config=with_secret({"providers": {"claude": {"x-api-key": "x"}}})
                ),
            ),
            "top-level reserved key": (
                ValueError,
                lambda: Provider(config=with_secret({"stream": True})),
            ),
            "per-provider reserved key": (
                ValueError,
                lambda: Provider(
                    config=with_secret({"providers": {"openai": {"input": []}}})
                ),
            ),
            "per-provider embed-reserved key": (
                ValueError,
                lambda: Provider(
                    config=with_secret({"providers": {"ollama-local": {"input": []}}})
                ),
            ),
            "third-party adapter headers": (
                ValueError,
                lambda: Provider(
                    adapters={"rec": RecordingAdapter()},
                    config=secret_headers("rec"),
                ),
            ),
            "timeout twice": (
                ValueError,
                lambda: Provider(timeout=5, config=with_secret({"timeout": 5})),
            ),
            "adapter copying to itself": (
                TypeError,
                lambda: Provider(
                    adapters={"claude": SameInstanceCopyAdapter()},
                    config=secret_headers("claude"),
                ),
            ),
            "bad shape plus secret api_keys": (
                TypeError,
                lambda: Provider(api_keys={"claude": SENTINEL}, config=[]),  # ty: ignore[invalid-argument-type]
            ),
            "bare-str api_keys plus secret config": (
                TypeError,
                lambda: Provider(api_keys=SENTINEL, config=with_secret({})),  # ty: ignore[invalid-argument-type]
            ),
            "invalid timeout plus secret config": (
                ValueError,
                lambda: Provider(
                    timeout=-1,
                    api_keys={"claude": SENTINEL},
                    config=with_secret({}),
                ),
            ),
            "api_keys items() raises partway": (
                RuntimeError,
                lambda: Provider(
                    api_keys=ExplodingItems({"claude": SENTINEL, "openai": "k"}),
                    config=with_secret({}),
                ),
            ),
            "headers items() raises partway": (
                RuntimeError,
                lambda: Provider(
                    api_keys={"claude": SENTINEL},
                    config={
                        "providers": {
                            "ollama-local": {
                                "headers": ExplodingItems({HEADER: SENTINEL, "X": "v"})
                            }
                        }
                    },
                ),
            ),
            "providers items() raises partway": (
                RuntimeError,
                lambda: Provider(
                    config={
                        "providers": ExplodingItems({"ollama-local": secret(), "x": {}})
                    }
                ),
            ),
        }

        def resolved(x: object) -> object:
            """`x()` for a zero-arg callable, `x` itself otherwise — used below
            for the bad names/values that embed SENTINEL: kept as a closure-free
            lambda re-deriving it from the global on each call, instead of a
            pre-built string, so `call`'s own `__defaults__` (a frame local of
            `raised`, alongside every other test's `call`) never holds it.
            """
            return x() if callable(x) else x

        bad_names: list[tuple[str, str | Callable[[], str]]] = [
            ("empty", ""),
            ("leading space", " X"),
            ("trailing space", "X "),
            ("colon", "X:Y"),
            ("internal space", "X Y"),
            ("embedded newline", "X\nY"),
            ("non-ascii", "Xé"),
            ("secret with CRLF", lambda: f"{SENTINEL}\r\n"),
        ]
        for name_label, name in bad_names:
            failures[f"header name ({name_label})"] = (
                ValueError,
                lambda name=name: Provider(
                    config={
                        "providers": {
                            "ollama-local": {"headers": {resolved(name): "v"}}
                        }
                    }
                ),
            )
        bad_values: dict[str, tuple[type[Exception], object]] = {
            "empty": (ValueError, ""),
            "CR": (ValueError, lambda: f"{SENTINEL}\r"),
            "LF": (ValueError, lambda: f"{SENTINEL}\n"),
            "folded": (ValueError, lambda: f"{SENTINEL}\r\n x"),
            "NUL": (ValueError, lambda: f"{SENTINEL}\x00"),
            "non-latin-1": (ValueError, lambda: f"{SENTINEL}€"),
            "bytes": (TypeError, lambda: SENTINEL.encode()),
            "int": (TypeError, 4242),
            "None": (TypeError, None),
        }
        for label, (error, value) in bad_values.items():
            failures[f"header value {label}"] = (
                error,
                lambda value=value: Provider(
                    config={
                        "providers": {
                            "ollama-local": {
                                "headers": {"X-A": SENTINEL, "X-B": resolved(value)}
                            }
                        }
                    }
                ),
            )
        for label, (error, call) in failures.items():
            with self.subTest(label):
                exc = raised(self, error, call)
                self.assertIs(type(exc), error)
                assert_no_secret_in_messages(self, exc, HEADER, "X-A", "X-B", "4242")
                assert_no_sentinel_in_frames(self, exc)
                if label in ("third-party adapter headers", "top-level headers"):
                    self.assertIn("per", str(exc))

    def test_header_mapping_is_read_once(self):
        class Lazy(Mapping[str, str]):
            reads = 0

            def __getitem__(self, key: str) -> str:
                Lazy.reads += 1
                return f"v{Lazy.reads}"

            def __iter__(self) -> Iterator[str]:
                return iter([HEADER])

            def __len__(self) -> int:
                return 1

        provider = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()},
            config={"providers": {"ollama-local": {"headers": Lazy()}}},
        )
        self.assertEqual(Lazy.reads, 1)
        with patch("urllib.request.urlopen", side_effect=ollama_urlopen) as m:
            provider.chat(OLLAMA_MODEL, MESSAGES)
            provider.chat(OLLAMA_MODEL, MESSAGES, provider="ollama-local")
        self.assertEqual([sent_header(c.args[0]) for c in m.call_args_list], ["v1"] * 4)
        self.assertEqual(Lazy.reads, 1)

    def test_configured_state_is_redacted_and_unpicklable(self):
        for target in TARGETS:
            for value in (SENTINEL, lambda: SENTINEL):
                with self.subTest(target.name, value=type(value).__name__):
                    provider = target.provider({HEADER: value}, key=SENTINEL)
                    adapter = adapter_of(provider, target.name)
                    for state in (vars(adapter), vars(provider)):
                        self.assertNotIn(SENTINEL, repr(state))
                        self.assertFalse(_sentinel_in_value(state, set(), 0))
                    for obj in (adapter, provider):
                        with self.assertRaises(TypeError) as ctx:
                            pickle.dumps(obj)
                        self.assertNotIn(SENTINEL, str(ctx.exception))

    def test_passed_adapters_are_left_unchanged(self):
        for target in TARGETS:
            with self.subTest(target.name):
                original = target.cls()
                stale: Any = (
                    {"stale": {"context_window": None, "max_output_tokens": None}}
                    if isinstance(original, ClaudeAdapter)
                    else {"stale"}
                )
                original._models_cache = stale
                original._warned_transport = True
                provider = Provider(
                    adapters={target.name: original},
                    config=secret_headers(target.name, "t"),
                )
                copied = adapter_of(provider, target.name)
                self.assertIsNot(copied, original)
                self.assertEqual(original._provider_headers, ())
                self.assertIsNone(original._provider_name)
                self.assertEqual(original._models_cache, stale)
                self.assertIsNone(copied._models_cache)
                self.assertFalse(copied._warned_transport)
                self.assertEqual(copied._provider_name, target.name)
                self.assertEqual(len(copied._provider_headers), 1)

    def test_headers_apply_to_plugins_subclassing_a_built_in_adapter(self):
        from test_provider_discovery import ENTRY_POINTS, FakeEntryPoint

        from ducktape_provider.provider import _clear_discovery_cache

        _clear_discovery_cache()
        self.addCleanup(_clear_discovery_cache)
        plugin = FakeEntryPoint("myvendor", OllamaLocalAdapter)
        with patch(ENTRY_POINTS, return_value=[plugin]):
            provider = Provider(
                autodiscover=True, config=secret_headers("myvendor", "t")
            )
        self.assertEqual(len(adapter_of(provider, "myvendor")._provider_headers), 1)
        self.assertEqual(adapter_of(provider, "ollama-local")._provider_headers, ())


class ProviderHeaderWireTests(unittest.TestCase):
    """What configured headers put on built requests, with urlopen mocked."""

    def setUp(self):
        self.enterContext(clean_env(OLLAMA_HOST="http://127.0.0.1:9", no_proxy="*"))

    def test_headers_reach_requests_and_probes(self):
        for target in TARGETS:
            for value in ("t", lambda: "t"):
                provider = target.provider({HEADER: value})
                adapter = adapter_of(provider, target.name)
                calls = {**target.requests(adapter), **target.probes(adapter)}
                with (
                    self.subTest(target.name, value=type(value).__name__),
                    patch("urllib.request.urlopen", side_effect=target.urlopen) as m,
                ):
                    for call in calls.values():
                        call()
                    self.assertEqual(
                        [sent_header(c.args[0]) for c in m.call_args_list],
                        ["t"] * len(calls),
                    )
                    self.assertEqual(
                        [c.kwargs["timeout"] for c in m.call_args_list][-1], 3
                    )

    def test_function_is_called_once_per_request(self):
        for target in TARGETS:
            counter = itertools.count(1)
            fn = Mock(side_effect=lambda counter=counter: f"t{next(counter)}")
            provider = target.provider({HEADER: fn})
            adapter = adapter_of(provider, target.name)
            calls = {**target.requests(adapter), **target.probes(adapter)}
            with (
                self.subTest(target.name),
                patch("urllib.request.urlopen", side_effect=target.urlopen) as m,
            ):
                for call in calls.values():
                    call()
                    adapter._models_cache = None
            self.assertEqual(
                [sent_header(c.args[0]) for c in m.call_args_list],
                [f"t{i}" for i in range(1, len(calls) + 1)],
            )

    def test_claude_models_pagination_calls_function_once(self):
        fn = Mock(return_value="t")
        provider = Provider(
            adapters={"claude": ClaudeAdapter()},
            api_keys={"claude": "k"},
            config=secret_headers("claude", fn),
        )
        pages = [
            buffered_response(
                b'{"data": [{"id": "a"}], "has_more": true, "last_id": "a"}'
            ),
            buffered_response(b'{"data": [{"id": "b"}], "has_more": false}'),
        ]
        with patch("urllib.request.urlopen", side_effect=pages) as m:
            self.assertEqual(adapter_of(provider, "claude").models(), {"a", "b"})
        fn.assert_called_once_with()
        self.assertEqual([sent_header(c.args[0]) for c in m.call_args_list], ["t"] * 2)

    def test_auth_header_works_without_a_key(self):
        auth_values = {"claude": "hdr-key", "openai": "Bearer hdr-key"}
        for target in TARGETS[:2]:
            assert target.auth_header is not None
            cls = cast("type[ClaudeAdapter | OpenAIAdapter]", target.cls)
            for source in (None, Mock(return_value="k")):
                adapter_in = cls(api_key=source)
                provider = Provider(
                    adapters={target.name: adapter_in},
                    config={
                        "providers": {
                            target.name: {
                                "headers": {
                                    target.auth_header.upper(): auth_values[target.name]
                                }
                            }
                        }
                    },
                )
                adapter = adapter_of(provider, target.name)
                with (
                    self.subTest(target.name, source=type(source).__name__),
                    patch.object(
                        os.environ, "get", side_effect=AssertionError("env read")
                    ),
                    patch("urllib.request.urlopen", side_effect=target.urlopen) as m,
                ):
                    self.assertTrue(adapter.is_available())
                    self.assertEqual(adapter.models(), {target.model})
                    for call in target.requests(adapter).values():
                        call()
                    self.assertEqual(
                        [
                            sent_header(c.args[0], target.auth_header)
                            for c in m.call_args_list
                        ],
                        [auth_values[target.name]] * 3,
                    )
                if source is not None:
                    source.assert_not_called()
                with (
                    self.subTest(target.name, auto_match=True),
                    patch("urllib.request.urlopen", side_effect=target.urlopen),
                ):
                    provider.chat(target.model, MESSAGES)

    def test_call_header_overrides_without_calling_the_function(self):
        for target in TARGETS:
            fn = Mock(return_value="configured")
            provider = target.provider({HEADER: fn, "X-Other": "other"})
            adapter = adapter_of(provider, target.name)
            config = {"headers": {HEADER.lower(): "call"}}
            with (
                self.subTest(target.name),
                patch("urllib.request.urlopen", side_effect=target.urlopen) as m,
            ):
                adapter.chat(target.model, MESSAGES, config=config)
                list(adapter.stream_chat(target.model, MESSAGES, config=config))
            fn.assert_not_called()
            for c in m.call_args_list:
                self.assertEqual(sent_header(c.args[0]), "call")
                self.assertEqual(sent_header(c.args[0], "X-Other"), "other")

    def test_headers_only_reach_their_own_provider(self):
        fn = Mock(return_value="t")
        provider = Provider(
            api_keys={"claude": "k", "openai": "k"},
            config=secret_headers("ollama-local", fn),
        )
        with patch("urllib.request.urlopen", side_effect=any_urlopen) as m:
            self.assertEqual(
                provider.providers(),
                {"claude": True, "openai": True, "ollama-local": True},
            )
            self.assertEqual(len(provider.models()), 3)
            asyncio.run(provider.async_models())
            for target in TARGETS:
                provider.chat(target.model, MESSAGES, provider=target.name)
                list(provider.stream_chat(target.model, MESSAGES, provider=target.name))
        ollama = [
            c.args[0] for c in m.call_args_list if "127.0.0.1" in c.args[0].full_url
        ]
        others = [c.args[0] for c in m.call_args_list if c.args[0] not in ollama]
        self.assertTrue(ollama)
        self.assertTrue(others)
        self.assertEqual([sent_header(r) for r in ollama], ["t"] * len(ollama))
        self.assertEqual([sent_header(r) for r in others], [None] * len(others))
        self.assertEqual(fn.call_count, len(ollama))

    def test_headers_are_not_forwarded_on_a_redirect(self):
        """`req.headers` is what urllib copies onto a redirected request; a
        configured header must land only in `req.unredirected_hdrs`, same as
        an API key, never here."""
        for target in TARGETS:
            provider = target.provider({HEADER: "t"})
            adapter = adapter_of(provider, target.name)
            calls = {**target.requests(adapter), **target.probes(adapter)}
            with (
                self.subTest(target.name),
                patch("urllib.request.urlopen", side_effect=target.urlopen) as m,
            ):
                for call in calls.values():
                    call()
            for c in m.call_args_list:
                self.assertEqual(c.args[0].headers, {})


class ProviderHeaderProbeTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(clean_env(OLLAMA_HOST="http://127.0.0.1:9", no_proxy="*"))

    def test_ollama_behind_an_auth_proxy(self):
        unauthorized = [Reply(status=401, chunked=False) for _ in range(2)]
        with LocalServer(*unauthorized) as server:
            os.environ["OLLAMA_HOST"] = server.url
            provider = Provider(adapters={"ollama-local": OllamaLocalAdapter()})
            self.assertEqual(provider.providers(), {"ollama-local": False})
            self.assertEqual(adapter_of(provider, "ollama-local").models(), set())
        ok = [
            Reply([b'{"version": "0.0.0"}'], chunked=False),
            Reply([b'{"version": "0.0.0"}'], chunked=False),
            Reply([b'{"models": [{"name": "llama3"}]}'], chunked=False),
        ]
        with LocalServer(*ok) as server:
            os.environ["OLLAMA_HOST"] = f"{server.url}/"
            provider = Provider(
                adapters={"ollama-local": OllamaLocalAdapter()},
                config=secret_headers("ollama-local", "Bearer proxy"),
            )
            self.assertEqual(provider.providers(), {"ollama-local": True})
            self.assertEqual(provider.models(), {"ollama-local": ["llama3"]})
        self.assertEqual(
            [(r.path, r.headers.get(HEADER)) for r in server.requests],
            [
                ("/api/version", "Bearer proxy"),
                ("/api/version", "Bearer proxy"),
                ("/api/tags", "Bearer proxy"),
            ],
        )

    def test_bad_header_values_make_probes_report_unusable(self):
        for target in TARGETS:
            for value in ("", "a\nb", "a\x00b", b"t", None):
                provider = target.provider({HEADER: lambda value=value: value})
                adapter = adapter_of(provider, target.name)
                for label, probe in target.probes(adapter).items():
                    with (
                        self.subTest(target.name, probe=label, value=value),
                        patch("urllib.request.urlopen", side_effect=no_request),
                    ):
                        self.assertIn(probe(), (set(), False))

    def test_header_function_errors_propagate_out_of_probes(self):
        for target in TARGETS:
            for error in (OSError("x"), KeyError("x"), ValueError("x")):
                provider = target.provider({HEADER: Mock(side_effect=error)})
                adapter = adapter_of(provider, target.name)
                for label, probe in target.probes(adapter).items():
                    with (
                        self.subTest(target.name, probe=label, error=type(error)),
                        patch("urllib.request.urlopen", side_effect=no_request),
                    ):
                        self.assertIs(raised(self, type(error), probe), error)

    def test_ollama_probe_timeout(self):
        for config in (None, secret_headers("ollama-local", "t")):
            provider = Provider(
                adapters={"ollama-local": OllamaLocalAdapter()}, config=config
            )
            adapter = adapter_of(provider, "ollama-local")
            with (
                self.subTest(configured=config is not None),
                patch("urllib.request.urlopen", side_effect=ollama_urlopen) as m,
            ):
                self.assertTrue(adapter.is_available())
                self.assertEqual(adapter.models(), {OLLAMA_MODEL})
            self.assertEqual([c.kwargs["timeout"] for c in m.call_args_list], [3, 3])


class ProviderHeaderTransportTests(unittest.TestCase):
    """Configured headers only go over https, or http to an unproxied loopback IP."""

    def test_refused_urls(self):
        refused = {
            "http non-loopback": ("http://gateway.example:8080", {}),
            "localhost": ("http://localhost:8080", {}),
            "userinfo": ("https://user:pass@gateway.example", {}),
            "proxied loopback": (
                "http://127.0.0.1:8080",
                {"http_proxy": "http://proxy.invalid:3128"},
            ),
        }
        for target in TARGETS:
            for label, (url, env) in refused.items():
                fn = Mock(return_value="t")
                with (
                    self.subTest(target.name, url=label),
                    clean_env(OLLAMA_HOST="http://127.0.0.1:9", **env),
                    patch("urllib.request.urlopen", side_effect=no_request),
                ):
                    provider = target.provider({HEADER: fn})
                    adapter = adapter_of(provider, target.name)
                    with pointed_at(provider, target.name, url):
                        with self.assertRaises(ValueError) as ctx:
                            adapter.chat(target.model, MESSAGES)
                        assert_no_secret_in_messages(
                            self, ctx.exception, "pass", HEADER
                        )
                        stream = adapter.stream_chat(target.model, MESSAGES)
                        with self.assertRaises(ValueError):
                            next(stream)
                        with self.assertLogs("ducktape_provider", "WARNING") as logs:
                            for _ in range(2):
                                for probe in target.probes(adapter).values():
                                    self.assertIn(probe(), (set(), False))
                    [record] = logs.records
                    self.assertIn(target.name, record.getMessage())
                    self.assertNotIn("pass", record.getMessage())
                    self.assertNotIn("gateway", record.getMessage())
                    fn.assert_not_called()

    def test_allowed_urls(self):
        allowed = {
            "https": ("https://gateway.example", {}),
            "bypassed loopback": (
                "http://127.0.0.1:8080",
                {"http_proxy": "http://proxy.invalid:3128", "no_proxy": "127.0.0.1"},
            ),
            "unproxied IPv6 loopback": ("http://[::1]:8080", {}),
        }
        for target in TARGETS:
            for label, (url, env) in allowed.items():
                with (
                    self.subTest(target.name, url=label),
                    clean_env(OLLAMA_HOST="http://127.0.0.1:9", **env),
                    patch("urllib.request.urlopen", side_effect=target.urlopen) as m,
                ):
                    provider = target.provider({HEADER: "t"})
                    adapter = adapter_of(provider, target.name)
                    calls = {**target.requests(adapter), **target.probes(adapter)}
                    with pointed_at(provider, target.name, url):
                        for call in calls.values():
                            call()
                    self.assertEqual(
                        [sent_header(c.args[0]) for c in m.call_args_list],
                        ["t"] * len(calls),
                    )


class ProviderHeaderFrameTests(unittest.TestCase):
    """No frame local on a failure's traceback, or any exception chained to it,
    holds a configured header's value, for every call that sends one."""

    def setUp(self):
        self.enterContext(clean_env(OLLAMA_HOST="http://127.0.0.1:9", no_proxy="*"))

    def calls(
        self, target: Target, provider: Provider
    ) -> dict[str, Callable[[], object]]:
        adapter = adapter_of(provider, target.name)
        name, model = target.name, target.model
        return {
            **target.requests(adapter),
            "Provider.chat": functools.partial(
                provider.chat, model, MESSAGES, provider=name
            ),
            "async_chat": lambda: asyncio.run(
                provider.async_chat(model, MESSAGES, provider=name)
            ),
            "async_stream_chat": lambda: asyncio.run(
                consume_stream(provider, model, name)
            ),
        }

    def check(
        self,
        label: str,
        target: Target,
        make: Callable[[], Provider],
        error: type[BaseException],
        probe_error: type[BaseException] | None,
    ) -> None:
        """Runs every call on a fresh `make()` provider: requests raise `error`, probes
        raise `probe_error` or, when None, report the provider unusable."""
        provider = make()
        adapter = adapter_of(provider, target.name)
        for call_label, call in self.calls(target, provider).items():
            with self.subTest(target.name, failure=label, call=call_label):
                exc = raised(self, error, call)
                assert_no_secret_in_messages(self, exc, HEADER, "X-A", "X-B")
                assert_no_sentinel_in_frames(self, exc)
        for probe_label, probe in target.probes(adapter).items():
            with self.subTest(target.name, failure=label, call=probe_label):
                if probe_error is None:
                    self.assertIn(probe(), (set(), False))
                else:
                    exc = raised(self, probe_error, probe)
                    assert_no_sentinel_in_frames(self, exc)

    def test_request_failures(self):
        for target in TARGETS:
            first = {"X-A": lambda: SENTINEL}
            with patch("urllib.request.urlopen", side_effect=no_request):
                if target.auth_header is not None:
                    self.check(
                        "missing key",
                        target,
                        lambda target=target, first=first: target.provider(
                            first, key=None
                        ),
                        AuthError,
                        None,
                    )
                self.check(
                    "second function raises",
                    target,
                    lambda target=target, first=first: target.provider(
                        {**first, "X-B": vault_down}, key=SENTINEL
                    ),
                    OSError,
                    OSError,
                )
                for value_label, value in (
                    ("CR", f"{SENTINEL}\r"),
                    ("LF", f"{SENTINEL}\n"),
                    ("NUL", f"{SENTINEL}\x00"),
                    ("bytes", SENTINEL.encode()),
                    ("empty", ""),
                ):
                    self.check(
                        f"second value ({value_label})",
                        target,
                        lambda target=target, first=first, value=value: target.provider(
                            {**first, "X-B": lambda: value}, key=SENTINEL
                        ),
                        TypeError if isinstance(value, bytes) else ValueError,
                        None,
                    )

                def refused(target: Target = target) -> Provider:
                    provider = target.provider({"X-A": SENTINEL}, key=SENTINEL)
                    stack = contextlib.ExitStack()
                    self.addCleanup(stack.close)
                    stack.enter_context(
                        pointed_at(provider, target.name, "http://gateway.example")
                    )
                    return provider

                with self.assertLogs("ducktape_provider", "WARNING"):
                    self.check("transport refused", target, refused, ValueError, None)

    def test_transport_failures(self):
        for target in TARGETS:
            provider = target.provider(
                {"X-A": SENTINEL, "X-B": lambda: SENTINEL}, SENTINEL
            )
            with pointed_at(provider, target.name, closed_loopback_url()):
                for call_label, call in self.calls(target, provider).items():
                    with self.subTest(target.name, failure="refused", call=call_label):
                        exc = raised(self, APIError, call)
                        assert_no_sentinel_in_frames(self, exc)
                for probe_label, probe in target.probes(
                    adapter_of(provider, target.name)
                ).items():
                    with self.subTest(target.name, failure="refused", call=probe_label):
                        self.assertIn(probe(), (set(), False))
            for call_label, reply in (
                ("chat", lambda *a, **k: buffered_response(b"not json")),
                (
                    "stream_chat",
                    lambda *a, target=target, **k: FakeStreamResponse(
                        target.bad_stream
                    ),
                ),
            ):
                with (
                    self.subTest(target.name, failure="malformed", call=call_label),
                    patch("urllib.request.urlopen", side_effect=reply),
                ):
                    call = target.requests(adapter_of(provider, target.name))[
                        call_label
                    ]
                    exc = raised(self, MalformedResponseError, call)
                    assert_no_sentinel_in_frames(self, exc)

    def test_logged_probe_failures(self):
        for target in TARGETS:
            provider = target.provider({"X-A": SENTINEL, "X-B": vault_down}, SENTINEL)
            probes: dict[str, Callable[[], object]] = {
                "providers": provider.providers,
                "models": provider.models,
                "async_providers": lambda provider=provider: asyncio.run(
                    provider.async_providers()
                ),
                "async_models": lambda provider=provider: asyncio.run(
                    provider.async_models()
                ),
            }
            for label, probe in probes.items():
                expect_warning = target.auth_header is None or "models" in label
                log_cm = (
                    self.assertLogs("ducktape_provider", "WARNING")
                    if expect_warning
                    else contextlib.nullcontext()
                )
                with (
                    self.subTest(target.name, call=label),
                    patch("urllib.request.urlopen", side_effect=no_request),
                    log_cm as logs,
                ):
                    probe()
                if not expect_warning:
                    continue
                assert logs is not None
                failures = [r for r in logs.records if r.exc_info]
                self.assertTrue(failures)
                for record in failures:
                    assert record.exc_info is not None
                    exc = record.exc_info[1]
                    assert exc is not None
                    self.assertIsInstance(exc, OSError)
                    assert_no_sentinel_in_frames(self, exc)


class UserTracebackTests(unittest.TestCase):
    """Failing inside a caller's own `except` block leaves the caller's traceback."""

    def setUp(self):
        self.enterContext(clean_env(OLLAMA_HOST="http://127.0.0.1:9", no_proxy="*"))

    def test_callers_exception_keeps_its_traceback(self):
        ollama = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()},
            config=secret_headers("ollama-local", "t"),
        )
        claude = Provider(
            adapters={"claude": ClaudeAdapter()},
            config=secret_headers("claude", "t"),
        )
        failures: dict[str, Callable[[], object]] = {
            "Provider config": lambda: Provider(
                config={"providers": {"ollama-local": {"headers": {"X": b"t"}}}}
            ),
            "Provider api_keys": lambda: Provider(
                api_keys=ExplodingItems({"claude": "k"})
            ),
            "missing key": lambda: claude.chat("claude-x", MESSAGES),
            "missing key in stream": lambda: list(
                claude.stream_chat("claude-x", MESSAGES)
            ),
            "connection error": lambda: ollama.chat(OLLAMA_MODEL, MESSAGES),
            "interrupted probe": lambda: ollama.providers(),
        }
        for label, call in failures.items():
            with (
                self.subTest(label),
                patch.dict(os.environ, {"OLLAMA_HOST": closed_loopback_url()}),
                patch(
                    "http.client.HTTPConnection.request",
                    side_effect=SystemExit(1),
                )
                if label == "interrupted probe"
                else contextlib.nullcontext(),
            ):
                try:
                    raise LookupError("the caller's own")
                except LookupError as own:
                    try:
                        call()
                    except BaseException as exc:  # noqa: BLE001
                        self.assertIs(exc.__context__, own)
                    else:
                        self.fail("no error")
                    self.assertIsNotNone(own.__traceback__)

    def test_clear_tracebacks_stops_at_the_given_exception(self):
        try:
            raise LookupError("the caller's own")
        except LookupError as own:
            try:
                raise OSError("ours")
            except OSError as ours:
                _clear_tracebacks(ours, own)
                self.assertIsNone(ours.__traceback__)
                self.assertIs(ours.__context__, own)
            self.assertIsNotNone(own.__traceback__)


class TestProviderEmbedConfigResolution(unittest.TestCase):
    def test_provider_level_temperature_not_in_embed_but_in_chat(self):
        """Provider-level chat defaults do not reach embed but do reach chat."""
        for provider_config in (
            {"temperature": 0.2},
            {"providers": {"rec": {"temperature": 0.2}}},
        ):
            with self.subTest(provider_config=provider_config):
                adapter = RecordingAdapter()
                provider = Provider(
                    adapters={"rec": adapter, "other": RecordingAdapter()},
                    config=provider_config,
                )
                provider.chat("model", MESSAGES, provider="rec")
                provider.embed("rec-embed", "hi", provider="rec")
                chat_config = adapter.configs[0]
                embed_config = adapter.embed_configs[0]
                assert chat_config is not None
                assert embed_config is not None
                self.assertIn("temperature", chat_config)
                self.assertNotIn("temperature", embed_config)

    def test_provider_level_timeout_reaches_embed(self):
        """Provider-level timeout reaches embed."""
        self.assertEqual(resolve_embed(provider_config={"timeout": 5}), {"timeout": 5})
        self.assertEqual(
            resolve_embed(provider_config={"providers": {"rec": {"timeout": 7}}}),
            {"timeout": 7},
        )
        self.assertEqual(resolve_embed(timeout=10), {"timeout": 10})

    def test_provider_level_timeout_none_disables(self):
        """timeout=None disables timeout for embed."""
        self.assertEqual(resolve_embed(timeout=None), {"timeout": None})
        self.assertEqual(
            resolve_embed(provider_config={"timeout": None}), {"timeout": None}
        )

    def test_call_level_body_keys_reach_embed(self):
        """Call-level body keys reach embed."""
        self.assertEqual(resolve_embed({"temperature": 0.7}), {"temperature": 0.7})
        self.assertEqual(
            resolve_embed({"providers": {"rec": {"top_p": 0.9}}}), {"top_p": 0.9}
        )

    def test_call_level_per_provider_wins_over_top_level(self):
        """Call-level per-provider wins over top-level for embed."""
        config: dict[str, Any] = {
            "temperature": 0.1,
            "providers": {"rec": {"temperature": 0.9}},
        }
        resolved = resolve_embed(config)
        assert resolved is not None
        self.assertEqual(resolved["temperature"], 0.9)

    def test_call_level_headers_merge_key_by_key(self):
        """Call-level headers merge key by key for embed."""
        config = {
            "headers": {"X-Global": "1", "X-Shared": "call"},
            "providers": {"rec": {"headers": {"X-Prov": "2", "X-Shared": "prov"}}},
        }
        resolved = resolve_embed(config)
        assert resolved is not None
        self.assertEqual(
            resolved["headers"], {"X-Global": "1", "X-Prov": "2", "X-Shared": "prov"}
        )

    def test_provider_level_body_filtered_but_timeout_kept_with_call_keys(self):
        """Provider body filtered but timeout kept together with call keys for embed."""
        provider_config: dict[str, Any] = {"temperature": 0.2, "timeout": 9}
        call_config: dict[str, Any] = {"temperature": 0.7}
        resolved = resolve_embed(call_config, provider_config=provider_config)
        assert resolved is not None
        self.assertEqual(resolved, {"timeout": 9, "temperature": 0.7})

    def test_merged_headers_reach_embed_request(self):
        """Call-level merged headers reach the embed request."""
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
            provider.embed(OLLAMA_MODEL, "hi", config=config, provider="ollama-local")
        [req] = sent
        headers = {k.lower(): v for k, v in req.header_items()}
        self.assertEqual(headers["x-global"], "1")
        self.assertEqual(headers["x-prov"], "2")

    def test_provider_configured_headers_reach_embed_request(self):
        """Provider-configured headers reach embed request."""
        sent: list[urllib.request.Request] = []

        def capture(req: urllib.request.Request, timeout: float | None = None):
            sent.append(req)
            raise StopBeforeSending

        provider = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()},
            config=secret_headers("ollama-local", "t"),
        )
        with (
            patch.dict(os.environ, {"OLLAMA_HOST": "http://127.0.0.1:9"}),
            patch("urllib.request.urlopen", capture),
            self.assertRaises(StopBeforeSending),
        ):
            provider.embed(OLLAMA_MODEL, "hi", provider="ollama-local")
        [req] = sent
        self.assertEqual(sent_header(req), "t")


class TestProviderEmbedSecretRedaction(unittest.TestCase):
    def setUp(self):
        self.enterContext(clean_env(OLLAMA_HOST="http://127.0.0.1:9", no_proxy="*"))

    def test_secret_not_in_repr_or_exception_for_embed(self):
        """Secret header not in repr or exception for embed."""
        provider = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()},
            config=secret_headers("ollama-local", SENTINEL),
        )
        adapter = adapter_of(provider, "ollama-local")
        for obj in (provider, adapter):
            self.assertNotIn(SENTINEL, repr(vars(obj)))
            self.assertFalse(_sentinel_in_value(vars(obj), set(), 0))
        bad_provider = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()},
            config=secret_headers("ollama-local", lambda: f"{SENTINEL}\n"),
        )
        bad_adapter = adapter_of(bad_provider, "ollama-local")
        with patch("urllib.request.urlopen", side_effect=no_request):
            for call in (
                lambda: bad_provider.embed(OLLAMA_MODEL, "hi", provider="ollama-local"),
                lambda: bad_adapter.embed(OLLAMA_MODEL, ["hi"]),
            ):
                exc = raised(self, ValueError, call)
                assert_no_secret_in_messages(self, exc)
                assert_no_sentinel_in_frames(self, exc)

    def test_configured_state_redacted_for_embed_provider(self):
        """Configured state redacted for embed provider."""
        for target in TARGETS:
            provider = target.provider({HEADER: SENTINEL}, key=SENTINEL)
            adapter = adapter_of(provider, target.name)
            for state in (vars(adapter), vars(provider)):
                self.assertNotIn(SENTINEL, repr(state))
                self.assertFalse(_sentinel_in_value(state, set(), 0))


class TestProviderEmbedHeaderWireTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(clean_env(no_proxy="*"))

    def test_missing_key_raises_auth_error_before_request_for_embed(self):
        """Missing key raises AuthError before request for embed."""
        for case in (c for c in KEYED if c.name == "openai"):
            provider = Provider(
                adapters={case.name: case.cls()},
                api_keys={case.name: lambda: None},  # type: ignore[dict-item]
            )
            adapter = provider._adapters[case.name]
            with (
                self.subTest(case.name),
                patch("urllib.request.urlopen", side_effect=no_request) as m,
            ):
                with self.assertRaises(AuthError):
                    provider.embed(case.model, "hi", provider=case.name)
                with self.assertRaises(AuthError):
                    adapter.embed(case.model, ["hi"])
                m.assert_not_called()

    def test_configured_key_and_header_sent_on_embed_request(self):
        """Configured key and header sent on embed request."""
        for case in (c for c in KEYED if c.name == "openai"):
            with self.subTest(case.name):
                provider = Provider(
                    adapters={case.name: case.cls()},
                    api_keys={case.name: "k"},
                    config=secret_headers(case.name, "t"),
                )
                adapter = adapter_of(provider, case.name)
                with patch(
                    "urllib.request.urlopen", side_effect=fake_urlopen(case)
                ) as m:
                    provider.embed(case.model, "hi", provider=case.name)
                    adapter.embed(case.model, ["hi"])
                headers = [sent_header(c.args[0]) for c in m.call_args_list]
                self.assertTrue(all(h == "t" for h in headers))
                keys = [case.sent_key(c.args[0]) for c in m.call_args_list]
                self.assertTrue(all(k == "k" for k in keys))

    def test_ollama_embed_sends_configured_header(self):
        """Ollama embed sends configured header."""

        def embed_aware(req: urllib.request.Request, timeout: float | None = None):
            if req.full_url.endswith("/api/embed"):
                payload = ollama_embed_response(OLLAMA_MODEL, [[0.1]])
                return buffered_response(json.dumps(payload).encode())
            return ollama_urlopen(req, timeout=timeout)

        provider = Provider(
            adapters={"ollama-local": OllamaLocalAdapter()},
            config=secret_headers("ollama-local", "t"),
        )
        adapter = adapter_of(provider, "ollama-local")
        with patch("urllib.request.urlopen", side_effect=embed_aware) as m:
            provider.embed(OLLAMA_MODEL, "hi", provider="ollama-local")
            adapter.embed(OLLAMA_MODEL, ["hi"])
        self.assertEqual([sent_header(c.args[0]) for c in m.call_args_list], ["t"] * 2)


if __name__ == "__main__":
    unittest.main()
