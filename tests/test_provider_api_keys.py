"""Tests for API keys passed without env vars: `Provider(api_keys=...)`, the adapters'
`api_key=`, and keeping a key or configured header out of reprs, pickles and
traceback frame locals.

urlopen is mocked, except for connection errors against a closed port on 127.0.0.1;
nothing reaches a vendor or a local Ollama daemon.
"""

import asyncio
import contextlib
import copy
import functools
import http.client
import json
import os
import pickle
import socket
import types
import unittest
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NoReturn, cast
from unittest.mock import Mock, patch

from http_test_utils import (
    FakeStreamResponse,
    buffered_response,
    ndjson_lines,
    sse_lines,
)
from test_provider_discovery import ENTRY_POINTS, FakeEntryPoint

from ducktape_provider import (
    Adapter,
    APIError,
    AuthError,
    ClaudeAdapter,
    MalformedResponseError,
    Message,
    ModelInfo,
    OllamaLocalAdapter,
    OpenAIAdapter,
    Provider,
)
from ducktape_provider.adapter import _Secret
from ducktape_provider.provider import _clear_discovery_cache, _should_evict_cache
from ducktape_provider.streaming import _clear_tracebacks

MESSAGES: list[Message] = [
    {"role": "user", "content": [{"type": "text", "text": "hi"}]}
]


def no_request(*_args: object, **_kwargs: object) -> NoReturn:
    """A urlopen stand-in for requests that must never be sent.

    Raising rather than answering: a mock body would make the read loop forever.
    A fresh error per call, so repeated raises don't pile onto one traceback.
    """
    raise AssertionError("request sent")


SENTINEL = "sk-SENTINEL-4f1c9a"

_CLEARED_ENV = {
    "anthropic_api_key",
    "openai_api_key",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
}


@contextmanager
def clean_env(**values: str) -> Iterator[None]:
    """os.environ without API keys or proxies, plus `values`; restored afterwards."""
    with patch.dict(os.environ):
        for name in list(os.environ):
            if name.lower() in _CLEARED_ENV:
                del os.environ[name]
        os.environ.update(values)
        yield


@dataclass(frozen=True)
class Keyed:
    cls: type[ClaudeAdapter] | type[OpenAIAdapter]
    name: str
    env_var: str
    chat_url_attr: str
    header: str
    prefix: str
    model: str
    chat_body: dict[str, Any]
    stream_lines: list[bytes]

    def sent_key(self, req: Any) -> str | None:
        value = req.get_header(self.header.capitalize())
        return None if value is None else value.removeprefix(self.prefix)


CLAUDE_BODY = {"content": [], "stop_reason": "end_turn", "usage": {}}
OPENAI_BODY = {"status": "completed", "output": [], "usage": {}}

KEYED = [
    Keyed(
        ClaudeAdapter,
        "claude",
        "ANTHROPIC_API_KEY",
        "_MESSAGES_URL",
        "x-api-key",
        "",
        "claude-x",
        CLAUDE_BODY,
        sse_lines({"type": "message_start", "message": {}}, {"type": "message_stop"}),
    ),
    Keyed(
        OpenAIAdapter,
        "openai",
        "OPENAI_API_KEY",
        "_RESPONSES_URL",
        "Authorization",
        "Bearer ",
        "gpt-x",
        OPENAI_BODY,
        sse_lines({"type": "response.completed", "response": OPENAI_BODY}),
    ),
]


def fake_urlopen(case: Keyed) -> Callable[..., Any]:
    """A urlopen stand-in answering models, chat and stream requests for `case`."""

    def urlopen(req: Any, timeout: float | None = None) -> Any:
        if "/embeddings" in req.full_url:
            return buffered_response(
                json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "object": "embedding",
                                "embedding": [0.1, 0.2, 0.3],
                                "index": 0,
                            }
                        ],
                        "model": case.model,
                        "usage": {"prompt_tokens": 5, "total_tokens": 5},
                    }
                ).encode()
            )
        if "/models" in req.full_url:
            return buffered_response(
                json.dumps({"data": [{"id": case.model}]}).encode()
            )
        data = req.data and json.loads(cast(bytes, req.data))
        if isinstance(data, dict) and data.get("stream"):
            return FakeStreamResponse(case.stream_lines)
        return buffered_response(json.dumps(case.chat_body).encode())

    return urlopen


OLLAMA_MODEL = "llama3"
OLLAMA_BODY = {"message": {"content": "hi"}, "done": True, "done_reason": "stop"}


def ollama_urlopen(req: Any, timeout: float | None = None) -> Any:
    """A urlopen stand-in answering Ollama's version, tags, chat and stream requests."""
    if req.full_url.endswith("/api/version"):
        return buffered_response(b'{"version": "0.0.0"}')
    if req.full_url.endswith("/api/tags"):
        return buffered_response(
            json.dumps({"models": [{"name": OLLAMA_MODEL}]}).encode()
        )
    if json.loads(req.data)["stream"]:
        return FakeStreamResponse(ndjson_lines(OLLAMA_BODY))
    return buffered_response(json.dumps(OLLAMA_BODY).encode())


def secret_headers(name: str, value: object = SENTINEL) -> dict[str, Any]:
    """A Provider config giving provider `name` one configured header."""
    return {"providers": {name: {"headers": {"X-Proxy-Token": value}}}}


def secured_providers() -> list[tuple[str, Provider, str, str]]:
    """(label, provider, provider name, model) for each way a provider can hold
    SENTINEL: a keyed adapter's own `api_key=`, or `api_keys` plus a configured
    header function, and a configured str header for Ollama."""
    setups: list[tuple[str, Provider, str, str]] = []
    for case in KEYED:
        setups.append(
            (
                f"{case.name} api_key=",
                Provider(adapters={case.name: case.cls(api_key=SENTINEL)}),
                case.name,
                case.model,
            )
        )
        setups.append(
            (
                f"{case.name} api_keys and headers",
                Provider(
                    adapters={case.name: case.cls()},
                    api_keys={case.name: SENTINEL},
                    config=secret_headers(case.name, lambda: SENTINEL),
                ),
                case.name,
                case.model,
            )
        )
    setups.append(
        (
            "ollama-local headers",
            Provider(
                adapters={"ollama-local": OllamaLocalAdapter()},
                config=secret_headers("ollama-local"),
            ),
            "ollama-local",
            OLLAMA_MODEL,
        )
    )
    return setups


@contextmanager
def pointed_at(provider: Provider, name: str, url: str) -> Iterator[None]:
    """Sends every request from `provider`'s `name` adapter to `url` instead."""
    adapter = provider._adapters[name]
    if isinstance(adapter, OllamaLocalAdapter):
        with patch.dict(os.environ, {"OLLAMA_HOST": url}):
            yield
        return
    attrs = [
        attr
        for attr in (
            "_MESSAGES_URL",
            "_RESPONSES_URL",
            "_MODELS_URL",
            "_EMBEDDINGS_URL",
        )
        if hasattr(adapter, attr)
    ]
    with contextlib.ExitStack() as stack:
        for attr in attrs:
            stack.enter_context(patch.object(adapter, attr, url))
        yield


def drain(adapter: Adapter, model: str) -> list[Any]:
    return list(adapter.stream_chat(model, MESSAGES))


def request_calls(case: Keyed, adapter: Any) -> dict[str, Callable[[], object]]:
    calls: dict[str, Callable[[], object]] = {
        "chat": lambda: adapter.chat(case.model, MESSAGES),
        "stream_chat": lambda: list(adapter.stream_chat(case.model, MESSAGES)),
    }
    if isinstance(adapter, OpenAIAdapter):
        calls["embed"] = lambda: adapter.embed(case.model, ["hi"])
    return calls


class SecretTests(unittest.TestCase):
    def test_never_shows_the_value(self):
        for source in (SENTINEL, lambda: SENTINEL):
            secret = _Secret(source)
            for text in (repr(secret), str(secret), f"{secret}", f"{secret:>40}"):
                self.assertEqual(text, "<redacted>")
            self.assertEqual("%s %r" % (secret, secret), "<redacted> <redacted>")  # noqa: UP031

    def test_pickling_raises_without_the_value(self):
        class Vault:
            def read(self) -> str:
                return SENTINEL

        for source in (SENTINEL, Vault().read):
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                with (
                    self.subTest(type(source).__name__, protocol=protocol),
                    self.assertRaises(TypeError) as ctx,
                ):
                    pickle.dumps(_Secret(source), protocol)
                self.assertNotIn(SENTINEL, str(ctx.exception))
            with self.assertRaises(TypeError) as ctx:
                _Secret(source).__getstate__()
            self.assertNotIn(SENTINEL, str(ctx.exception))

    def test_copies_are_the_same_object(self):
        secret = _Secret(SENTINEL)
        self.assertIs(copy.copy(secret), secret)
        self.assertIs(copy.deepcopy(secret), secret)

    def test_adapter_vars_are_redacted(self):
        for case in KEYED:
            for source in (SENTINEL, lambda: SENTINEL):
                with self.subTest(case.name):
                    self.assertNotIn(SENTINEL, repr(vars(case.cls(api_key=source))))


class KeyedAdapterTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(clean_env())

    def test_explicit_key_is_sent_and_env_is_ignored(self):
        for case in KEYED:
            os.environ[case.env_var] = "from-env"
            for source in ("explicit", lambda: "explicit"):
                adapter = case.cls(api_key=source)
                for label, call in {
                    **request_calls(case, adapter),
                    "models": adapter.models,
                }.items():
                    with (
                        self.subTest(case.name, call=label, source=type(source)),
                        patch(
                            "urllib.request.urlopen", side_effect=fake_urlopen(case)
                        ) as mock_urlopen,
                    ):
                        call()
                        req = mock_urlopen.call_args.args[0]
                        self.assertEqual(case.sent_key(req), "explicit")

    def test_env_var_is_read_per_request_without_a_source(self):
        for case in KEYED:
            adapter = case.cls()
            with patch("urllib.request.urlopen", side_effect=fake_urlopen(case)) as m:
                for key in ("first", "second"):
                    with self.subTest(case.name, key=key):
                        os.environ[case.env_var] = key
                        adapter.chat(case.model, MESSAGES)
                        self.assertEqual(case.sent_key(m.call_args.args[0]), key)

    def test_function_source_is_called_once_per_request(self):
        for case in KEYED:
            with self.subTest(case.name):
                source = Mock(side_effect=["k1", "k2", "k3"])
                adapter = case.cls(api_key=source)
                with patch(
                    "urllib.request.urlopen", side_effect=fake_urlopen(case)
                ) as mock_urlopen:
                    adapter.chat(case.model, MESSAGES)
                    list(adapter.stream_chat(case.model, MESSAGES))
                    adapter.models()
                sent = [case.sent_key(c.args[0]) for c in mock_urlopen.call_args_list]
                self.assertEqual(sent, ["k1", "k2", "k3"])

    def test_missing_key_raises_auth_error_before_any_request(self):
        for case in KEYED:
            os.environ[case.env_var] = "from-env"
            for source in (lambda: None, lambda: ""):
                adapter = case.cls(api_key=source)
                with patch(
                    "urllib.request.urlopen", side_effect=no_request
                ) as mock_urlopen:
                    for label, call in request_calls(case, adapter).items():
                        with (
                            self.subTest(case.name, call=label),
                            self.assertRaises(AuthError) as ctx,
                        ):
                            call()
                        self.assertIsNone(ctx.exception.status)
                    self.assertEqual(adapter.models(), set())
                    self.assertIsNone(adapter._models_cache)
                    mock_urlopen.assert_not_called()
            del os.environ[case.env_var]
            with (
                self.subTest(case.name, source="unset env var"),
                patch("urllib.request.urlopen", side_effect=no_request) as mock_urlopen,
            ):
                with self.assertRaises(AuthError):
                    case.cls().chat(case.model, MESSAGES)
                self.assertEqual(case.cls().models(), set())
                mock_urlopen.assert_not_called()

    def test_source_exceptions_propagate_unwrapped(self):
        for case in KEYED:
            error = ConnectionError("vault down")
            adapter = case.cls(api_key=Mock(side_effect=error))
            calls = {**request_calls(case, adapter), "models": adapter.models}
            for label, call in calls.items():
                with (
                    self.subTest(case.name, call=label),
                    patch(
                        "urllib.request.urlopen", side_effect=no_request
                    ) as mock_urlopen,
                    self.assertRaises(ConnectionError) as ctx,
                ):
                    call()
                self.assertIs(ctx.exception, error)
                mock_urlopen.assert_not_called()

    def test_is_available_with_a_source_neither_calls_it_nor_reads_env(self):
        for case in KEYED:
            with self.subTest(case.name):
                source = Mock(return_value="k")
                adapter = case.cls(api_key=source)
                with patch.object(
                    os.environ, "get", side_effect=AssertionError("env read")
                ):
                    self.assertTrue(adapter.is_available())
                source.assert_not_called()
                self.assertFalse(case.cls().is_available())

    def test_unusable_keys_raise_without_the_key(self):
        bad: dict[str, object] = {
            "bytes": SENTINEL.encode(),
            "int": 4242424242,
            "CRLF": f"{SENTINEL}\r\n",
            "LF": f"{SENTINEL}\n",
            "folded header": f"{SENTINEL}\r\n x",
            "NUL": f"{SENTINEL}\x00",
            "DEL": f"{SENTINEL}\x7f",
            "space": f"{SENTINEL} x",
            "non-ASCII": f"{SENTINEL}é",
        }
        for case in KEYED:
            for label, value in bad.items():
                with (
                    self.subTest(case.name, value=label, at="construction"),
                    self.assertRaises((TypeError, ValueError)) as ctx,
                ):
                    case.cls(api_key=value)  # ty: ignore[invalid-argument-type]
                self.assertNotIn(SENTINEL, str(ctx.exception))
                self.assertNotIn("4242424242", str(ctx.exception))

                adapter = case.cls(api_key=lambda value=value: value)  # ty: ignore[invalid-argument-type]
                with patch(
                    "urllib.request.urlopen", side_effect=no_request
                ) as mock_urlopen:
                    for call_name, call in request_calls(case, adapter).items():
                        with (
                            self.subTest(case.name, value=label, at=call_name),
                            self.assertRaises((TypeError, ValueError)) as ctx,
                        ):
                            call()
                        self.assertNotIsInstance(ctx.exception, APIError)
                        self.assertNotIn(SENTINEL, str(ctx.exception))
                        self.assertIsNone(ctx.exception.__context__)
                    self.assertEqual(adapter.models(), set())
                    mock_urlopen.assert_not_called()
            with self.subTest(case.name, value="empty"), self.assertRaises(ValueError):
                case.cls(api_key="")

    def test_key_is_only_sent_over_https_or_to_unproxied_loopback(self):
        rejected = [
            "http://api.example.com/v1",
            "http://evil.com@127.0.0.1/v1",
            "https:api.anthropic.com/v1",
            "http://localhost:8080/v1",
            "ftp://127.0.0.1/v1",
        ]
        for case in KEYED:
            for url in rejected:
                source = Mock(return_value="k")
                adapter = case.cls(api_key=source)
                with (
                    self.subTest(case.name, url=url),
                    patch.object(adapter, case.chat_url_attr, url),
                    patch.object(adapter, "_MODELS_URL", url),
                    patch(
                        "urllib.request.urlopen", side_effect=no_request
                    ) as mock_urlopen,
                ):
                    with self.assertRaises(ValueError):
                        adapter.chat(case.model, MESSAGES)
                    stream = adapter.stream_chat(case.model, MESSAGES)
                    with self.assertRaises(ValueError):
                        next(stream)
                    self.assertEqual(adapter.models(), set())
                    source.assert_not_called()
                    mock_urlopen.assert_not_called()

    def test_plain_http_to_loopback_ip_is_allowed_unless_proxied(self):
        for case in KEYED:
            for url in ("http://127.0.0.1:1/v1", "http://[::1]:1/v1"):
                adapter = case.cls(api_key="k")
                with (
                    self.subTest(case.name, url=url),
                    patch.object(adapter, case.chat_url_attr, url),
                    patch("urllib.request.urlopen", side_effect=fake_urlopen(case)),
                ):
                    adapter.chat(case.model, MESSAGES)
            loopback = "http://127.0.0.1:1/v1"
            adapter = case.cls(api_key="k")
            self.enterContext(patch.object(adapter, case.chat_url_attr, loopback))
            self.enterContext(
                patch.object(adapter, "_MODELS_URL", f"{loopback}/models")
            )
            with (
                self.subTest(case.name, proxy="http_proxy without bypass"),
                clean_env(http_proxy="http://proxy.invalid:3128"),
                patch("urllib.request.urlopen", side_effect=no_request) as mock_urlopen,
            ):
                with self.assertRaises(ValueError):
                    adapter.chat(case.model, MESSAGES)
                stream = adapter.stream_chat(case.model, MESSAGES)
                with self.assertRaises(ValueError):
                    next(stream)
                self.assertEqual(adapter.models(), set())
                mock_urlopen.assert_not_called()
            with (
                self.subTest(case.name, proxy="bypassed"),
                clean_env(http_proxy="http://proxy.invalid:3128", no_proxy="127.0.0.1"),
                patch("urllib.request.urlopen", side_effect=fake_urlopen(case)),
            ):
                adapter.chat(case.model, MESSAGES)
                list(adapter.stream_chat(case.model, MESSAGES))
                self.assertEqual(adapter.models(), {case.model})

    def test_key_goes_to_the_url_it_was_checked_against(self):
        https, http = "https://vendor.example/v1", "http://evil.example/v1"

        def shifting(*urls: str) -> property:
            """A URL property returning `urls` in turn, then the last one."""
            reads = iter(urls)
            return property(lambda self: next(reads, urls[-1]))

        for case in KEYED:
            for attr, call in (
                (
                    case.chat_url_attr,
                    lambda a, model=case.model: a.chat(model, MESSAGES),
                ),
                ("_MODELS_URL", lambda a: a.models()),
            ):
                with self.subTest(case.name, attr=attr, order="checked https first"):
                    cls = type("Shifting", (case.cls,), {attr: shifting(https, http)})
                    with patch(
                        "urllib.request.urlopen", side_effect=fake_urlopen(case)
                    ) as mock_urlopen:
                        call(cls(api_key="k"))
                    [req] = [c.args[0] for c in mock_urlopen.call_args_list]
                    self.assertTrue(req.full_url.startswith(https), req.full_url)
                with self.subTest(case.name, attr=attr, order="http first"):
                    cls = type("Shifting", (case.cls,), {attr: shifting(http, https)})
                    with patch("urllib.request.urlopen", side_effect=no_request):
                        try:
                            result = call(cls(api_key="k"))
                        except ValueError:
                            pass
                        else:
                            self.assertEqual(result, set())

        ollama_calls: dict[str, Callable[[Adapter], object]] = {
            "chat": lambda a: a.chat(OLLAMA_MODEL, MESSAGES),
            "stream_chat": lambda a: drain(a, OLLAMA_MODEL),
            "models": lambda a: a.models(),
            "is_available": lambda a: a.is_available(),
        }

        def shifting_ollama(*urls: str) -> Adapter:
            """A configured Ollama adapter whose base URL is each of `urls` in turn."""
            reads = iter(urls)
            cls = type(
                "Shifting",
                (OllamaLocalAdapter,),
                {"_base_url": lambda self: next(reads, urls[-1])},
            )
            provider = Provider(
                adapters={"ollama-local": cls()},
                config=secret_headers("ollama-local", "t"),
            )
            return provider._adapters["ollama-local"]

        for label, call in ollama_calls.items():
            with self.subTest("ollama-local", call=label, order="checked https first"):
                adapter = shifting_ollama(https, http)
                with patch(
                    "urllib.request.urlopen", side_effect=ollama_urlopen
                ) as mock_urlopen:
                    call(adapter)
                [req] = [c.args[0] for c in mock_urlopen.call_args_list]
                self.assertTrue(req.full_url.startswith(https), req.full_url)
                self.assertEqual(req.get_header("X-proxy-token"), "t")
            with (
                self.subTest("ollama-local", call=label, order="http first"),
                patch("urllib.request.urlopen", side_effect=no_request),
                self.assertLogs("ducktape_provider", "WARNING")
                if label in ("models", "is_available")
                else contextlib.nullcontext(),
            ):
                adapter = shifting_ollama(http, https)
                try:
                    result = call(adapter)
                except ValueError:
                    self.assertIn(label, ("chat", "stream_chat"))
                else:
                    self.assertIn(result, (set(), False))


class ClearTracebacksTests(unittest.TestCase):
    def test_non_list_notes_do_not_stop_clearing(self):
        def fail() -> NoReturn:
            raise OSError("boom")

        try:
            try:
                fail()
            except OSError as inner:
                inner.__notes__ = "not a list"  # ty: ignore[invalid-assignment]
                raise ValueError("outer") from inner
        except ValueError as outer:
            outer.__notes__ = ()  # ty: ignore[invalid-assignment]
            _clear_tracebacks(outer)
            self.assertIsNone(outer.__traceback__)
            assert outer.__cause__ is not None
            self.assertIsNone(outer.__cause__.__traceback__)


class SameInstanceCopyAdapter(ClaudeAdapter):
    def __copy__(self):
        return self


class ProviderApiKeysTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(clean_env())

    def _sent_key(self, provider: Provider, name: str, case: Keyed) -> str | None:
        with patch(
            "urllib.request.urlopen", side_effect=fake_urlopen(case)
        ) as mock_urlopen:
            provider.chat(case.model, MESSAGES, provider=name)
        return case.sent_key(mock_urlopen.call_args.args[0])

    def test_mapping_reaches_adapters_by_registered_name(self):
        claude, openai = KEYED
        provider = Provider(
            adapters={
                "work": ClaudeAdapter(),
                "personal": OpenAIAdapter(),
                "spare": ClaudeAdapter(),
                "local": OllamaLocalAdapter(),
            },
            api_keys={"work": "k-work", "personal": lambda: "k-personal"},
        )
        self.assertEqual(self._sent_key(provider, "work", claude), "k-work")
        self.assertEqual(self._sent_key(provider, "personal", openai), "k-personal")
        with self.assertRaises(AuthError):
            self._sent_key(provider, "spare", claude)

    def test_function_is_called_with_each_adapters_own_name(self):
        claude, openai = KEYED
        fn = Mock(side_effect=lambda name: f"k-{name}")
        provider = Provider(
            adapters={
                "a": ClaudeAdapter(),
                "b": OpenAIAdapter(),
                "c": ClaudeAdapter(),
                "local": OllamaLocalAdapter(),
            },
            api_keys=fn,
        )
        fn.assert_not_called()
        for name, case in (("c", claude), ("a", claude), ("b", openai)):
            with self.subTest(name):
                self.assertEqual(self._sent_key(provider, name, case), f"k-{name}")
                self.assertEqual(fn.call_args.args, (name,))

    def test_built_in_adapters_get_keys(self):
        claude, openai = KEYED
        provider = Provider(api_keys={"claude": "k1", "openai": "k2"})
        self.assertEqual(self._sent_key(provider, "claude", claude), "k1")
        self.assertEqual(self._sent_key(provider, "openai", openai), "k2")

    def test_passed_adapter_is_left_unchanged(self):
        for case in KEYED:
            with self.subTest(case.name):
                if case.cls is ClaudeAdapter:
                    stale: dict[str, ModelInfo] = {
                        "stale": {"context_window": None, "max_output_tokens": None}
                    }
                    stale_raw: dict[str, dict[str, Any]] = {
                        "stale": {
                            "id": "stale",
                            "capabilities": {"image_input": {"supported": True}},
                        }
                    }
                    original = ClaudeAdapter()
                    original._models_cache = dict(stale)
                    original._models_raw = dict(stale_raw)
                    original._cache_time = 123.0
                    provider = Provider(
                        adapters={"claude": original}, api_keys={"claude": "k"}
                    )
                    copied = provider._adapters["claude"]
                    self.assertIsNot(copied, original)
                    self.assertIsNone(original._key_source)
                    self.assertEqual(original._models_cache, stale)
                    self.assertEqual(original._models_raw, stale_raw)
                    self.assertEqual(original._cache_time, 123.0)
                    assert isinstance(copied, ClaudeAdapter)
                    self.assertIsNone(copied._models_cache)
                    self.assertIsNone(copied._models_raw)
                    self.assertEqual(copied._cache_time, 0.0)
                    self.assertIsNotNone(copied._key_source)
                else:
                    original_oai = OpenAIAdapter()
                    original_oai._models_cache = {"stale"}  # type: ignore[assignment]
                    original_oai._embed_models_cache = {"stale-embed"}  # type: ignore[assignment]
                    original_oai._cache_time = 123.0
                    provider = Provider(
                        adapters={"openai": original_oai}, api_keys={"openai": "k"}
                    )
                    copied = provider._adapters["openai"]
                    self.assertIsNot(copied, original_oai)
                    self.assertIsNone(original_oai._key_source)
                    self.assertEqual(original_oai._models_cache, {"stale"})
                    self.assertEqual(original_oai._embed_models_cache, {"stale-embed"})
                    self.assertEqual(original_oai._cache_time, 123.0)
                    assert isinstance(copied, OpenAIAdapter)
                    self.assertIsNone(copied._models_cache)
                    self.assertIsNone(copied._embed_models_cache)
                    self.assertEqual(copied._cache_time, 0.0)
                    self.assertIsNotNone(copied._key_source)

    def test_invalid_api_keys_raise_without_keys_in_the_message(self):
        failures: dict[str, tuple[type[Exception], Callable[[], object]]] = {
            "unknown name": (ValueError, lambda: Provider(api_keys={"nope": SENTINEL})),
            "non-str name": (ValueError, lambda: Provider(api_keys={42: SENTINEL})),  # ty: ignore[invalid-argument-type]
            "keyless adapter": (
                ValueError,
                lambda: Provider(api_keys={"ollama-local": SENTINEL}),
            ),
            "plugin name": (
                ValueError,
                lambda: Provider(autodiscover=True, api_keys={"myvendor": SENTINEL}),
            ),
            "bytes value": (
                TypeError,
                lambda: Provider(api_keys={"claude": SENTINEL.encode()}),  # ty: ignore[invalid-argument-type]
            ),
            "None value": (TypeError, lambda: Provider(api_keys={"claude": None})),  # ty: ignore[invalid-argument-type]
            "bad str value": (
                ValueError,
                lambda: Provider(api_keys={"claude": f"{SENTINEL}\n"}),
            ),
            "empty value": (ValueError, lambda: Provider(api_keys={"claude": ""})),
            "bare str": (TypeError, lambda: Provider(api_keys=SENTINEL)),  # ty: ignore[invalid-argument-type]
            "function without keyed adapter": (
                ValueError,
                lambda: Provider(
                    adapters={"local": OllamaLocalAdapter()},
                    api_keys=lambda name: SENTINEL,
                ),
            ),
            "mapping for adapter with its own key": (
                ValueError,
                lambda: Provider(
                    adapters={"claude": ClaudeAdapter(api_key="own")},
                    api_keys={"claude": SENTINEL},
                ),
            ),
            "function for adapter with its own key": (
                ValueError,
                lambda: Provider(
                    adapters={"claude": ClaudeAdapter(api_key="own")},
                    api_keys=lambda name: SENTINEL,
                ),
            ),
            "adapter copying to itself": (
                TypeError,
                lambda: Provider(
                    adapters={"claude": SameInstanceCopyAdapter()},
                    api_keys={"claude": SENTINEL},
                ),
            ),
        }
        for label, (error, call) in failures.items():
            with self.subTest(label):
                exc = raised(self, error, call)
                self.assertNotIn(SENTINEL, str(exc))
                self.assertNotIsInstance(exc, APIError)
                assert_no_sentinel_in_frames(self, exc)

    def test_unknown_names_are_not_echoed_or_kept_in_frames(self):
        failures: dict[str, Callable[[], object]] = {
            "inverted mapping": lambda: Provider(api_keys={SENTINEL: "claude"}),
            "name with newline": lambda: Provider(api_keys={f"{SENTINEL}\n": "k"}),
            "later check fails": lambda: Provider(
                api_keys={SENTINEL: "claude"},
                timeout=-1,
            ),
        }
        for label, call in failures.items():
            with self.subTest(label):
                exc = raised(self, ValueError, call)
                self.assertNotIn(SENTINEL, str(exc))
                assert_no_sentinel_in_frames(self, exc)
                if label != "later check fails":
                    self.assertIn("'claude', 'openai'", str(exc))

    def test_str_subclass_cannot_impersonate_a_provider_name(self):
        class Impostor(str):
            def __hash__(self) -> int:
                return hash("claude")

            def __eq__(self, other: object) -> bool:
                return True

        exc = raised(
            self, ValueError, lambda: Provider(api_keys={Impostor(SENTINEL): "k"})
        )
        self.assertNotIn(SENTINEL, str(exc))
        assert_no_sentinel_in_frames(self, exc)

    def test_keys_are_applied_before_plugins_register(self):
        _clear_discovery_cache()
        self.addCleanup(_clear_discovery_cache)
        plugin = FakeEntryPoint("myvendor", ClaudeAdapter)
        with patch(ENTRY_POINTS, return_value=[plugin]):
            with self.assertRaises(ValueError):
                Provider(autodiscover=True, api_keys={"myvendor": "k"})
            provider = Provider(autodiscover=True, api_keys=lambda name: "k")
        keyed = provider._adapters["claude"]
        plugged = provider._adapters["myvendor"]
        assert isinstance(keyed, ClaudeAdapter) and isinstance(plugged, ClaudeAdapter)
        self.assertIsNotNone(keyed._key_source)
        self.assertIsNone(plugged._key_source)


def raised(
    test: unittest.TestCase, error: type[BaseException], call: Callable[[], object]
) -> BaseException:
    """The `error` that `call` raises, with its traceback's frame locals intact.

    Not assertRaises, which clears those locals and would make any scan pass.
    """
    try:
        call()
    except error as exc:
        return exc
    test.fail(f"{error.__name__} not raised")


def _cell_contents(cell: types.CellType) -> object:
    """A closure cell's value, or None while the variable is still unbound."""
    try:
        return cell.cell_contents
    except ValueError:
        return None


def _sentinel_in_value(value: object, seen: set[int], depth: int) -> bool:
    """A bounded, cycle-safe walk of `value` for SENTINEL.

    Reaches strings/bytes directly, dict keys and values, list/tuple/set/frozenset
    elements, `functools.partial` parts, a function's closure cells and defaults, a
    bound method's `__self__` and function, a suspended generator's, coroutine's or
    async generator's frame locals, and plain objects via their `__dict__`. Never
    `__slots__` (`_Secret`'s are deliberately opaque to this kind of introspection,
    same as to `repr`), nor a function's `__globals__`, which hold SENTINEL itself.
    """
    if depth > 8:
        return False
    if isinstance(value, str):
        return SENTINEL in value
    if isinstance(value, bytes | bytearray):
        return SENTINEL.encode() in value
    oid = id(value)
    if oid in seen:
        return False
    seen.add(oid)

    def found(*children: object) -> bool:
        return any(_sentinel_in_value(child, seen, depth + 1) for child in children)

    if isinstance(value, dict):
        return any(found(k, v) for k, v in value.items())
    if isinstance(value, list | tuple | set | frozenset):
        return found(*value)
    if isinstance(value, functools.partial):
        return found(value.func, value.args, value.keywords)
    if isinstance(value, types.MethodType):
        return found(value.__self__, value.__func__)
    if isinstance(value, types.FunctionType):
        return found(
            *(_cell_contents(cell) for cell in value.__closure__ or ()),
            value.__defaults__,
            value.__kwdefaults__,
        )
    if isinstance(value, types.GeneratorType):
        frame = value.gi_frame
    elif isinstance(value, types.CoroutineType):
        frame = value.cr_frame
    elif isinstance(value, types.AsyncGeneratorType):
        frame = value.ag_frame
    else:
        frame = None
    if frame is not None:
        return found(dict(frame.f_locals))
    obj_dict = getattr(value, "__dict__", None)
    if obj_dict:
        return found(obj_dict)
    return False


def assert_no_sentinel_in_frames(test: unittest.TestCase, exc: BaseException) -> None:
    """Fails if any frame local on `exc`'s traceback, or any chained exception's,
    has a repr containing SENTINEL, or holds it more deeply — e.g. a
    `urllib.request.Request` whose `unredirected_hdrs` dict has the raw key, which
    `Request.__repr__` doesn't show. See `_sentinel_in_value` for the scan's reach.
    """
    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        tb = current.__traceback__
        while tb is not None:
            frame = tb.tb_frame
            for name, value in frame.f_locals.items():
                label = (
                    f"local {name!r} of {frame.f_code.co_qualname} "
                    f"on {type(current).__name__}"
                )
                test.assertNotIn(SENTINEL, repr(value), label)
                test.assertFalse(_sentinel_in_value(value, set(), 0), label)
            tb = tb.tb_next
        pending += (current.__cause__, current.__context__)


def closed_loopback_url() -> str:
    """An http URL on 127.0.0.1 that refuses connections."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


class FrameLocalsTests(unittest.TestCase):
    """No frame local on a failure's traceback, or on any exception chained to it,
    shows the key: error reporters and debuggers capture exactly these."""

    def setUp(self):
        self.enterContext(clean_env(no_proxy="*"))

    def test_request_failures(self):
        for case in KEYED:
            os.environ[case.env_var] = f"{SENTINEL}\n"
            failures: dict[str, tuple[type[BaseException], Any, dict[str, Any]]] = {
                "bad config JSON": (TypeError, SENTINEL, {"x": object()}),
                "bad config header": (ValueError, SENTINEL, {"headers": {"x": "a\nb"}}),
                "bad key": (ValueError, lambda: f"{SENTINEL}\n", {}),
                "bad key from env": (ValueError, None, {}),
                "missing key": (AuthError, lambda: None, {}),
            }
            for label, (error, source, config) in failures.items():
                adapter = case.cls(api_key=source)
                calls: dict[str, Callable[[], object]] = {
                    "chat": functools.partial(
                        adapter.chat, case.model, MESSAGES, config=config
                    ),
                    "stream_chat": lambda adapter=adapter, model=case.model, config=config: (
                        list(adapter.stream_chat(model, MESSAGES, config=config))
                    ),
                }
                for call_name, call in calls.items():
                    with (
                        self.subTest(case.name, failure=label, call=call_name),
                        patch(
                            "urllib.request.urlopen", side_effect=no_request
                        ) as mock_urlopen,
                    ):
                        assert_no_sentinel_in_frames(self, raised(self, error, call))
                        mock_urlopen.assert_not_called()
            with self.subTest(case.name, failure="bad key at construction"):
                exc = raised(
                    self,
                    ValueError,
                    lambda cls=case.cls: cls(api_key=f"{SENTINEL}\r\n"),
                )
                assert_no_sentinel_in_frames(self, exc)
            del os.environ[case.env_var]

    def test_malformed_responses(self):
        for case in KEYED:
            adapter = case.cls(api_key=SENTINEL)
            calls: dict[str, tuple[Callable[[], object], Any]] = {
                "chat": (
                    functools.partial(adapter.chat, case.model, MESSAGES),
                    lambda: buffered_response(b"not json"),
                ),
                "stream_chat": (
                    lambda adapter=adapter, model=case.model: list(
                        adapter.stream_chat(model, MESSAGES)
                    ),
                    lambda: FakeStreamResponse([b"data: not json\n", b"\n"]),
                ),
            }
            for call_name, (call, make_response) in calls.items():
                with (
                    self.subTest(case.name, call=call_name),
                    patch(
                        "urllib.request.urlopen",
                        side_effect=lambda *a, r=make_response, **k: r(),
                    ),
                ):
                    exc = raised(self, MalformedResponseError, call)
                    assert_no_sentinel_in_frames(self, exc)

    def test_interruptions_mid_request(self):
        class Interrupted(Exception):
            pass

        for setup, provider, name, model in secured_providers():
            adapter = provider._adapters[name]
            calls: dict[str, Callable[[], object]] = {
                "chat": functools.partial(adapter.chat, model, MESSAGES),
                "stream_chat": functools.partial(drain, adapter, model),
                "models": adapter.models,
                "Provider.chat": functools.partial(
                    provider.chat, model, MESSAGES, provider=name
                ),
            }
            if isinstance(adapter, OllamaLocalAdapter):
                calls["is_available"] = adapter.is_available
            for error in (SystemExit(1), Interrupted()):
                with (
                    pointed_at(provider, name, "http://127.0.0.1:1/v1"),
                    patch.object(
                        http.client.HTTPConnection, "request", side_effect=error
                    ),
                ):
                    for label, call in calls.items():
                        with self.subTest(setup, call=label, error=type(error)):
                            exc = raised(self, type(error), call)
                            self.assertIs(exc, error)
                            assert_no_sentinel_in_frames(self, exc)

    def test_connection_errors(self):
        for setup, provider, name, model in secured_providers():
            adapter = provider._adapters[name]

            async def consume_stream(
                provider: Provider = provider, model: str = model, name: str = name
            ) -> None:
                async for _ in provider.async_stream_chat(
                    model, MESSAGES, provider=name
                ):
                    pass

            calls: dict[str, Callable[[], object]] = {
                "chat": functools.partial(adapter.chat, model, MESSAGES),
                "stream_chat": functools.partial(drain, adapter, model),
                "async_chat": lambda provider=provider, model=model, name=name: (
                    asyncio.run(provider.async_chat(model, MESSAGES, provider=name))
                ),
                "async_stream_chat": lambda consume_stream=consume_stream: asyncio.run(
                    consume_stream()
                ),
            }
            with pointed_at(provider, name, closed_loopback_url()):
                for label, call in calls.items():
                    with self.subTest(setup, call=label):
                        exc = raised(self, APIError, call)
                        assert_no_sentinel_in_frames(self, exc)
                        assert isinstance(exc, APIError)
                        self.assertTrue(_should_evict_cache(exc))
                        cause = exc.__cause__
                        assert cause is not None
                        self.assertIsNone(cause.__traceback__)
                        self.assertIn(
                            "do_open", "".join(getattr(cause, "__notes__", []))
                        )


if __name__ == "__main__":
    unittest.main()
