"""Tests for API keys passed without env vars: `Provider(api_keys=...)`, the adapters'
`api_key=`, and keeping a key out of reprs, pickles and traceback frame locals.

urlopen is mocked, except for connection errors against a closed port on 127.0.0.1;
nothing reaches a vendor or a local Ollama daemon.
"""

import asyncio
import copy
import functools
import http.client
import json
import os
import pickle
import socket
import unittest
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, NoReturn
from unittest.mock import Mock, patch

from http_test_utils import FakeStreamResponse, buffered_response, sse_lines
from test_provider_discovery import ENTRY_POINTS, FakeEntryPoint

from ducktape_provider import (
    APIError,
    AuthError,
    ClaudeAdapter,
    Message,
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
        if "/models" in req.full_url:
            return buffered_response(
                json.dumps({"data": [{"id": case.model}]}).encode()
            )
        if json.loads(req.data)["stream"]:
            return FakeStreamResponse(case.stream_lines)
        return buffered_response(json.dumps(case.chat_body).encode())

    return urlopen


def request_calls(case: Keyed, adapter: Any) -> dict[str, Callable[[], object]]:
    return {
        "chat": lambda: adapter.chat(case.model, MESSAGES),
        "stream_chat": lambda: list(adapter.stream_chat(case.model, MESSAGES)),
    }


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
                (case.chat_url_attr, lambda a: a.chat(case.model, MESSAGES)),  # noqa: B023
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
        original = ClaudeAdapter()
        original._models_cache = {"stale"}
        provider = Provider(adapters={"claude": original}, api_keys={"claude": "k"})
        copied = provider._adapters["claude"]
        self.assertIsNot(copied, original)
        self.assertIsNone(original._key_source)
        self.assertEqual(original._models_cache, {"stale"})
        assert isinstance(copied, ClaudeAdapter)
        self.assertIsNone(copied._models_cache)
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


def assert_no_sentinel_in_frames(test: unittest.TestCase, exc: BaseException) -> None:
    """Fails if any frame local on `exc`'s traceback, or any chained exception's,
    has a repr containing SENTINEL."""
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
                test.assertNotIn(
                    SENTINEL,
                    repr(value),
                    f"local {name!r} of {frame.f_code.co_qualname} "
                    f"on {type(current).__name__}",
                )
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
                    "stream_chat": lambda: list(
                        adapter.stream_chat(case.model, MESSAGES, config=config)  # noqa: B023
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
                    lambda: case.cls(api_key=f"{SENTINEL}\r\n"),  # noqa: B023
                )
                assert_no_sentinel_in_frames(self, exc)
            del os.environ[case.env_var]

    def test_interruptions_mid_request(self):
        class Interrupted(Exception):
            pass

        for case in KEYED:
            adapter = case.cls(api_key=SENTINEL)
            provider = Provider(adapters={case.name: adapter})
            model, name = case.model, case.name
            calls: dict[str, Callable[[], object]] = {
                "chat": functools.partial(adapter.chat, model, MESSAGES),
                "stream_chat": lambda: list(
                    adapter.stream_chat(model, MESSAGES)  # noqa: B023
                ),
                "models": adapter.models,
                "Provider.chat": functools.partial(
                    provider.chat, model, MESSAGES, provider=name
                ),
            }
            for error in (SystemExit(1), Interrupted()):
                with (
                    patch.object(adapter, case.chat_url_attr, "http://127.0.0.1:1/v1"),
                    patch.object(adapter, "_MODELS_URL", "http://127.0.0.1:1/v1"),
                    patch.object(
                        http.client.HTTPConnection, "request", side_effect=error
                    ),
                ):
                    for label, call in calls.items():
                        with self.subTest(case.name, call=label, error=type(error)):
                            exc = raised(self, type(error), call)
                            self.assertIs(exc, error)
                            assert_no_sentinel_in_frames(self, exc)

    def test_connection_errors(self):
        for case in KEYED:
            adapter = case.cls(api_key=SENTINEL)
            provider = Provider(adapters={case.name: adapter})
            model, name = case.model, case.name

            async def consume_stream(model: str = model, name: str = name) -> None:
                async for _ in provider.async_stream_chat(  # noqa: B023
                    model, MESSAGES, provider=name
                ):
                    pass

            calls: dict[str, Callable[[], object]] = {
                "chat": functools.partial(adapter.chat, model, MESSAGES),
                "stream_chat": lambda: list(
                    adapter.stream_chat(model, MESSAGES)  # noqa: B023
                ),
                "async_chat": lambda: asyncio.run(
                    provider.async_chat(model, MESSAGES, provider=name)  # noqa: B023
                ),
                "async_stream_chat": lambda: asyncio.run(consume_stream()),
            }
            with patch.object(adapter, case.chat_url_attr, closed_loopback_url()):
                for label, call in calls.items():
                    with self.subTest(case.name, call=label):
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
