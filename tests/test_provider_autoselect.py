import asyncio
import http.client
import threading
import unittest
import urllib.error
from collections.abc import Callable, Generator, Iterator
from typing import Any, cast

from ducktape_provider import errors as errors_module
from ducktape_provider.adapter import Adapter
from ducktape_provider.errors import (
    APIError,
    AuthError,
    ContextOverflowError,
    MalformedResponseError,
    RequestTimeoutError,
    ServerError,
)
from ducktape_provider.provider import Provider
from ducktape_provider.types import Message, Response, StreamEvent, ToolDef

FIXED_RESPONSE: Response = {
    "content": [{"type": "text", "text": "hi there"}],
    "stop_reason": "end_turn",
    "raw_stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 2},
    "raw": {},
    "latency_ms": 0.0,
}

FIXED_STREAM: list[StreamEvent] = [
    {"type": "text_delta", "index": 0, "text": "hi"},
    {"type": "message_stop", "response": FIXED_RESPONSE},
]

MESSAGES: list[Message] = [
    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
]

LOGGER = "ducktape_provider.provider"


def raise_(exc: BaseException) -> Any:
    raise exc


def _connection_error(cause: BaseException) -> APIError:
    """The APIError errors.raise_for_connection_error actually raises for `cause`,
    cause chain (`__cause__`) included — building one by hand would miss that."""
    try:
        errors_module.raise_for_connection_error("test", cause)
    except APIError as e:
        return e
    raise AssertionError("raise_for_connection_error did not raise APIError")


class CountingFakeAdapter(Adapter):
    """Adapter whose behavior is scripted, with call counters for is_available()/models()."""

    def __init__(
        self,
        chat: Callable[[], Response] = lambda: FIXED_RESPONSE,
        stream: Callable[[], Iterator[StreamEvent]] = lambda: iter(FIXED_STREAM),
        is_available: Callable[[], bool] = lambda: True,
        models: Callable[[], set[str]] = lambda: {"fake-model"},
    ):
        self._chat = chat
        self._stream = stream
        self._is_available = is_available
        self._models = models
        self.is_available_calls = 0
        self.models_calls = 0
        self.chat_threads: list[threading.Thread] = []
        self.models_threads: list[threading.Thread] = []

    def is_available(self) -> bool:
        self.is_available_calls += 1
        return self._is_available()

    def models(self) -> set[str]:
        self.models_calls += 1
        self.models_threads.append(threading.current_thread())
        return self._models()

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response:
        self.chat_threads.append(threading.current_thread())
        return self._chat()

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        self.chat_threads.append(threading.current_thread())
        return self._stream()


class TestExplicitProviderUnchanged(unittest.TestCase):
    def test_explicit_provider_keyword_still_works(self):
        adapter = CountingFakeAdapter()
        provider = Provider(adapters={"fake": adapter})
        response = provider.chat("fake-model", MESSAGES, provider="fake")
        self.assertEqual(response, FIXED_RESPONSE)
        self.assertEqual(adapter.is_available_calls, 0)
        self.assertEqual(adapter.models_calls, 0)


class TestAutoMatch(unittest.TestCase):
    def test_picks_first_available_matching_adapter_in_order(self):
        first = CountingFakeAdapter(models=lambda: {"shared-model"})
        second = CountingFakeAdapter(models=lambda: {"shared-model"})
        provider = Provider(adapters={"first": first, "second": second})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.chat("shared-model", MESSAGES)
        self.assertEqual(len(first.chat_threads), 1)
        self.assertEqual(second.models_calls, 0)

    def test_skips_unavailable_adapter_even_if_model_matches(self):
        down = CountingFakeAdapter(
            is_available=lambda: False, models=lambda: {"shared-model"}
        )
        up = CountingFakeAdapter(models=lambda: {"shared-model"})
        provider = Provider(adapters={"down": down, "up": up})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.chat("shared-model", MESSAGES)
        self.assertEqual(down.models_calls, 0)
        self.assertEqual(up.models_calls, 1)

    def test_warning_names_matched_provider(self):
        adapter = CountingFakeAdapter(models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING") as logs:
            provider.chat("fake-model", MESSAGES)
        self.assertTrue(
            any("fake-model" in line and "'fake'" in line for line in logs.output)
        )

    def test_no_match_raises_key_error(self):
        adapter = CountingFakeAdapter(models=lambda: {"other-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertRaisesRegex(KeyError, "no-such-model"):
            provider.chat("no-such-model", MESSAGES)

    def test_adapter_raising_during_models_is_skipped_not_fatal(self):
        broken = CountingFakeAdapter(models=lambda: raise_(RuntimeError("boom")))
        fine = CountingFakeAdapter(models=lambda: {"fake-model"})
        provider = Provider(adapters={"broken": broken, "fine": fine})
        with self.assertLogs(LOGGER, "WARNING"):
            response = provider.chat("fake-model", MESSAGES)
        self.assertEqual(response, FIXED_RESPONSE)

    def test_adapter_raising_during_is_available_is_skipped_not_fatal(self):
        broken = CountingFakeAdapter(
            is_available=lambda: raise_(RuntimeError("boom")),
            models=lambda: {"fake-model"},
        )
        fine = CountingFakeAdapter(models=lambda: {"fake-model"})
        provider = Provider(adapters={"broken": broken, "fine": fine})
        with self.assertLogs(LOGGER, "WARNING"):
            response = provider.chat("fake-model", MESSAGES)
        self.assertEqual(response, FIXED_RESPONSE)
        self.assertEqual(broken.models_calls, 0)

    def test_providers_reports_false_when_is_available_raises(self):
        broken = CountingFakeAdapter(is_available=lambda: raise_(RuntimeError("boom")))
        provider = Provider(adapters={"broken": broken})
        with self.assertLogs(LOGGER, "WARNING"):
            self.assertEqual(provider.providers(), {"broken": False})

    def test_models_omits_adapter_when_is_available_raises(self):
        broken = CountingFakeAdapter(is_available=lambda: raise_(RuntimeError("boom")))
        provider = Provider(adapters={"broken": broken})
        with self.assertLogs(LOGGER, "WARNING"):
            self.assertEqual(provider.models(), {})

    def test_models_omits_adapter_when_models_raises(self):
        broken = CountingFakeAdapter(models=lambda: raise_(RuntimeError("boom")))
        provider = Provider(adapters={"broken": broken})
        with self.assertLogs(LOGGER, "WARNING"):
            self.assertEqual(provider.models(), {})

    def test_per_provider_config_applies_to_resolved_provider(self):
        seen: list[dict[str, Any] | None] = []

        class RecordingAdapter(CountingFakeAdapter):
            def chat(self, model, messages, system=None, tools=None, config=None):
                seen.append(config)
                return FIXED_RESPONSE

        adapter = RecordingAdapter(models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        config = {"timeout": 5, "providers": {"fake": {"timeout": 7}}}
        with self.assertLogs(LOGGER, "WARNING"):
            provider.chat("fake-model", MESSAGES, config=config)
        self.assertEqual(seen, [{"timeout": 7}])


class TestAutoMatchAsync(unittest.IsolatedAsyncioTestCase):
    async def test_async_chat_matches_off_the_event_loop(self):
        adapter = CountingFakeAdapter(models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        loop_thread = threading.current_thread()
        with self.assertLogs(LOGGER, "WARNING"):
            await provider.async_chat("fake-model", MESSAGES)
        self.assertEqual(len(adapter.models_threads), 1)
        self.assertNotEqual(adapter.models_threads[0], loop_thread)

    async def test_async_chat_no_match_raises_key_error(self):
        adapter = CountingFakeAdapter(models=lambda: {"other-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertRaises(KeyError):
            await provider.async_chat("no-such-model", MESSAGES)

    async def test_async_stream_chat_matches_on_reader_thread(self):
        adapter = CountingFakeAdapter(models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        loop_thread = threading.current_thread()
        with self.assertLogs(LOGGER, "WARNING"):
            events = [
                event
                async for event in provider.async_stream_chat("fake-model", MESSAGES)
            ]
        self.assertEqual(events, FIXED_STREAM)
        self.assertEqual(len(adapter.models_threads), 1)
        self.assertNotEqual(adapter.models_threads[0], loop_thread)

    async def test_async_stream_chat_no_match_raises_on_iteration_not_call(self):
        adapter = CountingFakeAdapter(models=lambda: {"other-model"})
        provider = Provider(adapters={"fake": adapter})
        stream = provider.async_stream_chat("no-such-model", MESSAGES)
        self.assertEqual(adapter.models_calls, 0)
        with self.assertRaises(KeyError):
            async for _event in stream:
                pass

    async def test_explicit_unknown_provider_raises_eagerly(self):
        provider = Provider(adapters={"fake": CountingFakeAdapter()})
        with self.assertRaises(KeyError):
            provider.stream_chat("fake-model", MESSAGES, provider="nope")
        with self.assertRaises(KeyError):
            provider.async_stream_chat("fake-model", MESSAGES, provider="nope")

    async def test_async_providers_reports_false_when_is_available_raises(self):
        broken = CountingFakeAdapter(is_available=lambda: raise_(RuntimeError("boom")))
        provider = Provider(adapters={"broken": broken})
        with self.assertLogs(LOGGER, "WARNING"):
            self.assertEqual(await provider.async_providers(), {"broken": False})

    async def test_async_models_omits_adapter_when_is_available_raises(self):
        broken = CountingFakeAdapter(is_available=lambda: raise_(RuntimeError("boom")))
        provider = Provider(adapters={"broken": broken})
        with self.assertLogs(LOGGER, "WARNING"):
            self.assertEqual(await provider.async_models(), {})


class TestAutoMatchCache(unittest.TestCase):
    def test_second_call_skips_probing(self):
        adapter = CountingFakeAdapter(models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.chat("fake-model", MESSAGES)
        self.assertEqual(adapter.is_available_calls, 1)
        self.assertEqual(adapter.models_calls, 1)

        provider.chat("fake-model", MESSAGES)
        self.assertEqual(adapter.is_available_calls, 1)
        self.assertEqual(adapter.models_calls, 1)

    def test_warning_logged_only_on_first_call(self):
        adapter = CountingFakeAdapter(models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING") as logs:
            provider.chat("fake-model", MESSAGES)
        first_call_warnings = len(logs.output)
        self.assertGreaterEqual(first_call_warnings, 1)

        with self.assertNoLogs(LOGGER, "WARNING"):
            provider.chat("fake-model", MESSAGES)

    def test_explicit_provider_bypasses_cache(self):
        adapter = CountingFakeAdapter(models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        provider.chat("fake-model", MESSAGES, provider="fake")
        provider.chat("fake-model", MESSAGES, provider="fake")
        self.assertEqual(adapter.is_available_calls, 0)
        self.assertEqual(adapter.models_calls, 0)
        self.assertEqual(provider._auto_match_cache, {})

    def test_404_through_cached_resolution_evicts_entry(self):
        calls = {"n": 0}

        def chat() -> Response:
            calls["n"] += 1
            if calls["n"] == 2:
                raise APIError("gone", status=404)
            return FIXED_RESPONSE

        adapter = CountingFakeAdapter(chat=chat, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.chat("fake-model", MESSAGES)
        self.assertIn("fake-model", provider._auto_match_cache)

        with self.assertRaises(APIError):
            provider.chat("fake-model", MESSAGES)
        self.assertNotIn("fake-model", provider._auto_match_cache)

        self.assertEqual(adapter.models_calls, 1)
        with self.assertLogs(LOGGER, "WARNING"):
            provider.chat("fake-model", MESSAGES)
        self.assertEqual(adapter.models_calls, 2)

    def test_non_404_error_does_not_evict_cache_entry(self):
        def chat() -> Response:
            raise APIError("server broke", status=500)

        adapter = CountingFakeAdapter(chat=chat, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(APIError):
            provider.chat("fake-model", MESSAGES)
        self.assertIn("fake-model", provider._auto_match_cache)

    def test_stream_404_evicts_when_error_reaches_consumer(self):
        def stream() -> Iterator[StreamEvent]:
            yield {"type": "text_delta", "index": 0, "text": "x"}
            raise APIError("gone", status=404)

        adapter = CountingFakeAdapter(stream=stream, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(APIError):
            list(provider.stream_chat("fake-model", MESSAGES))
        self.assertNotIn("fake-model", provider._auto_match_cache)

    def test_stream_404_does_not_evict_when_consumer_stops_before_error(self):
        def stream() -> Iterator[StreamEvent]:
            yield {"type": "text_delta", "index": 0, "text": "x"}
            raise APIError("gone", status=404)

        adapter = CountingFakeAdapter(stream=stream, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            gen = cast(
                Generator[StreamEvent, None, None],
                provider.stream_chat("fake-model", MESSAGES),
            )
            next(gen)
            gen.close()
        self.assertIn("fake-model", provider._auto_match_cache)

    def test_404_through_async_chat_evicts_entry(self):
        calls = {"n": 0}

        def chat() -> Response:
            calls["n"] += 1
            if calls["n"] == 2:
                raise APIError("gone", status=404)
            return FIXED_RESPONSE

        adapter = CountingFakeAdapter(chat=chat, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})

        async def run() -> None:
            await provider.async_chat("fake-model", MESSAGES)
            with self.assertRaises(APIError):
                await provider.async_chat("fake-model", MESSAGES)

        with self.assertLogs(LOGGER, "WARNING"):
            asyncio.run(run())
        self.assertNotIn("fake-model", provider._auto_match_cache)

    def test_404_through_async_stream_chat_evicts_entry(self):
        def stream() -> Iterator[StreamEvent]:
            yield {"type": "text_delta", "index": 0, "text": "x"}
            raise APIError("gone", status=404)

        adapter = CountingFakeAdapter(stream=stream, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})

        async def run() -> None:
            with self.assertRaises(APIError):
                async for _event in provider.async_stream_chat("fake-model", MESSAGES):
                    pass

        with self.assertLogs(LOGGER, "WARNING"):
            asyncio.run(run())
        self.assertNotIn("fake-model", provider._auto_match_cache)

    def test_auth_error_evicts_cache_entry(self):
        def chat() -> Response:
            raise AuthError("no key", status=401)

        adapter = CountingFakeAdapter(chat=chat, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(AuthError):
            provider.chat("fake-model", MESSAGES)
        self.assertNotIn("fake-model", provider._auto_match_cache)

    def test_url_error_caused_failure_evicts_cache_entry(self):
        def chat() -> Response:
            raise _connection_error(urllib.error.URLError(ConnectionRefusedError()))

        adapter = CountingFakeAdapter(chat=chat, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(APIError):
            provider.chat("fake-model", MESSAGES)
        self.assertNotIn("fake-model", provider._auto_match_cache)

    def test_truncated_stream_does_not_evict_cache_entry(self):
        def chat() -> Response:
            errors_module.raise_for_truncated_stream("test", "message_stop")

        adapter = CountingFakeAdapter(chat=chat, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(APIError):
            provider.chat("fake-model", MESSAGES)
        self.assertIn("fake-model", provider._auto_match_cache)

    def test_incomplete_read_does_not_evict_cache_entry(self):
        def chat() -> Response:
            raise _connection_error(http.client.IncompleteRead(b""))

        adapter = CountingFakeAdapter(chat=chat, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(APIError):
            provider.chat("fake-model", MESSAGES)
        self.assertIn("fake-model", provider._auto_match_cache)

    def test_raw_oserror_mid_read_does_not_evict_cache_entry(self):
        def chat() -> Response:
            raise _connection_error(ConnectionResetError())

        adapter = CountingFakeAdapter(chat=chat, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(APIError):
            provider.chat("fake-model", MESSAGES)
        self.assertIn("fake-model", provider._auto_match_cache)

    def test_unmapped_vendor_error_does_not_evict_cache_entry(self):
        def chat() -> Response:
            errors_module.raise_for_vendor_error("test", "weird event", status=None)

        adapter = CountingFakeAdapter(chat=chat, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(APIError):
            provider.chat("fake-model", MESSAGES)
        self.assertIn("fake-model", provider._auto_match_cache)

    def test_stream_chat_auth_error_evicts_cache_entry(self):
        def stream() -> Iterator[StreamEvent]:
            yield {"type": "text_delta", "index": 0, "text": "x"}
            raise AuthError("revoked", status=403)

        adapter = CountingFakeAdapter(stream=stream, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(AuthError):
            list(provider.stream_chat("fake-model", MESSAGES))
        self.assertNotIn("fake-model", provider._auto_match_cache)

    def test_stream_url_error_caused_failure_evicts_cache_entry(self):
        def stream() -> Iterator[StreamEvent]:
            yield {"type": "text_delta", "index": 0, "text": "x"}
            raise _connection_error(urllib.error.URLError(ConnectionRefusedError()))

        adapter = CountingFakeAdapter(stream=stream, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(APIError):
            list(provider.stream_chat("fake-model", MESSAGES))
        self.assertNotIn("fake-model", provider._auto_match_cache)

    def test_non_evicting_error_types_leave_cache_entry(self):
        non_evicting: list[APIError] = [
            ServerError("down", status=500),
            RequestTimeoutError("too slow"),
            MalformedResponseError("bad json"),
            ContextOverflowError("too long", status=400),
        ]
        for exc in non_evicting:
            with self.subTest(type=type(exc).__name__):
                adapter = CountingFakeAdapter(
                    chat=lambda exc=exc: raise_(exc), models=lambda: {"fake-model"}
                )
                provider = Provider(adapters={"fake": adapter})
                with (
                    self.assertLogs(LOGGER, "WARNING"),
                    self.assertRaises(type(exc)),
                ):
                    provider.chat("fake-model", MESSAGES)
                self.assertIn("fake-model", provider._auto_match_cache)


if __name__ == "__main__":
    unittest.main()
