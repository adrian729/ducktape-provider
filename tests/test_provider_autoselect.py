import asyncio
import http.client
import threading
import time
import unittest
import urllib.error
import urllib.request
from collections.abc import Callable, Generator, Iterator
from typing import Any, Self, cast
from unittest.mock import patch

from ducktape_provider import ClaudeAdapter, OpenAIAdapter
from ducktape_provider import errors as errors_module
from ducktape_provider.adapter import Adapter
from ducktape_provider.errors import (
    APIError,
    AuthError,
    ContextOverflowError,
    MalformedResponseError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    UnsupportedOperationError,
)
from ducktape_provider.provider import Provider, _EvictingStream
from ducktape_provider.types import (
    Capabilities,
    EmbedResponse,
    Message,
    ModelInfo,
    Response,
    StreamEvent,
    SystemBlock,
    ToolDef,
)

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


class LoopTicker:
    """Counts event-loop turns; a blocked loop stops the count."""

    def __init__(self) -> None:
        self.count = 0
        self._task: asyncio.Task[None] | None = None

    async def _tick(self) -> None:
        while True:
            self.count += 1
            await asyncio.sleep(0)

    def __enter__(self) -> Self:
        self._task = asyncio.ensure_future(self._tick())
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._task is not None
        self._task.cancel()


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


FIXED_EMBED: EmbedResponse = {
    "embeddings": [[0.1, 0.2, 0.3]],
    "usage": {"input_tokens": 1},
    "raw": {},
    "latency_ms": 0.0,
}

FILLED_EMBED: EmbedResponse = {**FIXED_EMBED, "dimensions": 3}


class CountingFakeAdapter(Adapter):
    """Adapter whose behavior is scripted, with call counters for is_available()/models()."""

    def __init__(
        self,
        chat: Callable[[], Response] = lambda: FIXED_RESPONSE,
        stream: Callable[[], Iterator[StreamEvent]] = lambda: iter(FIXED_STREAM),
        is_available: Callable[[], bool] = lambda: True,
        models: Callable[[], set[str]] = lambda: {"fake-model"},
        model_info: Callable[[], ModelInfo | None] = lambda: None,
        capabilities: Callable[[str], Capabilities | None]
        | Callable[[], Capabilities | None]
        | None = None,
        embed: Callable[[str, list[str], dict[str, Any] | None], EmbedResponse]
        | None = None,
        embed_models: Callable[[], set[str]] = lambda: set(),
    ):
        self._chat = chat
        self._stream = stream
        self._is_available = is_available
        self._models = models
        self._model_info = model_info
        self._capabilities = capabilities
        self._embed = embed
        self._embed_models = embed_models
        self.is_available_calls = 0
        self.models_calls = 0
        self.model_info_calls = 0
        self.capabilities_calls = 0
        self.embed_calls = 0
        self.embed_models_calls = 0
        self.chat_threads: list[threading.Thread] = []
        self.models_threads: list[threading.Thread] = []
        self.embed_models_threads: list[threading.Thread] = []
        self.embed_configs: list[dict[str, Any] | None] = []
        self._capabilities_cache: dict[str, Capabilities | None] = {}

    def is_available(self) -> bool:
        self.is_available_calls += 1
        return self._is_available()

    def models(self) -> set[str]:
        self.models_calls += 1
        self.models_threads.append(threading.current_thread())
        return self._models()

    def model_info(self, model: str) -> ModelInfo | None:
        self.model_info_calls += 1
        return self._model_info()

    def capabilities(self, model: str) -> Capabilities | None:
        if model in self._capabilities_cache:
            return self._capabilities_cache[model]
        self.capabilities_calls += 1
        if self._capabilities is None:
            result = super().capabilities(model)
        else:
            try:
                result = self._capabilities(model)  # ty: ignore
            except TypeError:
                result = self._capabilities()  # ty: ignore
        self._capabilities_cache[model] = result
        return result

    def embed_models(self) -> set[str]:
        self.embed_models_calls += 1
        self.embed_models_threads.append(threading.current_thread())
        return self._embed_models()

    def embed(
        self,
        model: str,
        input: list[str],
        config: dict[str, Any] | None = None,
    ) -> EmbedResponse:
        self.embed_calls += 1
        self.embed_configs.append(config)
        if self._embed is not None:
            return self._embed(model, input, config)
        return super().embed(model, input, config)

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | list[SystemBlock] | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response:
        self.chat_threads.append(threading.current_thread())
        return self._chat()

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | list[SystemBlock] | None = None,
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

    def test_explicit_provider_short_circuits_model_info_resolution(self):
        info: ModelInfo = {"context_window": 100, "max_output_tokens": None}
        adapter = CountingFakeAdapter(model_info=lambda: info)
        provider = Provider(adapters={"fake": adapter})
        result = provider.model_info("fake-model", provider="fake")
        self.assertEqual(result, info)
        self.assertEqual(adapter.is_available_calls, 0)
        self.assertEqual(adapter.models_calls, 0)
        self.assertEqual(adapter.model_info_calls, 1)


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

    def test_model_info_reuses_the_auto_match_cache_chat_populates(self):
        adapter = CountingFakeAdapter(models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.chat("fake-model", MESSAGES)
        self.assertEqual(adapter.is_available_calls, 1)
        self.assertEqual(adapter.models_calls, 1)

        with self.assertNoLogs(LOGGER, "WARNING"):
            provider.model_info("fake-model")
        self.assertEqual(adapter.is_available_calls, 1)
        self.assertEqual(adapter.models_calls, 1)
        self.assertEqual(adapter.model_info_calls, 1)

    def test_model_info_populates_the_auto_match_cache_chat_then_reuses(self):
        adapter = CountingFakeAdapter(models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.model_info("fake-model")
        self.assertEqual(adapter.is_available_calls, 1)
        self.assertEqual(adapter.models_calls, 1)

        with self.assertNoLogs(LOGGER, "WARNING"):
            provider.chat("fake-model", MESSAGES)
        self.assertEqual(adapter.is_available_calls, 1)
        self.assertEqual(adapter.models_calls, 1)

    def test_model_info_raising_propagates_unwrapped(self):
        adapter = CountingFakeAdapter(
            models=lambda: {"fake-model"},
            model_info=lambda: raise_(RuntimeError("boom")),
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(RuntimeError):
            provider.model_info("fake-model")

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


class RecordingStream:
    """An inner stream that records the calls `_EvictingStream` should pass through."""

    def __init__(self, events: list[StreamEvent]):
        self._events = iter(events)
        self.closed = False
        self.cancelled = False

    def __iter__(self) -> "RecordingStream":
        return self

    def __next__(self) -> StreamEvent:
        return next(self._events)

    def close(self) -> None:
        self.closed = True

    def cancel(self) -> None:
        self.cancelled = True


class EvictingStreamTests(unittest.TestCase):
    """Wrapping an auto-matched stream must not hide what the consumer holds it for:
    a generator wrapper couldn't carry `close`/`cancel` at all, which is the bug this
    class exists to avoid."""

    def _provider(self) -> Provider:
        provider = Provider(
            adapters={"fake": CountingFakeAdapter(models=lambda: {"fake-model"})}
        )
        provider._auto_match_cache["fake-model"] = "fake"
        return provider

    def test_close_and_cancel_forward_to_the_inner_stream(self):
        inner = RecordingStream(list(FIXED_STREAM))
        stream = _EvictingStream(inner, self._provider(), "fake-model", "fake")
        stream.close()
        stream.cancel()
        self.assertTrue(inner.closed)
        self.assertTrue(inner.cancelled)

    def test_forwarding_degrades_when_the_inner_stream_lacks_the_methods(self):
        inner = iter(FIXED_STREAM)
        stream = _EvictingStream(inner, self._provider(), "fake-model", "fake")
        stream.close()
        # `cancel` isn't just a no-op here: it's genuinely absent, so
        # `getattr(stream, "cancel", None)` — the README's documented
        # feature check for third-party adapter streams — correctly reports
        # "not supported" instead of a method that silently does nothing.
        self.assertIsNone(getattr(stream, "cancel", None))
        self.assertEqual(list(stream), FIXED_STREAM)

    def test_cancel_degrades_on_a_plain_generator_stream(self):
        def inner() -> Iterator[StreamEvent]:
            yield from FIXED_STREAM

        generator = inner()
        stream = _EvictingStream(generator, self._provider(), "fake-model", "fake")
        self.assertIsNone(getattr(stream, "cancel", None))
        self.assertEqual(list(stream), FIXED_STREAM)

    def test_exhausted_inner_stream_ends_the_loop_without_evicting(self):
        provider = self._provider()
        stream = _EvictingStream(
            RecordingStream(list(FIXED_STREAM)), provider, "fake-model", "fake"
        )
        self.assertEqual(list(stream), FIXED_STREAM)
        with self.assertRaises(StopIteration):
            next(stream)
        self.assertEqual(provider._auto_match_cache, {"fake-model": "fake"})

    def test_cancel_through_stream_chat_reaches_the_adapter_stream(self):
        inner = RecordingStream(list(FIXED_STREAM))
        adapter = CountingFakeAdapter(
            stream=lambda: inner, models=lambda: {"fake-model"}
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            stream = provider.stream_chat("fake-model", MESSAGES)
        cancel = getattr(stream, "cancel", None)
        self.assertIsNotNone(cancel)
        cast(Callable[[], None], cancel)()
        self.assertTrue(inner.cancelled)


def _fail_urlopen(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("urllib.request.urlopen should not be called")


class TestEmbedUnsupportedAdapter(unittest.TestCase):
    """Provider.embed on an adapter without embed support."""

    def test_plain_fake_raises_without_any_request(self):
        fake = CountingFakeAdapter()
        provider = Provider(adapters={"fake": fake})
        with patch("urllib.request.urlopen", side_effect=_fail_urlopen) as mock:
            with self.assertRaises(UnsupportedOperationError) as ctx:
                provider.embed("m", "hi", provider="fake")
            self.assertEqual(
                str(ctx.exception), "provider 'fake' does not support embed()"
            )
            mock.assert_not_called()
        self.assertIsInstance(ctx.exception.__cause__, UnsupportedOperationError)

    def test_message_uses_registered_name_not_class(self):
        fake = CountingFakeAdapter()
        provider = Provider(adapters={"other": fake})
        with (
            patch("urllib.request.urlopen", side_effect=_fail_urlopen),
            self.assertRaises(UnsupportedOperationError) as ctx,
        ):
            provider.embed("m", "hi", provider="other")
        self.assertEqual(
            str(ctx.exception), "provider 'other' does not support embed()"
        )
        self.assertNotIn("CountingFakeAdapter", str(ctx.exception))
        self.assertNotIn("anthropic", str(ctx.exception).lower())
        self.assertNotIn("openai", str(ctx.exception).lower())
        self.assertNotIn("ollama", str(ctx.exception).lower())

    def test_minimal_adapter_instantiates_with_default_embed_hooks(self):
        class Minimal(Adapter):
            def is_available(self) -> bool:
                return True

            def models(self) -> set[str]:
                return set()

            def chat(
                self,
                model: str,
                messages: list[Message],
                system: str | list[SystemBlock] | None = None,
                tools: list[ToolDef] | None = None,
                config: dict[str, Any] | None = None,
            ) -> Response:
                return FIXED_RESPONSE

            def stream_chat(
                self,
                model: str,
                messages: list[Message],
                system: str | list[SystemBlock] | None = None,
                tools: list[ToolDef] | None = None,
                config: dict[str, Any] | None = None,
            ) -> Iterator[StreamEvent]:
                return iter(())

        adapter = Minimal()
        self.assertEqual(adapter.embed_models(), set())
        with self.assertRaises(UnsupportedOperationError) as ctx:
            adapter.embed("m", ["hi"])
        self.assertEqual(
            str(ctx.exception), "provider 'Minimal' does not support embed()"
        )

    def test_claude_adapter_behaves_identically_to_plain_fake(self):
        for name in ("my-claude", "alt"):
            with self.subTest(name=name):
                provider = Provider(adapters={name: ClaudeAdapter()})
                with patch("urllib.request.urlopen", side_effect=_fail_urlopen) as mock:
                    with self.assertRaises(UnsupportedOperationError) as ctx:
                        provider.embed("m", "hi", provider=name)
                    self.assertEqual(
                        str(ctx.exception),
                        f"provider {name!r} does not support embed()",
                    )
                    mock.assert_not_called()


class TestEmbedAutoMatch(unittest.TestCase):
    """Auto-match via embed_models()."""

    def test_first_available_match_via_embed_models(self):
        first = CountingFakeAdapter(
            embed=lambda _m, _i, _c: FIXED_EMBED,
            embed_models=lambda: {"shared-embed"},
        )
        second = CountingFakeAdapter(
            embed=lambda _m, _i, _c: FIXED_EMBED,
            embed_models=lambda: {"shared-embed"},
        )
        provider = Provider(adapters={"first": first, "second": second})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.embed("shared-embed", "hi")
        self.assertEqual(first.embed_calls, 1)
        self.assertEqual(second.embed_models_calls, 0)
        self.assertEqual(second.embed_calls, 0)

    def test_dimensions_filled_by_provider_not_adapter(self):
        adapter = CountingFakeAdapter(
            embed=lambda _m, _i, _c: FIXED_EMBED,
            embed_models=lambda: {"dims-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        direct = adapter.embed("dims-model", ["hi"])
        self.assertNotIn("dimensions", direct)
        with self.assertLogs(LOGGER, "WARNING"):
            result = provider.embed("dims-model", "hi")
        self.assertEqual(result["dimensions"], 3)
        self.assertNotIn("dimensions", FIXED_EMBED)

    def test_ragged_vectors_raise_malformed(self):
        ragged: EmbedResponse = {**FIXED_EMBED, "embeddings": [[0.1], [0.1, 0.2]]}
        adapter = CountingFakeAdapter(
            embed=lambda _m, _i, _c: ragged,
            embed_models=lambda: {"ragged-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        with (
            self.assertLogs(LOGGER, "WARNING"),
            self.assertRaises(MalformedResponseError),
        ):
            provider.embed("ragged-model", ["a", "b"])

    def test_unsized_vector_raises_malformed(self):
        payload = cast(EmbedResponse, {**FIXED_EMBED, "embeddings": [[0.1], None]})
        adapter = CountingFakeAdapter(
            embed=lambda _m, _i, _c: payload,
            embed_models=lambda: {"unsized-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        with (
            self.assertLogs(LOGGER, "WARNING"),
            self.assertRaises(MalformedResponseError),
        ):
            provider.embed("unsized-model", ["a", "b"])

    def test_missing_embeddings_key_raises_malformed(self):
        payload = cast(EmbedResponse, {"usage": None, "raw": {}})
        adapter = CountingFakeAdapter(
            embed=lambda _m, _i, _c: payload,
            embed_models=lambda: {"missing-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        with (
            self.assertLogs(LOGGER, "WARNING"),
            self.assertRaises(MalformedResponseError),
        ):
            provider.embed("missing-model", "hi")

    def test_extra_keys_and_tuple_vectors_are_normalized(self):
        payload = cast(
            EmbedResponse,
            {
                **FIXED_EMBED,
                "embeddings": [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)],
                "vendor_extra": 1,
            },
        )
        adapter = CountingFakeAdapter(
            embed=lambda _m, _i, _c: payload,
            embed_models=lambda: {"extra-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            result = provider.embed("extra-model", ["a", "b"])
        self.assertEqual(result["embeddings"], [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.assertIsInstance(result["embeddings"][0], list)
        self.assertEqual(cast(dict[str, object], result)["vendor_extra"], 1)

    def test_string_vector_raises_malformed(self):
        payload = cast(EmbedResponse, {**FIXED_EMBED, "embeddings": ["ab"]})
        adapter = CountingFakeAdapter(
            embed=lambda _m, _i, _c: payload,
            embed_models=lambda: {"string-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        with (
            self.assertLogs(LOGGER, "WARNING"),
            self.assertRaises(MalformedResponseError),
        ):
            provider.embed("string-model", "hi")

    def test_count_mismatch_raises_malformed(self):
        payload: EmbedResponse = {**FIXED_EMBED, "embeddings": [[0.1]]}
        adapter = CountingFakeAdapter(
            embed=lambda _m, _i, _c: payload,
            embed_models=lambda: {"count-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        with (
            self.assertLogs(LOGGER, "WARNING"),
            self.assertRaises(MalformedResponseError),
        ):
            provider.embed("count-model", ["a", "b"])

    def test_empty_embeddings_for_nonempty_input_raises(self):
        payload: EmbedResponse = {**FIXED_EMBED, "embeddings": []}
        adapter = CountingFakeAdapter(
            embed=lambda _m, _i, _c: payload,
            embed_models=lambda: {"empty-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        with (
            self.assertLogs(LOGGER, "WARNING"),
            self.assertRaises(MalformedResponseError),
        ):
            provider.embed("empty-model", "hi")

    def test_embed_override_without_embed_models_is_not_auto_matched(self):
        trap = CountingFakeAdapter(
            embed=lambda _m, _i, _c: FIXED_EMBED,
        )
        good = CountingFakeAdapter(
            embed=lambda _m, _i, _c: FIXED_EMBED,
            embed_models=lambda: {"trap-model"},
        )
        provider = Provider(adapters={"trap": trap, "good": good})
        with self.assertLogs(LOGGER, "WARNING"):
            result = provider.embed("trap-model", "hi")
        self.assertEqual(result, FILLED_EMBED)
        self.assertEqual(trap.embed_calls, 0)
        self.assertEqual(good.embed_calls, 1)
        result2 = provider.embed("trap-model", "hi", provider="trap")
        self.assertEqual(result2, FILLED_EMBED)
        self.assertEqual(trap.embed_calls, 1)

    def test_verb_isolation_text_embedding_3_small(self):
        openai_like = CountingFakeAdapter(
            models=lambda: {"gpt-4o"},
            embed_models=lambda: {"text-embedding-3-small"},
            embed=lambda _m, _i, _c: FIXED_EMBED,
        )
        provider = Provider(adapters={"openai-like": openai_like})
        with self.assertLogs(LOGGER, "WARNING"):
            result = provider.embed("text-embedding-3-small", "hi")
        self.assertEqual(result, FILLED_EMBED)
        with self.assertRaises(KeyError):
            provider.chat("text-embedding-3-small", MESSAGES)

    def test_embed_and_chat_caches_never_cross_pollute(self):
        adapter = CountingFakeAdapter(
            models=lambda: {"chat-model"},
            embed_models=lambda: {"embed-model"},
            embed=lambda _m, _i, _c: FIXED_EMBED,
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.chat("chat-model", MESSAGES)
        self.assertEqual(provider._auto_match_cache, {"chat-model": "fake"})
        self.assertEqual(provider._auto_match_embed_cache, {})
        self.assertEqual(adapter.models_calls, 1)
        self.assertEqual(adapter.embed_models_calls, 0)
        with self.assertLogs(LOGGER, "WARNING"):
            provider.embed("embed-model", "hi")
        self.assertEqual(provider._auto_match_embed_cache, {"embed-model": "fake"})
        self.assertEqual(provider._auto_match_cache, {"chat-model": "fake"})
        provider.chat("chat-model", MESSAGES)
        provider.embed("embed-model", "hi")
        self.assertEqual(adapter.models_calls, 1)
        self.assertEqual(adapter.embed_models_calls, 1)
        second = CountingFakeAdapter(
            models=lambda: {"chat-model-2"},
            embed_models=lambda: {"embed-model-2"},
            embed=lambda _m, _i, _c: FIXED_EMBED,
        )
        provider2 = Provider(adapters={"fake": second})
        with self.assertLogs(LOGGER, "WARNING"):
            provider2.embed("embed-model-2", "hi")
        self.assertEqual(provider2._auto_match_embed_cache, {"embed-model-2": "fake"})
        self.assertEqual(provider2._auto_match_cache, {})
        with self.assertLogs(LOGGER, "WARNING"):
            provider2.chat("chat-model-2", MESSAGES)
        self.assertEqual(provider2._auto_match_cache, {"chat-model-2": "fake"})
        self.assertEqual(provider2._auto_match_embed_cache, {"embed-model-2": "fake"})

    def test_adapter_raising_during_embed_models_is_skipped(self):
        broken = CountingFakeAdapter(
            embed_models=lambda: raise_(RuntimeError("boom")),
        )
        fine = CountingFakeAdapter(
            embed_models=lambda: {"ok-model"},
            embed=lambda _m, _i, _c: FIXED_EMBED,
        )
        provider = Provider(adapters={"broken": broken, "fine": fine})
        with self.assertLogs(LOGGER, "WARNING"):
            result = provider.embed("ok-model", "hi")
        self.assertEqual(result, FILLED_EMBED)
        self.assertEqual(broken.embed_models_calls, 1)
        self.assertEqual(fine.embed_calls, 1)

    def test_404_through_cached_embed_evicts_only_embed_cache(self):
        calls = {"n": 0}

        def embed(
            model: str, input: list[str], config: dict[str, Any] | None = None
        ) -> EmbedResponse:
            calls["n"] += 1
            if calls["n"] == 2:
                raise APIError("gone", status=404)
            return FIXED_EMBED

        adapter = CountingFakeAdapter(
            embed=embed,
            embed_models=lambda: {"evict-model"},
            models=lambda: {"chat-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.chat("chat-model", MESSAGES)
        with self.assertLogs(LOGGER, "WARNING"):
            provider.embed("evict-model", "hi")
        self.assertIn("evict-model", provider._auto_match_embed_cache)
        self.assertIn("chat-model", provider._auto_match_cache)
        with self.assertRaises(APIError):
            provider.embed("evict-model", "hi")
        self.assertNotIn("evict-model", provider._auto_match_embed_cache)
        self.assertIn("chat-model", provider._auto_match_cache)

    def test_auth_error_through_cached_embed_evicts_embed_cache(self):
        adapter = CountingFakeAdapter(
            embed=lambda _m, _i, _c: raise_(AuthError("no key", status=401)),
            embed_models=lambda: {"secure-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(AuthError):
            provider.embed("secure-model", "hi")
        self.assertNotIn("secure-model", provider._auto_match_embed_cache)

    def test_unreachable_through_cached_embed_evicts_embed_cache(self):
        adapter = CountingFakeAdapter(
            embed=lambda _m, _i, _c: raise_(
                _connection_error(urllib.error.URLError(ConnectionRefusedError()))
            ),
            embed_models=lambda: {"u-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(APIError):
            provider.embed("u-model", "hi")
        self.assertNotIn("u-model", provider._auto_match_embed_cache)

    def test_429_and_500_do_not_evict_embed_cache(self):
        for exc in (
            RateLimitError("limited", status=429),
            ServerError("down", status=500),
        ):
            with self.subTest(type=type(exc).__name__):
                adapter = CountingFakeAdapter(
                    embed=lambda _m, _i, _c, exc=exc: raise_(exc),
                    embed_models=lambda: {"keep-model"},
                )
                provider = Provider(adapters={"fake": adapter})
                with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(type(exc)):
                    provider.embed("keep-model", "hi")
                self.assertIn("keep-model", provider._auto_match_embed_cache)

    def test_explicit_provider_never_evicts(self):
        for exc in (
            APIError("gone", status=404),
            AuthError("no key", status=401),
            _connection_error(urllib.error.URLError(ConnectionRefusedError())),
        ):
            with self.subTest(type=type(exc).__name__):
                adapter = CountingFakeAdapter(
                    embed=lambda _m, _i, _c, exc=exc: raise_(exc),
                    embed_models=lambda: {"explicit-model"},
                )
                provider = Provider(adapters={"fake": adapter})
                provider._auto_match_embed_cache["explicit-model"] = "fake"
                with self.assertRaises(type(exc)):
                    provider.embed("explicit-model", "hi", provider="fake")
                self.assertIn("explicit-model", provider._auto_match_embed_cache)

    def test_key_error_lists_every_available_provider_including_empty(self):
        empty = CountingFakeAdapter(embed_models=lambda: set())
        other = CountingFakeAdapter(embed_models=lambda: {"other-model"})
        provider = Provider(adapters={"empty": empty, "other": other})
        with self.assertRaises(KeyError) as ctx:
            provider.embed("missing-model", "hi")
        msg = str(ctx.exception)
        self.assertIn("'empty'", msg)
        self.assertIn("'other'", msg)

    def test_embed_raising_propagates_unwrapped(self):
        adapter = CountingFakeAdapter(
            embed=lambda _m, _i, _c: raise_(RuntimeError("boom")),
            embed_models=lambda: {"embed-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(RuntimeError):
            provider.embed("embed-model", "hi")


class TestEmbedInputValidation(unittest.TestCase):
    """Input validation before any adapter I/O."""

    def _provider_with_recording(self) -> tuple[Provider, CountingFakeAdapter]:
        adapter = CountingFakeAdapter(
            embed=lambda _m, _i, _c: FIXED_EMBED,
            embed_models=lambda: {"m"},
        )
        provider = Provider(adapters={"fake": adapter})
        return provider, adapter

    def test_empty_list_raises_before_adapter(self):
        provider, adapter = self._provider_with_recording()
        with (
            patch("urllib.request.urlopen", side_effect=_fail_urlopen),
            self.assertRaises(ValueError),
        ):
            provider.embed("m", [])
        self.assertEqual(adapter.embed_calls, 0)

    def test_non_str_element_raises_before_adapter(self):
        provider, adapter = self._provider_with_recording()
        with (
            patch("urllib.request.urlopen", side_effect=_fail_urlopen),
            self.assertRaises(TypeError),
        ):
            provider.embed("m", ["hi", 123])  # ty: ignore[invalid-argument-type]
        self.assertEqual(adapter.embed_calls, 0)

    def test_empty_string_bare_raises_before_adapter(self):
        provider, adapter = self._provider_with_recording()
        with (
            patch("urllib.request.urlopen", side_effect=_fail_urlopen),
            self.assertRaises(ValueError),
        ):
            provider.embed("m", "")
        self.assertEqual(adapter.embed_calls, 0)

    def test_empty_string_inside_list_raises_before_adapter(self):
        provider, adapter = self._provider_with_recording()
        with (
            patch("urllib.request.urlopen", side_effect=_fail_urlopen),
            self.assertRaises(ValueError),
        ):
            provider.embed("m", ["hi", ""])
        self.assertEqual(adapter.embed_calls, 0)

    def test_dict_raises_before_adapter(self):
        provider, adapter = self._provider_with_recording()
        with (
            patch("urllib.request.urlopen", side_effect=_fail_urlopen),
            self.assertRaises(TypeError),
        ):
            provider.embed("m", {"a": "b"})  # ty: ignore[invalid-argument-type]
        self.assertEqual(adapter.embed_calls, 0)

    def test_bytes_raises_before_adapter(self):
        provider, adapter = self._provider_with_recording()
        with (
            patch("urllib.request.urlopen", side_effect=_fail_urlopen),
            self.assertRaises(TypeError),
        ):
            provider.embed("m", b"hi")  # ty: ignore[invalid-argument-type]
        self.assertEqual(adapter.embed_calls, 0)

    def test_bytearray_raises_before_adapter(self):
        provider, adapter = self._provider_with_recording()
        with (
            patch("urllib.request.urlopen", side_effect=_fail_urlopen),
            self.assertRaises(TypeError),
        ):
            provider.embed("m", bytearray(b"hi"))  # ty: ignore[invalid-argument-type]
        self.assertEqual(adapter.embed_calls, 0)

    def test_memoryview_and_set_raise_before_adapter(self):
        provider, adapter = self._provider_with_recording()
        for value in (memoryview(b"hi"), {"a", "b"}, frozenset({"a"})):
            with (
                self.subTest(type=type(value).__name__),
                patch("urllib.request.urlopen", side_effect=_fail_urlopen),
                self.assertRaises(TypeError),
            ):
                provider.embed("m", value)  # ty: ignore[invalid-argument-type]
        self.assertEqual(adapter.embed_calls, 0)

    def test_str_normalizes_to_one_element_list(self):
        seen: list[list[str]] = []

        def capture(
            model: str, input: list[str], config: dict[str, Any] | None = None
        ) -> EmbedResponse:
            seen.append(list(input))
            return FIXED_EMBED

        adapter = CountingFakeAdapter(embed=capture, embed_models=lambda: {"m"})
        provider = Provider(adapters={"fake": adapter})
        result = provider.embed("m", "hello")
        self.assertEqual(result["embeddings"], FIXED_EMBED["embeddings"])
        self.assertEqual(seen, [["hello"]])

    def test_tuple_is_accepted_and_passed_as_list(self):
        seen: list[list[str]] = []
        payload: EmbedResponse = {
            **FIXED_EMBED,
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        }

        def capture(
            model: str, input: list[str], config: dict[str, Any] | None = None
        ) -> EmbedResponse:
            seen.append(input)
            self.assertIsInstance(input, list)
            return payload

        adapter = CountingFakeAdapter(embed=capture, embed_models=lambda: {"m"})
        provider = Provider(adapters={"fake": adapter})
        provider.embed("m", ("a", "b"))
        self.assertEqual(seen, [["a", "b"]])

    def test_generator_is_materialized(self):
        seen: list[list[str]] = []
        payload: EmbedResponse = {
            **FIXED_EMBED,
            "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        }

        def capture(
            model: str, input: list[str], config: dict[str, Any] | None = None
        ) -> EmbedResponse:
            seen.append(input)
            return payload

        adapter = CountingFakeAdapter(embed=capture, embed_models=lambda: {"m"})
        provider = Provider(adapters={"fake": adapter})
        result = provider.embed("m", (t for t in ("a", "b")))  # ty: ignore[invalid-argument-type]
        self.assertEqual(seen, [["a", "b"]])
        self.assertEqual(result["dimensions"], 3)


class TestKindMismatch(unittest.TestCase):
    """Kind-mismatch between chat and embed via auto-match."""

    def test_embed_on_chat_only_raises_key_error_and_leaves_caches_empty(self):
        """embed() on chat-only id raises KeyError."""
        adapter = CountingFakeAdapter(
            models=lambda: {"chat-only"}, embed_models=lambda: set()
        )
        provider = Provider(adapters={"chat": adapter})
        with self.assertRaises(KeyError):
            provider.embed("chat-only", "hi")
        self.assertEqual(provider._auto_match_cache, {})
        self.assertEqual(provider._auto_match_embed_cache, {})

    def test_chat_on_embed_only_raises_key_error(self):
        """chat() on embed-only id raises KeyError."""
        adapter = CountingFakeAdapter(
            models=lambda: set(), embed_models=lambda: {"embed-only"}
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertRaises(KeyError):
            provider.chat("embed-only", MESSAGES)
        self.assertEqual(provider._auto_match_cache, {})
        self.assertEqual(provider._auto_match_embed_cache, {})

    def test_stream_chat_on_embed_only_raises_key_error(self):
        """stream_chat() on embed-only id raises KeyError."""
        adapter = CountingFakeAdapter(
            models=lambda: set(), embed_models=lambda: {"embed-only"}
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertRaises(KeyError):
            list(provider.stream_chat("embed-only", MESSAGES))
        self.assertEqual(provider._auto_match_cache, {})
        self.assertEqual(provider._auto_match_embed_cache, {})

    def test_model_info_on_embed_only_raises_key_error(self):
        """model_info() on embed-only id raises KeyError."""
        adapter = CountingFakeAdapter(
            models=lambda: set(), embed_models=lambda: {"embed-only"}
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertRaises(KeyError):
            provider.model_info("embed-only")
        self.assertEqual(provider._auto_match_cache, {})
        self.assertEqual(provider._auto_match_embed_cache, {})

    def test_embed_only_ids_through_chat_all_raise_key_error(self):
        """Parametrized embed-only ids through chat verbs raise KeyError."""
        cases = ["embed-only", "text-embedding-3-small", "nomic-embed-text:latest"]
        for model in cases:
            with self.subTest(model=model):
                adapter = CountingFakeAdapter(
                    models=lambda: {"chat-model"},
                    embed_models=lambda m=model: {m},
                )
                provider = Provider(adapters={"fake": adapter})
                with self.assertRaises(KeyError):
                    provider.chat(model, MESSAGES)
                with self.assertRaises(KeyError):
                    list(provider.stream_chat(model, MESSAGES))
                with self.assertRaises(KeyError):
                    provider.model_info(model)
                self.assertEqual(provider._auto_match_cache, {})
                self.assertEqual(provider._auto_match_embed_cache, {})

    def test_explicit_provider_embed_on_chat_only_surfaces_404(self):
        """Explicit embed on chat-only id issues request and surfaces 404."""
        adapter = CountingFakeAdapter(
            models=lambda: {"chat-only"},
            embed=lambda _m, _i, _c: raise_(APIError("not found", status=404)),
            embed_models=lambda: set(),
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertRaises(APIError) as ctx:
            provider.embed("chat-only", "hi", provider="fake")
        self.assertEqual(ctx.exception.status, 404)
        self.assertEqual(adapter.embed_calls, 1)
        self.assertEqual(provider._auto_match_cache, {})
        self.assertEqual(provider._auto_match_embed_cache, {})

    def test_explicit_provider_chat_surfaces_404_individually(self):
        """Explicit chat/stream/model_info on embed-only each surface 404."""
        for verb in ("chat", "stream_chat", "model_info"):
            with self.subTest(verb=verb):
                if verb == "chat":

                    def chat() -> Response:
                        raise APIError("not found", status=404)

                    adapter = CountingFakeAdapter(
                        chat=chat,
                        models=lambda: set(),
                        embed_models=lambda: {"embed-only"},
                    )
                    provider = Provider(adapters={"fake": adapter})
                    with self.assertRaises(APIError) as ctx:
                        provider.chat("embed-only", MESSAGES, provider="fake")
                    self.assertEqual(ctx.exception.status, 404)
                elif verb == "stream_chat":

                    def stream() -> Iterator[StreamEvent]:
                        raise APIError("not found", status=404)
                        yield from ()

                    adapter = CountingFakeAdapter(
                        stream=stream,
                        models=lambda: set(),
                        embed_models=lambda: {"embed-only"},
                    )
                    provider = Provider(adapters={"fake": adapter})
                    with self.assertRaises(APIError) as ctx:
                        list(
                            provider.stream_chat(
                                "embed-only", MESSAGES, provider="fake"
                            )
                        )
                    self.assertEqual(ctx.exception.status, 404)
                else:

                    def model_info() -> ModelInfo | None:
                        raise APIError("not found", status=404)

                    adapter = CountingFakeAdapter(
                        model_info=model_info,
                        models=lambda: set(),
                        embed_models=lambda: {"embed-only"},
                    )
                    provider = Provider(adapters={"fake": adapter})
                    with self.assertRaises(APIError) as ctx:
                        provider.model_info("embed-only", provider="fake")
                    self.assertEqual(ctx.exception.status, 404)
                self.assertEqual(provider._auto_match_cache, {})
                self.assertEqual(provider._auto_match_embed_cache, {})

    def test_explicit_provider_404_does_not_evict_existing_cache(self):
        """Explicit 404 never evicts either cache."""
        adapter = CountingFakeAdapter(
            chat=lambda: raise_(APIError("gone", status=404)),
            stream=lambda: raise_(APIError("gone", status=404)),
            embed=lambda _m, _i, _c: raise_(APIError("gone", status=404)),
            models=lambda: {"chat-only"},
            embed_models=lambda: {"embed-only"},
        )
        provider = Provider(adapters={"fake": adapter})
        provider._auto_match_cache["chat-only"] = "fake"
        provider._auto_match_embed_cache["embed-only"] = "fake"
        with self.assertRaises(APIError):
            provider.chat("other", MESSAGES, provider="fake")
        with self.assertRaises(APIError):
            provider.embed("other", "hi", provider="fake")
        with self.assertRaises(APIError):
            list(provider.stream_chat("other", MESSAGES, provider="fake"))
        self.assertEqual(provider._auto_match_cache, {"chat-only": "fake"})
        self.assertEqual(provider._auto_match_embed_cache, {"embed-only": "fake"})


class TestModelsEmbeddingsMode(unittest.IsolatedAsyncioTestCase):
    """models(embeddings=) mode switch."""

    def test_models_default_is_chat_only(self):
        """models() returns chat ids only."""
        adapter = CountingFakeAdapter(
            models=lambda: {"b", "a"}, embed_models=lambda: {"embed-z"}
        )
        provider = Provider(adapters={"fake": adapter})
        self.assertEqual(provider.models(), {"fake": ["a", "b"]})

    def test_models_embeddings_returns_sorted_embed_models(self):
        """models(embeddings=True) returns sorted embed ids."""
        adapter = CountingFakeAdapter(
            models=lambda: {"chat"}, embed_models=lambda: {"z", "a", "m"}
        )
        provider = Provider(adapters={"fake": adapter})
        self.assertEqual(provider.models(embeddings=True), {"fake": ["a", "m", "z"]})

    def test_modes_are_isolated(self):
        """Each mode isolates its ids; other mode shows []."""
        chat_only = CountingFakeAdapter(
            models=lambda: {"chat-id"}, embed_models=lambda: set()
        )
        embed_only = CountingFakeAdapter(
            models=lambda: set(), embed_models=lambda: {"embed-id"}
        )
        provider = Provider(adapters={"chat": chat_only, "embed": embed_only})
        self.assertEqual(provider.models(), {"chat": ["chat-id"], "embed": []})
        self.assertEqual(
            provider.models(embeddings=True), {"chat": [], "embed": ["embed-id"]}
        )
        single = CountingFakeAdapter(
            models=lambda: {"chat-id"}, embed_models=lambda: set()
        )
        provider2 = Provider(adapters={"single": single})
        self.assertEqual(provider2.models(), {"single": ["chat-id"]})
        self.assertEqual(provider2.models(embeddings=True), {"single": []})

    def test_empty_embed_appears_unavailable_and_raising_omitted(self):
        """Empty appears as []; unavailable and raising are omitted."""
        empty = CountingFakeAdapter(embed_models=lambda: set())
        unavailable = CountingFakeAdapter(
            is_available=lambda: False, embed_models=lambda: {"x"}
        )
        raising = CountingFakeAdapter(embed_models=lambda: raise_(RuntimeError("boom")))
        provider = Provider(
            adapters={"empty": empty, "unavailable": unavailable, "raising": raising}
        )
        with self.assertLogs(LOGGER, "WARNING"):
            result = provider.models(embeddings=True)
        self.assertEqual(result, {"empty": []})
        with self.assertLogs(LOGGER, "WARNING"):
            raising_only = Provider(adapters={"raising": raising})
            self.assertEqual(raising_only.models(embeddings=True), {})

    def test_listers_not_called_in_wrong_mode(self):
        """Each mode calls only its lister."""
        adapter = CountingFakeAdapter(
            models=lambda: {"chat"}, embed_models=lambda: {"embed"}
        )
        provider = Provider(adapters={"fake": adapter})
        adapter.models_calls = 0
        adapter.embed_models_calls = 0
        provider.models()
        self.assertEqual(adapter.models_calls, 1)
        self.assertEqual(adapter.embed_models_calls, 0)
        adapter.models_calls = 0
        adapter.embed_models_calls = 0
        provider.models(embeddings=True)
        self.assertEqual(adapter.models_calls, 0)
        self.assertEqual(adapter.embed_models_calls, 1)

    def test_embeddings_flag_is_keyword_only(self):
        """embeddings flag is keyword-only."""
        provider = Provider(adapters={"fake": CountingFakeAdapter()})
        with self.assertRaises(TypeError):
            provider.models(True)  # ty: ignore[too-many-positional-arguments]
        with self.assertRaises(TypeError):
            provider.async_models(True)  # ty: ignore[too-many-positional-arguments, unused-awaitable]

    async def test_async_models_embeddings_matches_sync_and_is_non_blocking(self):
        """async_models(embeddings=True) matches sync and stays non-blocking."""
        adapter = CountingFakeAdapter(
            models=lambda: {"chat"}, embed_models=lambda: {"b", "a"}
        )
        provider = Provider(adapters={"fake": adapter})
        expected = provider.models(embeddings=True)
        self.assertEqual(await provider.async_models(embeddings=True), expected)
        loop_thread = threading.current_thread()
        adapter.embed_models_threads.clear()
        result = await provider.async_models(embeddings=True)
        self.assertEqual(result, expected)
        self.assertEqual(len(adapter.embed_models_threads), 1)
        self.assertNotEqual(adapter.embed_models_threads[-1], loop_thread)

        def slow_embed_models() -> set[str]:
            time.sleep(0.2)
            return {"b", "a"}

        slow = CountingFakeAdapter(embed_models=slow_embed_models)
        slow_provider = Provider(adapters={"slow": slow})
        with LoopTicker() as ticker:
            slow_result = await slow_provider.async_models(embeddings=True)
        self.assertEqual(slow_result, {"slow": ["a", "b"]})
        self.assertGreater(ticker.count, 5)


class TestCapabilitiesSurface(unittest.TestCase):
    """Provider capabilities surface via explicit provider."""

    def test_capabilities_returns_fake_dict_and_supports_tri_state(self):
        """capabilities via explicit provider returns dict and supports projects."""
        caps: Capabilities = {
            "tools": True,
            "vision": False,
            "pdf_input": None,
            "thinking": True,
            "raw": {"capabilities": ["tools", "thinking"]},
        }
        adapter = CountingFakeAdapter(
            capabilities=lambda _m: caps, models=lambda: {"m"}
        )
        provider = Provider(adapters={"fake": adapter})
        result = provider.capabilities("m", provider="fake")
        self.assertEqual(result, caps)
        self.assertTrue(provider.supports("m", "tools", provider="fake"))
        self.assertFalse(provider.supports("m", "vision", provider="fake"))
        self.assertIsNone(provider.supports("m", "pdf_input", provider="fake"))
        self.assertTrue(provider.supports("m", "thinking", provider="fake"))

    def test_capabilities_default_returns_none(self):
        """adapter with no override returns None."""
        adapter = CountingFakeAdapter(models=lambda: {"m"})
        provider = Provider(adapters={"fake": adapter})
        with patch("urllib.request.urlopen", side_effect=_fail_urlopen) as mock:
            self.assertIsNone(provider.capabilities("m", provider="fake"))
            self.assertIsNone(provider.supports("m", "tools", provider="fake"))
            mock.assert_not_called()
        self.assertEqual(adapter.capabilities_calls, 1)

    def test_supports_validation_before_io(self):
        """supports with bad name raises before any I/O."""
        caps: Capabilities = {
            "tools": True,
            "vision": True,
            "pdf_input": None,
            "thinking": None,
        }
        for bad in ("visoin", "embedding"):
            with self.subTest(bad=bad):
                adapter = CountingFakeAdapter(
                    capabilities=lambda _m: caps, models=lambda: {"m"}
                )
                provider = Provider(adapters={"fake": adapter})
                with patch("urllib.request.urlopen", side_effect=_fail_urlopen) as mock:
                    with self.assertRaises(ValueError) as ctx:
                        provider.supports("m", bad, provider="fake")  # ty: ignore[invalid-argument-type]
                    msg = str(ctx.exception)
                    self.assertIn(repr(bad), msg)
                    self.assertIn("pdf_input", msg)
                    self.assertIn("thinking", msg)
                    self.assertIn("tools", msg)
                    self.assertIn("vision", msg)
                    mock.assert_not_called()
                self.assertEqual(adapter.capabilities_calls, 0)
                self.assertEqual(adapter.models_calls, 0)

    def test_unknown_explicit_provider_raises_key_error(self):
        """unknown explicit provider raises KeyError."""
        provider = Provider(adapters={"fake": CountingFakeAdapter()})
        with self.assertRaises(KeyError) as ctx:
            provider.capabilities("m", provider="nope")
        self.assertIn("nope", str(ctx.exception))
        with self.assertRaises(KeyError):
            provider.supports("m", "tools", provider="nope")


class TestCapabilitiesAutoMatch(unittest.TestCase):
    """Auto-match resolution and caching for capabilities."""

    def test_first_available_provider_matched(self):
        """first available provider whose chat models lists model is used."""
        caps_first: Capabilities = {
            "tools": True,
            "vision": True,
            "pdf_input": None,
            "thinking": None,
        }
        caps_second: Capabilities = {
            "tools": False,
            "vision": False,
            "pdf_input": None,
            "thinking": None,
        }
        first = CountingFakeAdapter(
            capabilities=lambda _m: caps_first, models=lambda: {"shared-model"}
        )
        second = CountingFakeAdapter(
            capabilities=lambda _m: caps_second, models=lambda: {"shared-model"}
        )
        provider = Provider(adapters={"first": first, "second": second})
        with self.assertLogs(LOGGER, "WARNING"):
            result = provider.capabilities("shared-model")
        self.assertEqual(result, caps_first)
        self.assertEqual(first.capabilities_calls, 1)
        self.assertEqual(second.capabilities_calls, 0)
        self.assertEqual(second.models_calls, 0)

    def test_second_call_skips_probing(self):
        """second capabilities reuses auto-match cache."""
        caps: Capabilities = {
            "tools": True,
            "vision": None,
            "pdf_input": None,
            "thinking": None,
        }
        adapter = CountingFakeAdapter(
            capabilities=lambda _m: caps, models=lambda: {"fake-model"}
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.capabilities("fake-model")
        self.assertEqual(adapter.models_calls, 1)
        self.assertEqual(adapter.capabilities_calls, 1)
        provider.capabilities("fake-model")
        self.assertEqual(adapter.models_calls, 1)
        self.assertEqual(adapter.capabilities_calls, 1)

    def test_warm_cache_hit_probe_failure_still_raises(self):
        """warm auto-match hit then probe failure still raises."""
        caps: Capabilities = {
            "tools": True,
            "vision": None,
            "pdf_input": None,
            "thinking": None,
        }

        def cap(model: str) -> Capabilities | None:
            if cap.calls == 0:  # ty: ignore[unresolved-attribute]
                cap.calls += 1  # ty: ignore[unresolved-attribute]
                return caps
            raise ServerError("down", status=500)

        cap.calls = 0  # ty: ignore[unresolved-attribute]
        adapter = CountingFakeAdapter(capabilities=cap, models=lambda: {"fake-model"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.capabilities("fake-model")
        self.assertEqual(adapter.models_calls, 1)
        adapter._capabilities_cache.clear()
        with self.assertRaises(ServerError):
            provider.capabilities("fake-model")
        self.assertEqual(adapter.models_calls, 1)
        self.assertEqual(adapter.capabilities_calls, 2)


class TestCapabilitiesErrors(unittest.TestCase):
    """Error matrix for capabilities and supports."""

    def test_auto_match_unknown_raises_key_error(self):
        """auto-match id no provider lists raises KeyError."""
        adapter = CountingFakeAdapter(models=lambda: {"other"})
        provider = Provider(adapters={"fake": adapter})
        with self.assertRaises(KeyError) as ctx:
            provider.capabilities("missing-model")
        self.assertIn("missing-model", str(ctx.exception))
        with self.assertRaises(KeyError):
            provider.supports("missing-model", "tools")

    def test_resolved_but_unknown_returns_none_explicit(self):
        """explicit provider unknown model returns None not KeyError."""
        adapter = CountingFakeAdapter(
            capabilities=lambda _m: None, models=lambda: {"known"}
        )
        provider = Provider(adapters={"fake": adapter})
        with patch("urllib.request.urlopen", side_effect=_fail_urlopen) as mock:
            self.assertIsNone(provider.capabilities("unknown", provider="fake"))
            self.assertIsNone(provider.supports("unknown", "tools", provider="fake"))
            mock.assert_not_called()

    def test_resolved_but_unknown_returns_none_auto_match(self):
        """auto-match resolved but vendor unknown returns None."""
        adapter = CountingFakeAdapter(
            capabilities=lambda _m: None, models=lambda: {"known"}
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            self.assertIsNone(provider.capabilities("known"))
        adapter._capabilities_cache.clear()
        self.assertIsNone(provider.supports("known", "vision"))

    def test_explicit_unknown_returns_none_via_none(self):
        """explicit unknown via None return stays None."""
        adapter = CountingFakeAdapter(models=lambda: {"a"})
        provider = Provider(adapters={"fake": adapter})
        self.assertIsNone(provider.capabilities("missing", provider="fake"))

    def test_error_matrix_raises(self):
        """401 500 connection timeout raise mapped errors not None."""
        errors = [
            (AuthError("no key", status=401), AuthError),
            (ServerError("down", status=500), ServerError),
            (
                _connection_error(urllib.error.URLError(ConnectionRefusedError())),
                APIError,
            ),
            (RequestTimeoutError("slow"), RequestTimeoutError),
        ]
        for exc, typ in errors:
            with self.subTest(type=type(exc).__name__):
                adapter = CountingFakeAdapter(
                    capabilities=lambda _m, exc=exc: raise_(exc),
                    models=lambda: {"m"},
                )
                provider = Provider(adapters={"fake": adapter})
                with patch("urllib.request.urlopen", side_effect=_fail_urlopen):
                    with self.assertRaises(typ):
                        provider.capabilities("m", provider="fake")
                    adapter._capabilities_cache.clear()
                    with self.assertRaises(typ):
                        provider.supports("m", "tools", provider="fake")
                adapter2 = CountingFakeAdapter(
                    capabilities=lambda _m, exc=exc: raise_(exc),
                    models=lambda: {"m"},
                )
                provider2 = Provider(adapters={"fake": adapter2})
                with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(typ):
                    provider2.capabilities("m")

    def test_no_failure_returns_none(self):
        """no failure path returns None for explicit unknown."""
        adapter = CountingFakeAdapter(models=lambda: {"known"})
        provider = Provider(adapters={"fake": adapter})
        self.assertIsNone(provider.capabilities("unknown", provider="fake"))


class TestCapabilitiesKindMismatch(unittest.TestCase):
    """Kind mismatch between chat and embed via capabilities."""

    def test_auto_match_embed_only_raises_key_error(self):
        """capabilities supports model_info on embed-only raise KeyError auto-match."""
        for model in ("embed-only", "text-embedding-3-small"):
            with self.subTest(model=model):
                adapter = CountingFakeAdapter(
                    models=lambda: {"chat-model"},
                    embed_models=lambda m=model: {m},
                    embed=lambda _m, _i, _c: FIXED_EMBED,
                )
                provider = Provider(adapters={"fake": adapter})
                with self.assertRaises(KeyError):
                    provider.capabilities(model)
                with self.assertRaises(KeyError):
                    provider.supports(model, "tools")
                with self.assertRaises(KeyError):
                    provider.model_info(model)
                with self.assertLogs(LOGGER, "WARNING"):
                    result = provider.embed(model, "hi")
                self.assertEqual(result, FILLED_EMBED)
                self.assertEqual(provider._auto_match_cache, {})
                self.assertIn(model, provider._auto_match_embed_cache)

    def test_explicit_provider_returns_adapter_answer(self):
        """explicit provider hands to adapter which returns None for fake."""
        for model in ("embed-only", "text-embedding-3-small"):
            with self.subTest(model=model):
                adapter = CountingFakeAdapter(
                    models=lambda: set(),
                    embed_models=lambda m=model: {m},
                    embed=lambda _m, _i, _c: FIXED_EMBED,
                )
                provider = Provider(adapters={"fake": adapter})
                self.assertIsNone(provider.capabilities(model, provider="fake"))
                self.assertIsNone(provider.supports(model, "tools", provider="fake"))

    def test_embed_only_ids_through_chat_all_raise_key_error(self):
        """parametrized embed-only ids through capabilities all raise KeyError."""
        cases = ["embed-only", "text-embedding-3-small", "nomic-embed-text:latest"]
        for model in cases:
            with self.subTest(model=model):
                adapter = CountingFakeAdapter(
                    models=lambda: {"chat-model"},
                    embed_models=lambda m=model: {m},
                )
                provider = Provider(adapters={"fake": adapter})
                with self.assertRaises(KeyError):
                    provider.capabilities(model)
                with self.assertRaises(KeyError):
                    provider.supports(model, "vision")
                with self.assertRaises(KeyError):
                    provider.model_info(model)
                self.assertEqual(provider._auto_match_cache, {})
                self.assertEqual(provider._auto_match_embed_cache, {})


class TestCapabilitiesPropagation(unittest.TestCase):
    """Non-probe exceptions propagate unwrapped."""

    def test_capabilities_raising_propagates_unwrapped(self):
        """capabilities RuntimeError propagates unwrapped."""
        adapter = CountingFakeAdapter(
            capabilities=lambda _m: raise_(RuntimeError("boom")),
            models=lambda: {"fake-model"},
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(RuntimeError) as ctx:
            provider.capabilities("fake-model")
        self.assertEqual(str(ctx.exception), "boom")
        adapter._capabilities_cache.clear()
        with self.assertRaises(RuntimeError):
            provider.supports("fake-model", "tools")
        with self.assertRaises(RuntimeError):
            provider.capabilities("fake-model", provider="fake")

    def test_model_info_raising_still_propagates(self):
        """sanity that model_info raising still unwrapped."""
        adapter = CountingFakeAdapter(
            models=lambda: {"fake-model"},
            model_info=lambda: raise_(RuntimeError("boom")),
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(RuntimeError):
            provider.model_info("fake-model")


class TestCapabilitiesOpenAI(unittest.TestCase):
    """OpenAI default is ABC None with zero network."""

    def test_openai_capabilities_returns_none_with_zero_urlopen(self):
        """OpenAI capabilities returns None with zero urlopen."""
        with patch("urllib.request.urlopen", side_effect=_fail_urlopen) as mock:
            provider = Provider(adapters={"openai": OpenAIAdapter()})
            self.assertIsNone(provider.capabilities("gpt-4o", provider="openai"))
            self.assertIsNone(provider.supports("gpt-4o", "tools", provider="openai"))
            mock.assert_not_called()

    def test_openai_async_capabilities_returns_none_with_zero_urlopen(self):
        """async variant also zero urlopen."""
        with patch("urllib.request.urlopen", side_effect=_fail_urlopen) as mock:
            provider = Provider(adapters={"openai": OpenAIAdapter()})

            async def run() -> None:
                self.assertIsNone(
                    await provider.async_capabilities("gpt-4o", provider="openai")
                )
                self.assertIsNone(
                    await provider.async_supports("gpt-4o", "tools", provider="openai")
                )

            asyncio.run(run())
            mock.assert_not_called()


class TestCapabilitiesCaching(unittest.TestCase):
    """Two calls share one adapter fetch."""

    def test_two_capabilities_one_adapter_call(self):
        """two capabilities for same model hits one adapter call."""
        caps: Capabilities = {
            "tools": True,
            "vision": False,
            "pdf_input": None,
            "thinking": None,
        }
        adapter = CountingFakeAdapter(
            capabilities=lambda _m: caps, models=lambda: {"m"}
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.capabilities("m")
        provider.capabilities("m")
        self.assertEqual(adapter.capabilities_calls, 1)
        self.assertEqual(adapter.models_calls, 1)

    def test_capabilities_then_supports_one_adapter_call(self):
        """capabilities then supports for other keys still one adapter call."""
        caps: Capabilities = {
            "tools": True,
            "vision": True,
            "pdf_input": False,
            "thinking": None,
        }
        adapter = CountingFakeAdapter(
            capabilities=lambda _m: caps, models=lambda: {"m"}
        )
        provider = Provider(adapters={"fake": adapter})
        with self.assertLogs(LOGGER, "WARNING"):
            provider.capabilities("m")
        self.assertEqual(provider.supports("m", "vision"), True)
        self.assertEqual(provider.supports("m", "pdf_input"), False)
        self.assertEqual(adapter.capabilities_calls, 1)


class TestCapabilitiesIntegration(unittest.TestCase):
    """Per-model metadata fake via explicit provider."""

    def test_per_model_metadata(self):
        """one id returns caps, another returns None."""
        caps: Capabilities = {
            "tools": True,
            "vision": True,
            "pdf_input": None,
            "thinking": False,
            "raw": {"capabilities": ["tools", "vision"]},
        }

        def per_model(model: str) -> Capabilities | None:
            if model == "has-caps":
                return caps
            return None

        adapter = CountingFakeAdapter(
            capabilities=per_model, models=lambda: {"has-caps", "no-caps"}
        )
        provider = Provider(adapters={"fake": adapter})
        self.assertEqual(provider.capabilities("has-caps", provider="fake"), caps)
        self.assertTrue(provider.supports("has-caps", "tools", provider="fake"))
        self.assertIsNone(provider.capabilities("no-caps", provider="fake"))
        self.assertIsNone(provider.supports("no-caps", "tools", provider="fake"))


class TestCapabilityEviction(unittest.TestCase):
    """A chat failure that evicts auto-match also drops that model's caps entry."""

    def _provider(self, invalidated: list[str], **kwargs: Any) -> Provider:
        class Recording(CountingFakeAdapter):
            def _invalidate_model_capabilities(self, model: str) -> None:
                invalidated.append(model)

        adapter = Recording(**kwargs)
        return Provider(adapters={"fake": adapter})

    def test_evicting_failures_drop_the_model_entry(self):
        for exc, should in [
            (APIError("gone", status=404), True),
            (AuthError("no key", status=401), True),
            (_connection_error(urllib.error.URLError(ConnectionRefusedError())), True),
            (RateLimitError("slow", status=429), False),
            (ServerError("down", status=500), False),
        ]:
            with self.subTest(type=type(exc).__name__):
                invalidated: list[str] = []
                provider = self._provider(
                    invalidated,
                    chat=lambda exc=exc: raise_(exc),
                    models=lambda: {"m"},
                )
                with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(type(exc)):
                    provider.chat("m", MESSAGES)
                self.assertEqual(invalidated, ["m"] if should else [])

    def test_explicit_provider_never_drops_caps(self):
        invalidated: list[str] = []
        provider = self._provider(
            invalidated,
            chat=lambda: raise_(APIError("gone", status=404)),
            models=lambda: {"m"},
        )
        with self.assertRaises(APIError):
            provider.chat("m", MESSAGES, provider="fake")
        self.assertEqual(invalidated, [])

    def test_probe_failure_does_not_drop_caps(self):
        invalidated: list[str] = []
        provider = self._provider(
            invalidated,
            capabilities=lambda _m: raise_(AuthError("no key", status=401)),
            models=lambda: {"m"},
        )
        with self.assertRaises(AuthError):
            provider.capabilities("m", provider="fake")
        self.assertEqual(invalidated, [])

    def test_embed_failure_does_not_drop_caps(self):
        invalidated: list[str] = []
        provider = self._provider(
            invalidated,
            embed=lambda _m, _i, _c: raise_(APIError("gone", status=404)),
            embed_models=lambda: {"em"},
            models=lambda: {"m"},
        )
        with self.assertLogs(LOGGER, "WARNING"), self.assertRaises(APIError):
            provider.embed("em", "hi")
        self.assertEqual(invalidated, [])


class TestCapabilitiesAsync(unittest.IsolatedAsyncioTestCase):
    """Async variants mirror sync."""

    async def test_async_capabilities_matches_sync(self):
        """async_capabilities returns same as sync."""
        caps: Capabilities = {
            "tools": True,
            "vision": None,
            "pdf_input": None,
            "thinking": None,
        }
        adapter = CountingFakeAdapter(
            capabilities=lambda _m: caps, models=lambda: {"m"}
        )
        provider = Provider(adapters={"fake": adapter})
        self.assertEqual(await provider.async_capabilities("m", provider="fake"), caps)
        self.assertTrue(await provider.async_supports("m", "tools", provider="fake"))

    async def test_async_supports_validation_before_io(self):
        """async supports validation before I/O."""
        adapter = CountingFakeAdapter(models=lambda: {"m"})
        provider = Provider(adapters={"fake": adapter})
        with patch("urllib.request.urlopen", side_effect=_fail_urlopen) as mock:
            with self.assertRaises(ValueError):
                await provider.async_supports("m", "visoin", provider="fake")  # ty: ignore[invalid-argument-type]
            mock.assert_not_called()
        self.assertEqual(adapter.capabilities_calls, 0)

    async def test_async_warm_cache_hit_still_raises(self):
        """async warm cache hit still raises probe failure."""
        caps: Capabilities = {
            "tools": True,
            "vision": None,
            "pdf_input": None,
            "thinking": None,
        }

        def cap(model: str) -> Capabilities | None:
            if cap.calls == 0:  # ty: ignore[unresolved-attribute]
                cap.calls += 1  # ty: ignore[unresolved-attribute]
                return caps
            raise ServerError("down", status=500)

        cap.calls = 0  # ty: ignore[unresolved-attribute]
        adapter = CountingFakeAdapter(capabilities=cap, models=lambda: {"m"})
        provider = Provider(adapters={"fake": adapter})
        await provider.async_capabilities("m")
        adapter._capabilities_cache.clear()
        with self.assertRaises(ServerError):
            await provider.async_capabilities("m")
