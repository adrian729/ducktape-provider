import asyncio
import concurrent.futures
import contextlib
import contextvars
import gc
import inspect
import sys
import threading
import time
import unittest
from collections.abc import Callable, Iterator
from typing import Any, Self
from unittest.mock import patch

from ducktape_provider.adapter import Adapter
from ducktape_provider.errors import APIError, UnsupportedOperationError
from ducktape_provider.provider import _STREAM_BUFFER_SIZE, Provider
from ducktape_provider.types import (
    EmbedResponse,
    Message,
    ModelInfo,
    Response,
    StreamEvent,
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
    {"type": "text_delta", "index": 0, "text": " there"},
    {"type": "block_stop", "index": 0},
    {"type": "message_stop", "response": FIXED_RESPONSE},
]

FIXED_EMBED: EmbedResponse = {
    "embeddings": [[0.1, 0.2, 0.3]],
    "usage": {"input_tokens": 1},
    "raw": {},
    "latency_ms": 0.0,
}

MESSAGES: list[Message] = [
    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
]

WAIT_SECONDS = 5.0
LOGGER = "ducktape_provider.provider"
POLL_SECONDS = "ducktape_provider.provider._READER_POLL_SECONDS"


def text_event(text: str) -> StreamEvent:
    return {"type": "text_delta", "index": 0, "text": text}


class FakeAdapter(Adapter):
    """Adapter whose behavior each test scripts through plain callables."""

    def __init__(
        self,
        chat: Callable[[], Response] = lambda: FIXED_RESPONSE,
        stream: Callable[[], Iterator[StreamEvent]] = lambda: iter(FIXED_STREAM),
        is_available: Callable[[], bool] = lambda: True,
        models: Callable[[], set[str]] = lambda: {"fake-model"},
        model_info: Callable[[], ModelInfo | None] = lambda: None,
        embed: Callable[[str, list[str], dict[str, Any] | None], EmbedResponse]
        | None = None,
        embed_models: Callable[[], set[str]] = lambda: set(),
    ):
        self._chat = chat
        self._stream = stream
        self._is_available = is_available
        self._models = models
        self._model_info = model_info
        self._embed = embed
        self._embed_models = embed_models
        self.embed_calls = 0
        self.embed_models_calls = 0
        self.embed_models_threads: list[threading.Thread] = []

    def is_available(self) -> bool:
        return self._is_available()

    def models(self) -> set[str]:
        return self._models()

    def model_info(self, model: str) -> ModelInfo | None:
        return self._model_info()

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
        if self._embed is not None:
            return self._embed(model, input, config)
        return super().embed(model, input, config)

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response:
        return self._chat()

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        return self._stream()


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


async def wait_for_event(event: threading.Event) -> bool:
    return await asyncio.to_thread(event.wait, WAIT_SECONDS)


def raise_(exc: BaseException) -> Any:
    raise exc


def endless_stream(closed: threading.Event) -> Callable[[], Iterator[StreamEvent]]:
    def stream() -> Iterator[StreamEvent]:
        try:
            while True:
                yield text_event("x")
        finally:
            closed.set()

    return stream


def stream_until_blocked(
    blocked: threading.Event, closed: threading.Event, counter: list[int]
) -> Callable[[], Iterator[StreamEvent]]:
    """An endless stream that sets `blocked` when it yields the event that can't fit.

    The consumer takes one event, so the buffer's slots plus that one released
    slot admit _STREAM_BUFFER_SIZE + 1 events; the next one blocks the reader.
    """

    def stream() -> Iterator[StreamEvent]:
        try:
            while True:
                counter[0] += 1
                if counter[0] == _STREAM_BUFFER_SIZE + 2:
                    blocked.set()
                yield text_event("x")
        finally:
            closed.set()

    return stream


def more_than_the_buffer() -> Iterator[StreamEvent]:
    for _ in range(_STREAM_BUFFER_SIZE * 3):
        yield text_event("x")


class FailingIterator:
    """A non-generator iterator whose close() also fails, as a buggy adapter's might."""

    def __init__(self, error: BaseException | None) -> None:
        self._error = error

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> StreamEvent:
        if self._error is not None:
            raise self._error
        raise StopIteration

    def close(self) -> None:
        raise RuntimeError("close broke")


class TestProviderAsyncChat(unittest.IsolatedAsyncioTestCase):
    async def test_async_chat_matches_sync_chat(self):
        provider = Provider(adapters={"fake": FakeAdapter()})
        expected = provider.chat("fake-model", MESSAGES, provider="fake")
        actual = await provider.async_chat("fake-model", MESSAGES, provider="fake")
        self.assertEqual(actual, expected)

    async def test_async_chat_does_not_block_event_loop(self):
        def slow_chat() -> Response:
            time.sleep(0.2)
            return FIXED_RESPONSE

        provider = Provider(adapters={"slow": FakeAdapter(chat=slow_chat)})
        with LoopTicker() as ticker:
            await provider.async_chat("fake-model", MESSAGES, provider="slow")
        self.assertGreater(ticker.count, 5)

    async def test_async_chat_propagates_error(self):
        failing = FakeAdapter(chat=lambda: raise_(APIError("boom", status=500)))
        provider = Provider(adapters={"failing": failing})
        with self.assertRaises(APIError):
            await provider.async_chat("fake-model", MESSAGES, provider="failing")

    async def test_async_model_info_matches_sync_model_info(self):
        info: ModelInfo = {"context_window": 100, "max_output_tokens": 10}
        provider = Provider(adapters={"fake": FakeAdapter(model_info=lambda: info)})
        expected = provider.model_info("fake-model", provider="fake")
        actual = await provider.async_model_info("fake-model", provider="fake")
        self.assertEqual(actual, expected)
        self.assertEqual(actual, info)

    async def test_async_model_info_does_not_block_event_loop(self):
        def slow_model_info() -> ModelInfo | None:
            time.sleep(0.2)
            return None

        provider = Provider(adapters={"slow": FakeAdapter(model_info=slow_model_info)})
        with LoopTicker() as ticker:
            await provider.async_model_info("fake-model", provider="slow")
        self.assertGreater(ticker.count, 5)

    async def test_async_model_info_propagates_error(self):
        failing = FakeAdapter(model_info=lambda: raise_(RuntimeError("boom")))
        provider = Provider(adapters={"failing": failing})
        with self.assertRaises(RuntimeError):
            await provider.async_model_info("fake-model", provider="failing")

    async def test_async_model_info_resolves_auto_match_off_the_loop(self):
        provider = Provider(adapters={"fake": FakeAdapter()})
        with self.assertLogs("ducktape_provider.provider", "WARNING"):
            result = await provider.async_model_info("fake-model")
        self.assertIsNone(result)

    async def test_async_chat_propagates_contextvars(self):
        request_id = contextvars.ContextVar("request_id", default="unset")
        seen: list[str] = []

        def chat() -> Response:
            seen.append(request_id.get())
            return FIXED_RESPONSE

        provider = Provider(adapters={"fake": FakeAdapter(chat=chat)})
        request_id.set("abc")
        await provider.async_chat("fake-model", MESSAGES, provider="fake")
        self.assertEqual(seen, ["abc"])


class TestProviderAsyncStream(unittest.IsolatedAsyncioTestCase):
    async def test_async_stream_chat_matches_sync_stream_chat(self):
        provider = Provider(adapters={"fake": FakeAdapter()})
        expected = list(provider.stream_chat("fake-model", MESSAGES, provider="fake"))
        actual = [
            event
            async for event in provider.async_stream_chat(
                "fake-model", MESSAGES, provider="fake"
            )
        ]
        self.assertEqual(actual, expected)

    async def test_async_stream_chat_propagates_mid_stream_error(self):
        def stream() -> Iterator[StreamEvent]:
            yield text_event("hi")
            raise APIError("stream broke", status=500, body="mid-stream failure")

        provider = Provider(adapters={"failing": FakeAdapter(stream=stream)})
        received: list[StreamEvent] = []
        with self.assertRaises(APIError):
            async for event in provider.async_stream_chat(
                "fake-model", MESSAGES, provider="failing"
            ):
                received.append(event)
        self.assertEqual(received, [text_event("hi")])

    async def test_async_stream_chat_propagates_error_from_eager_adapter(self):
        failing = FakeAdapter(stream=lambda: raise_(APIError("refused", status=401)))
        provider = Provider(adapters={"failing": failing})
        with self.assertRaises(APIError):
            async for _event in provider.async_stream_chat(
                "fake-model", MESSAGES, provider="failing"
            ):
                pass

    async def test_generator_stream_does_not_block_event_loop(self):
        def stream() -> Iterator[StreamEvent]:
            for text in ("a", "b"):
                time.sleep(0.1)
                yield text_event(text)

        provider = Provider(adapters={"slow": FakeAdapter(stream=stream)})
        with LoopTicker() as ticker:
            async for _event in provider.async_stream_chat(
                "fake-model", MESSAGES, provider="slow"
            ):
                pass
        self.assertGreater(ticker.count, 5)

    async def test_eager_stream_is_opened_off_the_event_loop(self):
        opened_on: list[int] = []

        def eager_stream() -> Iterator[StreamEvent]:
            opened_on.append(threading.get_ident())
            time.sleep(0.2)
            return iter(FIXED_STREAM)

        provider = Provider(adapters={"eager": FakeAdapter(stream=eager_stream)})
        with LoopTicker() as ticker:
            stream = provider.async_stream_chat(
                "fake-model", MESSAGES, provider="eager"
            )
            self.assertEqual(opened_on, [])
            events = [event async for event in stream]
        self.assertEqual(events, FIXED_STREAM)
        self.assertNotEqual(opened_on, [threading.get_ident()])
        self.assertGreater(ticker.count, 5)

    async def test_early_break_closes_generator_off_the_event_loop(self):
        closed = threading.Event()
        closed_on: list[int] = []

        def stream() -> Iterator[StreamEvent]:
            try:
                while True:
                    yield text_event("x")
            finally:
                closed_on.append(threading.get_ident())
                closed.set()

        provider = Provider(adapters={"fake": FakeAdapter(stream=stream)})
        async for _event in provider.async_stream_chat(
            "fake-model", MESSAGES, provider="fake"
        ):
            break
        self.assertTrue(await wait_for_event(closed))
        self.assertNotEqual(closed_on, [threading.get_ident()])

    async def test_aclosing_closes_generator_on_consumer_exception(self):
        closed = threading.Event()

        def stream() -> Iterator[StreamEvent]:
            try:
                while True:
                    yield text_event("x")
            finally:
                closed.set()

        provider = Provider(adapters={"fake": FakeAdapter(stream=stream)})
        with self.assertRaises(RuntimeError):
            async with contextlib.aclosing(
                provider.async_stream_chat("fake-model", MESSAGES, provider="fake")
            ) as events:
                async for _event in events:
                    raise RuntimeError("consumer failed")
        self.assertTrue(await wait_for_event(closed))

    async def test_cancellation_returns_promptly_and_closes_after_read(self):
        first_sent = threading.Event()
        release = threading.Event()
        closed = threading.Event()
        produced_after_release: list[StreamEvent] = []

        def stream() -> Iterator[StreamEvent]:
            try:
                yield text_event("first")
                first_sent.set()
                release.wait(WAIT_SECONDS)
                produced_after_release.append(text_event("second"))
                yield text_event("second")
                yield text_event("never consumed")
            finally:
                closed.set()

        self.addCleanup(release.set)
        provider = Provider(adapters={"fake": FakeAdapter(stream=stream)})
        received: list[StreamEvent] = []

        async def consume() -> None:
            async for event in provider.async_stream_chat(
                "fake-model", MESSAGES, provider="fake"
            ):
                received.append(event)

        task = asyncio.ensure_future(consume())
        self.assertTrue(await wait_for_event(first_sent))
        await asyncio.sleep(0.05)
        started = time.monotonic()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertLess(time.monotonic() - started, 0.5)
        self.assertFalse(closed.is_set())

        release.set()
        self.assertTrue(await wait_for_event(closed))
        self.assertEqual(received, [text_event("first")])
        self.assertEqual(len(produced_after_release), 1)

    async def test_slow_consumer_applies_backpressure(self):
        produced = [0]
        blocked = threading.Event()
        closed = threading.Event()
        stream = stream_until_blocked(blocked, closed, produced)

        provider = Provider(adapters={"fake": FakeAdapter(stream=stream)})
        async with contextlib.aclosing(
            provider.async_stream_chat("fake-model", MESSAGES, provider="fake")
        ) as events:
            async for _event in events:
                self.assertTrue(await wait_for_event(blocked))
                break
        self.assertTrue(await wait_for_event(closed))
        self.assertEqual(produced[0], _STREAM_BUFFER_SIZE + 2)

    async def test_reader_exits_promptly_when_consumer_leaves_a_full_buffer(self):
        produced = [0]
        blocked = threading.Event()
        closed = threading.Event()
        stream = stream_until_blocked(blocked, closed, produced)
        readers: list[threading.Thread] = []

        def recording_stream() -> Iterator[StreamEvent]:
            readers.append(threading.current_thread())
            return stream()

        provider = Provider(adapters={"fake": FakeAdapter(stream=recording_stream)})
        with patch(POLL_SECONDS, WAIT_SECONDS * 10):
            async with contextlib.aclosing(
                provider.async_stream_chat("fake-model", MESSAGES, provider="fake")
            ) as events:
                async for _event in events:
                    self.assertTrue(await wait_for_event(blocked))
                    break
            [reader] = readers
            await asyncio.to_thread(reader.join, WAIT_SECONDS)
        self.assertTrue(closed.is_set())
        self.assertFalse(reader.is_alive())

    async def test_stream_propagates_contextvars(self):
        request_id = contextvars.ContextVar("request_id", default="unset")
        seen: list[str] = []

        def stream() -> Iterator[StreamEvent]:
            seen.append(request_id.get())
            yield from FIXED_STREAM

        provider = Provider(adapters={"fake": FakeAdapter(stream=stream)})
        request_id.set("abc")
        async for _event in provider.async_stream_chat(
            "fake-model", MESSAGES, provider="fake"
        ):
            pass
        self.assertEqual(seen, ["abc"])

    async def test_async_stream_chat_returns_an_async_generator(self):
        provider = Provider(adapters={"fake": FakeAdapter()})
        events = provider.async_stream_chat("fake-model", MESSAGES, provider="fake")
        self.assertTrue(inspect.isasyncgen(events))
        await events.aclose()

    async def test_consumer_can_await_the_executor_mid_stream(self):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.addCleanup(executor.shutdown, wait=False, cancel_futures=True)
        provider = Provider(
            adapters={"fake": FakeAdapter(stream=more_than_the_buffer)},
            executor=executor,
        )

        async def consume() -> int:
            count = 0
            async for _event in provider.async_stream_chat(
                "fake-model", MESSAGES, provider="fake"
            ):
                count += 1
                if count == 1:
                    await provider.async_chat("fake-model", MESSAGES, provider="fake")
            return count

        count = await asyncio.wait_for(consume(), WAIT_SECONDS)
        self.assertEqual(count, _STREAM_BUFFER_SIZE * 3)

    async def test_streams_do_not_starve_the_default_executor(self):
        streams = 2
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=streams)
        asyncio.get_running_loop().set_default_executor(executor)
        provider = Provider(adapters={"fake": FakeAdapter(stream=more_than_the_buffer)})

        async def consume() -> int:
            count = 0
            async for event in provider.async_stream_chat(
                "fake-model", MESSAGES, provider="fake"
            ):
                count += 1
                await asyncio.to_thread(len, event.get("text", ""))
            return count

        counts = await asyncio.wait_for(
            asyncio.gather(*(consume() for _ in range(streams))), WAIT_SECONDS
        )
        self.assertEqual(counts, [_STREAM_BUFFER_SIZE * 3] * streams)

    async def test_garbage_collected_stream_stops_its_reader(self):
        closed = threading.Event()
        provider = Provider(
            adapters={"fake": FakeAdapter(stream=endless_stream(closed))}
        )
        events = provider.async_stream_chat("fake-model", MESSAGES, provider="fake")
        await anext(events)
        del events
        gc.collect()
        self.assertTrue(await wait_for_event(closed))

    async def test_stream_error_wins_over_close_error(self):
        failing = FailingIterator(APIError("stream broke", status=500))
        provider = Provider(adapters={"fake": FakeAdapter(stream=lambda: failing)})
        with (
            self.assertLogs(LOGGER, "WARNING") as logs,
            self.assertRaisesRegex(APIError, "stream broke"),
        ):
            async for _event in provider.async_stream_chat(
                "fake-model", MESSAGES, provider="fake"
            ):
                pass
        self.assertIn("close broke", logs.output[0])

    async def test_close_error_after_clean_stream_is_raised(self):
        failing = FailingIterator(None)
        provider = Provider(adapters={"fake": FakeAdapter(stream=lambda: failing)})
        with self.assertRaisesRegex(RuntimeError, "close broke"):
            async for _event in provider.async_stream_chat(
                "fake-model", MESSAGES, provider="fake"
            ):
                pass

    async def test_reader_stops_when_loop_rejects_handoff(self):
        closed = threading.Event()
        thread_errors: list[threading.ExceptHookArgs] = []
        loop = asyncio.get_running_loop()
        real_call_soon_threadsafe = loop.call_soon_threadsafe

        def reject_handoff(callback: Callable[..., Any], *args: Any, **kwargs: Any):
            if getattr(callback, "__name__", None) == "put_nowait":
                raise RuntimeError("Event loop is closed")
            return real_call_soon_threadsafe(callback, *args, **kwargs)

        provider = Provider(
            adapters={"fake": FakeAdapter(stream=endless_stream(closed))}
        )
        events = provider.async_stream_chat("fake-model", MESSAGES, provider="fake")
        with (
            patch.object(loop, "call_soon_threadsafe", reject_handoff),
            patch.object(threading, "excepthook", thread_errors.append),
        ):
            pending = asyncio.ensure_future(anext(events))
            self.assertTrue(await wait_for_event(closed))
        pending.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertEqual(thread_errors, [])


class CancellableFakeStream:
    """Stands in for a built-in adapter's stream: after its first event it blocks the
    reader thread, and only `cancel()` ends that block — the same shape as a real
    stream parked in a socket read that a force-close has to break out of."""

    def __init__(self) -> None:
        self.cancelled = threading.Event()
        self.closed = threading.Event()
        self.blocked = threading.Event()
        self._sent_first = False

    def __iter__(self) -> Self:
        return self

    def __next__(self) -> StreamEvent:
        if not self._sent_first:
            self._sent_first = True
            return text_event("first")
        self.blocked.set()
        if not self.cancelled.wait(WAIT_SECONDS):
            raise AssertionError("stream was never cancelled")
        raise StopIteration

    def close(self) -> None:
        self.closed.set()

    def cancel(self) -> None:
        self.cancelled.set()


class TestStreamCancelPropagation(unittest.IsolatedAsyncioTestCase):
    """A consumer that stops early must reach the stream's own `cancel()`, not just
    stop reading: without it the reader thread stays parked in its current read."""

    def _provider(self, stream: CancellableFakeStream) -> Provider:
        self.addCleanup(stream.cancel)
        return Provider(adapters={"fake": FakeAdapter(stream=lambda: stream)})

    async def test_task_cancellation_cancels_the_underlying_stream(self):
        stream = CancellableFakeStream()
        provider = self._provider(stream)
        received: list[StreamEvent] = []
        got_first = asyncio.Event()

        async def consume() -> None:
            async for event in provider.async_stream_chat(
                "fake-model", MESSAGES, provider="fake"
            ):
                received.append(event)
                got_first.set()

        task = asyncio.ensure_future(consume())
        await asyncio.wait_for(got_first.wait(), WAIT_SECONDS)
        self.assertTrue(await wait_for_event(stream.blocked))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(stream.cancelled.is_set())
        self.assertTrue(await wait_for_event(stream.closed))
        self.assertEqual(received, [text_event("first")])

    async def test_breaking_out_of_the_loop_cancels_the_underlying_stream(self):
        stream = CancellableFakeStream()
        provider = self._provider(stream)
        async with contextlib.aclosing(
            provider.async_stream_chat("fake-model", MESSAGES, provider="fake")
        ) as events:
            async for _event in events:
                self.assertTrue(await wait_for_event(stream.blocked))
                break
        self.assertTrue(stream.cancelled.is_set())
        self.assertTrue(await wait_for_event(stream.closed))

    async def test_cancel_after_a_stream_completes_normally_is_harmless(self):
        cancelled = threading.Event()

        class FinishingStream:
            """Completes on its own, then still gets the `finally`'s cancel()."""

            def __init__(self) -> None:
                self._events = iter(FIXED_STREAM)

            def __iter__(self) -> Self:
                return self

            def __next__(self) -> StreamEvent:
                return next(self._events)

            def cancel(self) -> None:
                cancelled.set()

        provider = Provider(adapters={"fake": FakeAdapter(stream=FinishingStream)})
        received = [
            event
            async for event in provider.async_stream_chat(
                "fake-model", MESSAGES, provider="fake"
            )
        ]
        self.assertEqual(received, FIXED_STREAM)
        self.assertTrue(cancelled.is_set())

    async def test_a_stream_without_cancel_is_left_alone(self):
        released = threading.Event()
        self.addCleanup(released.set)
        blocked = threading.Event()

        def stream() -> Iterator[StreamEvent]:
            yield text_event("first")
            blocked.set()
            released.wait(WAIT_SECONDS)

        provider = Provider(adapters={"fake": FakeAdapter(stream=stream)})
        async with contextlib.aclosing(
            provider.async_stream_chat("fake-model", MESSAGES, provider="fake")
        ) as events:
            async for _event in events:
                self.assertTrue(await wait_for_event(blocked))
                break


class TestProviderAsyncExecutorAndDiscovery(unittest.IsolatedAsyncioTestCase):
    async def test_injected_executor_runs_non_stream_blocking_work(self):
        threads: list[str] = []

        def record(result: Any) -> Callable[[], Any]:
            def call() -> Any:
                threads.append(threading.current_thread().name)
                return result

            return call

        stream_threads: list[threading.Thread] = []

        def stream() -> Iterator[StreamEvent]:
            stream_threads.append(threading.current_thread())
            yield from FIXED_STREAM

        executor = concurrent.futures.ThreadPoolExecutor(thread_name_prefix="injected")
        self.addCleanup(executor.shutdown, wait=True)
        adapter = FakeAdapter(
            chat=record(FIXED_RESPONSE),
            stream=stream,
            is_available=record(True),
            models=record({"fake-model"}),
        )
        provider = Provider(adapters={"fake": adapter}, executor=executor)

        await provider.async_chat("fake-model", MESSAGES, provider="fake")
        async for _event in provider.async_stream_chat(
            "fake-model", MESSAGES, provider="fake"
        ):
            pass
        await provider.async_providers()
        await provider.async_models()

        self.assertEqual(len(threads), 4)
        self.assertTrue(all(name.startswith("injected") for name in threads), threads)
        [stream_thread] = stream_threads
        self.assertFalse(stream_thread.name.startswith("injected"))
        self.assertTrue(stream_thread.daemon)

    async def test_async_providers_and_models_match_sync(self):
        provider = Provider(
            adapters={
                "up": FakeAdapter(models=lambda: {"b", "a"}),
                "down": FakeAdapter(is_available=lambda: False),
            }
        )
        self.assertEqual(await provider.async_providers(), provider.providers())
        self.assertEqual(await provider.async_models(), provider.models())
        self.assertEqual(await provider.async_models(), {"up": ["a", "b"]})

    async def test_async_providers_probes_adapters_concurrently(self):
        barrier = threading.Barrier(2, timeout=WAIT_SECONDS)

        def probe() -> bool:
            barrier.wait()
            return True

        provider = Provider(
            adapters={
                "one": FakeAdapter(is_available=probe),
                "two": FakeAdapter(is_available=probe),
            }
        )
        result = await provider.async_providers()
        self.assertEqual(result, {"one": True, "two": True})

    async def test_unknown_provider_raises_eagerly_everywhere(self):
        provider = Provider(adapters={"fake": FakeAdapter()})

        with self.assertRaisesRegex(KeyError, "unknown provider 'nope'.*'fake'"):
            provider.chat("fake-model", MESSAGES, provider="nope")
        with self.assertRaisesRegex(KeyError, "unknown provider 'nope'"):
            provider.stream_chat("fake-model", MESSAGES, provider="nope")
        with self.assertRaisesRegex(KeyError, "unknown provider 'nope'"):
            provider.async_stream_chat("fake-model", MESSAGES, provider="nope")
        with self.assertRaisesRegex(KeyError, "unknown provider 'nope'"):
            await provider.async_chat("fake-model", MESSAGES, provider="nope")


class TestProviderAsyncEmbed(unittest.IsolatedAsyncioTestCase):
    """Async embed mirrors sync embed but off the event loop."""

    async def test_async_embed_matches_sync_embed_for_str_and_batch(self):
        """async_embed returns the same result as sync embed."""

        def embed(
            model: str, input: list[str], config: dict[str, Any] | None = None
        ) -> EmbedResponse:
            return {
                "embeddings": [[float(len(t))] * 2 for t in input],
                "usage": {"input_tokens": len(input)},
                "raw": {},
                "latency_ms": 0.0,
            }

        provider = Provider(
            adapters={
                "fake": FakeAdapter(embed=embed, embed_models=lambda: {"embed-model"})
            }
        )
        for inp in ["hello", ["a", "b", "c"]]:
            with self.subTest(input=inp):
                expected = provider.embed("embed-model", inp, provider="fake")
                actual = await provider.async_embed("embed-model", inp, provider="fake")
                self.assertEqual(actual, expected)

    async def test_async_embed_does_not_block_event_loop(self):
        """A slow embed does not stall the event loop."""

        def slow_embed(
            model: str, input: list[str], config: dict[str, Any] | None = None
        ) -> EmbedResponse:
            time.sleep(0.2)
            return FIXED_EMBED

        provider = Provider(adapters={"slow": FakeAdapter(embed=slow_embed)})
        with LoopTicker() as ticker:
            await provider.async_embed("m", "hi", provider="slow")
        self.assertGreater(ticker.count, 5)

    async def test_async_embed_propagates_api_error(self):
        """APIError from embed propagates unwrapped."""

        failing = FakeAdapter(
            embed=lambda *_a, **_k: raise_(APIError("boom", status=500))
        )
        provider = Provider(adapters={"failing": failing})
        with self.assertRaises(APIError):
            await provider.async_embed("m", "hi", provider="failing")

    async def test_async_embed_propagates_unsupported_operation(self):
        """UnsupportedOperationError propagates with provider name."""

        provider = Provider(adapters={"fake": FakeAdapter()})
        with self.assertRaises(UnsupportedOperationError) as ctx:
            await provider.async_embed("m", "hi", provider="fake")
        self.assertEqual(str(ctx.exception), "provider 'fake' does not support embed()")
        self.assertIsInstance(ctx.exception.__cause__, UnsupportedOperationError)

    async def test_async_embed_propagates_contextvars(self):
        """Contextvars propagate into the embed worker."""

        request_id = contextvars.ContextVar("request_id", default="unset")
        seen: list[str] = []

        def embed(
            model: str, input: list[str], config: dict[str, Any] | None = None
        ) -> EmbedResponse:
            seen.append(request_id.get())
            return FIXED_EMBED

        provider = Provider(adapters={"fake": FakeAdapter(embed=embed)})
        request_id.set("abc")
        await provider.async_embed("m", "hi", provider="fake")
        self.assertEqual(seen, ["abc"])

    async def test_async_embed_resolves_auto_match_off_the_loop(self):
        """Auto-match via embed_models runs off the event loop."""

        provider = Provider(
            adapters={
                "fake": FakeAdapter(
                    embed=lambda _m, _i, _c: FIXED_EMBED,
                    embed_models=lambda: {"embed-model"},
                )
            }
        )
        loop_thread = threading.current_thread()
        fake = provider._adapters["fake"]
        assert isinstance(fake, FakeAdapter)
        fake.embed_models_threads.clear()
        with self.assertLogs(LOGGER, "WARNING"):
            result = await provider.async_embed("embed-model", "hi")
        self.assertEqual(result["embeddings"], FIXED_EMBED["embeddings"])
        self.assertEqual(len(fake.embed_models_threads), 1)
        self.assertNotEqual(fake.embed_models_threads[0], loop_thread)

    async def test_async_embed_input_validation_raises_before_adapter_call(self):
        """ValueError and TypeError raise before any adapter call."""

        cases: list[tuple[str, Any, type[BaseException]]] = [
            ("empty list", [], ValueError),
            ("empty string", "", ValueError),
            ("empty string in batch", ["hi", ""], ValueError),
            ("non-str element", ["hi", 123], TypeError),
            ("dict input", {"a": "b"}, TypeError),
            ("bytes input", b"hi", TypeError),
            ("bytearray input", bytearray(b"hi"), TypeError),
        ]
        for label, bad_input, exc_type in cases:
            with self.subTest(label=label):
                adapter = FakeAdapter(
                    embed=lambda _m, _i, _c: FIXED_EMBED,
                    embed_models=lambda: {"m"},
                )
                provider = Provider(adapters={"fake": adapter})
                with self.assertRaises(exc_type):
                    await provider.async_embed("m", bad_input, provider="fake")
                self.assertEqual(adapter.embed_calls, 0)
                with self.assertRaises(exc_type):
                    await provider.async_embed("m", bad_input)
                self.assertEqual(adapter.embed_models_calls, 0)
                self.assertEqual(adapter.embed_calls, 0)


class TestStreamReaderAfterLoopCloses(unittest.TestCase):
    """Loops closed without finalizing the stream, which asyncio.run never does."""

    def test_reader_blocked_on_full_buffer_exits_when_loop_closes(self):
        produced = [0]
        blocked = threading.Event()
        closed = threading.Event()
        stream = stream_until_blocked(blocked, closed, produced)

        provider = Provider(adapters={"fake": FakeAdapter(stream=stream)})
        loop = asyncio.new_event_loop()
        self.addCleanup(loop.close)
        events = provider.async_stream_chat("fake-model", MESSAGES, provider="fake")
        with patch(POLL_SECONDS, 0.01):
            loop.run_until_complete(anext(events))
            self.assertTrue(blocked.wait(WAIT_SECONDS))
            self.assertFalse(closed.is_set())
            loop.close()
            self.assertTrue(closed.wait(WAIT_SECONDS))
        self.assertEqual(produced[0], _STREAM_BUFFER_SIZE + 2)

        unraisable: list[Any] = []
        with patch.object(sys, "unraisablehook", unraisable.append):
            del events
            gc.collect()
        self.assertEqual(unraisable, [])


if __name__ == "__main__":
    unittest.main()
