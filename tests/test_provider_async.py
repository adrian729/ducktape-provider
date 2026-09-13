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
from ducktape_provider.errors import APIError
from ducktape_provider.provider import _STREAM_BUFFER_SIZE, Provider
from ducktape_provider.types import (
    Message,
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
    ):
        self._chat = chat
        self._stream = stream
        self._is_available = is_available
        self._models = models

    def is_available(self) -> bool:
        return self._is_available()

    def models(self) -> set[str]:
        return self._models()

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
    # Long enough that a reader blocked on backpressure can't just finish.
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
                # Stands in for a blocking HTTP read that cancellation can't abort.
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
        # Exactly the events that fit, plus the one the blocked reader held:
        # fewer means the buffer never filled, more means it didn't block.
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
        # A poll far longer than the join below: only an exit that doesn't wait
        # on the buffer again can pass.
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
        # One worker: a stream reader holding it would starve async_chat forever.
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

        # Stands in for the loop closing between the reader's is_closed() check
        # and its hand-off; other callers (e.g. asyncio.to_thread) still work.
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
        # Streams read on their own daemon thread so they never hold a worker.
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
        # Each probe only returns once both are running at the same time.
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
        # A sequential probe would break the barrier and raise instead.
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
            # Neither stopped nor closed, the blocked reader must keep waiting.
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
