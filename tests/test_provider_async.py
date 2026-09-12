import asyncio
import time
import unittest
from collections.abc import Iterator
from typing import Any

from ducktape_provider.adapter import Adapter
from ducktape_provider.errors import APIError
from ducktape_provider.provider import Provider
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


class FakeAdapter(Adapter):
    def is_available(self) -> bool:
        return True

    def models(self) -> set[str]:
        return {"fake-model"}

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response:
        return FIXED_RESPONSE

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        yield from FIXED_STREAM


class SlowAdapter(Adapter):
    def is_available(self) -> bool:
        return True

    def models(self) -> set[str]:
        return {"fake-model"}

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response:
        time.sleep(0.2)
        return FIXED_RESPONSE

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        yield from FIXED_STREAM


class FailingChatAdapter(Adapter):
    def is_available(self) -> bool:
        return True

    def models(self) -> set[str]:
        return {"fake-model"}

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response:
        raise APIError("boom", status=500, body="server exploded")

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        yield from FIXED_STREAM


class FailingStreamAdapter(Adapter):
    def is_available(self) -> bool:
        return True

    def models(self) -> set[str]:
        return {"fake-model"}

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response:
        return FIXED_RESPONSE

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]:
        yield {"type": "text_delta", "index": 0, "text": "hi"}
        raise APIError("stream broke", status=500, body="mid-stream failure")


MESSAGES: list[Message] = [
    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
]


class TestProviderAsync(unittest.IsolatedAsyncioTestCase):
    async def test_async_chat_matches_sync_chat(self):
        provider = Provider(adapters={"fake": FakeAdapter()})
        expected = provider.chat("fake", "fake-model", MESSAGES)
        actual = await provider.async_chat("fake", "fake-model", MESSAGES)
        self.assertEqual(actual, expected)

    async def test_async_stream_chat_matches_sync_stream_chat(self):
        provider = Provider(adapters={"fake": FakeAdapter()})
        expected = list(provider.stream_chat("fake", "fake-model", MESSAGES))
        actual = [
            event
            async for event in provider.async_stream_chat(
                "fake", "fake-model", MESSAGES
            )
        ]
        self.assertEqual(actual, expected)

    async def test_async_chat_does_not_block_event_loop(self):
        provider = Provider(adapters={"slow": SlowAdapter()})
        counter = 0

        async def increment_loop():
            nonlocal counter
            while True:
                counter += 1
                await asyncio.sleep(0)

        incrementer = asyncio.ensure_future(increment_loop())
        await provider.async_chat("slow", "fake-model", MESSAGES)
        incrementer.cancel()

        self.assertGreater(counter, 5)

    async def test_async_chat_propagates_error(self):
        provider = Provider(adapters={"failing": FailingChatAdapter()})
        with self.assertRaises(APIError):
            await provider.async_chat("failing", "fake-model", MESSAGES)

    async def test_async_stream_chat_propagates_error(self):
        provider = Provider(adapters={"failing": FailingStreamAdapter()})

        async def consume():
            async for _event in provider.async_stream_chat(
                "failing", "fake-model", MESSAGES
            ):
                pass

        with self.assertRaises(APIError):
            await consume()


if __name__ == "__main__":
    unittest.main()
