import asyncio
import importlib.metadata
import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any, TypedDict, cast

from .adapter import Adapter
from .adapters.claude import ClaudeAdapter
from .adapters.ollama import OllamaLocalAdapter
from .adapters.openai import OpenAIAdapter
from .types import Message, Response, StreamEvent, ToolDef

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "ducktape_provider.adapters"


class Config(TypedDict, total=False):
    timeout: float
    providers: dict[str, dict[str, Any]]


def _discover_adapters() -> dict[str, Adapter]:
    discovered: dict[str, Adapter] = {}
    for entry_point in importlib.metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            discovered[entry_point.name] = entry_point.load()()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Skipping third-party adapter %r: failed to load (%s)",
                entry_point.name,
                exc,
            )
    return discovered


class Provider:
    def __init__(
        self,
        adapters: dict[str, Adapter] | None = None,
        timeout: float | None = None,
        autodiscover: bool = False,
    ):
        self._adapters: dict[str, Adapter] = (
            adapters
            if adapters is not None
            else {
                "claude": ClaudeAdapter(),
                "openai": OpenAIAdapter(),
                "ollama-local": OllamaLocalAdapter(),
            }
        )
        if autodiscover:
            for name, adapter in _discover_adapters().items():
                self._adapters.setdefault(name, adapter)
        self._config: dict[str, Any] = {}
        if timeout is not None:
            self._config["timeout"] = timeout

    def providers(self) -> dict[str, bool]:
        """Configured providers, available or not."""
        return {
            name: adapter.is_available() for name, adapter in self._adapters.items()
        }

    def models(self) -> dict[str, list[str]]:
        """Usable model ids by provider; nothing from unreachable vendors."""
        return {
            name: sorted(adapter.models())
            for name, adapter in self._adapters.items()
            if adapter.is_available()
        }

    def _merge_config(self, provider: str, config: Config | None) -> dict[str, Any]:
        """Merges self._config, config, config["providers"][provider] in order — each overrides the previous."""
        config = config or {}
        merged = dict(self._config)
        merged.update({k: v for k, v in config.items() if k != "providers"})
        merged.update(config.get("providers", {}).get(provider, {}))
        return merged

    def chat(
        self,
        provider: str,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: Config | None = None,
    ) -> Response:
        return self._adapters[provider].chat(
            model, messages, system, tools, self._merge_config(provider, config)
        )

    def stream_chat(
        self,
        provider: str,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: Config | None = None,
    ) -> Iterator[StreamEvent]:
        return self._adapters[provider].stream_chat(
            model, messages, system, tools, self._merge_config(provider, config)
        )

    async def async_chat(
        self,
        provider: str,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: Config | None = None,
    ) -> Response:
        return await asyncio.to_thread(
            self.chat, provider, model, messages, system, tools, config
        )

    async def async_stream_chat(
        self,
        provider: str,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: Config | None = None,
    ) -> AsyncIterator[StreamEvent]:
        sync_iter = self.stream_chat(provider, model, messages, system, tools, config)
        sentinel = object()
        while True:
            event = await asyncio.to_thread(next, sync_iter, sentinel)
            if event is sentinel:
                break
            yield cast(StreamEvent, event)
