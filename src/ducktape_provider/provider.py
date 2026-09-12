from collections.abc import Iterator
from typing import Any, TypedDict

from .adapter import Adapter
from .adapters.claude import ClaudeAdapter
from .adapters.ollama import OllamaLocalAdapter
from .adapters.openai import OpenAIAdapter
from .types import Message, Response, StreamEvent, ToolDef


class Config(TypedDict, total=False):
    timeout: float
    providers: dict[str, dict[str, Any]]


class Provider:
    def __init__(
        self,
        adapters: dict[str, Adapter] | None = None,
        timeout: float | None = None,
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
