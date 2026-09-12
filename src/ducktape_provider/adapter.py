"""The abstract per-vendor adapter interface every concrete adapter implements."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

from .types import Message, Response, StreamEvent, ToolDef


class Adapter(ABC):
    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def models(self) -> set[str]:
        """Unprefixed ids servable now; empty when unreachable."""
        ...

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Response: ...

    @abstractmethod
    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: dict[str, Any] | None = None,
    ) -> Iterator[StreamEvent]: ...
