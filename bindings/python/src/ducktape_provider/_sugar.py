"""Shortcuts over the normalized types."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

from ._types import (
    Block,
    CacheMark,
    DocumentBlock,
    ImageBlock,
    Message,
    Model,
    Response,
    TextBlock,
    ToolResultBlock,
    decode,
)

Part = Union[str, bytes, Block, dict]


def _b64(data: Union[bytes, str]) -> str:
    return base64.b64encode(data).decode() if isinstance(data, (bytes, bytearray, memoryview)) else data


def text(value: str, *, cache: Optional[str] = None) -> TextBlock:
    """A text block, with a cache mark when `cache` names a ttl."""
    return TextBlock(text=value, cache=CacheMark(ttl=cache) if cache else None)


def image(
    data: Union[bytes, str, None] = None,
    *,
    url: Optional[str] = None,
    file_id: Optional[str] = None,
    media_type: Optional[str] = None,
) -> ImageBlock:
    """An image block from bytes, base64 text, a URL or a vendor file id."""
    return ImageBlock(data=_b64(data) if data is not None else None, url=url, file_id=file_id, media_type=media_type)


def document(
    data: Union[bytes, str, None] = None,
    *,
    url: Optional[str] = None,
    file_id: Optional[str] = None,
    text: Optional[str] = None,
    media_type: Optional[str] = None,
    name: Optional[str] = None,
) -> DocumentBlock:
    """A document block from bytes, base64 text, a URL, a vendor file id or plain text."""
    return DocumentBlock(
        data=_b64(data) if data is not None else None,
        url=url,
        file_id=file_id,
        text=text,
        media_type=media_type,
        name=name,
    )


def blocks(parts: tuple[Part, ...]) -> list[Block]:
    """Turns strings, bytes, blocks and dicts into blocks."""
    out: list[Block] = []
    for part in parts:
        if isinstance(part, str):
            out.append(TextBlock(text=part))
        elif isinstance(part, (bytes, bytearray, memoryview)):
            out.append(ImageBlock(data=_b64(bytes(part))))
        elif isinstance(part, Model):
            out.append(part)
        elif isinstance(part, dict):
            out.append(decode(Block, part))
        elif isinstance(part, (list, tuple)):
            out.extend(blocks(tuple(part)))
        else:
            raise TypeError(f"cannot make a block from {type(part).__name__}")
    return out


def tool_result(tool_call_id: str, *content: Part, is_error: Optional[bool] = None) -> ToolResultBlock:
    """The result of one tool call."""
    return ToolResultBlock(tool_call_id=tool_call_id, content=blocks(content), is_error=is_error)


def user(*parts: Part) -> Message:
    """A user turn."""
    return Message(role="user", content=blocks(parts))


def assistant(*parts: Union[Part, Response]) -> Message:
    """An assistant turn, from parts or a whole response."""
    content: list[Block] = []
    for part in parts:
        if isinstance(part, Response):
            content.extend(part.content)
        else:
            content.extend(blocks((part,)))
    return Message(role="assistant", content=content)


def system(*parts: Part) -> Message:
    """A mid-conversation system turn."""
    return Message(role="system", content=blocks(parts))


def tool(*results: Union[ToolResultBlock, dict]) -> Message:
    """A tool turn holding every result of one round."""
    return Message(role="tool", content=blocks(results))


@dataclass(frozen=True)
class EnvAuth:
    """Credential read from the first set environment variable."""

    names: tuple[str, ...]
    kind: Optional[str] = None
    placement: Optional[str] = None


@dataclass(frozen=True)
class SessionAuth:
    """Credential from a login session."""

    preset: str
    store: Optional[str] = None
    client_id: Optional[str] = None
    profile: Optional[str] = None
    refresh: Optional[str] = None


@dataclass(frozen=True)
class ResolverAuth:
    """Credential fetched by a callable at request time."""

    fn: Callable[[], Any]
    kind: Optional[str] = None
    placement: Optional[str] = None


def env(*names: str, kind: Optional[str] = None, placement: Optional[str] = None) -> EnvAuth:
    """Reads the credential from the first of these environment variables that is set."""
    if not names:
        raise ValueError("env() needs at least one variable name")
    return EnvAuth(tuple(names), kind, placement)


def session(
    preset: str,
    *,
    store: Optional[str] = None,
    client_id: Optional[str] = None,
    profile: Optional[str] = None,
    refresh: Optional[str] = None,
) -> SessionAuth:
    """Uses a login session of the given preset."""
    if refresh is not None and refresh not in ("never", "memory", "write_back"):
        raise ValueError(f"refresh must be 'never', 'memory' or 'write_back', not {refresh!r}")
    return SessionAuth(preset, store, client_id, profile, refresh)


def resolver(fn: Callable[[], Any], *, kind: Optional[str] = None, placement: Optional[str] = None) -> ResolverAuth:
    """Calls `fn` for the credential; it returns a string or `{secret}`."""
    return ResolverAuth(fn, kind, placement)
