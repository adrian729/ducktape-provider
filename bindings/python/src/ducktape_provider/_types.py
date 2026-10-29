"""Normalized types as dataclasses, mirroring schema/provider.schema.json."""
from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Union


class _Unknown:
    """Marker for a capability the provider does not report."""

    _instance: ClassVar[Optional[_Unknown]] = None

    def __new__(cls) -> _Unknown:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNKNOWN"

    def __bool__(self) -> bool:
        return False


UNKNOWN: Any = _Unknown()

_TAGS: dict[type, str] = {}


def _hints(cls: type) -> dict[str, Any]:
    cached = cls.__dict__.get("_hint_cache")
    if cached is None:
        cached = typing.get_type_hints(cls)
        cls._hint_cache = cached
    return cached


def _args(hint: Any) -> tuple[Any, ...]:
    return typing.get_args(hint)


def _is_union(hint: Any) -> bool:
    origin = typing.get_origin(hint)
    return origin is Union or origin is types.UnionType


def _tag_of(cls: type) -> Optional[str]:
    return _TAGS.get(cls)


def _decode(hint: Any, value: Any) -> Any:
    if value is None or hint is Any:
        return value
    if _is_union(hint):
        members = [m for m in _args(hint) if m is not type(None)]
        tagged = {_tag_of(m): m for m in members if _tag_of(m)}
        if tagged and isinstance(value, dict):
            tag = value.get("type")
            if tag not in tagged:
                raise ValueError(f"unknown type tag {tag!r}")
            return _decode(tagged[tag], value)
        return _decode(members[0], value) if len(members) == 1 else value
    origin = typing.get_origin(hint)
    if origin is list:
        (item,) = _args(hint)
        return [_decode(item, v) for v in value]
    if origin is dict:
        _, item = _args(hint)
        return {k: _decode(item, v) for k, v in value.items()}
    if dataclasses.is_dataclass(hint) and isinstance(value, dict):
        return hint.from_dict(value)
    return value


def _encode(value: Any) -> Any:
    if isinstance(value, Model):
        return value.to_dict()
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, tuple):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {k: _encode(v) for k, v in value.items()}
    return value


class Model:
    """Base of every normalized type, converting to and from plain data."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Any:
        if isinstance(data, cls):
            return data
        hints = _hints(cls)
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if not f.init:
                continue
            if f.name in data:
                kwargs[f.name] = _decode(hints[f.name], data[f.name])
            elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                raise ValueError(f"{cls.__name__} is missing {f.name!r}")
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        tag = _tag_of(type(self))
        if tag:
            out["type"] = tag
        for f in dataclasses.fields(self):
            if not f.init:
                continue
            value = getattr(self, f.name)
            if value is UNKNOWN:
                continue
            if value is None and f.default is None:
                continue
            out[f.name] = _encode(value)
        return out


def tagged(tag: str):
    """Registers a dataclass as the member of a tagged union."""

    def wrap(cls: type) -> type:
        _TAGS[cls] = tag
        cls.type = tag
        return cls

    return wrap


@dataclass
class Origin(Model):
    """Where a reply or an opaque block came from."""

    wire: str
    ns: str
    base_url: str
    model: str
    auth_source: Optional[str] = None


@dataclass
class CacheMark(Model):
    """A prompt cache breakpoint."""

    ttl: Optional[str] = None


@dataclass
class _Meta(Model):
    id: Optional[str] = None
    cache: Optional[CacheMark] = None
    vendor: Optional[dict[str, Any]] = None
    provider_executed: Optional[bool] = None


@tagged("text")
@dataclass
class TextBlock(_Meta):
    """A run of text."""

    text: str = ""


@tagged("image")
@dataclass
class ImageBlock(_Meta):
    """An image by bytes, URL or vendor file id."""

    data: Optional[str] = None
    url: Optional[str] = None
    file_id: Optional[str] = None
    media_type: Optional[str] = None


@tagged("document")
@dataclass
class DocumentBlock(_Meta):
    """A document by bytes, URL, vendor file id or plain text."""

    data: Optional[str] = None
    url: Optional[str] = None
    file_id: Optional[str] = None
    text: Optional[str] = None
    media_type: Optional[str] = None
    name: Optional[str] = None


@tagged("tool_call")
@dataclass
class ToolCallBlock(_Meta):
    """A tool the model asked to run."""

    id: str = ""
    name: str = ""
    arguments: Any = field(default_factory=dict)
    truncated: Optional[bool] = None
    arguments_json: Optional[str] = None
    status: Optional[str] = None


@tagged("tool_result")
@dataclass
class ToolResultBlock(_Meta):
    """What a tool returned."""

    tool_call_id: str = ""
    content: list[Block] = field(default_factory=list)
    is_error: Optional[bool] = None


@tagged("reasoning")
@dataclass
class ReasoningBlock(_Meta):
    """The model's reasoning, with opaque vendor state when any."""

    text: str = ""
    vendor_state: Any = None
    origin: Optional[Origin] = None


@tagged("compaction")
@dataclass
class CompactionBlock(_Meta):
    """Vendor-summarized history."""

    vendor_state: Any = None
    text: Optional[str] = None
    origin: Optional[Origin] = None


@tagged("custom")
@dataclass
class CustomBlock(_Meta):
    """A vendor-specific block kept as data."""

    kind: str = ""
    data: Any = None
    origin: Optional[Origin] = None


Block = Union[
    TextBlock,
    ImageBlock,
    DocumentBlock,
    ToolCallBlock,
    ToolResultBlock,
    ReasoningBlock,
    CompactionBlock,
    CustomBlock,
]


@dataclass
class Message(Model):
    """One turn of a conversation."""

    role: str
    content: list[Block]


@tagged("function")
@dataclass
class ToolDef(Model):
    """A function tool the caller runs."""

    name: str
    description: str
    input_schema: Any
    strict: Optional[bool] = None
    cache: Optional[CacheMark] = None
    vendor: Optional[dict[str, Any]] = None


@tagged("provider")
@dataclass
class ProviderTool(Model):
    """A tool the vendor runs itself."""

    provider_tool: str
    config: Any = None


Tool = Union[ToolDef, ProviderTool]


@dataclass
class ToolChoice(Model):
    """How the model may pick tools."""

    mode: str
    name: Optional[str] = None


@dataclass
class OutputFormat(Model):
    """A structured output request."""

    json_schema: Any
    name: Optional[str] = None


@dataclass
class Timeout(Model):
    """Per-phase timeouts in seconds."""

    connect: Optional[float] = None
    read: Optional[float] = None
    total: Optional[float] = None


@dataclass
class Options(Model):
    """Portable chat options."""

    max_output_tokens: Optional[int] = None
    reasoning: Optional[str] = None
    reasoning_summary: Optional[str] = None
    tool_choice: Optional[ToolChoice] = None
    parallel_tools: Optional[bool] = None
    output: Optional[OutputFormat] = None
    stop_sequences: Optional[list[str]] = None
    cache: Optional[str] = None
    retries: Optional[int] = None
    user_id: Optional[str] = None


@dataclass
class Stop(Model):
    """Why generation stopped."""

    reason: str
    raw: Any = None


@dataclass
class InputUsage(Model):
    """Input token counts."""

    total: int
    cache_read: Optional[int] = None
    cache_write: Optional[int] = None


@dataclass
class OutputUsage(Model):
    """Output token counts."""

    total: int
    reasoning: Optional[int] = None


@dataclass
class Usage(Model):
    """Token usage of a call."""

    input: InputUsage
    output: OutputUsage
    raw: Any = None


@dataclass
class Warning(Model):
    """Something the provider could not do exactly as asked."""

    kind: str
    feature: str
    detail: Optional[str] = None


@dataclass
class Response(Model):
    """A finished chat reply."""

    content: list[Block]
    stop: Stop
    usage: Optional[Usage]
    warnings: list[Warning]
    origin: Origin
    latency_ms: float
    first_token_ms: Optional[float] = None
    raw: Any = None

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def reasoning(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, ReasoningBlock))

    @property
    def tool_calls(self) -> list[ToolCallBlock]:
        return [b for b in self.content if isinstance(b, ToolCallBlock)]


@tagged("block_start")
@dataclass
class BlockStart(Model):
    """A block began streaming."""

    index: int
    kind: str
    id: Optional[str] = None
    name: Optional[str] = None
    parent_id: Optional[str] = None


@tagged("text_delta")
@dataclass
class TextDelta(Model):
    """A piece of text."""

    index: int
    text: str
    parent_id: Optional[str] = None


@tagged("reasoning_delta")
@dataclass
class ReasoningDelta(Model):
    """A piece of reasoning text."""

    index: int
    text: str
    parent_id: Optional[str] = None


@tagged("tool_call_delta")
@dataclass
class ToolCallDelta(Model):
    """A fragment of a tool call's arguments JSON."""

    index: int
    arguments_json: str
    parent_id: Optional[str] = None


@tagged("block_stop")
@dataclass
class BlockStop(Model):
    """A block finished, carrying its final form."""

    index: int
    block: Block
    parent_id: Optional[str] = None


@tagged("finish")
@dataclass
class Finish(Model):
    """The stream ended with the full response."""

    response: Response
    parent_id: Optional[str] = None


@tagged("raw")
@dataclass
class Raw(Model):
    """A vendor event, delivered only with include_raw."""

    event: Any
    parent_id: Optional[str] = None


Event = Union[BlockStart, TextDelta, ReasoningDelta, ToolCallDelta, BlockStop, Finish, Raw]


@dataclass
class EmbedUsage(Model):
    """Token usage of an embedding call."""

    input: dict[str, Any]


@dataclass
class EmbedResponse(Model):
    """Vectors for a batch of inputs."""

    embeddings: list[list[float]]
    usage: Optional[EmbedUsage]
    warnings: list[Warning]
    origin: Origin
    latency_ms: float
    raw: Any = None

    @property
    def dimensions(self) -> int:
        return len(self.embeddings[0]) if self.embeddings else 0


@dataclass
class ReasoningCapability(Model):
    """Reasoning support of a model."""

    levels: Optional[list[str]] = UNKNOWN
    can_disable: Optional[bool] = UNKNOWN


@dataclass
class ParallelToolsCapability(Model):
    """Parallel tool call support."""

    supported: Optional[bool] = UNKNOWN
    toggle: Optional[bool] = UNKNOWN


@dataclass
class OutputCapability(Model):
    """Structured output support."""

    json_schema: Optional[bool] = UNKNOWN


@dataclass
class CacheCapability(Model):
    """Prompt cache support."""

    breakpoints: Optional[bool] = UNKNOWN
    ttls: Optional[list[str]] = UNKNOWN


@dataclass
class ToolsCapability(Model):
    """Tool support."""

    functions: Optional[bool] = UNKNOWN
    hosted: Optional[list[str]] = UNKNOWN


@dataclass
class Limits(Model):
    """Size limits."""

    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None


@dataclass
class Capabilities(Model):
    """What a model can do; UNKNOWN where nobody knows."""

    source: str
    reasoning: Optional[ReasoningCapability] = None
    tool_choice: Optional[list[str]] = UNKNOWN
    parallel_tools: Optional[ParallelToolsCapability] = None
    output: Optional[OutputCapability] = None
    cache: Optional[CacheCapability] = None
    inputs: Optional[list[str]] = UNKNOWN
    tools: Optional[ToolsCapability] = None
    limits: Optional[Limits] = None
    embeddings: Optional[bool] = UNKNOWN


@dataclass
class ModelInfo(Model):
    """What a provider says about one model."""

    id: str
    kind: str
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    capabilities: Optional[Capabilities] = None
    raw: Any = None


@dataclass
class AuthStatus(Model):
    """Whether and how the provider is logged in."""

    source: Optional[str]
    expires_at: Optional[str] = None
    session: Optional[str] = None


@tagged("open_url")
@dataclass
class OpenUrl(Model):
    """Ask the user to open a URL."""

    url: str
    instructions: str


@tagged("show_code")
@dataclass
class ShowCode(Model):
    """Ask the user to enter a code at a URL."""

    url: str
    code: str
    instructions: str


@tagged("paste")
@dataclass
class Paste(Model):
    """Ask the user to paste a value back."""

    prompt: str


LoginAction = Union[OpenUrl, ShowCode, Paste]

_DEFS: dict[str, Any] = {
    "Message": Message,
    "Block": Block,
    "Tool": Tool,
    "Response": Response,
    "Event": Event,
    "EmbedResponse": EmbedResponse,
    "Capabilities": Capabilities,
    "ModelInfo": ModelInfo,
    "AuthStatus": AuthStatus,
    "LoginAction": LoginAction,
    "Options": Options,
}


def decode(hint: Any, value: Any) -> Any:
    """Builds a typed value from plain data."""
    return _decode(hint, value)


def encode(value: Any) -> Any:
    """Turns typed values into plain data."""
    return _encode(value)
