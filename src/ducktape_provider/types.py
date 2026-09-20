"""Every public type of ducktape_provider: what you pass in, what you get back, what
can be raised, and the `Adapter` base class for adding providers.

Lazy annotations let types read top-down (a type first, then what it's built from)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any, Literal, NotRequired, TypedDict

"""1. Interface types

What `Provider` methods take and return. A third-party adapter's `chat`/`stream_chat`
take and return these same types (see section 3).

Inputs: `messages`, `tools` and `config` arguments.
"""


class Message(TypedDict):
    """One conversation turn."""

    role: Literal["user", "assistant"]
    content: list[Block]


type Block = (
    TextBlock
    | ImageBlock
    | DocumentBlock
    | ToolUseBlock
    | ToolResultBlock
    | ThinkingBlock
)


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


type ImageBlock = Base64ImageBlock | UrlImageBlock


class Base64ImageBlock(TypedDict):
    type: Literal["image"]
    source: Literal["base64"]
    media_type: MimeType
    data: Base64Str


class UrlImageBlock(TypedDict):
    type: Literal["image"]
    source: Literal["url"]
    url: str


type DocumentBlock = Base64DocumentBlock | UrlDocumentBlock


class Base64DocumentBlock(TypedDict):
    type: Literal["document"]
    source: Literal["base64"]
    media_type: MimeType
    data: Base64Str


class UrlDocumentBlock(TypedDict):
    type: Literal["document"]
    source: Literal["url"]
    url: str


class ToolUseBlock(TypedDict):
    """A tool call the model made; answer it with a `ToolResultBlock`.

    `truncated` is set when the model's `input` couldn't be parsed — almost
    always a `max_tokens` cutoff mid-call; treat the call as incomplete, not as
    a legitimate empty-argument call.
    """

    type: Literal["tool_use"]
    id: ToolCallId
    name: str
    input: dict[str, Any]
    truncated: NotRequired[bool]


class ToolResultBlock(TypedDict):
    """Your tool's output, sent back in a `user` message. `content` can be
    plain text, or a list of text/image blocks for tools that return images
    (e.g. a screenshot, a rendered chart)."""

    type: Literal["tool_result"]
    tool_use_id: ToolCallId
    name: str
    content: str | list[TextBlock | ImageBlock]
    is_error: NotRequired[bool]


class ThinkingBlock(TypedDict):
    """Model reasoning. Send it back unchanged in follow-up turns.

    The OpenAI adapter drops thinking blocks given to it: it has no equivalent input field.
    """

    type: Literal["thinking"]
    thinking: str
    signature: NotRequired[str]


type Base64Str = str
type MimeType = str
type ToolCallId = str


class ToolDef(TypedDict):
    """A tool the model may call."""

    name: str
    description: str
    parameters: JsonSchema


type JsonSchema = dict[str, Any]


class Config(TypedDict, total=False):
    """Per-call options. `timeout` and `headers` apply to the HTTP request; any other
    key is sent as-is to the vendor. `providers` maps a provider name to overrides
    for that provider only.

    Call sites accept `Config | Mapping[str, Any]`, since a closed TypedDict would
    reject vendor fields like `temperature` (PEP 728's `extra_items` would keep both,
    but not all type checkers support it yet).
    """

    timeout: float | None
    headers: dict[str, str]
    providers: dict[str, dict[str, Any]]


"""Outputs: what `chat` returns and what `stream_chat` yields."""


class Response(TypedDict):
    """A complete model turn, from `chat()` or a stream's final `message_stop` event.

    Only `content`, `stop_reason` and `usage` are required; `usage` is `None` when
    not known. Adapters set the rest when they can.

    - `latency_ms`: from sending the request until the reply was fully read.
    - `ttft_ms`: streaming only, until the first content event was read.
    """

    content: list[Block]
    stop_reason: StopReason
    raw_stop_reason: NotRequired[str]
    usage: Usage | None
    raw: NotRequired[dict[str, Any]]
    latency_ms: NotRequired[float]
    ttft_ms: NotRequired[float]


type StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "content_filter",
    "refusal",
    "pause_turn",
    "other",
]


class Usage(TypedDict):
    """Token counts. `input_tokens` includes cached tokens."""

    input_tokens: int
    output_tokens: int
    cache_read_tokens: NotRequired[int]
    cache_write_tokens: NotRequired[int]


class EmbedUsage(TypedDict):
    """Token counts for an embed() call."""

    input_tokens: int


class EmbedResponse(TypedDict):
    """Normalized result of embed()."""

    embeddings: list[list[float]]
    usage: EmbedUsage | None
    raw: NotRequired[dict[str, Any]]
    dimensions: NotRequired[int]
    latency_ms: NotRequired[float]


type StreamEvent = (
    TextDeltaEvent
    | ThinkingDeltaEvent
    | ToolUseStartEvent
    | ToolUseDeltaEvent
    | BlockStopEvent
    | MessageStopEvent
)
"""What `stream_chat` yields. `index` is the block's position in the final
`response["content"]`; the stream always ends with one `MessageStopEvent`."""


class TextDeltaEvent(TypedDict):
    type: Literal["text_delta"]
    index: int
    text: str


class ThinkingDeltaEvent(TypedDict):
    type: Literal["thinking_delta"]
    index: int
    thinking: str


class ToolUseStartEvent(TypedDict):
    type: Literal["tool_use_start"]
    index: int
    id: ToolCallId
    name: str


class ToolUseDeltaEvent(TypedDict):
    """A chunk of the tool call's JSON arguments; concatenate them."""

    type: Literal["tool_use_delta"]
    index: int
    partial_json: str


class BlockStopEvent(TypedDict):
    type: Literal["block_stop"]
    index: int


class MessageStopEvent(TypedDict):
    type: Literal["message_stop"]
    response: Response


"""2. Errors

Everything raised by `Provider` calls, streams included. Catch `DucktapeError` for
all of them. Adapters should raise these too, so every provider fails the same way.
"""


class DucktapeError(Exception):
    """Base of every error below."""


class APIError(DucktapeError):
    """A request failed with no more specific error below, e.g. a dropped connection.

    `status` is the HTTP status (or `None`); `body` is the error body.
    """

    def __init__(self, message: str, *, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class AuthError(APIError):
    """API key missing or rejected (401/403)."""


class RateLimitError(APIError):
    """Rate limited (429). Retry after `retry_after` seconds, if set."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        retry_after: float | None = None,
    ):
        super().__init__(message, status=status, body=body)
        self.retry_after = retry_after


class ServerError(APIError):
    """Vendor server error (5xx). Retry after `retry_after` seconds, if set."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        retry_after: float | None = None,
    ):
        super().__init__(message, status=status, body=body)
        self.retry_after = retry_after


class RequestTimeoutError(APIError):
    """The request exceeded `timeout`."""


class ContextOverflowError(APIError):
    """The input is too long for the model."""


class MalformedResponseError(APIError):
    """The vendor's reply couldn't be parsed. Retrying rarely helps."""


class UnsupportedBlockError(DucktapeError):
    """The vendor can't accept a content block (e.g. a URL image for Ollama).

    Raised before any request is sent.
    """


class UnsupportedOperationError(DucktapeError):
    """The resolved provider does not support the requested operation."""


"""3. Everything else: extending

Subclass `Adapter` to add a provider, then pass it to `Provider(adapters=...)` or
register it as a plugin (see the README's "Third-party adapters").
"""


class ModelInfo(TypedDict):
    """Best-effort model metadata. Either field is `None` when the provider
    doesn't expose it for that model."""

    context_window: int | None
    max_output_tokens: int | None


class Adapter(ABC):
    """One vendor behind the `Provider` interface.

    - `chat` returns a `Response`; only `content`, `stop_reason` and `usage` are
      required (`usage` is `None` when not known).
    - `stream_chat` yields `StreamEvent`s and must end with one `MessageStopEvent`.
    - `config` arrives merged (per-provider overrides applied): apply its `timeout`
      and `headers` to the HTTP request and send every other key to the vendor.
    - Raise the errors from section 2.
    - Plugins are instantiated with no arguments.
    """

    @abstractmethod
    def is_available(self) -> bool:
        """Whether the vendor is configured and usable, e.g. its API key is set."""
        ...

    @abstractmethod
    def models(self) -> set[str]:
        """Model ids servable now; empty when unreachable."""
        ...

    def model_info(self, model: str) -> ModelInfo | None:
        """Context window and max output for `model`, when known.

        Optional: the default returns `None`. Override to add real data — see the
        built-in adapters for examples of reading it from the vendor rather than
        guessing.
        """
        return None

    def embed(
        self, model: str, input: list[str], config: dict[str, Any] | None = None
    ) -> EmbedResponse:
        """Embed texts; raises UnsupportedOperationError unless overridden."""
        raise UnsupportedOperationError(
            f"provider {type(self).__name__!r} does not support embed()"
        )

    def embed_models(self) -> set[str]:
        """Model ids this adapter can embed with."""
        return set()

    def _reset_caches(self) -> None:
        """Clear cached metadata."""

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
