from typing import Any, Literal, NotRequired, TypedDict


class TextBlock(TypedDict):
    type: Literal["text"]
    text: str


type Base64Str = str
type MimeType = str
type ToolCallId = str


class ImageBlock(TypedDict):
    type: Literal["image"]
    media_type: MimeType
    data: Base64Str


class ToolUseBlock(TypedDict):
    type: Literal["tool_use"]
    id: ToolCallId
    name: str
    input: dict[str, Any]


class ToolResultBlock(TypedDict):
    type: Literal["tool_result"]
    tool_use_id: ToolCallId
    name: str
    content: str
    is_error: NotRequired[bool]


class ThinkingBlock(TypedDict):
    type: Literal["thinking"]
    thinking: str
    signature: NotRequired[str]


Block = TextBlock | ImageBlock | ToolUseBlock | ToolResultBlock | ThinkingBlock


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: list[Block]


type JsonSchema = dict[str, Any]


class ToolDef(TypedDict):
    name: str
    description: str
    parameters: JsonSchema


StopReason = Literal[
    "end_turn",
    "tool_use",
    "max_tokens",
    "stop_sequence",
    "content_filter",
    "refusal",
    "other",
]


class Usage(TypedDict):
    input_tokens: int
    output_tokens: int


class Response(TypedDict):
    content: list[Block]
    stop_reason: StopReason
    raw_stop_reason: str
    usage: Usage
    raw: dict[str, Any]


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
    type: Literal["tool_use_delta"]
    index: int
    partial_json: str


class BlockStopEvent(TypedDict):
    type: Literal["block_stop"]
    index: int


class MessageStopEvent(TypedDict):
    type: Literal["message_stop"]
    response: Response


StreamEvent = (
    TextDeltaEvent
    | ThinkingDeltaEvent
    | ToolUseStartEvent
    | ToolUseDeltaEvent
    | BlockStopEvent
    | MessageStopEvent
)
