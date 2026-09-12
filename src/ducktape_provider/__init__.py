"""One normalized chat/tool-use interface over Claude, OpenAI, and local Ollama."""

from .adapter import Adapter
from .adapters.claude import ClaudeAdapter
from .adapters.ollama import OllamaLocalAdapter
from .adapters.openai import OpenAIAdapter
from .provider import Config, Provider
from .types import (
    Base64Str,
    Block,
    BlockStopEvent,
    ImageBlock,
    JsonSchema,
    Message,
    MessageStopEvent,
    MimeType,
    Response,
    StopReason,
    StreamEvent,
    TextBlock,
    TextDeltaEvent,
    ThinkingBlock,
    ThinkingDeltaEvent,
    ToolCallId,
    ToolDef,
    ToolResultBlock,
    ToolUseBlock,
    ToolUseDeltaEvent,
    ToolUseStartEvent,
    Usage,
)

__all__ = [
    "Adapter",
    "Base64Str",
    "Block",
    "BlockStopEvent",
    "ClaudeAdapter",
    "Config",
    "ImageBlock",
    "JsonSchema",
    "Message",
    "MessageStopEvent",
    "MimeType",
    "OllamaLocalAdapter",
    "OpenAIAdapter",
    "Provider",
    "Response",
    "StopReason",
    "StreamEvent",
    "TextBlock",
    "TextDeltaEvent",
    "ThinkingBlock",
    "ThinkingDeltaEvent",
    "ToolCallId",
    "ToolDef",
    "ToolResultBlock",
    "ToolUseBlock",
    "ToolUseDeltaEvent",
    "ToolUseStartEvent",
    "Usage",
]
