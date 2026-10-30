use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use ducktape_core::types::{
    Block, BlockMeta, DocumentBlock, ImageBlock, Message, Response, Role, TextBlock, Tool, ToolDef,
    ToolResultBlock,
};
use serde_json::Value;

/// Anything that becomes the content of a message: text, one block, several blocks or a reply.
pub trait IntoBlocks {
    fn into_blocks(self) -> Vec<Block>;
}

impl IntoBlocks for &str {
    fn into_blocks(self) -> Vec<Block> {
        vec![text(self)]
    }
}

impl IntoBlocks for String {
    fn into_blocks(self) -> Vec<Block> {
        vec![text(self)]
    }
}

impl IntoBlocks for Block {
    fn into_blocks(self) -> Vec<Block> {
        vec![self]
    }
}

impl IntoBlocks for Vec<Block> {
    fn into_blocks(self) -> Vec<Block> {
        self
    }
}

impl<const N: usize> IntoBlocks for [Block; N] {
    fn into_blocks(self) -> Vec<Block> {
        self.into()
    }
}

impl IntoBlocks for &Response {
    fn into_blocks(self) -> Vec<Block> {
        self.content.clone()
    }
}

impl IntoBlocks for Response {
    fn into_blocks(self) -> Vec<Block> {
        self.content
    }
}

/// A text block.
pub fn text(text: impl Into<String>) -> Block {
    Block::text(text)
}

/// An image from bytes.
pub fn image(data: &[u8], media_type: impl Into<String>) -> Block {
    Block::Image(ImageBlock {
        meta: BlockMeta::default(),
        data: Some(STANDARD.encode(data)),
        media_type: Some(media_type.into()),
        ..ImageBlock::default()
    })
}

/// An image at a URL.
pub fn image_url(url: impl Into<String>) -> Block {
    Block::Image(ImageBlock {
        url: Some(url.into()),
        ..ImageBlock::default()
    })
}

/// A document from bytes.
pub fn document(data: &[u8], media_type: impl Into<String>, name: Option<&str>) -> Block {
    Block::Document(DocumentBlock {
        data: Some(STANDARD.encode(data)),
        media_type: Some(media_type.into()),
        name: name.map(str::to_owned),
        ..DocumentBlock::default()
    })
}

/// A document at a URL.
pub fn document_url(url: impl Into<String>, name: Option<&str>) -> Block {
    Block::Document(DocumentBlock {
        url: Some(url.into()),
        name: name.map(str::to_owned),
        ..DocumentBlock::default()
    })
}

/// A tool result.
pub fn tool_result(tool_call_id: impl Into<String>, content: impl IntoBlocks) -> Block {
    Block::ToolResult(ToolResultBlock {
        meta: BlockMeta::default(),
        tool_call_id: tool_call_id.into(),
        content: content.into_blocks(),
        is_error: None,
    })
}

/// A tool result that reports a failure.
pub fn tool_error(tool_call_id: impl Into<String>, content: impl IntoBlocks) -> Block {
    Block::ToolResult(ToolResultBlock {
        meta: BlockMeta::default(),
        tool_call_id: tool_call_id.into(),
        content: content.into_blocks(),
        is_error: Some(true),
    })
}

fn message(role: Role, content: impl IntoBlocks) -> Message {
    Message {
        role,
        content: content.into_blocks(),
    }
}

/// A user message.
pub fn user(content: impl IntoBlocks) -> Message {
    message(Role::User, content)
}

/// An assistant message, from blocks or a reply.
pub fn assistant(content: impl IntoBlocks) -> Message {
    message(Role::Assistant, content)
}

/// A system message.
pub fn system(content: impl IntoBlocks) -> Message {
    message(Role::System, content)
}

/// A tool message carrying tool results.
pub fn tool(results: impl IntoBlocks) -> Message {
    message(Role::Tool, results)
}

/// A function tool with a JSON schema for its input.
pub fn function_tool(
    name: impl Into<String>,
    description: impl Into<String>,
    input_schema: Value,
) -> Tool {
    Tool::Function(ToolDef {
        name: name.into(),
        description: description.into(),
        input_schema,
        strict: None,
        cache: None,
        vendor: None,
    })
}

/// A system prompt as the schema carries it.
pub fn system_blocks(text: impl Into<String>) -> Vec<ducktape_core::types::SystemBlock> {
    vec![ducktape_core::types::SystemBlock::Text(TextBlock {
        meta: BlockMeta::default(),
        text: text.into(),
    })]
}
