//! Normalized types, mirroring `schema/provider.schema.json`.
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

fn nullable<'de, D, T>(d: D) -> Result<Option<Option<T>>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(d).map(Some)
}

/// Vendor request fields by namespace, sent as-is.
pub type VendorMap = BTreeMap<String, Value>;

/// Extra HTTP headers by name.
pub type Headers = BTreeMap<String, String>;

macro_rules! open_enum {
    ($(#[$meta:meta])* $name:ident { $($variant:ident => $wire:literal),* $(,)? }) => {
        $(#[$meta])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        pub enum $name {
            $(#[serde(rename = $wire)] $variant,)*
            #[serde(untagged)]
            Other(String),
        }

        impl $name {
            /// The wire spelling.
            pub fn as_str(&self) -> &str {
                match self {
                    $(Self::$variant => $wire,)*
                    Self::Other(s) => s,
                }
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                match s {
                    $($wire => Self::$variant,)*
                    other => Self::Other(other.to_owned()),
                }
            }
        }
    };
}

open_enum! {
    /// Vendor namespace.
    Ns { Anthropic => "anthropic", Openai => "openai", Google => "google", Ollama => "ollama" }
}

open_enum! {
    /// Who speaks a message.
    Role { User => "user", Assistant => "assistant", System => "system", Tool => "tool" }
}

open_enum! {
    /// Cache lifetime.
    CacheTtl { Short => "short", Long => "long" }
}

open_enum! {
    /// How deep the model reasons.
    ReasoningLevel { Auto => "auto", None => "none", Minimal => "minimal", Low => "low", Medium => "medium", High => "high", Xhigh => "xhigh", Max => "max" }
}

open_enum! {
    /// Whether the vendor returns a reasoning summary.
    ReasoningSummary { None => "none", Auto => "auto" }
}

open_enum! {
    /// Tool selection mode.
    ToolChoiceMode { Auto => "auto", None => "none", Required => "required", Named => "named" }
}

open_enum! {
    /// Prompt cache policy.
    CachePolicy { None => "none", Auto => "auto", Explicit => "explicit" }
}

open_enum! {
    /// Why generation stopped.
    StopReason {
        Complete => "complete", ToolCall => "tool_call", MaxOutputTokens => "max_output_tokens", StopSequence => "stop_sequence",
        ContextWindowExceeded => "context_window_exceeded", Refusal => "refusal", ContentFiltered => "content_filtered",
        ContinueRequired => "continue_required", Compaction => "compaction", MalformedOutput => "malformed_output", OtherReason => "other"
    }
}

open_enum! {
    /// Kind of a loss warning.
    WarningKind { Unsupported => "unsupported", Compatibility => "compatibility", Deprecated => "deprecated" }
}

open_enum! {
    /// Kind of a streamed block.
    BlockKind { Text => "text", Reasoning => "reasoning", ToolCall => "tool_call", Compaction => "compaction", Custom => "custom" }
}

open_enum! {
    /// Error classification.
    ErrorKind {
        Auth => "auth", RateLimit => "rate_limit", Quota => "quota", Server => "server", Connection => "connection", Timeout => "timeout",
        Cancelled => "cancelled", NotFound => "not_found", ContextOverflow => "context_overflow", RequestTooLarge => "request_too_large",
        BadRequest => "bad_request", MalformedResponse => "malformed_response", Unsupported => "unsupported"
    }
}

open_enum! {
    /// State of a provider-executed tool call.
    ToolCallStatus { Pending => "pending", InProgress => "in_progress", Completed => "completed", Failed => "failed", Cancelled => "cancelled" }
}

open_enum! {
    /// Where a capabilities answer came from.
    CapabilitySource { Shipped => "shipped", Vendor => "vendor", User => "user" }
}

open_enum! {
    /// What a model is for.
    ModelKind { Chat => "chat", Embedding => "embedding", Unknown => "unknown" }
}

open_enum! {
    /// Input modality.
    InputKind { Text => "text", Image => "image", Document => "document", Audio => "audio", Video => "video" }
}

open_enum! {
    /// How a credential is sent.
    AuthKind { None => "none", ApiKey => "api_key", Bearer => "bearer", Sigv4 => "sigv4" }
}

open_enum! {
    /// Header that carries an API key.
    Placement { XApiKey => "x-api-key", Authorization => "authorization", XGoogApiKey => "x-goog-api-key", ApiKey => "api-key" }
}

open_enum! {
    /// Known login session.
    SessionPreset { AnthropicProfile => "anthropic_profile", ClaudeCode => "claude_code", ClaudeCodeTokenEnv => "claude_code_token_env", Codex => "codex", ChatgptPlan => "chatgpt_plan" }
}

open_enum! {
    /// What to do with a refreshed session token.
    RefreshMode { Never => "never", Memory => "memory", WriteBack => "write_back" }
}

open_enum! {
    /// Token exchange grant.
    Grant { RefreshToken => "refresh_token", JwtBearer => "jwt_bearer", ClientCredentials => "client_credentials", StsWebIdentity => "sts_web_identity", GcpMetadata => "gcp_metadata", AwsImds => "aws_imds", AwsContainer => "aws_container" }
}

/// Where a reply or an opaque block came from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Origin {
    pub wire: String,
    pub ns: Ns,
    pub base_url: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_source: Option<String>,
}

/// Explicit cache breakpoint.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CacheMark {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl: Option<CacheTtl>,
}

/// Fields every block may carry.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct BlockMeta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheMark>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor: Option<VendorMap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_executed: Option<bool>,
}

/// Plain text.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct TextBlock {
    #[serde(flatten)]
    pub meta: BlockMeta,
    pub text: String,
}

/// A system prompt block; only text is allowed there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SystemBlock {
    Text(TextBlock),
}

/// An image, by bytes, URL or vendor file id.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ImageBlock {
    #[serde(flatten)]
    pub meta: BlockMeta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

/// A document, by bytes, URL, vendor file id or text.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct DocumentBlock {
    #[serde(flatten)]
    pub meta: BlockMeta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// The model asks for a tool to run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCallBlock {
    #[serde(flatten)]
    pub meta: BlockMeta,
    pub id: String,
    pub name: String,
    pub arguments: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncated: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments_json: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<ToolCallStatus>,
}

/// What a tool returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultBlock {
    #[serde(flatten)]
    pub meta: BlockMeta,
    pub tool_call_id: String,
    pub content: Vec<Block>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

/// Model reasoning, with opaque vendor state.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ReasoningBlock {
    #[serde(flatten)]
    pub meta: BlockMeta,
    pub text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor_state: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Origin>,
}

/// Server-side summary of earlier turns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactionBlock {
    #[serde(flatten)]
    pub meta: BlockMeta,
    pub vendor_state: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    pub origin: Origin,
}

/// A vendor block the library does not model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CustomBlock {
    #[serde(flatten)]
    pub meta: BlockMeta,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Origin>,
}

/// One unit of message content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Block {
    Text(TextBlock),
    Image(ImageBlock),
    Document(DocumentBlock),
    ToolCall(ToolCallBlock),
    ToolResult(ToolResultBlock),
    Reasoning(ReasoningBlock),
    Compaction(CompactionBlock),
    Custom(CustomBlock),
}

impl Block {
    /// Convenience text block.
    pub fn text(text: impl Into<String>) -> Self {
        Block::Text(TextBlock {
            meta: BlockMeta::default(),
            text: text.into(),
        })
    }
}

/// One turn of a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: Vec<Block>,
}

/// A function the model may call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheMark>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor: Option<VendorMap>,
}

/// A tool the vendor runs itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderTool {
    pub provider_tool: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config: Option<Value>,
}

/// Any tool available to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Tool {
    Function(ToolDef),
    Provider(ProviderTool),
}

/// Which tool the model must use.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolChoice {
    pub mode: ToolChoiceMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Structured output request.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OutputFormat {
    pub json_schema: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Seconds per phase of a request.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Timeout {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connect: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub read: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total: Option<f64>,
}

/// Normalized per-call settings.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Options {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_summary: Option<ReasoningSummary>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tools: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CachePolicy>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retries: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

/// One chat or stream call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<Vec<SystemBlock>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<Options>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor: Option<VendorMap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<Timeout>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_raw: Option<bool>,
}

/// Why and how generation stopped.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stop {
    pub reason: StopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

/// Tokens read.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct InputUsage {
    pub total: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_write: Option<u64>,
}

/// Tokens written.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct OutputUsage {
    pub total: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
}

/// Token accounting for one call.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input: InputUsage,
    pub output: OutputUsage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

/// Something asked for that the vendor could not honor exactly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Warning {
    pub kind: WarningKind,
    pub feature: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A complete reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Response {
    pub content: Vec<Block>,
    pub stop: Stop,
    pub usage: Option<Usage>,
    pub warnings: Vec<Warning>,
    pub origin: Origin,
    pub latency_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_token_ms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

impl Response {
    /// All text blocks joined.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                Block::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The tool calls in order.
    pub fn tool_calls(&self) -> Vec<&ToolCallBlock> {
        self.content
            .iter()
            .filter_map(|b| match b {
                Block::ToolCall(t) => Some(t),
                _ => None,
            })
            .collect()
    }
}

/// One streamed event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    BlockStart {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        index: u32,
        kind: BlockKind,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    TextDelta {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        index: u32,
        text: String,
    },
    ReasoningDelta {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        index: u32,
        text: String,
    },
    ToolCallDelta {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        index: u32,
        arguments_json: String,
    },
    BlockStop {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        index: u32,
        block: Block,
    },
    Finish {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        response: Response,
    },
    Raw {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        parent_id: Option<String>,
        event: Value,
    },
}

/// A failed call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderError {
    pub kind: ErrorKind,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    pub retryable: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after: Option<f64>,
    pub attempts: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<Origin>,
}

impl ProviderError {
    /// A non-retryable error of the given kind.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        ProviderError {
            kind,
            message: message.into(),
            status: None,
            code: None,
            retryable: false,
            retry_after: None,
            attempts: 0,
            body: None,
            request_id: None,
            origin: None,
        }
    }
}

/// Settings of an embed call.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EmbedOptions {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retries: Option<u32>,
}

/// One embed call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub model: String,
    pub input: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub options: Option<EmbedOptions>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor: Option<VendorMap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<Timeout>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_raw: Option<bool>,
}

/// Tokens read by an embed call.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EmbedUsage {
    pub input: EmbedInputUsage,
}

/// Input token total of an embed call.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct EmbedInputUsage {
    pub total: u64,
}

/// Vectors for each input, in order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub embeddings: Vec<Vec<f64>>,
    pub usage: Option<EmbedUsage>,
    pub warnings: Vec<Warning>,
    pub origin: Origin,
    pub latency_ms: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

/// Reasoning support of a model.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ReasoningCapability {
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub levels: Option<Option<Vec<ReasoningLevel>>>,
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub can_disable: Option<Option<bool>>,
}

/// Parallel tool call support.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ParallelToolsCapability {
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub supported: Option<Option<bool>>,
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub toggle: Option<Option<bool>>,
}

/// Structured output support.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct OutputCapability {
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub json_schema: Option<Option<bool>>,
}

/// Prompt cache support.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct CacheCapability {
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub breakpoints: Option<Option<bool>>,
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub ttls: Option<Option<Vec<CacheTtl>>>,
}

/// Tool support.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ToolsCapability {
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub functions: Option<Option<bool>>,
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub hosted: Option<Option<Vec<String>>>,
}

/// Size limits.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Limits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
}

/// What a model can do; `None` inside an entry means unknown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Capabilities {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningCapability>,
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub tool_choice: Option<Option<Vec<ToolChoiceMode>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tools: Option<ParallelToolsCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<OutputCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache: Option<CacheCapability>,
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub inputs: Option<Option<Vec<InputKind>>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<ToolsCapability>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub limits: Option<Limits>,
    #[serde(
        default,
        deserialize_with = "nullable",
        skip_serializing_if = "Option::is_none"
    )]
    pub embeddings: Option<Option<bool>>,
    pub source: CapabilitySource,
}

/// What a provider says about one model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub kind: ModelKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Capabilities>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

/// Whether and how the provider is logged in.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthStatus {
    pub source: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session: Option<String>,
}

/// What the app must show the user during login.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LoginAction {
    OpenUrl {
        url: String,
        instructions: String,
    },
    ShowCode {
        url: String,
        code: String,
        instructions: String,
    },
    Paste {
        prompt: String,
    },
}

/// How a credential is sent, shared by the simple sources.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Transport {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<AuthKind>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<Placement>,
}

/// Where a credential comes from.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AuthSource {
    Inline {
        secret_ref: String,
        #[serde(flatten)]
        transport: Transport,
    },
    Env {
        names: Vec<String>,
        #[serde(flatten)]
        transport: Transport,
    },
    Resolver {
        resolver_id: String,
        #[serde(flatten)]
        transport: Transport,
    },
    Session {
        preset: SessionPreset,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        store: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        client_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        profile: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        refresh: Option<RefreshMode>,
    },
    Exchange {
        grant: Grant,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        params: Option<Value>,
        #[serde(flatten)]
        transport: Transport,
    },
}

/// Credential configuration of a provider.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Auth {
    #[serde(default)]
    pub sources: Vec<AuthSource>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origins: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extra_headers: Option<Headers>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dangerously_allow_browser: Option<bool>,
}

/// Consumer overrides of shipped knowledge.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Overrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<Capabilities>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dialect: Option<Value>,
}

/// One configured provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub ns: Ns,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth: Option<Auth>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub headers: Option<Headers>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<Timeout>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_embedding_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_system: Option<Vec<SystemBlock>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_tools: Option<Vec<Tool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_options: Option<Options>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_vendor: Option<VendorMap>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub overrides: Option<Overrides>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}
