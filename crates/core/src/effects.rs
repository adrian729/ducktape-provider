//! The core-to-host protocol, mirroring `schema/effects.schema.json`.
use crate::types::{
    AuthSource, AuthStatus, Capabilities, ChatRequest, EmbedRequest, EmbedResponse, Event, Headers,
    LoginAction, ModelInfo, ProviderConfig, ProviderError, Response, Timeout,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Bytes carried as base64 text.
pub mod b64 {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&STANDARD.encode(bytes))
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        let text = String::deserialize(d)?;
        STANDARD.decode(text).map_err(serde::de::Error::custom)
    }

    pub mod opt {
        use super::STANDARD;
        use base64::Engine;
        use serde::{Deserialize, Deserializer, Serializer};

        pub fn serialize<S: Serializer>(bytes: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
            match bytes {
                Some(b) => s.serialize_some(&STANDARD.encode(b)),
                None => s.serialize_none(),
            }
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<u8>>, D::Error> {
            match Option::<String>::deserialize(d)? {
                Some(text) => STANDARD
                    .decode(text)
                    .map(Some)
                    .map_err(serde::de::Error::custom),
                None => Ok(None),
            }
        }
    }
}

/// Time and randomness, supplied by the host on every call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Clock {
    pub now_ms: i64,
    #[serde(with = "b64")]
    pub entropy: Vec<u8>,
}

/// Opens an operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Start {
    pub clock: Clock,
    pub provider: ProviderConfig,
    pub operation: OperationKind,
}

/// What the host asked for.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OperationKind {
    Chat {
        request: ChatRequest,
    },
    Stream {
        request: ChatRequest,
    },
    Embed {
        request: EmbedRequest,
    },
    Models {
        #[serde(default)]
        refresh: bool,
    },
    ModelInfo {
        model: String,
    },
    Capabilities {
        model: String,
    },
    Login {
        #[serde(default)]
        callback: Callback,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_ms: Option<u64>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        app_name: Option<String>,
    },
    Logout,
    AuthStatus,
}

/// How a login redirect reaches the app.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Callback {
    #[default]
    Loopback,
    Paste,
}

/// Continues an operation with the result of the pending effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Resume {
    pub clock: Clock,
    pub input: Input,
}

/// The host's answer to an effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Input {
    HttpResponse {
        status: u16,
        headers: Headers,
    },
    HttpChunk {
        #[serde(with = "b64")]
        bytes: Vec<u8>,
    },
    HttpEnd,
    HttpError {
        error: TransportError,
    },
    Slept,
    Secret {
        secret_ref: String,
        found: bool,
    },
    Read {
        found: bool,
        #[serde(default, with = "b64::opt", skip_serializing_if = "Option::is_none")]
        bytes: Option<Vec<u8>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        etag: Option<String>,
    },
    Written {
        ok: bool,
        #[serde(default)]
        conflict: bool,
    },
    UserAction {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        value: Option<String>,
    },
    Callback {
        url: String,
    },
    Cancelled,
}

/// What the core returns from a start or a resume.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notify: Vec<Notification>,
    pub effect: Effect,
}

impl Step {
    /// A step with nothing to notify.
    pub fn effect(effect: Effect) -> Self {
        Step {
            notify: Vec::new(),
            effect,
        }
    }

    /// A step that ends the operation with an error.
    pub fn failed(error: ProviderError) -> Self {
        Step::effect(Effect::Failed { error })
    }
}

/// Something the host delivers before acting on the effect.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Notification {
    Event { event: Box<Event> },
    Observer { hook: ObserverHook, data: Value },
}

/// Observer callback names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObserverHook {
    OnRequest,
    OnResponse,
    OnEvent,
    OnRetry,
}

/// What the host must do next.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Effect {
    NeedHttp {
        request: HttpRequest,
    },
    Receive,
    NeedSleep {
        ms: u64,
    },
    NeedSecret {
        secret_ref: String,
        source: AuthSource,
    },
    NeedRead {
        locator: Locator,
    },
    NeedWrite {
        locator: Locator,
        #[serde(with = "b64")]
        bytes: Vec<u8>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        expect: Option<String>,
        #[serde(default = "default_true")]
        private: bool,
    },
    NeedUserAction {
        action: LoginAction,
    },
    NeedCallback {
        redirect_uri: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        timeout_ms: Option<u64>,
        action: LoginAction,
    },
    Done {
        result: OperationResult,
    },
    Failed {
        error: ProviderError,
    },
}

fn default_true() -> bool {
    true
}

/// The value of a finished operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum OperationResult {
    Response { response: Response },
    StreamEnd,
    Embed { response: EmbedResponse },
    Models { models: Vec<ModelInfo> },
    ModelInfo { info: Option<ModelInfo> },
    Capabilities { capabilities: Option<Capabilities> },
    AuthStatus { status: AuthStatus },
    None,
}

/// HTTP method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Method {
    Get,
    Post,
    Put,
    Delete,
}

/// Why a request is made; auth requests are never logged or retried by the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Purpose {
    Chat,
    Embed,
    Models,
    Auth,
    Fetch,
}

/// A request for the host transport to send as given.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HttpRequest {
    pub method: Method,
    pub url: String,
    pub headers: Headers,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout: Option<Timeout>,
    pub purpose: Purpose,
}

/// Failure of the host transport.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TransportError {
    pub kind: TransportErrorKind,
    pub message: String,
}

/// Transport failure classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransportErrorKind {
    Connect,
    Timeout,
    Tls,
    Dns,
    Reset,
    Cancelled,
    Other,
}

/// Where the host reads or writes bytes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Locator {
    File { path: String },
    Keychain { service: String, account: String },
    Env { name: String },
}
