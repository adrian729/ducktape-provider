//! Codecs: one per vendor wire protocol.
use crate::effects::HttpRequest;
use crate::types::*;
use serde_json::{Map, Value};
use std::collections::BTreeSet;

pub mod anthropic;

/// Everything a codec needs to know about the call besides the request.
#[derive(Debug, Clone)]
pub struct CallContext {
    pub ns: Ns,
    pub wire: &'static str,
    pub base_url: String,
    pub model: String,
    pub auth_source: Option<String>,
    pub api_version: Option<String>,
    pub headers: Headers,
    pub timeout: Option<Timeout>,
    pub strict: bool,
    pub include_raw: bool,
    pub overrides: Option<Overrides>,
}

impl CallContext {
    /// Origin stamp of this call.
    pub fn origin(&self) -> Origin {
        Origin {
            wire: self.wire.to_owned(),
            ns: self.ns.clone(),
            base_url: self.base_url.clone(),
            model: self.model.clone(),
            auth_source: self.auth_source.clone(),
        }
    }

    /// Whether an opaque block from `origin` may be replayed here.
    pub fn accepts(&self, origin: Option<&Origin>) -> bool {
        origin.is_some_and(|o| o.wire == self.wire && same_origin(&o.base_url, &self.base_url))
    }
}

/// A request ready to send, with what was lost on the way.
#[derive(Debug, Clone)]
pub struct Built {
    pub request: HttpRequest,
    pub warnings: Vec<Warning>,
}

/// What a decoder produced from one chunk.
#[derive(Debug, Default)]
pub struct Decoded {
    pub events: Vec<Event>,
    pub complete: bool,
}

/// Incremental decoder of one streamed chat reply.
pub trait ChatDecoder {
    /// Decodes a chunk into normalized events.
    fn feed(&mut self, bytes: &[u8]) -> Result<Decoded, ProviderError>;
    /// Builds the response once the body ended; `latency_ms` and `first_token_ms` are filled by the caller.
    fn finish(&mut self) -> Result<Response, ProviderError>;
}

/// A vendor wire protocol.
pub trait Wire {
    /// Builds the streamed chat request.
    fn build_chat(
        &self,
        ctx: &CallContext,
        req: &ChatRequest,
        index_offset: u32,
    ) -> Result<Built, ProviderError>;
    /// Decoder for a chat reply; `index_offset` shifts block indices after a continuation.
    fn chat_decoder(&self, ctx: &CallContext, index_offset: u32) -> Box<dyn ChatDecoder>;
    /// Whether this stop asks the core to continue the turn.
    fn wants_continuation(&self, response: &Response) -> bool;
    /// Builds the embed request.
    fn build_embed(&self, ctx: &CallContext, req: &EmbedRequest) -> Result<Built, ProviderError>;
    /// Decodes the embed reply body.
    fn decode_embed(&self, ctx: &CallContext, body: &[u8]) -> Result<EmbedResponse, ProviderError>;
    /// Builds the model listing request; `None` when the wire has no listing.
    fn build_models(&self, ctx: &CallContext) -> Option<HttpRequest>;
    /// Decodes a model listing page; returns the models and the URL of the next page.
    fn decode_models(
        &self,
        ctx: &CallContext,
        body: &[u8],
    ) -> Result<(Vec<ModelInfo>, Option<HttpRequest>), ProviderError>;
    /// Builds the single-model request; `None` when the wire has no such endpoint.
    fn build_model_info(&self, ctx: &CallContext, model: &str) -> Option<HttpRequest>;
    /// Decodes a single-model reply.
    fn decode_model_info(&self, ctx: &CallContext, body: &[u8])
        -> Result<ModelInfo, ProviderError>;
    /// Shipped capabilities of a model, when the table knows it.
    fn shipped_capabilities(&self, model: &str) -> Option<Capabilities>;
    /// Classifies a non-2xx reply.
    fn classify(&self, status: u16, headers: &Headers, body: &[u8]) -> ProviderError;
}

/// Codec for a namespace.
pub fn for_ns(ns: &Ns) -> Option<Box<dyn Wire>> {
    match ns {
        Ns::Anthropic => Some(Box::new(anthropic::AnthropicMessages)),
        _ => None,
    }
}

/// A JSON body under construction that remembers which keys the caller set.
#[derive(Debug, Default)]
pub struct Body {
    pub map: Map<String, Value>,
    pub user_keys: BTreeSet<String>,
    pub warnings: Vec<Warning>,
}

impl Body {
    /// An empty body.
    pub fn new() -> Self {
        Body::default()
    }

    /// Sets a key from a caller option.
    pub fn user(&mut self, key: &str, value: Value) {
        self.map.insert(key.to_owned(), value);
        self.user_keys.insert(key.to_owned());
    }

    /// Sets a key from a library default.
    pub fn lib(&mut self, key: &str, value: Value) {
        self.map.insert(key.to_owned(), value);
    }

    /// Records a loss.
    pub fn warn(&mut self, kind: WarningKind, feature: &str, detail: impl Into<String>) {
        self.warnings.push(Warning {
            kind,
            feature: feature.to_owned(),
            detail: Some(detail.into()),
        });
    }

    /// Merges the caller's raw vendor fields; a clash with a caller-set option is an argument error.
    pub fn merge_vendor(
        &mut self,
        vendor: Option<&Map<String, Value>>,
    ) -> Result<(), ProviderError> {
        let Some(vendor) = vendor else { return Ok(()) };
        for (key, value) in vendor {
            if self.user_keys.contains(key) {
                return Err(argument_error(format!(
                    "vendor field `{key}` clashes with an option that sets the same field"
                )));
            }
            match (self.map.get_mut(key), value) {
                (Some(Value::Object(existing)), Value::Object(incoming)) => {
                    for (k, v) in incoming {
                        existing.insert(k.clone(), v.clone());
                    }
                }
                _ => {
                    self.map.insert(key.clone(), value.clone());
                }
            }
        }
        Ok(())
    }

    /// Fails when `strict` and anything was lost.
    pub fn enforce_strict(&self, strict: bool) -> Result<(), ProviderError> {
        match (strict, self.warnings.first()) {
            (true, Some(w)) => {
                let mut e = ProviderError::new(
                    ErrorKind::Unsupported,
                    format!("{}: {}", w.feature, w.detail.clone().unwrap_or_default()),
                );
                e.code = Some("strict".to_owned());
                Err(e)
            }
            _ => Ok(()),
        }
    }
}

/// An invalid-argument error, mapped to the language's own error by bindings.
pub fn argument_error(message: impl Into<String>) -> ProviderError {
    let mut e = ProviderError::new(ErrorKind::BadRequest, message);
    e.code = Some("invalid_argument".to_owned());
    e
}

/// A reply the codec could not parse.
pub fn malformed(message: impl Into<String>) -> ProviderError {
    ProviderError::new(ErrorKind::MalformedResponse, message)
}

/// The vendor map entry for a namespace.
pub fn vendor_for<'a>(vendor: Option<&'a VendorMap>, ns: &Ns) -> Option<&'a Map<String, Value>> {
    vendor
        .and_then(|v| v.get(ns.as_str()))
        .and_then(Value::as_object)
}

/// Copies a vendor extras map into a block object.
pub fn apply_vendor(target: &mut Map<String, Value>, vendor: Option<&VendorMap>, ns: &Ns) {
    if let Some(extra) = vendor_for(vendor, ns) {
        for (k, v) in extra {
            target.insert(k.clone(), v.clone());
        }
    }
}

/// Media type of an image from the first bytes of its base64 data.
pub fn sniff_image(data_b64: &str) -> Option<&'static str> {
    use base64::Engine;
    let head: String = data_b64.chars().take(24).collect();
    let bytes = base64::engine::general_purpose::STANDARD_NO_PAD
        .decode(head.trim_end_matches('='))
        .ok()?;
    if bytes.starts_with(&[0x89, b'P', b'N', b'G']) {
        Some("image/png")
    } else if bytes.starts_with(&[0xFF, 0xD8, 0xFF]) {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF8") {
        Some("image/gif")
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        Some("image/webp")
    } else {
        None
    }
}

/// Parses a response body as JSON.
pub fn parse_json(body: &[u8]) -> Result<Value, ProviderError> {
    serde_json::from_slice(body).map_err(|e| malformed(format!("reply is not JSON: {e}")))
}

/// Whether two URLs share scheme and host.
pub fn same_origin(a: &str, b: &str) -> bool {
    fn origin(url: &str) -> (String, String) {
        let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
        let host = rest.split(['/', '?', '#']).next().unwrap_or("");
        (scheme.to_ascii_lowercase(), host.to_ascii_lowercase())
    }
    origin(a) == origin(b)
}

/// Joins a base URL and a path.
pub fn join_url(base: &str, path: &str) -> String {
    format!(
        "{}/{}",
        base.trim_end_matches('/'),
        path.trim_start_matches('/')
    )
}

/// Reads a string field.
pub fn str_field<'a>(v: &'a Value, key: &str) -> Option<&'a str> {
    v.get(key).and_then(Value::as_str)
}

/// Reads an unsigned integer field.
pub fn u64_field(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(Value::as_u64)
}
