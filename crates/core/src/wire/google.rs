//! The Google Gemini Interactions wire.
use super::*;
use crate::effects::{HttpRequest, Method, Purpose};
use crate::retry::parse_retry_after;
use crate::sse::SseParser;
use serde_json::json;
use std::collections::BTreeMap;

/// The `google.interactions` codec.
#[derive(Debug, Clone, Copy, Default)]
pub struct GoogleInteractions;

const API_VERSION: &str = "v1beta";
const HOSTED: &[&str] = &[
    "google.google_search",
    "google.google_maps",
    "google.code_execution",
    "google.url_context",
    "google.file_search",
    "google.computer_use",
    "google.mcp_server",
];
const BLOCKED: &[&str] = &[
    "safety",
    "recitation",
    "language",
    "prohibited_content",
    "spii",
    "blocklist",
    "image_safety",
    "image_prohibited_content",
    "image_recitation",
    "image_other",
    "content_blocked",
];
const MALFORMED: &[&str] = &[
    "malformed_function_call",
    "malformed_tool_call",
    "unexpected_tool_call",
    "too_many_tool_calls",
    "no_image",
];

fn version(ctx: &CallContext) -> String {
    ctx.api_version
        .clone()
        .unwrap_or_else(|| API_VERSION.to_owned())
}

fn base_headers(ctx: &CallContext) -> Headers {
    let mut h = Headers::new();
    h.insert("content-type".into(), "application/json".into());
    for (k, v) in &ctx.headers {
        h.insert(k.to_ascii_lowercase(), v.clone());
    }
    h
}

fn get(ctx: &CallContext, path: &str) -> HttpRequest {
    HttpRequest {
        method: Method::Get,
        url: join_url(&ctx.base_url, path),
        headers: base_headers(ctx),
        body: None,
        stream: false,
        timeout: ctx.timeout,
        purpose: Purpose::Models,
    }
}

fn blocked_stop(code: &str) -> Option<StopReason> {
    if BLOCKED.contains(&code) {
        Some(StopReason::ContentFiltered)
    } else if MALFORMED.contains(&code) {
        Some(StopReason::MalformedOutput)
    } else {
        None
    }
}

fn label_ok(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 63
        && id.starts_with(|c: char| c.is_ascii_lowercase())
        && id
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_' || c == '-')
}

struct Encoder<'a> {
    ctx: &'a CallContext,
    warnings: Vec<Warning>,
    call_names: Vec<(String, String)>,
    marks: bool,
}

impl Encoder<'_> {
    fn warn(&mut self, feature: &str, detail: impl Into<String>) {
        self.warnings.push(Warning {
            kind: WarningKind::Unsupported,
            feature: feature.to_owned(),
            detail: Some(detail.into()),
        });
    }

    fn part(&mut self, block: &Block) -> Option<Value> {
        let mut o = Map::new();
        match block {
            Block::Text(t) => {
                self.marks |= t.meta.cache.is_some();
                o.insert("type".into(), json!("text"));
                o.insert("text".into(), json!(t.text));
                apply_vendor(&mut o, t.meta.vendor.as_ref(), &self.ctx.vendor);
            }
            Block::Image(b) => {
                self.marks |= b.meta.cache.is_some();
                o.insert("type".into(), json!("image"));
                if let Some(data) = &b.data {
                    o.insert("data".into(), json!(data));
                    o.insert(
                        "mime_type".into(),
                        json!(b
                            .media_type
                            .clone()
                            .or_else(|| sniff_image(data).map(str::to_owned))
                            .unwrap_or_else(|| "image/png".to_owned())),
                    );
                } else if let Some(uri) = b.url.as_ref().or(b.file_id.as_ref()) {
                    o.insert("uri".into(), json!(uri));
                    if let Some(m) = &b.media_type {
                        o.insert("mime_type".into(), json!(m));
                    }
                } else {
                    self.warn("image", "image block needs data, url or file_id");
                    return None;
                }
                apply_vendor(&mut o, b.meta.vendor.as_ref(), &self.ctx.vendor);
            }
            Block::Document(b) => {
                self.marks |= b.meta.cache.is_some();
                if let Some(text) = &b.text {
                    o.insert("type".into(), json!("text"));
                    o.insert("text".into(), json!(text));
                } else {
                    o.insert("type".into(), json!("document"));
                    if let Some(data) = &b.data {
                        o.insert("data".into(), json!(data));
                    } else if let Some(uri) = b.url.as_ref().or(b.file_id.as_ref()) {
                        o.insert("uri".into(), json!(uri));
                    } else {
                        self.warn(
                            "document",
                            "document block needs data, url, file_id or text",
                        );
                        return None;
                    }
                    o.insert(
                        "mime_type".into(),
                        json!(b
                            .media_type
                            .clone()
                            .unwrap_or_else(|| "application/pdf".to_owned())),
                    );
                }
                apply_vendor(&mut o, b.meta.vendor.as_ref(), &self.ctx.vendor);
            }
            other => {
                self.warn(
                    "content",
                    format!("{} block dropped from this turn", kind_name(other)),
                );
                return None;
            }
        }
        Some(Value::Object(o))
    }

    fn steps(&mut self, messages: &[Message], system: &mut Vec<String>) -> Vec<Value> {
        let mut out = Vec::new();
        for m in messages {
            match &m.role {
                Role::User | Role::Other(_) => {
                    let parts: Vec<Value> = m.content.iter().filter_map(|b| self.part(b)).collect();
                    out.push(json!({"type": "user_input", "content": parts}));
                }
                Role::System => {
                    self.warnings.push(Warning {
                        kind: WarningKind::Compatibility,
                        feature: "system".into(),
                        detail: Some(
                            "mid-conversation system message folded into the system instruction"
                                .into(),
                        ),
                    });
                    for b in &m.content {
                        if let Block::Text(t) = b {
                            system.push(t.text.clone());
                        }
                    }
                }
                Role::Tool => {
                    for b in &m.content {
                        match b {
                            Block::ToolResult(r) => out.push(self.tool_result(r)),
                            other => self.warn(
                                "content",
                                format!("{} block ignored in a tool turn", kind_name(other)),
                            ),
                        }
                    }
                }
                Role::Assistant => self.assistant(&m.content, &mut out),
            }
        }
        out
    }

    fn tool_result(&mut self, r: &ToolResultBlock) -> Value {
        let name = self
            .call_names
            .iter()
            .rev()
            .find(|(id, _)| *id == r.tool_call_id)
            .map(|(_, n)| n.clone());
        let mut result = Vec::new();
        for b in &r.content {
            match b {
                Block::Text(_) | Block::Image(_) => {
                    if let Some(p) = self.part(b) {
                        result.push(p);
                    }
                }
                Block::Document(d) if d.text.is_some() => {
                    if let Some(p) = self.part(b) {
                        result.push(p);
                    }
                }
                other => self.warn(
                    "tool_result",
                    format!(
                        "{} block dropped from a tool result; Google takes text and images",
                        kind_name(other)
                    ),
                ),
            }
        }
        let mut o = Map::new();
        o.insert("type".into(), json!("function_result"));
        o.insert("call_id".into(), json!(r.tool_call_id));
        if let Some(n) = name {
            o.insert("name".into(), json!(n));
        }
        o.insert("result".into(), Value::Array(result));
        if let Some(true) = r.is_error {
            o.insert("is_error".into(), json!(true));
        }
        apply_vendor(&mut o, r.meta.vendor.as_ref(), &self.ctx.vendor);
        Value::Object(o)
    }

    fn assistant(&mut self, content: &[Block], out: &mut Vec<Value>) {
        let mut pending: Vec<Value> = Vec::new();
        let flush = |pending: &mut Vec<Value>, out: &mut Vec<Value>| {
            if !pending.is_empty() {
                out.push(json!({"type": "model_output", "content": Value::Array(std::mem::take(pending))}));
            }
        };
        for b in content {
            match b {
                Block::Text(_) | Block::Image(_) => {
                    if let Some(p) = self.part(b) {
                        pending.push(p);
                    }
                }
                Block::ToolCall(c) => {
                    flush(&mut pending, out);
                    self.call_names.push((c.id.clone(), c.name.clone()));
                    let mut o = Map::new();
                    o.insert("type".into(), json!("function_call"));
                    o.insert("id".into(), json!(c.id));
                    o.insert("name".into(), json!(c.name));
                    o.insert("arguments".into(), c.arguments.clone());
                    apply_vendor(&mut o, c.meta.vendor.as_ref(), &self.ctx.vendor);
                    out.push(Value::Object(o));
                }
                Block::Reasoning(r) => {
                    flush(&mut pending, out);
                    if !self.ctx.accepts(r.origin.as_ref()) {
                        self.warn("reasoning", "reasoning block from another endpoint dropped");
                        continue;
                    }
                    match &r.vendor_state {
                        Some(Value::Object(step)) => out.push(Value::Object(step.clone())),
                        _ => self.warn("reasoning", "reasoning block without vendor state dropped"),
                    }
                }
                Block::Custom(c) => {
                    flush(&mut pending, out);
                    let ours = c.kind.starts_with("google.");
                    if !ours || (c.origin.is_some() && !self.ctx.accepts(c.origin.as_ref())) {
                        self.warn("custom", format!("custom block `{}` dropped", c.kind));
                        continue;
                    }
                    match &c.data {
                        Some(Value::Object(d)) => out.push(Value::Object(d.clone())),
                        _ => self.warn("custom", format!("custom block `{}` has no data", c.kind)),
                    }
                }
                other => self.warn(
                    "content",
                    format!("{} block dropped from an assistant turn", kind_name(other)),
                ),
            }
        }
        flush(&mut pending, out);
    }

    fn tools(&mut self, tools: &[Tool]) -> Vec<Value> {
        let mut out = Vec::new();
        for t in tools {
            match t {
                Tool::Function(d) => {
                    let mut o = Map::new();
                    o.insert("type".into(), json!("function"));
                    o.insert("name".into(), json!(d.name));
                    o.insert("description".into(), json!(d.description));
                    o.insert("parameters".into(), d.input_schema.clone());
                    if d.strict.is_some() {
                        self.warn("strict", "strict tool schemas are not supported on Google");
                    }
                    if d.cache.is_some() {
                        self.marks = true;
                    }
                    apply_vendor(&mut o, d.vendor.as_ref(), &self.ctx.vendor);
                    out.push(Value::Object(o));
                }
                Tool::Provider(p) => {
                    let Some(name) = p.provider_tool.strip_prefix("google.") else {
                        self.warn(
                            "provider_tool",
                            format!("`{}` is not a Google tool", p.provider_tool),
                        );
                        continue;
                    };
                    let mut o = match &p.config {
                        Some(Value::Object(cfg)) => cfg.clone(),
                        _ => Map::new(),
                    };
                    o.entry("type").or_insert(json!(name));
                    out.push(Value::Object(o));
                }
            }
        }
        out
    }
}

fn kind_name(b: &Block) -> &'static str {
    match b {
        Block::Text(_) => "text",
        Block::Image(_) => "image",
        Block::Document(_) => "document",
        Block::ToolCall(_) => "tool_call",
        Block::ToolResult(_) => "tool_result",
        Block::Reasoning(_) => "reasoning",
        Block::Compaction(_) => "compaction",
        Block::Custom(_) => "custom",
    }
}

fn thinking_level(level: &ReasoningLevel) -> (Option<&'static str>, Option<String>) {
    match level {
        ReasoningLevel::Auto => (None, None),
        ReasoningLevel::None => (
            Some("minimal"),
            Some("reasoning cannot be disabled on Google; minimal used".into()),
        ),
        ReasoningLevel::Minimal => (Some("minimal"), None),
        ReasoningLevel::Low => (Some("low"), None),
        ReasoningLevel::Medium => (Some("medium"), None),
        ReasoningLevel::High => (Some("high"), None),
        ReasoningLevel::Xhigh => (Some("high"), Some("xhigh rounded to high".into())),
        ReasoningLevel::Max => (Some("high"), Some("max rounded to high".into())),
        ReasoningLevel::Other(_) => (
            None,
            Some(format!(
                "unknown reasoning level `{}` ignored",
                level.as_str()
            )),
        ),
    }
}

fn model_info(m: &Value) -> Option<ModelInfo> {
    let name = str_field(m, "name")?;
    let id = name.strip_prefix("models/").unwrap_or(name).to_owned();
    let methods: Vec<&str> = m
        .get("supportedGenerationMethods")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let kind = if methods.contains(&"embedContent") && !methods.contains(&"generateContent") {
        ModelKind::Embedding
    } else if methods.contains(&"generateContent")
        || methods.contains(&"createInteraction")
        || id.starts_with("gemini")
    {
        ModelKind::Chat
    } else {
        ModelKind::Unknown
    };
    Some(ModelInfo {
        id,
        kind,
        context_window: u64_field(m, "inputTokenLimit"),
        max_output_tokens: u64_field(m, "outputTokenLimit"),
        capabilities: None,
        raw: Some(m.clone()),
    })
}

impl Wire for GoogleInteractions {
    fn build_chat(
        &self,
        ctx: &CallContext,
        req: &ChatRequest,
        _index_offset: u32,
    ) -> Result<Built, ProviderError> {
        let opts = req.options.clone().unwrap_or_default();
        let cache_policy = opts.cache.clone().unwrap_or(CachePolicy::Auto);
        let mut enc = Encoder {
            ctx,
            warnings: Vec::new(),
            call_names: Vec::new(),
            marks: false,
        };
        let mut body = Body::new();
        body.user("model", json!(ctx.model));
        body.lib("stream", json!(true));
        body.lib("store", json!(false));
        let mut system: Vec<String> = Vec::new();
        if let Some(blocks) = req.system.as_ref() {
            for SystemBlock::Text(b) in blocks {
                enc.marks |= b.meta.cache.is_some();
                system.push(b.text.clone());
            }
        }
        let steps = enc.steps(&req.messages, &mut system);
        if !system.is_empty() {
            body.user("system_instruction", json!(system.join("\n\n")));
        }
        body.lib("input", Value::Array(steps));
        if let Some(tools) = req.tools.as_ref().filter(|t| !t.is_empty()) {
            let encoded = enc.tools(tools);
            if !encoded.is_empty() {
                body.user("tools", Value::Array(encoded));
            }
        }
        let mut gen = Map::new();
        let mut gen_lib = true;
        if let Some(n) = opts.max_output_tokens {
            gen.insert("max_output_tokens".into(), json!(n));
            gen_lib = false;
        }
        if let Some(stop) = opts.stop_sequences.as_ref().filter(|s| !s.is_empty()) {
            gen.insert("stop_sequences".into(), json!(stop));
            gen_lib = false;
        }
        let level = opts.reasoning.clone().unwrap_or(ReasoningLevel::Auto);
        let (wire_level, note) = thinking_level(&level);
        if let Some(note) = note {
            enc.warnings.push(Warning {
                kind: WarningKind::Compatibility,
                feature: "reasoning".into(),
                detail: Some(note),
            });
        }
        if let Some(l) = wire_level {
            gen.insert("thinking_level".into(), json!(l));
            gen_lib = false;
        }
        match opts
            .reasoning_summary
            .clone()
            .unwrap_or(ReasoningSummary::Auto)
        {
            ReasoningSummary::None => {
                gen.insert("thinking_summaries".into(), json!("none"));
                gen_lib = false;
            }
            _ => {
                gen.insert("thinking_summaries".into(), json!("auto"));
                gen_lib &= opts.reasoning_summary.is_none();
            }
        }
        if let Some(tc) = &opts.tool_choice {
            let v = match tc.mode.as_str() {
                "auto" => json!("auto"),
                "none" => json!("none"),
                "required" => json!("any"),
                "named" => {
                    json!({"allowed_tools": {"mode": "any", "tools": [tc.name.clone().unwrap_or_default()]}})
                }
                other => {
                    return Err(argument_error(format!(
                        "unknown tool_choice mode `{other}`"
                    )))
                }
            };
            gen.insert("tool_choice".into(), v);
            gen_lib = false;
        }
        if opts.parallel_tools.is_some() {
            enc.warn("parallel_tools", "Google has no parallel tool call toggle");
        }
        if let Some(out) = &opts.output {
            body.user(
                "response_format",
                json!({"type": "text", "mime_type": "application/json", "schema": out.json_schema}),
            );
        }
        if let Some(uid) = &opts.user_id {
            if label_ok(uid) {
                body.user("labels", json!({"safety_identifier": uid}));
            } else {
                enc.warn("user_id", "Google labels allow only lowercase letters, digits, `_` and `-`, at most 63 characters, starting with a letter");
            }
        }
        if enc.marks && !matches!(cache_policy, CachePolicy::None) {
            enc.warn("cache", "Google caches implicitly; cache marks are ignored");
        }
        let mut vendor = vendor_for(req.vendor.as_ref(), &ctx.vendor).cloned();
        if let Some(v) = vendor.as_mut() {
            if let Some(Value::Object(extra)) = v.remove("generation_config") {
                for (k, val) in extra {
                    if gen.contains_key(&k)
                        && !(gen_lib
                            || k == "thinking_summaries" && opts.reasoning_summary.is_none())
                    {
                        return Err(argument_error(format!("vendor field `generation_config.{k}` clashes with an option that sets the same field")));
                    }
                    gen.insert(k, val);
                }
            }
        }
        if !gen.is_empty() {
            if gen_lib {
                body.lib("generation_config", Value::Object(gen));
            } else {
                body.user("generation_config", Value::Object(gen));
            }
        }
        body.warnings.append(&mut enc.warnings);
        body.merge_vendor(vendor.as_ref())?;
        body.enforce_strict(ctx.strict)?;
        let mut headers = base_headers(ctx);
        for (k, v) in req.headers.iter().flatten() {
            headers.insert(k.to_ascii_lowercase(), v.clone());
        }
        let request = HttpRequest {
            method: Method::Post,
            url: join_url(&ctx.base_url, &format!("{}/interactions", version(ctx))),
            headers,
            body: Some(Value::Object(body.map).to_string()),
            stream: true,
            timeout: req.timeout.or(ctx.timeout),
            purpose: Purpose::Chat,
        };
        Ok(Built {
            request,
            warnings: body.warnings,
        })
    }

    fn chat_decoder(
        &self,
        ctx: &CallContext,
        _req: &ChatRequest,
        index_offset: u32,
    ) -> Box<dyn ChatDecoder> {
        Box::new(Decoder::new(ctx.clone(), index_offset))
    }

    fn wants_continuation(&self, _response: &Response) -> bool {
        false
    }

    fn build_embed(&self, ctx: &CallContext, req: &EmbedRequest) -> Result<Built, ProviderError> {
        let model_name = format!("models/{}", ctx.model);
        let vendor = vendor_for(req.vendor.as_ref(), &ctx.vendor);
        let dims = req.options.as_ref().and_then(|o| o.dimensions);
        let mut requests = Vec::with_capacity(req.input.len());
        for text in &req.input {
            let mut r = Map::new();
            r.insert("model".into(), json!(model_name));
            r.insert("content".into(), json!({"parts": [{"text": text}]}));
            if let Some(d) = dims {
                r.insert("outputDimensionality".into(), json!(d));
            }
            if let Some(v) = vendor {
                for (k, val) in v {
                    if k == "outputDimensionality" && dims.is_some() {
                        return Err(argument_error(
                            "vendor field `outputDimensionality` clashes with `dimensions`",
                        ));
                    }
                    r.insert(k.clone(), val.clone());
                }
            }
            requests.push(Value::Object(r));
        }
        let request = HttpRequest {
            method: Method::Post,
            url: join_url(
                &ctx.base_url,
                &format!("{}/models/{}:batchEmbedContents", version(ctx), ctx.model),
            ),
            headers: base_headers(ctx),
            body: Some(json!({"requests": requests}).to_string()),
            stream: false,
            timeout: req.timeout.or(ctx.timeout),
            purpose: Purpose::Embed,
        };
        Ok(Built {
            request,
            warnings: Vec::new(),
        })
    }

    fn decode_embed(&self, ctx: &CallContext, body: &[u8]) -> Result<EmbedResponse, ProviderError> {
        let v = parse_json(body)?;
        let rows = v
            .get("embeddings")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("embed reply has no `embeddings`"))?;
        let embeddings = rows
            .iter()
            .map(|row| {
                row.get("values")
                    .and_then(Value::as_array)
                    .map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<f64>>())
                    .ok_or_else(|| malformed("embedding row has no `values`"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let usage = v
            .get("usageMetadata")
            .and_then(|u| u64_field(u, "promptTokenCount"))
            .map(|t| EmbedUsage {
                input: EmbedInputUsage { total: t },
            });
        Ok(EmbedResponse {
            embeddings,
            usage,
            warnings: Vec::new(),
            origin: ctx.origin(),
            latency_ms: 0.0,
            raw: if ctx.include_raw { Some(v) } else { None },
        })
    }

    fn build_models(&self, ctx: &CallContext) -> Option<HttpRequest> {
        Some(get(ctx, &format!("{}/models?pageSize=1000", version(ctx))))
    }

    fn decode_models(
        &self,
        ctx: &CallContext,
        body: &[u8],
    ) -> Result<(Vec<ModelInfo>, Option<HttpRequest>), ProviderError> {
        let v = parse_json(body)?;
        let rows = v
            .get("models")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("model listing has no `models`"))?;
        let models = rows.iter().filter_map(model_info).collect();
        let next = str_field(&v, "nextPageToken").map(|token| {
            get(
                ctx,
                &format!("{}/models?pageSize=1000&pageToken={token}", version(ctx)),
            )
        });
        Ok((models, next))
    }

    fn build_model_info(&self, ctx: &CallContext, model: &str) -> Option<HttpRequest> {
        Some(get(ctx, &format!("{}/models/{model}", version(ctx))))
    }

    fn decode_model_info(
        &self,
        _ctx: &CallContext,
        body: &[u8],
    ) -> Result<ModelInfo, ProviderError> {
        let v = parse_json(body)?;
        model_info(&v).ok_or_else(|| malformed("model reply has no `name`"))
    }

    fn shipped_capabilities(&self, model: &str) -> Option<Capabilities> {
        let m = model.to_ascii_lowercase();
        if m.starts_with("gemini-embedding") || m.starts_with("text-embedding") {
            return Some(Capabilities {
                reasoning: None,
                tool_choice: None,
                parallel_tools: None,
                output: None,
                cache: None,
                inputs: Some(Some(vec![InputKind::Text])),
                tools: None,
                limits: None,
                embeddings: Some(Some(true)),
                source: CapabilitySource::Shipped,
            });
        }
        if !m.starts_with("gemini-") {
            return None;
        }
        Some(Capabilities {
            reasoning: Some(ReasoningCapability {
                levels: Some(Some(
                    ["auto", "minimal", "low", "medium", "high"]
                        .iter()
                        .map(|l| ReasoningLevel::from(*l))
                        .collect(),
                )),
                can_disable: Some(Some(false)),
            }),
            tool_choice: Some(Some(vec![
                ToolChoiceMode::Auto,
                ToolChoiceMode::None,
                ToolChoiceMode::Required,
                ToolChoiceMode::Named,
            ])),
            parallel_tools: Some(ParallelToolsCapability {
                supported: Some(Some(true)),
                toggle: Some(Some(false)),
            }),
            output: Some(OutputCapability {
                json_schema: Some(Some(true)),
            }),
            cache: Some(CacheCapability {
                breakpoints: Some(Some(false)),
                ttls: Some(None),
            }),
            inputs: Some(Some(vec![
                InputKind::Text,
                InputKind::Image,
                InputKind::Document,
                InputKind::Audio,
                InputKind::Video,
            ])),
            tools: Some(ToolsCapability {
                functions: Some(Some(true)),
                hosted: Some(Some(HOSTED.iter().map(|s| s.to_string()).collect())),
            }),
            limits: None,
            embeddings: Some(Some(false)),
            source: CapabilitySource::Shipped,
        })
    }

    fn classify(&self, status: u16, headers: &Headers, body: &[u8]) -> ProviderError {
        let parsed: Option<Value> = serde_json::from_slice(body).ok();
        let err = parsed.as_ref().and_then(|v| v.get("error"));
        let code = err
            .and_then(|e| str_field(e, "code"))
            .unwrap_or("")
            .to_owned();
        let message = err
            .and_then(|e| str_field(e, "message"))
            .unwrap_or("")
            .to_owned();
        let retry_after = headers
            .get("retry-after")
            .and_then(|v| parse_retry_after(v));
        let lower = message.to_ascii_lowercase();
        let (kind, retryable) = match (status, code.as_str()) {
            (401, _) | (403, _) => (ErrorKind::Auth, false),
            (402, _) => (ErrorKind::Quota, false),
            (404, _) => (ErrorKind::NotFound, false),
            (429, "quota_exceeded") => (ErrorKind::Quota, false),
            (429, _) => (ErrorKind::RateLimit, true),
            (499, _) => (ErrorKind::Cancelled, false),
            (400, _)
                if lower.contains("token")
                    && (lower.contains("exceed")
                        || lower.contains("too long")
                        || lower.contains("too large")) =>
            {
                (ErrorKind::ContextOverflow, false)
            }
            (413, _) => (ErrorKind::RequestTooLarge, false),
            (504, _) => (ErrorKind::Timeout, true),
            (s, _) if s >= 500 => (ErrorKind::Server, true),
            (s, _) if (400..500).contains(&s) => (ErrorKind::BadRequest, false),
            _ => (ErrorKind::MalformedResponse, false),
        };
        ProviderError {
            kind,
            message: if message.is_empty() {
                format!("HTTP {status}")
            } else {
                format!("HTTP {status}: {message}")
            },
            status: Some(status),
            code: if code.is_empty() { None } else { Some(code) },
            retryable,
            retry_after,
            attempts: 0,
            body: parsed,
            request_id: None,
            origin: None,
        }
    }

    fn error_as_response(&self, ctx: &CallContext, _status: u16, body: &[u8]) -> Option<Response> {
        let v: Value = serde_json::from_slice(body).ok()?;
        let code = v.get("error").and_then(|e| str_field(e, "code"))?;
        let reason = blocked_stop(code)?;
        Some(Response {
            content: Vec::new(),
            stop: Stop {
                reason,
                raw: Some(json!(code)),
            },
            usage: None,
            warnings: Vec::new(),
            origin: ctx.origin(),
            latency_ms: 0.0,
            first_token_ms: None,
            raw: if ctx.include_raw { Some(v) } else { None },
        })
    }
}

/// A step being streamed.
struct Partial {
    index: u32,
    kind: &'static str,
    step: Map<String, Value>,
    text: String,
    json: String,
    summary: Vec<Value>,
    started: bool,
}

/// Decoder of one Interactions SSE stream.
struct Decoder {
    ctx: CallContext,
    offset: u32,
    sse: SseParser,
    next_index: u32,
    open: BTreeMap<u64, Partial>,
    content: Vec<Block>,
    status: Option<String>,
    blocked: Option<(StopReason, String)>,
    usage: Option<Value>,
    complete: bool,
}

impl Decoder {
    fn new(ctx: CallContext, offset: u32) -> Self {
        Decoder {
            ctx,
            offset,
            sse: SseParser::default(),
            next_index: 0,
            open: BTreeMap::new(),
            content: Vec::new(),
            status: None,
            blocked: None,
            usage: None,
            complete: false,
        }
    }

    fn emit_custom(&mut self, step: Map<String, Value>, out: &mut Vec<Event>) {
        let index = self.next_index;
        self.next_index += 1;
        let kind = format!(
            "google.{}",
            step.get("type")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
        );
        let id = step
            .get("id")
            .or(step.get("call_id"))
            .and_then(Value::as_str)
            .map(str::to_owned);
        out.push(Event::BlockStart {
            parent_id: None,
            index: index + self.offset,
            kind: BlockKind::Custom,
            id: id.clone(),
            name: None,
        });
        let block = Block::Custom(CustomBlock {
            meta: BlockMeta {
                id,
                ..Default::default()
            },
            kind,
            data: Some(Value::Object(step)),
            origin: Some(self.ctx.origin()),
        });
        self.content.push(block.clone());
        out.push(Event::BlockStop {
            parent_id: None,
            index: index + self.offset,
            block,
        });
    }

    fn start(&mut self, step_index: u64, step: Value, out: &mut Vec<Event>) {
        let Value::Object(step) = step else { return };
        let ty = step
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        let (kind, block_kind) = match ty.as_str() {
            "model_output" => ("text", BlockKind::Text),
            "thought" => ("reasoning", BlockKind::Reasoning),
            "function_call" => ("tool_call", BlockKind::ToolCall),
            _ => ("custom", BlockKind::Custom),
        };
        let (id, name) = match kind {
            "tool_call" => (
                step.get("id").and_then(Value::as_str).map(str::to_owned),
                step.get("name").and_then(Value::as_str).map(str::to_owned),
            ),
            "custom" => (
                step.get("id")
                    .or(step.get("call_id"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
                None,
            ),
            _ => (None, None),
        };
        let index = if kind == "text" {
            0
        } else {
            let index = self.next_index;
            self.next_index += 1;
            out.push(Event::BlockStart {
                parent_id: None,
                index: index + self.offset,
                kind: block_kind,
                id,
                name,
            });
            index
        };
        let mut p = Partial {
            index,
            kind,
            step: step.clone(),
            text: String::new(),
            json: String::new(),
            summary: Vec::new(),
            started: kind != "text",
        };
        match kind {
            "text" => {
                let parts: Vec<Value> = step
                    .get("content")
                    .and_then(Value::as_array)
                    .cloned()
                    .unwrap_or_default();
                for part in parts {
                    self.text_part(&mut p, &part, out);
                }
            }
            "reasoning" => {
                if let Some(summary) = step.get("summary").and_then(Value::as_array) {
                    for s in summary {
                        if let Some(t) = str_field(s, "text") {
                            self.append_reasoning(&mut p, t, out);
                        }
                        p.summary.push(s.clone());
                    }
                }
            }
            "tool_call" => {
                if let Some(args) = step
                    .get("arguments")
                    .filter(|a| a.as_object().is_some_and(|o| !o.is_empty()))
                {
                    p.json = args.to_string();
                }
            }
            _ => {}
        }
        self.open.insert(step_index, p);
    }

    fn open_text(&mut self, p: &mut Partial, out: &mut Vec<Event>) {
        if p.started {
            return;
        }
        p.index = self.next_index;
        self.next_index += 1;
        p.started = true;
        out.push(Event::BlockStart {
            parent_id: None,
            index: p.index + self.offset,
            kind: BlockKind::Text,
            id: None,
            name: None,
        });
    }

    fn close_text(&mut self, p: &mut Partial, out: &mut Vec<Event>) {
        if !p.started {
            return;
        }
        let block = Block::Text(TextBlock {
            meta: BlockMeta::default(),
            text: std::mem::take(&mut p.text),
        });
        self.content.push(block.clone());
        out.push(Event::BlockStop {
            parent_id: None,
            index: p.index + self.offset,
            block,
        });
        p.started = false;
    }

    fn text_part(&mut self, p: &mut Partial, part: &Value, out: &mut Vec<Event>) {
        match str_field(part, "type") {
            Some("text") => {
                let t = str_field(part, "text").unwrap_or("");
                if !t.is_empty() {
                    self.open_text(p, out);
                    p.text.push_str(t);
                    out.push(Event::TextDelta {
                        parent_id: None,
                        index: p.index + self.offset,
                        text: t.to_owned(),
                    });
                }
            }
            _ => {
                self.close_text(p, out);
                if let Value::Object(o) = part {
                    self.emit_custom(o.clone(), out);
                }
            }
        }
    }

    fn append_reasoning(&self, p: &mut Partial, t: &str, out: &mut Vec<Event>) {
        if t.is_empty() {
            return;
        }
        if !p.text.is_empty() {
            p.text.push_str("\n\n");
        }
        p.text.push_str(t);
        out.push(Event::ReasoningDelta {
            parent_id: None,
            index: p.index + self.offset,
            text: t.to_owned(),
        });
    }

    fn delta(&mut self, step_index: u64, delta: Value, out: &mut Vec<Event>) {
        let Some(mut p) = self.open.remove(&step_index) else {
            return;
        };
        let dtype = str_field(&delta, "type").unwrap_or("").to_owned();
        match (p.kind, dtype.as_str()) {
            ("text", "text") => self.text_part(&mut p, &delta, out),
            ("text", "text_annotation_delta") => {
                if let Some(a) = delta.get("annotations") {
                    let entry = p.step.entry("annotations").or_insert_with(|| json!([]));
                    if let (Some(existing), Some(incoming)) = (entry.as_array_mut(), a.as_array()) {
                        existing.extend(incoming.iter().cloned());
                    }
                }
            }
            ("text", _) => self.text_part(&mut p, &delta, out),
            ("reasoning", "thought_summary") => {
                if let Some(c) = delta.get("content") {
                    if let Some(t) = str_field(c, "text") {
                        self.append_reasoning(&mut p, t, out);
                    }
                    p.summary.push(c.clone());
                }
            }
            ("reasoning", "thought_signature") => {
                p.step.insert(
                    "signature".into(),
                    delta.get("signature").cloned().unwrap_or(json!("")),
                );
            }
            ("tool_call", "arguments_delta") => {
                let a = str_field(&delta, "arguments").unwrap_or("").to_owned();
                p.json.push_str(&a);
                out.push(Event::ToolCallDelta {
                    parent_id: None,
                    index: p.index + self.offset,
                    arguments_json: a,
                });
            }
            _ => {
                if let Value::Object(o) = delta {
                    for (k, v) in o {
                        if k != "type" {
                            p.step.insert(k, v);
                        }
                    }
                }
            }
        }
        self.open.insert(step_index, p);
    }

    fn stop(&mut self, step_index: u64, out: &mut Vec<Event>) {
        let Some(p) = self.open.remove(&step_index) else {
            return;
        };
        if p.kind == "text" {
            let mut p = p;
            self.close_text(&mut p, out);
            return;
        }
        let block = match p.kind {
            "reasoning" => {
                let mut step = p.step;
                step.entry("signature").or_insert(json!(""));
                step.insert("summary".into(), Value::Array(p.summary));
                Block::Reasoning(ReasoningBlock {
                    meta: BlockMeta::default(),
                    text: p.text,
                    vendor_state: Some(Value::Object(step)),
                    origin: Some(self.ctx.origin()),
                })
            }
            "tool_call" => {
                let (arguments, truncated, arguments_json) = parse_arguments(&p.json);
                Block::ToolCall(ToolCallBlock {
                    meta: BlockMeta::default(),
                    id: p
                        .step
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                    name: p
                        .step
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_owned(),
                    arguments,
                    truncated,
                    arguments_json,
                    status: None,
                })
            }
            _ => {
                let kind = format!(
                    "google.{}",
                    p.step
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown")
                );
                let id = p
                    .step
                    .get("id")
                    .or(p.step.get("call_id"))
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                Block::Custom(CustomBlock {
                    meta: BlockMeta {
                        id,
                        ..Default::default()
                    },
                    kind,
                    data: Some(Value::Object(p.step)),
                    origin: Some(self.ctx.origin()),
                })
            }
        };
        self.content.push(block.clone());
        out.push(Event::BlockStop {
            parent_id: None,
            index: p.index + self.offset,
            block,
        });
    }

    fn close_all(&mut self, out: &mut Vec<Event>) {
        let keys: Vec<u64> = self.open.keys().copied().collect();
        for k in keys {
            self.stop(k, out);
        }
    }

    fn event(&mut self, ev: &Value, out: &mut Vec<Event>) -> Result<(), ProviderError> {
        if self.ctx.include_raw {
            out.push(Event::Raw {
                parent_id: None,
                event: ev.clone(),
            });
        }
        let kind = str_field(ev, "event_type").unwrap_or("");
        let index = u64_field(ev, "index").unwrap_or(0);
        match kind {
            "step.start" => self.start(index, ev.get("step").cloned().unwrap_or(Value::Null), out),
            "step.delta" => self.delta(index, ev.get("delta").cloned().unwrap_or(Value::Null), out),
            "step.stop" => self.stop(index, out),
            "interaction.status_update" => {
                if let Some(s) = str_field(ev, "status") {
                    self.status = Some(s.to_owned());
                }
            }
            "interaction.completed" => {
                let interaction = ev.get("interaction").cloned().unwrap_or(Value::Null);
                self.close_all(out);
                self.status = str_field(&interaction, "status").map(str::to_owned);
                self.usage = interaction.get("usage").cloned();
                self.complete = true;
            }
            "error" => {
                let code = ev
                    .get("error")
                    .and_then(|e| str_field(e, "code"))
                    .unwrap_or("")
                    .to_owned();
                let message = ev
                    .get("error")
                    .and_then(|e| str_field(e, "message"))
                    .unwrap_or("stream error")
                    .to_owned();
                if let Some(reason) = blocked_stop(&code) {
                    self.close_all(out);
                    self.blocked = Some((reason, code));
                    self.complete = true;
                    return Ok(());
                }
                let kind = match code.as_str() {
                    "rate_limit_exceeded" | "too_many_requests" => ErrorKind::RateLimit,
                    "quota_exceeded" => ErrorKind::Quota,
                    "api_error" | "service_unavailable" => ErrorKind::Server,
                    "deadline_exceeded" | "gateway_timeout" => ErrorKind::Timeout,
                    "cancelled" => ErrorKind::Cancelled,
                    "not_found" | "model_not_found" => ErrorKind::NotFound,
                    "authentication" | "permission_denied" => ErrorKind::Auth,
                    _ => ErrorKind::BadRequest,
                };
                let mut e = ProviderError::new(kind, format!("stream error: {message}"));
                e.code = Some(code);
                e.body = Some(ev.clone());
                return Err(e);
            }
            _ => {}
        }
        Ok(())
    }
}

fn parse_arguments(json: &str) -> (Value, Option<bool>, Option<String>) {
    if json.trim().is_empty() {
        return (json!({}), None, None);
    }
    match serde_json::from_str::<Value>(json) {
        Ok(v) if v.is_object() => (v, None, None),
        _ => (json!({}), Some(true), Some(json.to_owned())),
    }
}

impl ChatDecoder for Decoder {
    fn feed(&mut self, bytes: &[u8]) -> Result<Decoded, ProviderError> {
        let mut out = Decoded::default();
        for sse in self.sse.feed(bytes) {
            if sse.data.trim() == "[DONE]" || sse.data.trim().is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(&sse.data)
                .map_err(|e| malformed(format!("bad SSE JSON: {e}")))?;
            self.event(&v, &mut out.events)?;
        }
        out.complete = self.complete;
        Ok(out)
    }

    fn finish(&mut self) -> Result<Response, ProviderError> {
        if let Some(sse) = self.sse.finish() {
            if sse.data.trim() != "[DONE]" && !sse.data.trim().is_empty() {
                let v: Value = serde_json::from_str(&sse.data)
                    .map_err(|e| malformed(format!("bad SSE JSON: {e}")))?;
                let mut events = Vec::new();
                self.event(&v, &mut events)?;
            }
        }
        if !self.complete {
            return Err(malformed("stream ended before interaction.completed"));
        }
        let content = std::mem::take(&mut self.content);
        let status = self.status.clone().unwrap_or_default();
        let (vendor_stop, raw) = match self.blocked.take() {
            Some((reason, code)) => (reason, json!(code)),
            None => {
                let r = match status.as_str() {
                    "completed" | "requires_action" => StopReason::Complete,
                    "incomplete" => StopReason::MaxOutputTokens,
                    "cancelled" => StopReason::OtherReason,
                    "failed" => StopReason::OtherReason,
                    _ => StopReason::OtherReason,
                };
                (r, json!(status))
            }
        };
        let reason = derive_stop(&content, vendor_stop);
        let usage = self.usage.take().filter(Value::is_object).map(|u| {
            let thought = u64_field(&u, "total_thought_tokens");
            Usage {
                input: InputUsage {
                    total: u64_field(&u, "total_input_tokens").unwrap_or(0),
                    cache_read: u64_field(&u, "total_cached_tokens"),
                    cache_write: None,
                },
                output: OutputUsage {
                    total: u64_field(&u, "total_output_tokens").unwrap_or(0) + thought.unwrap_or(0),
                    reasoning: thought,
                },
                raw: Some(u),
            }
        });
        Ok(Response {
            content,
            stop: Stop {
                reason,
                raw: Some(raw),
            },
            usage,
            warnings: Vec::new(),
            origin: self.ctx.origin(),
            latency_ms: 0.0,
            first_token_ms: None,
            raw: None,
        })
    }
}
