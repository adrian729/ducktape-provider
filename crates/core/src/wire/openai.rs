//! The OpenAI Responses wire.
use super::*;
use crate::effects::{HttpRequest, Method, Purpose};
use crate::retry::parse_retry_after;
use crate::sse::SseParser;
use serde_json::json;
use std::collections::BTreeMap;

/// The `openai.responses` codec.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenaiResponses;

const HOSTED: &[&str] = &[
    "openai.web_search",
    "openai.file_search",
    "openai.code_interpreter",
    "openai.computer",
    "openai.image_generation",
    "openai.mcp",
    "openai.shell",
    "openai.apply_patch",
];

fn breakpoint(mark: &CacheMark) -> Value {
    match mark.ttl.as_ref().map(|t| t.as_str()) {
        Some("long") => json!({"mode": "explicit", "ttl": "12h"}),
        _ => json!({"mode": "explicit"}),
    }
}

fn data_url(data: &str, media_type: Option<&str>, fallback: &str) -> String {
    let media = media_type
        .map(str::to_owned)
        .or_else(|| sniff_image(data).map(str::to_owned))
        .unwrap_or_else(|| fallback.to_owned());
    format!("data:{media};base64,{data}")
}

struct Encoder<'a> {
    ctx: &'a CallContext,
    cache_policy: CachePolicy,
    warnings: Vec<Warning>,
}

impl Encoder<'_> {
    fn warn(&mut self, feature: &str, detail: impl Into<String>) {
        self.warnings.push(Warning {
            kind: WarningKind::Unsupported,
            feature: feature.to_owned(),
            detail: Some(detail.into()),
        });
    }

    fn mark(&self, target: &mut Map<String, Value>, meta: &BlockMeta) {
        if let Some(mark) = &meta.cache {
            if !matches!(self.cache_policy, CachePolicy::None) {
                target.insert("prompt_cache_breakpoint".into(), breakpoint(mark));
            }
        }
        apply_vendor(target, meta.vendor.as_ref(), &self.ctx.vendor);
    }

    fn input_part(&mut self, block: &Block) -> Result<Option<Value>, ProviderError> {
        let part = match block {
            Block::Text(b) => {
                let mut o = Map::new();
                o.insert("type".into(), json!("input_text"));
                o.insert("text".into(), json!(b.text));
                self.mark(&mut o, &b.meta);
                Value::Object(o)
            }
            Block::Image(b) => {
                let mut o = Map::new();
                o.insert("type".into(), json!("input_image"));
                if let Some(data) = &b.data {
                    o.insert(
                        "image_url".into(),
                        json!(data_url(data, b.media_type.as_deref(), "image/png")),
                    );
                } else if let Some(url) = &b.url {
                    o.insert("image_url".into(), json!(url));
                } else if let Some(id) = &b.file_id {
                    o.insert("file_id".into(), json!(id));
                } else {
                    return Err(argument_error("image block needs data, url or file_id"));
                }
                self.mark(&mut o, &b.meta);
                Value::Object(o)
            }
            Block::Document(b) => {
                let mut o = Map::new();
                o.insert("type".into(), json!("input_file"));
                let name = b.name.clone().unwrap_or_else(|| "document".to_owned());
                if let Some(text) = &b.text {
                    o.insert("type".into(), json!("input_text"));
                    o.insert("text".into(), json!(text));
                } else if let Some(data) = &b.data {
                    o.insert(
                        "file_data".into(),
                        json!(data_url(data, b.media_type.as_deref(), "application/pdf")),
                    );
                    o.insert("filename".into(), json!(name));
                } else if let Some(url) = &b.url {
                    o.insert("file_url".into(), json!(url));
                } else if let Some(id) = &b.file_id {
                    o.insert("file_id".into(), json!(id));
                } else {
                    return Err(argument_error(
                        "document block needs data, url, file_id or text",
                    ));
                }
                self.mark(&mut o, &b.meta);
                Value::Object(o)
            }
            other => {
                self.warn(
                    "content",
                    format!(
                        "{} block not allowed in this turn on OpenAI",
                        block_kind(other)
                    ),
                );
                return Ok(None);
            }
        };
        Ok(Some(part))
    }

    fn items(&mut self, messages: &[Message]) -> Result<Vec<Value>, ProviderError> {
        let mut out = Vec::new();
        for m in messages {
            match &m.role {
                Role::User | Role::System => {
                    let mut parts = Vec::new();
                    for b in &m.content {
                        if let Some(p) = self.input_part(b)? {
                            parts.push(p);
                        }
                    }
                    let role = if matches!(m.role, Role::User) {
                        "user"
                    } else {
                        "developer"
                    };
                    out.push(json!({"type": "message", "role": role, "content": parts}));
                }
                Role::Tool => {
                    for b in &m.content {
                        match b {
                            Block::ToolResult(r) => out.push(self.tool_output(r)?),
                            other => self.warn(
                                "content",
                                format!("{} block ignored in a tool turn", block_kind(other)),
                            ),
                        }
                    }
                }
                Role::Assistant => self.assistant_items(&m.content, &mut out)?,
                Role::Other(role) => {
                    let mut parts = Vec::new();
                    for b in &m.content {
                        if let Some(p) = self.input_part(b)? {
                            parts.push(p);
                        }
                    }
                    out.push(json!({"type": "message", "role": role, "content": parts}));
                }
            }
        }
        Ok(out)
    }

    fn tool_output(&mut self, r: &ToolResultBlock) -> Result<Value, ProviderError> {
        let only_text = r.content.iter().all(|b| matches!(b, Block::Text(_)));
        let output = if only_text {
            let mut text: String = r
                .content
                .iter()
                .filter_map(|b| {
                    if let Block::Text(t) = b {
                        Some(t.text.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            if r.is_error == Some(true) {
                text = format!("[error] {text}");
            }
            json!(text)
        } else {
            let mut parts = Vec::new();
            if r.is_error == Some(true) {
                parts.push(json!({"type": "input_text", "text": "[error]"}));
            }
            for b in &r.content {
                if let Some(p) = self.input_part(b)? {
                    parts.push(p);
                }
            }
            Value::Array(parts)
        };
        let mut o = Map::new();
        o.insert("type".into(), json!("function_call_output"));
        o.insert("call_id".into(), json!(r.tool_call_id));
        o.insert("output".into(), output);
        self.mark(&mut o, &r.meta);
        Ok(Value::Object(o))
    }

    fn assistant_items(
        &mut self,
        content: &[Block],
        out: &mut Vec<Value>,
    ) -> Result<(), ProviderError> {
        let mut pending_text: Vec<Value> = Vec::new();
        let mut pending_vendor: Map<String, Value> = Map::new();
        let flush =
            |pending: &mut Vec<Value>, vendor: &mut Map<String, Value>, out: &mut Vec<Value>| {
                if !pending.is_empty() {
                    let mut item = Map::new();
                    item.insert("type".into(), json!("message"));
                    item.insert("role".into(), json!("assistant"));
                    item.insert("content".into(), Value::Array(std::mem::take(pending)));
                    for (k, v) in std::mem::take(vendor) {
                        item.insert(k, v);
                    }
                    out.push(Value::Object(item));
                }
            };
        for b in content {
            match b {
                Block::Text(t) => {
                    pending_text
                        .push(json!({"type": "output_text", "text": t.text, "annotations": []}));
                    if let Some(extra) = vendor_for(t.meta.vendor.as_ref(), &self.ctx.vendor) {
                        for (k, v) in extra {
                            pending_vendor.insert(k.clone(), v.clone());
                        }
                    }
                }
                Block::ToolCall(c) => {
                    flush(&mut pending_text, &mut pending_vendor, out);
                    let mut o = Map::new();
                    o.insert("type".into(), json!("function_call"));
                    o.insert("call_id".into(), json!(c.id));
                    o.insert("name".into(), json!(c.name));
                    o.insert(
                        "arguments".into(),
                        json!(c
                            .arguments_json
                            .clone()
                            .unwrap_or_else(|| c.arguments.to_string())),
                    );
                    if let Some(id) = &c.meta.id {
                        o.insert("id".into(), json!(id));
                    }
                    apply_vendor(&mut o, c.meta.vendor.as_ref(), &self.ctx.vendor);
                    out.push(Value::Object(o));
                }
                Block::Reasoning(r) => {
                    flush(&mut pending_text, &mut pending_vendor, out);
                    if !self.ctx.accepts(r.origin.as_ref()) {
                        self.warn("reasoning", "reasoning block from another endpoint dropped");
                        continue;
                    }
                    match &r.vendor_state {
                        Some(Value::Object(item)) => out.push(Value::Object(item.clone())),
                        _ => self.warn("reasoning", "reasoning block without vendor state dropped"),
                    }
                }
                Block::Compaction(c) => {
                    flush(&mut pending_text, &mut pending_vendor, out);
                    if !self.ctx.accepts(Some(&c.origin)) {
                        self.warn("compaction", "compaction block from another endpoint dropped; summarized history is lost");
                        continue;
                    }
                    out.push(c.vendor_state.clone());
                }
                Block::Custom(c) => {
                    flush(&mut pending_text, &mut pending_vendor, out);
                    let ours = c.kind.starts_with("openai.");
                    if !ours || (c.origin.is_some() && !self.ctx.accepts(c.origin.as_ref())) {
                        self.warn("custom", format!("custom block `{}` dropped", c.kind));
                        continue;
                    }
                    match &c.data {
                        Some(Value::Object(d)) => out.push(Value::Object(d.clone())),
                        _ => out.push(json!({"type": c.kind.trim_start_matches("openai.")})),
                    }
                }
                other => {
                    self.warn(
                        "content",
                        format!("{} block dropped from an assistant turn", block_kind(other)),
                    );
                }
            }
        }
        flush(&mut pending_text, &mut pending_vendor, out);
        Ok(())
    }

    fn tools(&mut self, tools: &[Tool]) -> Result<Vec<Value>, ProviderError> {
        let mut out = Vec::new();
        for t in tools {
            match t {
                Tool::Function(d) => {
                    let mut o = Map::new();
                    o.insert("type".into(), json!("function"));
                    o.insert("name".into(), json!(d.name));
                    o.insert("description".into(), json!(d.description));
                    o.insert("parameters".into(), d.input_schema.clone());
                    if let Some(strict) = d.strict {
                        o.insert("strict".into(), json!(strict));
                    }
                    if d.cache.is_some() {
                        self.warn("cache", "cache marks on tools are not supported on OpenAI");
                    }
                    apply_vendor(&mut o, d.vendor.as_ref(), &self.ctx.vendor);
                    out.push(Value::Object(o));
                }
                Tool::Provider(p) => {
                    let Some(name) = p.provider_tool.strip_prefix("openai.") else {
                        self.warn(
                            "provider_tool",
                            format!("`{}` is not an OpenAI tool", p.provider_tool),
                        );
                        continue;
                    };
                    let value = match &p.config {
                        Some(Value::Object(cfg)) => {
                            let mut cfg = cfg.clone();
                            cfg.entry("type").or_insert(json!(name));
                            Value::Object(cfg)
                        }
                        _ => {
                            match name {
                                "web_search" => json!({"type": "web_search"}),
                                "code_interpreter" => {
                                    json!({"type": "code_interpreter", "container": {"type": "auto"}})
                                }
                                "image_generation" => json!({"type": "image_generation"}),
                                "shell" => json!({"type": "shell"}),
                                "apply_patch" => json!({"type": "apply_patch"}),
                                other => {
                                    let mut e = ProviderError::new(ErrorKind::Unsupported, format!("provider tool `openai.{other}` needs an explicit `config`"));
                                    e.code = Some("provider_tool".to_owned());
                                    return Err(e);
                                }
                            }
                        }
                    };
                    out.push(value);
                }
            }
        }
        Ok(out)
    }
}

fn block_kind(b: &Block) -> &'static str {
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

fn base_headers(ctx: &CallContext) -> Headers {
    let mut h = Headers::new();
    h.insert("content-type".into(), "application/json".into());
    for (k, v) in &ctx.headers {
        h.insert(k.to_ascii_lowercase(), v.clone());
    }
    h
}

fn path(ctx: &CallContext, rest: &str) -> String {
    match ctx.api_version.as_deref() {
        Some("") => rest.to_owned(),
        Some(v) => format!("{}/{rest}", v.trim_matches('/')),
        None => format!("v1/{rest}"),
    }
}

fn get(ctx: &CallContext, path: &str, purpose: Purpose) -> HttpRequest {
    HttpRequest {
        method: Method::Get,
        url: join_url(&ctx.base_url, path),
        headers: base_headers(ctx),
        body: None,
        stream: false,
        timeout: ctx.timeout,
        purpose,
    }
}

fn model_kind(id: &str) -> ModelKind {
    if id.contains("embedding") {
        ModelKind::Embedding
    } else {
        ModelKind::Chat
    }
}

impl Wire for OpenaiResponses {
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
            cache_policy: cache_policy.clone(),
            warnings: Vec::new(),
        };
        let mut body = Body::new();
        body.user("model", json!(ctx.model));
        body.lib("stream", json!(true));
        body.lib("store", json!(false));
        body.lib("include", json!(["reasoning.encrypted_content"]));
        let mut input = Vec::new();
        if let Some(system) = req.system.as_ref().filter(|s| !s.is_empty()) {
            let marked = system
                .iter()
                .any(|SystemBlock::Text(b)| b.meta.cache.is_some())
                && !matches!(cache_policy, CachePolicy::None);
            if marked {
                let parts: Vec<Value> = system
                    .iter()
                    .map(|SystemBlock::Text(b)| {
                        let mut o = Map::new();
                        o.insert("type".into(), json!("input_text"));
                        o.insert("text".into(), json!(b.text));
                        enc.mark(&mut o, &b.meta);
                        Value::Object(o)
                    })
                    .collect();
                input.push(json!({"type": "message", "role": "developer", "content": parts}));
            } else {
                let text: Vec<&str> = system
                    .iter()
                    .map(|SystemBlock::Text(b)| b.text.as_str())
                    .collect();
                body.user("instructions", json!(text.join("\n\n")));
            }
        }
        input.extend(enc.items(&req.messages)?);
        body.lib("input", Value::Array(input));
        if let Some(tools) = req.tools.as_ref().filter(|t| !t.is_empty()) {
            body.user("tools", Value::Array(enc.tools(tools)?));
        }
        if let Some(n) = opts.max_output_tokens {
            body.user("max_output_tokens", json!(n));
        }
        let mut reasoning = Map::new();
        let mut reasoning_is_user = false;
        match opts.reasoning.clone().unwrap_or(ReasoningLevel::Auto) {
            ReasoningLevel::Auto => {}
            level => {
                reasoning.insert("effort".into(), json!(level.as_str()));
                reasoning_is_user = true;
            }
        }
        let effort_none = reasoning.get("effort").and_then(Value::as_str) == Some("none");
        match opts
            .reasoning_summary
            .clone()
            .unwrap_or(ReasoningSummary::Auto)
        {
            ReasoningSummary::None => {}
            _ if effort_none => {}
            _ => {
                reasoning.insert("summary".into(), json!("auto"));
                reasoning_is_user |= opts.reasoning_summary.is_some();
            }
        }
        if !reasoning.is_empty() {
            if reasoning_is_user {
                body.user("reasoning", Value::Object(reasoning));
            } else {
                body.lib("reasoning", Value::Object(reasoning));
            }
        }
        if let Some(tc) = &opts.tool_choice {
            let v = match tc.mode.as_str() {
                "auto" => json!("auto"),
                "none" => json!("none"),
                "required" => json!("required"),
                "named" => json!({"type": "function", "name": tc.name.clone().unwrap_or_default()}),
                other => {
                    return Err(argument_error(format!(
                        "unknown tool_choice mode `{other}`"
                    )))
                }
            };
            body.user("tool_choice", v);
        }
        if let Some(parallel) = opts.parallel_tools {
            body.user("parallel_tool_calls", json!(parallel));
        }
        if let Some(out) = &opts.output {
            let mut format = json!({"type": "json_schema", "name": out.name.clone().unwrap_or_else(|| "output".to_owned()), "schema": out.json_schema});
            if let Some(obj) = format.as_object_mut() {
                obj.insert("strict".into(), json!(true));
            }
            body.user("text", json!({"format": format}));
        }
        if opts.stop_sequences.as_ref().is_some_and(|s| !s.is_empty()) {
            enc.warn("stop_sequences", "OpenAI Responses has no stop sequences");
        }
        if let Some(uid) = &opts.user_id {
            body.user("safety_identifier", json!(uid));
        }
        body.warnings.append(&mut enc.warnings);
        body.merge_vendor(vendor_for(req.vendor.as_ref(), &ctx.vendor))?;
        body.enforce_strict(ctx.strict)?;
        let mut headers = base_headers(ctx);
        for (k, v) in req.headers.iter().flatten() {
            headers.insert(k.to_ascii_lowercase(), v.clone());
        }
        let request = HttpRequest {
            method: Method::Post,
            url: join_url(&ctx.base_url, &path(ctx, "responses")),
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
        let mut body = Body::new();
        body.user("model", json!(ctx.model));
        body.user("input", json!(req.input));
        body.lib("encoding_format", json!("float"));
        if let Some(d) = req.options.as_ref().and_then(|o| o.dimensions) {
            body.user("dimensions", json!(d));
        }
        body.merge_vendor(vendor_for(req.vendor.as_ref(), &ctx.vendor))?;
        if body.map.get("encoding_format").and_then(Value::as_str) == Some("base64") {
            return Err(argument_error(
                "encoding_format base64 is not supported; vectors are always floats",
            ));
        }
        let request = HttpRequest {
            method: Method::Post,
            url: join_url(&ctx.base_url, &path(ctx, "embeddings")),
            headers: base_headers(ctx),
            body: Some(Value::Object(body.map).to_string()),
            stream: false,
            timeout: req.timeout.or(ctx.timeout),
            purpose: Purpose::Embed,
        };
        Ok(Built {
            request,
            warnings: body.warnings,
        })
    }

    fn decode_embed(&self, ctx: &CallContext, body: &[u8]) -> Result<EmbedResponse, ProviderError> {
        let v = parse_json(body)?;
        let data = v
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("embeddings reply has no `data`"))?;
        let mut rows: Vec<(u64, Vec<f64>)> = Vec::with_capacity(data.len());
        for (i, row) in data.iter().enumerate() {
            let index = u64_field(row, "index").unwrap_or(i as u64);
            let vector = row
                .get("embedding")
                .and_then(Value::as_array)
                .ok_or_else(|| malformed("embedding row has no vector"))?;
            rows.push((index, vector.iter().filter_map(Value::as_f64).collect()));
        }
        rows.sort_by_key(|(i, _)| *i);
        let usage = v
            .get("usage")
            .and_then(|u| u64_field(u, "prompt_tokens"))
            .map(|t| EmbedUsage {
                input: EmbedInputUsage { total: t },
            });
        Ok(EmbedResponse {
            embeddings: rows.into_iter().map(|(_, e)| e).collect(),
            usage,
            warnings: Vec::new(),
            origin: ctx.origin(),
            latency_ms: 0.0,
            raw: if ctx.include_raw { Some(v) } else { None },
        })
    }

    fn build_models(&self, ctx: &CallContext) -> Option<HttpRequest> {
        Some(get(ctx, &path(ctx, "models"), Purpose::Models))
    }

    fn decode_models(
        &self,
        _ctx: &CallContext,
        body: &[u8],
    ) -> Result<(Vec<ModelInfo>, Option<HttpRequest>), ProviderError> {
        let v = parse_json(body)?;
        let data = v
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("model listing has no `data`"))?;
        let models = data
            .iter()
            .filter_map(|m| {
                str_field(m, "id").map(|id| ModelInfo {
                    id: id.to_owned(),
                    kind: model_kind(id),
                    context_window: None,
                    max_output_tokens: None,
                    capabilities: None,
                    raw: Some(m.clone()),
                })
            })
            .collect();
        Ok((models, None))
    }

    fn build_model_info(&self, ctx: &CallContext, model: &str) -> Option<HttpRequest> {
        Some(get(
            ctx,
            &path(ctx, &format!("models/{model}")),
            Purpose::Models,
        ))
    }

    fn decode_model_info(
        &self,
        _ctx: &CallContext,
        body: &[u8],
    ) -> Result<ModelInfo, ProviderError> {
        let v = parse_json(body)?;
        let id = str_field(&v, "id")
            .ok_or_else(|| malformed("model reply has no `id`"))?
            .to_owned();
        Ok(ModelInfo {
            kind: model_kind(&id),
            id,
            context_window: None,
            max_output_tokens: None,
            capabilities: None,
            raw: Some(v.clone()),
        })
    }

    fn shipped_capabilities(&self, model: &str) -> Option<Capabilities> {
        let m = model.to_ascii_lowercase();
        if m.starts_with("text-embedding") {
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
        if !(m.starts_with("gpt-") || m.starts_with("o")) {
            return None;
        }
        Some(Capabilities {
            reasoning: Some(ReasoningCapability {
                levels: Some(Some(
                    [
                        "auto", "none", "minimal", "low", "medium", "high", "xhigh", "max",
                    ]
                    .iter()
                    .map(|l| ReasoningLevel::from(*l))
                    .collect(),
                )),
                can_disable: Some(None),
            }),
            tool_choice: Some(Some(vec![
                ToolChoiceMode::Auto,
                ToolChoiceMode::None,
                ToolChoiceMode::Required,
                ToolChoiceMode::Named,
            ])),
            parallel_tools: Some(ParallelToolsCapability {
                supported: Some(Some(true)),
                toggle: Some(Some(true)),
            }),
            output: Some(OutputCapability {
                json_schema: Some(Some(true)),
            }),
            cache: Some(CacheCapability {
                breakpoints: Some(None),
                ttls: Some(Some(vec![CacheTtl::Short, CacheTtl::Long])),
            }),
            inputs: Some(Some(vec![
                InputKind::Text,
                InputKind::Image,
                InputKind::Document,
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
        let vendor_type = err.and_then(|e| str_field(e, "type")).unwrap_or("");
        let code = err.and_then(|e| str_field(e, "code")).unwrap_or("");
        let vendor_message = err.and_then(|e| str_field(e, "message")).unwrap_or("");
        let retry_after = headers
            .get("retry-after")
            .and_then(|v| parse_retry_after(v));
        let billing = matches!(
            code,
            "credit_balance_exhausted"
                | "organization_spend_limit_exceeded"
                | "project_spend_limit_exceeded"
                | "organization_usage_limit_exceeded"
                | "insufficient_quota"
        ) || vendor_type == "insufficient_quota";
        let (kind, retryable) = match status {
            401 | 403 => (ErrorKind::Auth, false),
            402 => (ErrorKind::Quota, false),
            404 => (ErrorKind::NotFound, false),
            413 => (ErrorKind::RequestTooLarge, false),
            429 if billing => (ErrorKind::Quota, false),
            429 => (ErrorKind::RateLimit, true),
            408 => (ErrorKind::Timeout, true),
            400 if code == "context_length_exceeded" => (ErrorKind::ContextOverflow, false),
            s if s >= 500 => (ErrorKind::Server, true),
            s if (400..500).contains(&s) => (ErrorKind::BadRequest, false),
            _ => (ErrorKind::MalformedResponse, false),
        };
        let message = if vendor_message.is_empty() {
            format!("HTTP {status}")
        } else {
            format!("HTTP {status}: {vendor_message}")
        };
        ProviderError {
            kind,
            message,
            status: Some(status),
            code: if !code.is_empty() {
                Some(code.to_owned())
            } else if !vendor_type.is_empty() {
                Some(vendor_type.to_owned())
            } else {
                None
            },
            retryable,
            retry_after,
            attempts: 0,
            body: parsed.clone(),
            request_id: headers.get("x-request-id").cloned(),
            origin: None,
        }
    }
}

/// A block being streamed, keyed by the vendor's item and part position.
#[derive(Debug)]
struct Partial {
    index: u32,
    kind: &'static str,
    item: Value,
    text: String,
    json: String,
    refusal: bool,
}

/// Decoder of one Responses stream.
struct Decoder {
    ctx: CallContext,
    offset: u32,
    sse: SseParser,
    next_index: u32,
    open: BTreeMap<(u64, u64), Partial>,
    content: Vec<Block>,
    response: Option<Value>,
    status: Option<String>,
    incomplete_reason: Option<String>,
    failed_code: Option<String>,
    refused: bool,
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
            response: None,
            status: None,
            incomplete_reason: None,
            failed_code: None,
            refused: false,
            complete: false,
        }
    }

    fn start_block(
        &mut self,
        key: (u64, u64),
        kind: &'static str,
        item: Value,
        block_kind: BlockKind,
        label: (Option<String>, Option<String>),
        out: &mut Vec<Event>,
    ) {
        let index = self.next_index;
        self.next_index += 1;
        self.open.insert(
            key,
            Partial {
                index,
                kind,
                item,
                text: String::new(),
                json: String::new(),
                refusal: false,
            },
        );
        out.push(Event::BlockStart {
            parent_id: None,
            index: index + self.offset,
            kind: block_kind,
            id: label.0,
            name: label.1,
        });
    }

    fn stop_block(&mut self, key: (u64, u64), done_item: Option<&Value>, out: &mut Vec<Event>) {
        let Some(p) = self.open.remove(&key) else {
            return;
        };
        let block = self.finish_block(&p, done_item);
        self.content.push(block.clone());
        out.push(Event::BlockStop {
            parent_id: None,
            index: p.index + self.offset,
            block,
        });
    }

    fn finish_block(&self, p: &Partial, done_item: Option<&Value>) -> Block {
        let item = done_item.unwrap_or(&p.item);
        let item_id = str_field(item, "id").map(str::to_owned);
        match p.kind {
            "text" => {
                let mut vendor = None;
                if p.refusal {
                    let mut m = VendorMap::new();
                    m.insert("openai".into(), json!({"refusal": true}));
                    vendor = Some(m);
                }
                Block::Text(TextBlock {
                    meta: BlockMeta {
                        id: item_id,
                        vendor,
                        ..Default::default()
                    },
                    text: p.text.clone(),
                })
            }
            "reasoning" => {
                let mut text = p.text.clone();
                if let Some(summary) = item.get("summary").and_then(Value::as_array) {
                    let joined: Vec<&str> = summary
                        .iter()
                        .filter_map(|s| str_field(s, "text"))
                        .collect();
                    if !joined.is_empty() {
                        text = joined.join("\n\n");
                    }
                }
                Block::Reasoning(ReasoningBlock {
                    meta: BlockMeta {
                        id: item_id,
                        ..Default::default()
                    },
                    text,
                    vendor_state: Some(item.clone()),
                    origin: Some(self.ctx.origin()),
                })
            }
            "tool_call" => {
                let args = str_field(item, "arguments")
                    .map(str::to_owned)
                    .unwrap_or_else(|| p.json.clone());
                let (arguments, truncated, arguments_json) = parse_arguments(&args);
                Block::ToolCall(ToolCallBlock {
                    meta: BlockMeta::default(),
                    id: str_field(item, "call_id").unwrap_or("").to_owned(),
                    name: str_field(item, "name").unwrap_or("").to_owned(),
                    arguments,
                    truncated,
                    arguments_json,
                    status: None,
                })
            }
            "compaction" => Block::Compaction(CompactionBlock {
                meta: BlockMeta {
                    id: item_id,
                    ..Default::default()
                },
                vendor_state: item.clone(),
                text: None,
                origin: self.ctx.origin(),
            }),
            _ => Block::Custom(CustomBlock {
                meta: BlockMeta {
                    id: item_id,
                    ..Default::default()
                },
                kind: format!("openai.{}", str_field(item, "type").unwrap_or("unknown")),
                data: Some(item.clone()),
                origin: Some(self.ctx.origin()),
            }),
        }
    }

    fn event(&mut self, ev: &Value, out: &mut Vec<Event>) -> Result<(), ProviderError> {
        if self.ctx.include_raw {
            out.push(Event::Raw {
                parent_id: None,
                event: ev.clone(),
            });
        }
        let kind = str_field(ev, "type").unwrap_or("");
        let oi = u64_field(ev, "output_index").unwrap_or(0);
        let ci = u64_field(ev, "content_index").unwrap_or(0);
        match kind {
            "response.created" | "response.in_progress" | "response.queued" => {
                self.response = ev.get("response").cloned();
            }
            "response.output_item.added" => {
                let item = ev.get("item").cloned().unwrap_or(Value::Null);
                match str_field(&item, "type").unwrap_or("") {
                    "message" => {}
                    "reasoning" => self.start_block(
                        (oi, 0),
                        "reasoning",
                        item,
                        BlockKind::Reasoning,
                        (None, None),
                        out,
                    ),
                    "function_call" => {
                        let id = str_field(&item, "call_id").map(str::to_owned);
                        let name = str_field(&item, "name").map(str::to_owned);
                        self.start_block(
                            (oi, 0),
                            "tool_call",
                            item,
                            BlockKind::ToolCall,
                            (id, name),
                            out,
                        );
                    }
                    "compaction" => self.start_block(
                        (oi, 0),
                        "compaction",
                        item,
                        BlockKind::Compaction,
                        (None, None),
                        out,
                    ),
                    _ => {
                        let id = str_field(&item, "id").map(str::to_owned);
                        self.start_block(
                            (oi, 0),
                            "custom",
                            item,
                            BlockKind::Custom,
                            (id, None),
                            out,
                        );
                    }
                }
            }
            "response.content_part.added" => {
                let part = ev.get("part").cloned().unwrap_or(Value::Null);
                let refusal = str_field(&part, "type") == Some("refusal");
                let id = str_field(ev, "item_id").map(str::to_owned);
                let item = json!({"id": id, "type": "message"});
                self.start_block((oi, ci), "text", item, BlockKind::Text, (None, None), out);
                if let Some(p) = self.open.get_mut(&(oi, ci)) {
                    p.refusal = refusal;
                    p.text = str_field(&part, "text")
                        .or(str_field(&part, "refusal"))
                        .unwrap_or("")
                        .to_owned();
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                let delta = str_field(ev, "delta").unwrap_or("").to_owned();
                let offset = self.offset;
                if let Some(p) = self.open.get_mut(&(oi, ci)) {
                    p.text.push_str(&delta);
                    out.push(Event::TextDelta {
                        parent_id: None,
                        index: p.index + offset,
                        text: delta,
                    });
                }
            }
            "response.output_text.done" | "response.refusal.done" => {
                if let Some(p) = self.open.get_mut(&(oi, ci)) {
                    if let Some(text) = str_field(ev, "text").or(str_field(ev, "refusal")) {
                        p.text = text.to_owned();
                    }
                    if kind == "response.refusal.done" {
                        self.refused = true;
                    }
                }
            }
            "response.content_part.done" => {
                let part = ev.get("part").cloned();
                if let Some(p) = self.open.get_mut(&(oi, ci)) {
                    if let Some(text) = part
                        .as_ref()
                        .and_then(|x| str_field(x, "text").or(str_field(x, "refusal")))
                    {
                        p.text = text.to_owned();
                    }
                    if let Some(part) = part {
                        if let Some(obj) = p.item.as_object_mut() {
                            obj.insert("part".into(), part);
                        }
                    }
                }
                self.stop_block((oi, ci), None, out);
            }
            "response.function_call_arguments.delta" => {
                let delta = str_field(ev, "delta").unwrap_or("").to_owned();
                let offset = self.offset;
                if let Some(p) = self.open.get_mut(&(oi, 0)) {
                    p.json.push_str(&delta);
                    out.push(Event::ToolCallDelta {
                        parent_id: None,
                        index: p.index + offset,
                        arguments_json: delta,
                    });
                }
            }
            "response.function_call_arguments.done" => {
                if let Some(p) = self.open.get_mut(&(oi, 0)) {
                    if let Some(args) = str_field(ev, "arguments") {
                        p.json = args.to_owned();
                    }
                }
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let delta = str_field(ev, "delta").unwrap_or("").to_owned();
                let offset = self.offset;
                if let Some(p) = self.open.get_mut(&(oi, 0)) {
                    p.text.push_str(&delta);
                    out.push(Event::ReasoningDelta {
                        parent_id: None,
                        index: p.index + offset,
                        text: delta,
                    });
                }
            }
            "response.reasoning_summary_part.done" => {
                if let Some(p) = self.open.get_mut(&(oi, 0)) {
                    if u64_field(ev, "summary_index").unwrap_or(0) > 0 || !p.text.is_empty() {
                        p.text.push_str("\n\n");
                    }
                }
            }
            "response.output_item.done" => {
                let item = ev.get("item").cloned().unwrap_or(Value::Null);
                if str_field(&item, "type") == Some("message") {
                    let keys: Vec<(u64, u64)> = self
                        .open
                        .keys()
                        .filter(|(o, _)| *o == oi)
                        .copied()
                        .collect();
                    for k in keys {
                        self.stop_block(k, None, out);
                    }
                } else {
                    self.stop_block((oi, 0), Some(&item), out);
                }
            }
            "response.completed" | "response.incomplete" | "response.failed" => {
                let response = ev.get("response").cloned().unwrap_or(Value::Null);
                self.status = str_field(&response, "status").map(str::to_owned);
                self.incomplete_reason = response
                    .get("incomplete_details")
                    .and_then(|d| str_field(d, "reason"))
                    .map(str::to_owned);
                self.failed_code = response
                    .get("error")
                    .and_then(|e| str_field(e, "code"))
                    .map(str::to_owned);
                if kind == "response.failed"
                    && !self.failed_code.as_deref().is_some_and(is_filter_code)
                {
                    let message = response
                        .get("error")
                        .and_then(|e| str_field(e, "message"))
                        .unwrap_or("response failed");
                    let mut e = ProviderError::new(
                        ErrorKind::Server,
                        format!("response failed: {message}"),
                    );
                    e.code = self.failed_code.clone();
                    e.body = Some(ev.clone());
                    return Err(e);
                }
                let keys: Vec<(u64, u64)> = self.open.keys().copied().collect();
                for k in keys {
                    self.stop_block(k, None, out);
                }
                self.response = Some(response);
                self.complete = true;
            }
            "error" => {
                let code = str_field(ev, "code").unwrap_or("");
                let message = str_field(ev, "message").unwrap_or("stream error");
                let kind = match code {
                    "rate_limit_exceeded" | "slow_down" => ErrorKind::RateLimit,
                    "server_error" | "server_is_overloaded" => ErrorKind::Server,
                    _ => ErrorKind::BadRequest,
                };
                let mut e = ProviderError::new(kind, format!("stream error: {message}"));
                e.code = Some(code.to_owned());
                e.body = Some(ev.clone());
                return Err(e);
            }
            _ => {}
        }
        Ok(())
    }
}

fn is_filter_code(code: &str) -> bool {
    code.contains("content_filter") || code.contains("moderation") || code.contains("safety")
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
            if sse.data.trim() == "[DONE]" {
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
            if sse.data.trim() != "[DONE]" {
                let v: Value = serde_json::from_str(&sse.data)
                    .map_err(|e| malformed(format!("bad SSE JSON: {e}")))?;
                let mut events = Vec::new();
                self.event(&v, &mut events)?;
            }
        }
        if !self.complete {
            return Err(malformed("stream ended before response.completed"));
        }
        let content = std::mem::take(&mut self.content);
        let status = self.status.clone().unwrap_or_default();
        let vendor_stop = match (
            status.as_str(),
            self.incomplete_reason.as_deref(),
            self.failed_code.as_deref(),
        ) {
            ("completed", _, _) if self.refused => StopReason::Refusal,
            ("completed", _, _) => StopReason::Complete,
            ("incomplete", Some("max_output_tokens"), _) => StopReason::MaxOutputTokens,
            ("incomplete", Some("content_filter"), _) => StopReason::ContentFiltered,
            ("incomplete", Some("steered"), _) => StopReason::ContinueRequired,
            ("failed", _, Some(_)) => StopReason::ContentFiltered,
            _ => StopReason::OtherReason,
        };
        let reason = derive_stop(&content, vendor_stop);
        let raw_stop = match (&self.incomplete_reason, &self.failed_code) {
            (Some(r), _) => json!(format!("{status}/{r}")),
            (_, Some(c)) => json!(format!("{status}/{c}")),
            _ => json!(status),
        };
        let response = self.response.clone().unwrap_or(Value::Null);
        let usage = response.get("usage").filter(|u| u.is_object()).map(|u| {
            let details_in = u.get("input_tokens_details");
            let details_out = u.get("output_tokens_details");
            Usage {
                input: InputUsage {
                    total: u64_field(u, "input_tokens").unwrap_or(0),
                    cache_read: details_in.and_then(|d| u64_field(d, "cached_tokens")),
                    cache_write: details_in.and_then(|d| u64_field(d, "cache_write_tokens")),
                },
                output: OutputUsage {
                    total: u64_field(u, "output_tokens").unwrap_or(0),
                    reasoning: details_out.and_then(|d| u64_field(d, "reasoning_tokens")),
                },
                raw: Some(u.clone()),
            }
        });
        Ok(Response {
            content,
            stop: Stop {
                reason,
                raw: Some(raw_stop),
            },
            usage,
            warnings: Vec::new(),
            origin: self.ctx.origin(),
            latency_ms: 0.0,
            first_token_ms: None,
            raw: if self.ctx.include_raw {
                Some(response)
            } else {
                None
            },
        })
    }
}
