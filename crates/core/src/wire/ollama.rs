//! The Ollama native wire.
use super::*;
use crate::effects::{HttpRequest, Method, Purpose};
use crate::retry::parse_retry_after;
use serde_json::json;

/// The `ollama.native` codec.
#[derive(Debug, Clone, Copy, Default)]
pub struct OllamaNative;

struct Encoder<'a> {
    ctx: &'a CallContext,
    warnings: Vec<Warning>,
    call_names: Vec<(String, String)>,
}

impl Encoder<'_> {
    fn warn(&mut self, feature: &str, detail: impl Into<String>) {
        self.warnings.push(Warning {
            kind: WarningKind::Unsupported,
            feature: feature.to_owned(),
            detail: Some(detail.into()),
        });
    }

    fn user_parts(&mut self, content: &[Block], text: &mut Vec<String>, images: &mut Vec<String>) {
        for b in content {
            match b {
                Block::Text(t) => text.push(t.text.clone()),
                Block::Image(img) => match &img.data {
                    Some(data) => images.push(data.clone()),
                    None => self.warn(
                        "image",
                        "image needs bytes or a url on Ollama; file ids are not supported",
                    ),
                },
                Block::Document(d) => match &d.text {
                    Some(t) => text.push(t.clone()),
                    None => self.warn(
                        "document",
                        "documents are not supported on Ollama; only text documents are inlined",
                    ),
                },
                other => self.warn(
                    "content",
                    format!("{} block dropped from this turn", kind_name(other)),
                ),
            }
        }
    }

    fn messages(&mut self, system: Option<&[SystemBlock]>, messages: &[Message]) -> Vec<Value> {
        let mut out = Vec::new();
        if let Some(system) = system.filter(|s| !s.is_empty()) {
            let text: Vec<&str> = system
                .iter()
                .map(|SystemBlock::Text(b)| b.text.as_str())
                .collect();
            out.push(json!({"role": "system", "content": text.join("\n\n")}));
        }
        for m in messages {
            match &m.role {
                Role::User | Role::System | Role::Other(_) => {
                    let mut text = Vec::new();
                    let mut images = Vec::new();
                    self.user_parts(&m.content, &mut text, &mut images);
                    let mut o = Map::new();
                    o.insert("role".into(), json!(m.role.as_str()));
                    o.insert("content".into(), json!(text.join("\n\n")));
                    if !images.is_empty() {
                        o.insert("images".into(), json!(images));
                    }
                    out.push(Value::Object(o));
                }
                Role::Assistant => out.push(self.assistant(&m.content)),
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
            }
        }
        out
    }

    fn assistant(&mut self, content: &[Block]) -> Value {
        let mut text = Vec::new();
        let mut thinking = Vec::new();
        let mut calls = Vec::new();
        let mut images = Vec::new();
        for b in content {
            match b {
                Block::Text(t) => text.push(t.text.clone()),
                Block::Reasoning(r) => {
                    if !r.text.is_empty() {
                        thinking.push(r.text.clone());
                    }
                }
                Block::ToolCall(c) => {
                    self.call_names.push((c.id.clone(), c.name.clone()));
                    let mut f = Map::new();
                    f.insert("name".into(), json!(c.name));
                    f.insert("arguments".into(), c.arguments.clone());
                    let mut o = Map::new();
                    o.insert("type".into(), json!("function"));
                    o.insert("function".into(), Value::Object(f));
                    apply_vendor(&mut o, c.meta.vendor.as_ref(), &self.ctx.ns);
                    calls.push(Value::Object(o));
                }
                Block::Image(img) if img.data.is_some() => {
                    images.push(img.data.clone().unwrap_or_default())
                }
                other => self.warn(
                    "content",
                    format!("{} block dropped from an assistant turn", kind_name(other)),
                ),
            }
        }
        let mut o = Map::new();
        o.insert("role".into(), json!("assistant"));
        o.insert("content".into(), json!(text.join("")));
        if !thinking.is_empty() {
            o.insert("thinking".into(), json!(thinking.join("\n\n")));
        }
        if !calls.is_empty() {
            o.insert("tool_calls".into(), Value::Array(calls));
        }
        if !images.is_empty() {
            o.insert("images".into(), json!(images));
        }
        Value::Object(o)
    }

    fn tool_result(&mut self, r: &ToolResultBlock) -> Value {
        let name = self
            .call_names
            .iter()
            .rev()
            .find(|(id, _)| *id == r.tool_call_id)
            .map(|(_, n)| n.clone());
        let name = match name {
            Some(n) => n,
            None => {
                self.warn(
                    "tool_result",
                    format!(
                        "no earlier tool call with id `{}`; the id is sent as the tool name",
                        r.tool_call_id
                    ),
                );
                r.tool_call_id.clone()
            }
        };
        let mut text = Vec::new();
        for b in &r.content {
            match b {
                Block::Text(t) => text.push(t.text.clone()),
                Block::Document(d) if d.text.is_some() => {
                    text.push(d.text.clone().unwrap_or_default())
                }
                other => self.warn(
                    "tool_result",
                    format!(
                        "{} block dropped from a tool result; Ollama takes text only",
                        kind_name(other)
                    ),
                ),
            }
        }
        let mut content = text.join("\n\n");
        if r.is_error == Some(true) {
            content = format!("[error] {content}");
        }
        let mut o = Map::new();
        o.insert("role".into(), json!("tool"));
        o.insert("tool_name".into(), json!(name));
        o.insert("content".into(), json!(content));
        apply_vendor(&mut o, r.meta.vendor.as_ref(), &self.ctx.ns);
        Value::Object(o)
    }

    fn tools(&mut self, tools: &[Tool]) -> Vec<Value> {
        let mut out = Vec::new();
        for t in tools {
            match t {
                Tool::Function(d) => {
                    let mut f = Map::new();
                    f.insert("name".into(), json!(d.name));
                    f.insert("description".into(), json!(d.description));
                    f.insert("parameters".into(), d.input_schema.clone());
                    apply_vendor(&mut f, d.vendor.as_ref(), &self.ctx.ns);
                    out.push(json!({"type": "function", "function": Value::Object(f)}));
                }
                Tool::Provider(p) => self.warn(
                    "provider_tool",
                    format!("`{}` is not available on Ollama", p.provider_tool),
                ),
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

fn base_headers(ctx: &CallContext) -> Headers {
    let mut h = Headers::new();
    h.insert("content-type".into(), "application/json".into());
    for (k, v) in &ctx.headers {
        h.insert(k.to_ascii_lowercase(), v.clone());
    }
    h
}

fn think_level(level: &ReasoningLevel) -> (Value, Option<&'static str>) {
    match level {
        ReasoningLevel::None => (json!(false), None),
        ReasoningLevel::Minimal => (json!("low"), Some("minimal rounded to low")),
        ReasoningLevel::Low => (json!("low"), None),
        ReasoningLevel::Medium => (json!("medium"), None),
        ReasoningLevel::High => (json!("high"), None),
        ReasoningLevel::Xhigh => (json!("high"), Some("xhigh rounded to high")),
        ReasoningLevel::Max => (json!("high"), Some("max rounded to high")),
        ReasoningLevel::Auto => (Value::Null, None),
        ReasoningLevel::Other(s) => (json!(s), None),
    }
}

impl Wire for OllamaNative {
    fn build_chat(
        &self,
        ctx: &CallContext,
        req: &ChatRequest,
        _index_offset: u32,
    ) -> Result<Built, ProviderError> {
        let opts = req.options.clone().unwrap_or_default();
        let mut enc = Encoder {
            ctx,
            warnings: Vec::new(),
            call_names: Vec::new(),
        };
        let mut body = Body::new();
        body.user("model", json!(ctx.model));
        body.lib("stream", json!(true));
        let no_tools = opts
            .tool_choice
            .as_ref()
            .is_some_and(|tc| tc.mode.as_str() == "none");
        body.lib(
            "messages",
            Value::Array(enc.messages(req.system.as_deref(), &req.messages)),
        );
        if let Some(tools) = req.tools.as_ref().filter(|t| !t.is_empty() && !no_tools) {
            let encoded = enc.tools(tools);
            if !encoded.is_empty() {
                body.user("tools", Value::Array(encoded));
            }
        }
        let mut options = Map::new();
        if let Some(n) = opts.max_output_tokens {
            options.insert("num_predict".into(), json!(n));
        }
        if let Some(stop) = opts.stop_sequences.as_ref().filter(|s| !s.is_empty()) {
            options.insert("stop".into(), json!(stop));
        }
        match opts.reasoning.clone().unwrap_or(ReasoningLevel::Auto) {
            ReasoningLevel::Auto => {}
            level => {
                let (value, note) = think_level(&level);
                if let Some(note) = note {
                    enc.warnings.push(Warning {
                        kind: WarningKind::Compatibility,
                        feature: "reasoning".into(),
                        detail: Some(note.to_owned()),
                    });
                }
                body.user("think", value);
            }
        }
        if matches!(opts.reasoning_summary, Some(ReasoningSummary::None)) {
            enc.warn(
                "reasoning_summary",
                "Ollama always returns the raw thinking text",
            );
        }
        if let Some(tc) = &opts.tool_choice {
            match tc.mode.as_str() {
                "auto" | "none" => {}
                other => enc.warn(
                    "tool_choice",
                    format!("`{other}` is not supported on Ollama; the model decides"),
                ),
            }
        }
        if opts.parallel_tools.is_some() {
            enc.warn("parallel_tools", "Ollama has no parallel tool call toggle");
        }
        if let Some(out) = &opts.output {
            body.user("format", out.json_schema.clone());
        }
        if opts.user_id.is_some() {
            enc.warn("user_id", "Ollama has no user identifier field");
        }
        if matches!(opts.cache, Some(CachePolicy::Explicit)) {
            enc.warn("cache", "Ollama has no prompt cache control");
        }
        let mut vendor = vendor_for(req.vendor.as_ref(), &ctx.ns).cloned();
        if let Some(v) = vendor.as_mut() {
            if let Some(Value::Object(extra)) = v.remove("options") {
                for (k, val) in extra {
                    if options.contains_key(&k) {
                        return Err(argument_error(format!("vendor field `options.{k}` clashes with an option that sets the same field")));
                    }
                    options.insert(k, val);
                }
            }
        }
        if !options.is_empty() {
            body.user("options", Value::Object(options));
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
            url: join_url(&ctx.base_url, "api/chat"),
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

    fn chat_decoder(&self, ctx: &CallContext, index_offset: u32) -> Box<dyn ChatDecoder> {
        Box::new(Decoder::new(ctx.clone(), index_offset))
    }

    fn wants_continuation(&self, _response: &Response) -> bool {
        false
    }

    fn build_embed(&self, ctx: &CallContext, req: &EmbedRequest) -> Result<Built, ProviderError> {
        let mut body = Body::new();
        body.user("model", json!(ctx.model));
        body.user("input", json!(req.input));
        body.lib("truncate", json!(false));
        if let Some(d) = req.options.as_ref().and_then(|o| o.dimensions) {
            body.user("dimensions", json!(d));
        }
        body.merge_vendor(vendor_for(req.vendor.as_ref(), &ctx.ns))?;
        let request = HttpRequest {
            method: Method::Post,
            url: join_url(&ctx.base_url, "api/embed"),
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
        let rows = v
            .get("embeddings")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("embed reply has no `embeddings`"))?;
        let embeddings = rows
            .iter()
            .map(|row| {
                row.as_array()
                    .map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<f64>>())
                    .ok_or_else(|| malformed("embedding row is not a vector"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let usage = u64_field(&v, "prompt_eval_count").map(|t| EmbedUsage {
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
        Some(HttpRequest {
            method: Method::Get,
            url: join_url(&ctx.base_url, "api/tags"),
            headers: base_headers(ctx),
            body: None,
            stream: false,
            timeout: ctx.timeout,
            purpose: Purpose::Models,
        })
    }

    fn decode_models(
        &self,
        _ctx: &CallContext,
        body: &[u8],
    ) -> Result<(Vec<ModelInfo>, Option<HttpRequest>), ProviderError> {
        let v = parse_json(body)?;
        let rows = v
            .get("models")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("model listing has no `models`"))?;
        let models = rows
            .iter()
            .filter_map(|m| {
                let id = str_field(m, "name").or(str_field(m, "model"))?;
                let family = m
                    .get("details")
                    .and_then(|d| str_field(d, "family"))
                    .unwrap_or("");
                let kind = if family.contains("bert") || id.contains("embed") {
                    ModelKind::Embedding
                } else {
                    ModelKind::Unknown
                };
                Some(ModelInfo {
                    id: id.to_owned(),
                    kind,
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
        Some(HttpRequest {
            method: Method::Post,
            url: join_url(&ctx.base_url, "api/show"),
            headers: base_headers(ctx),
            body: Some(json!({"model": model}).to_string()),
            stream: false,
            timeout: ctx.timeout,
            purpose: Purpose::Models,
        })
    }

    fn decode_model_info(
        &self,
        ctx: &CallContext,
        body: &[u8],
    ) -> Result<ModelInfo, ProviderError> {
        let v = parse_json(body)?;
        let caps: Vec<&str> = v
            .get("capabilities")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        let info = v.get("model_info").and_then(Value::as_object);
        let arch = info
            .and_then(|i| i.get("general.architecture"))
            .and_then(Value::as_str);
        let context_window = info.and_then(|i| {
            arch.and_then(|a| i.get(&format!("{a}.context_length")))
                .or_else(|| {
                    i.iter()
                        .find(|(k, _)| k.ends_with(".context_length"))
                        .map(|(_, val)| val)
                })
                .and_then(Value::as_u64)
        });
        let embedding = caps.contains(&"embedding");
        let levels = v
            .get("thinking")
            .and_then(|t| t.get("values"))
            .and_then(Value::as_array)
            .map(|vals| {
                let mut out = vec![ReasoningLevel::Auto];
                for val in vals {
                    match val {
                        Value::Bool(false) => out.push(ReasoningLevel::None),
                        Value::String(s) => out.push(ReasoningLevel::from(s.as_str())),
                        _ => {}
                    }
                }
                out
            });
        let thinking_off_only = v
            .get("thinking")
            .and_then(|t| t.get("values"))
            .and_then(Value::as_array)
            .is_some_and(|vals| vals.len() == 1 && vals[0] == Value::Bool(false));
        let mut inputs = vec![InputKind::Text];
        if caps.contains(&"vision") {
            inputs.push(InputKind::Image);
        }
        let can_disable = !caps.contains(&"thinking")
            || levels
                .as_ref()
                .is_some_and(|l| l.contains(&ReasoningLevel::None));
        let capabilities = Capabilities {
            reasoning: Some(ReasoningCapability {
                levels: Some(if caps.contains(&"thinking") && !thinking_off_only {
                    levels.or_else(|| Some(vec![ReasoningLevel::Auto, ReasoningLevel::None]))
                } else {
                    None
                }),
                can_disable: Some(Some(can_disable)),
            }),
            tool_choice: Some(Some(vec![ToolChoiceMode::Auto, ToolChoiceMode::None])),
            parallel_tools: Some(ParallelToolsCapability {
                supported: Some(Some(caps.contains(&"tools"))),
                toggle: Some(Some(false)),
            }),
            output: Some(OutputCapability {
                json_schema: Some(Some(!embedding)),
            }),
            cache: Some(CacheCapability {
                breakpoints: Some(None),
                ttls: Some(None),
            }),
            inputs: Some(Some(inputs)),
            tools: Some(ToolsCapability {
                functions: Some(Some(caps.contains(&"tools"))),
                hosted: Some(Some(Vec::new())),
            }),
            limits: context_window.map(|c| Limits {
                context_window: Some(c),
                max_output_tokens: None,
            }),
            embeddings: Some(Some(embedding)),
            source: CapabilitySource::Vendor,
        };
        Ok(ModelInfo {
            id: ctx.model.clone(),
            kind: if embedding {
                ModelKind::Embedding
            } else {
                ModelKind::Chat
            },
            context_window,
            max_output_tokens: None,
            capabilities: Some(capabilities),
            raw: Some(v.clone()),
        })
    }

    fn shipped_capabilities(&self, _model: &str) -> Option<Capabilities> {
        None
    }

    fn classify(&self, status: u16, headers: &Headers, body: &[u8]) -> ProviderError {
        let parsed: Option<Value> = serde_json::from_slice(body).ok();
        let message = parsed
            .as_ref()
            .and_then(|v| str_field(v, "error"))
            .unwrap_or("")
            .to_owned();
        let lower = message.to_ascii_lowercase();
        let retry_after = headers
            .get("retry-after")
            .and_then(|v| parse_retry_after(v));
        let (kind, retryable, code) = match status {
            401 | 403 => (ErrorKind::Auth, false, None),
            404 if lower.contains("model") => (ErrorKind::NotFound, false, Some("model_not_found")),
            404 => (ErrorKind::NotFound, false, None),
            429 => (ErrorKind::RateLimit, true, None),
            400 if lower.contains("context length") || lower.contains("exceeds") => {
                (ErrorKind::ContextOverflow, false, Some("context_length"))
            }
            400 if lower.contains("does not support") => {
                (ErrorKind::Unsupported, false, Some("model_capability"))
            }
            400 => (ErrorKind::BadRequest, false, None),
            s if s >= 500 => (ErrorKind::Server, true, None),
            s if (400..500).contains(&s) => (ErrorKind::BadRequest, false, None),
            _ => (ErrorKind::MalformedResponse, false, None),
        };
        ProviderError {
            kind,
            message: if message.is_empty() {
                format!("HTTP {status}")
            } else {
                format!("HTTP {status}: {message}")
            },
            status: Some(status),
            code: code.map(str::to_owned),
            retryable,
            retry_after,
            attempts: 0,
            body: parsed,
            request_id: None,
            origin: None,
        }
    }

    fn auth_optional(&self) -> bool {
        true
    }

    fn inline_images(&self) -> bool {
        true
    }
}

/// Decoder of one `/api/chat` NDJSON stream.
struct Decoder {
    ctx: CallContext,
    offset: u32,
    buffer: Vec<u8>,
    next_index: u32,
    open: Option<(u32, &'static str, String)>,
    content: Vec<Block>,
    calls: u32,
    done: Option<Value>,
    complete: bool,
}

impl Decoder {
    fn new(ctx: CallContext, offset: u32) -> Self {
        Decoder {
            ctx,
            offset,
            buffer: Vec::new(),
            next_index: 0,
            open: None,
            content: Vec::new(),
            calls: 0,
            done: None,
            complete: false,
        }
    }

    fn close(&mut self, out: &mut Vec<Event>) {
        let Some((index, kind, text)) = self.open.take() else {
            return;
        };
        let block = match kind {
            "reasoning" => Block::Reasoning(ReasoningBlock {
                meta: BlockMeta::default(),
                text,
                vendor_state: None,
                origin: None,
            }),
            _ => Block::Text(TextBlock {
                meta: BlockMeta::default(),
                text,
            }),
        };
        self.content.push(block.clone());
        out.push(Event::BlockStop {
            parent_id: None,
            index: index + self.offset,
            block,
        });
    }

    fn delta(&mut self, kind: &'static str, text: &str, out: &mut Vec<Event>) {
        if text.is_empty() {
            return;
        }
        if self.open.as_ref().is_some_and(|(_, k, _)| *k != kind) {
            self.close(out);
        }
        if self.open.is_none() {
            let index = self.next_index;
            self.next_index += 1;
            self.open = Some((index, kind, String::new()));
            let block_kind = if kind == "reasoning" {
                BlockKind::Reasoning
            } else {
                BlockKind::Text
            };
            out.push(Event::BlockStart {
                parent_id: None,
                index: index + self.offset,
                kind: block_kind,
                id: None,
                name: None,
            });
        }
        let (index, _, buf) = self.open.as_mut().expect("open");
        buf.push_str(text);
        let index = *index + self.offset;
        if kind == "reasoning" {
            out.push(Event::ReasoningDelta {
                parent_id: None,
                index,
                text: text.to_owned(),
            });
        } else {
            out.push(Event::TextDelta {
                parent_id: None,
                index,
                text: text.to_owned(),
            });
        }
    }

    fn tool_call(&mut self, call: &Value, out: &mut Vec<Event>) {
        self.close(out);
        let f = call.get("function").cloned().unwrap_or(Value::Null);
        let name = str_field(&f, "name").unwrap_or("").to_owned();
        let arguments = f
            .get("arguments")
            .cloned()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({}));
        let id = format!("call_{}", self.calls);
        self.calls += 1;
        let index = self.next_index;
        self.next_index += 1;
        let json = arguments.to_string();
        out.push(Event::BlockStart {
            parent_id: None,
            index: index + self.offset,
            kind: BlockKind::ToolCall,
            id: Some(id.clone()),
            name: Some(name.clone()),
        });
        out.push(Event::ToolCallDelta {
            parent_id: None,
            index: index + self.offset,
            arguments_json: json,
        });
        let block = Block::ToolCall(ToolCallBlock {
            meta: BlockMeta::default(),
            id,
            name,
            arguments,
            truncated: None,
            arguments_json: None,
            status: None,
        });
        self.content.push(block.clone());
        out.push(Event::BlockStop {
            parent_id: None,
            index: index + self.offset,
            block,
        });
    }

    fn line(&mut self, line: &[u8], out: &mut Vec<Event>) -> Result<(), ProviderError> {
        let v: Value =
            serde_json::from_slice(line).map_err(|e| malformed(format!("bad NDJSON line: {e}")))?;
        if self.ctx.include_raw {
            out.push(Event::Raw {
                parent_id: None,
                event: v.clone(),
            });
        }
        if let Some(err) = str_field(&v, "error") {
            let mut e = ProviderError::new(ErrorKind::Server, format!("stream error: {err}"));
            e.body = Some(v.clone());
            return Err(e);
        }
        if let Some(message) = v.get("message") {
            if let Some(t) = str_field(message, "thinking") {
                self.delta("reasoning", t, out);
            }
            if let Some(t) = str_field(message, "content") {
                self.delta("text", t, out);
            }
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                for call in calls {
                    self.tool_call(call, out);
                }
            }
        }
        if v.get("done").and_then(Value::as_bool) == Some(true) {
            self.close(out);
            self.done = Some(v);
            self.complete = true;
        }
        Ok(())
    }
}

impl ChatDecoder for Decoder {
    fn feed(&mut self, bytes: &[u8]) -> Result<Decoded, ProviderError> {
        let mut out = Decoded::default();
        self.buffer.extend_from_slice(bytes);
        while let Some(pos) = self.buffer.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buffer.drain(..=pos).collect();
            let line = &line[..line.len() - 1];
            let line = line.strip_suffix(b"\r").unwrap_or(line);
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            self.line(line, &mut out.events)?;
        }
        out.complete = self.complete;
        Ok(out)
    }

    fn finish(&mut self) -> Result<Response, ProviderError> {
        if !self.buffer.iter().all(u8::is_ascii_whitespace) {
            let rest = std::mem::take(&mut self.buffer);
            let mut events = Vec::new();
            self.line(&rest, &mut events)?;
        }
        if !self.complete {
            return Err(malformed("stream ended before the final line"));
        }
        let done = self.done.take().unwrap_or(Value::Null);
        let content = std::mem::take(&mut self.content);
        let raw_reason = str_field(&done, "done_reason").unwrap_or("").to_owned();
        let vendor_stop = match raw_reason.as_str() {
            "stop" | "" => StopReason::Complete,
            "length" => StopReason::MaxOutputTokens,
            _ => StopReason::OtherReason,
        };
        let reason = derive_stop(&content, vendor_stop);
        let usage = u64_field(&done, "prompt_eval_count")
            .or(u64_field(&done, "eval_count"))
            .map(|_| {
                let mut raw = Map::new();
                for key in [
                    "prompt_eval_count",
                    "prompt_eval_cached_count",
                    "eval_count",
                    "total_duration",
                    "load_duration",
                    "prompt_eval_duration",
                    "eval_duration",
                ] {
                    if let Some(val) = done.get(key) {
                        raw.insert(key.to_owned(), val.clone());
                    }
                }
                Usage {
                    input: InputUsage {
                        total: u64_field(&done, "prompt_eval_count").unwrap_or(0),
                        cache_read: u64_field(&done, "prompt_eval_cached_count"),
                        cache_write: None,
                    },
                    output: OutputUsage {
                        total: u64_field(&done, "eval_count").unwrap_or(0),
                        reasoning: None,
                    },
                    raw: Some(Value::Object(raw)),
                }
            });
        Ok(Response {
            content,
            stop: Stop {
                reason,
                raw: Some(json!(raw_reason)),
            },
            usage,
            warnings: Vec::new(),
            origin: self.ctx.origin(),
            latency_ms: 0.0,
            first_token_ms: None,
            raw: if self.ctx.include_raw {
                Some(done)
            } else {
                None
            },
        })
    }
}
