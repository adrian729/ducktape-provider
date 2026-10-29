//! The Anthropic Messages wire.
use super::*;
use crate::effects::{HttpRequest, Method, Purpose};
use crate::retry::parse_retry_after;
use crate::sse::SseParser;
use serde_json::json;

/// Default API version sent as `anthropic-version`.
pub const API_VERSION: &str = "2026-10-21";

/// The `anthropic.messages` codec.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnthropicMessages;

/// How thinking can be turned off on a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Disable {
    Yes,
    No,
    BetweenTools,
}

/// What the shipped table knows about a Claude model.
#[derive(Debug, Clone, Copy)]
struct Profile {
    adaptive: bool,
    max_output: u64,
    context_window: u64,
    disable: Disable,
    forced_tool_choice: bool,
    efforts: &'static [&'static str],
    known: bool,
}

const EFFORTS_FULL: &[&str] = &["low", "medium", "high", "xhigh", "max"];
const EFFORTS_4_6: &[&str] = &["low", "medium", "high", "max"];

fn profile(model: &str) -> Profile {
    let m = model.to_ascii_lowercase();
    let starts = |p: &str| m.starts_with(p);
    let full = Profile {
        adaptive: true,
        max_output: 128_000,
        context_window: 1_000_000,
        disable: Disable::Yes,
        forced_tool_choice: true,
        efforts: EFFORTS_FULL,
        known: true,
    };
    if starts("claude-fable-5") || starts("claude-mythos-5") || starts("claude-opus-5-5") {
        return Profile {
            disable: Disable::No,
            forced_tool_choice: false,
            ..full
        };
    }
    if starts("claude-sonnet-5-5") {
        return Profile {
            disable: Disable::BetweenTools,
            forced_tool_choice: false,
            ..full
        };
    }
    if starts("claude-opus-5") || starts("claude-sonnet-5") || starts("claude-haiku-5") {
        return full;
    }
    if starts("claude-opus-4-8") || starts("claude-opus-4-7") {
        return Profile {
            context_window: 1_000_000,
            ..full
        };
    }
    if starts("claude-opus-4-6") || starts("claude-sonnet-4-6") {
        return Profile {
            context_window: 1_000_000,
            efforts: EFFORTS_4_6,
            ..full
        };
    }
    if starts("claude-haiku-4-5") {
        return Profile {
            adaptive: false,
            max_output: 64_000,
            context_window: 200_000,
            efforts: &[],
            ..full
        };
    }
    if starts("claude-sonnet-4-5") || starts("claude-opus-4-5") {
        return Profile {
            adaptive: false,
            max_output: 64_000,
            context_window: 200_000,
            efforts: &["low", "medium", "high"],
            ..full
        };
    }
    if starts("claude-opus-4") || starts("claude-sonnet-4") {
        return Profile {
            adaptive: false,
            max_output: 32_000,
            context_window: 200_000,
            efforts: &[],
            ..full
        };
    }
    if starts("claude-3-7") {
        return Profile {
            adaptive: false,
            max_output: 64_000,
            context_window: 200_000,
            efforts: &[],
            ..full
        };
    }
    if starts("claude-3") {
        return Profile {
            adaptive: false,
            max_output: 8_192,
            context_window: 200_000,
            efforts: &[],
            ..full
        };
    }
    Profile {
        known: false,
        ..full
    }
}

fn ttl(mark: &CacheMark) -> Value {
    match mark.ttl.as_ref().map(|t| t.as_str()) {
        Some("long") => json!({"type": "ephemeral", "ttl": "1h"}),
        _ => json!({"type": "ephemeral"}),
    }
}

fn block_meta(
    target: &mut Map<String, Value>,
    meta: &BlockMeta,
    ns: &Ns,
    cache_policy: &CachePolicy,
) {
    if let Some(mark) = &meta.cache {
        if !matches!(cache_policy, CachePolicy::None) {
            target.insert("cache_control".to_owned(), ttl(mark));
        }
    }
    apply_vendor(target, meta.vendor.as_ref(), ns);
}

fn source_from(
    data: &Option<String>,
    url: &Option<String>,
    file_id: &Option<String>,
    media_type: Option<&str>,
    default_media: &str,
) -> Option<Value> {
    if let Some(data) = data {
        let media = media_type
            .map(str::to_owned)
            .or_else(|| sniff_image(data).map(str::to_owned))
            .unwrap_or_else(|| default_media.to_owned());
        return Some(json!({"type": "base64", "media_type": media, "data": data}));
    }
    if let Some(url) = url {
        return Some(json!({"type": "url", "url": url}));
    }
    file_id
        .as_ref()
        .map(|id| json!({"type": "file", "file_id": id}))
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

    fn block(&mut self, block: &Block) -> Result<Option<Value>, ProviderError> {
        let ns = &self.ctx.ns;
        let out = match block {
            Block::Text(b) => {
                let mut o = Map::new();
                o.insert("type".into(), json!("text"));
                o.insert("text".into(), json!(b.text));
                block_meta(&mut o, &b.meta, ns, &self.cache_policy);
                Value::Object(o)
            }
            Block::Image(b) => {
                let Some(source) = source_from(
                    &b.data,
                    &b.url,
                    &b.file_id,
                    b.media_type.as_deref(),
                    "image/png",
                ) else {
                    return Err(argument_error("image block needs data, url or file_id"));
                };
                let mut o = Map::new();
                o.insert("type".into(), json!("image"));
                o.insert("source".into(), source);
                block_meta(&mut o, &b.meta, ns, &self.cache_policy);
                Value::Object(o)
            }
            Block::Document(b) => {
                let source = match &b.text {
                    Some(text) => json!({"type": "text", "media_type": "text/plain", "data": text}),
                    None => match source_from(
                        &b.data,
                        &b.url,
                        &b.file_id,
                        b.media_type.as_deref(),
                        "application/pdf",
                    ) {
                        Some(s) => s,
                        None => {
                            return Err(argument_error(
                                "document block needs data, url, file_id or text",
                            ))
                        }
                    },
                };
                let mut o = Map::new();
                o.insert("type".into(), json!("document"));
                o.insert("source".into(), source);
                if let Some(name) = &b.name {
                    o.insert("title".into(), json!(name));
                }
                block_meta(&mut o, &b.meta, ns, &self.cache_policy);
                Value::Object(o)
            }
            Block::ToolCall(b) => {
                let mut o = Map::new();
                o.insert("type".into(), json!("tool_use"));
                o.insert("id".into(), json!(b.id));
                o.insert("name".into(), json!(b.name));
                o.insert("input".into(), b.arguments.clone());
                block_meta(&mut o, &b.meta, ns, &self.cache_policy);
                Value::Object(o)
            }
            Block::ToolResult(b) => {
                let mut content = Vec::new();
                for inner in &b.content {
                    if let Some(v) = self.block(inner)? {
                        content.push(v);
                    }
                }
                let mut o = Map::new();
                o.insert("type".into(), json!("tool_result"));
                o.insert("tool_use_id".into(), json!(b.tool_call_id));
                o.insert("content".into(), Value::Array(content));
                if let Some(err) = b.is_error {
                    o.insert("is_error".into(), json!(err));
                }
                block_meta(&mut o, &b.meta, ns, &self.cache_policy);
                Value::Object(o)
            }
            Block::Reasoning(b) => {
                if !self.ctx.accepts(b.origin.as_ref()) {
                    self.warn("reasoning", "reasoning block from another endpoint dropped");
                    return Ok(None);
                }
                let state = b.vendor_state.as_ref().and_then(Value::as_object);
                match state.and_then(|s| s.get("type")).and_then(Value::as_str) {
                    Some("redacted_thinking") => {
                        json!({"type": "redacted_thinking", "data": state.and_then(|s| s.get("data")).cloned().unwrap_or(Value::Null)})
                    }
                    _ => {
                        let mut o = Map::new();
                        o.insert("type".into(), json!("thinking"));
                        o.insert("thinking".into(), json!(b.text));
                        if let Some(sig) = state.and_then(|s| s.get("signature")) {
                            o.insert("signature".into(), sig.clone());
                        }
                        Value::Object(o)
                    }
                }
            }
            Block::Compaction(b) => {
                if !self.ctx.accepts(Some(&b.origin)) {
                    self.warn("compaction", "compaction block from another endpoint dropped; summarized history is lost");
                    return Ok(None);
                }
                b.vendor_state.clone()
            }
            Block::Custom(b) => {
                let ours = b.kind.starts_with("anthropic.");
                if !ours || (b.origin.is_some() && !self.ctx.accepts(b.origin.as_ref())) {
                    self.warn("custom", format!("custom block `{}` dropped", b.kind));
                    return Ok(None);
                }
                match &b.data {
                    Some(Value::Object(data)) => Value::Object(data.clone()),
                    _ => {
                        let mut o = Map::new();
                        o.insert(
                            "type".into(),
                            json!(b.kind.trim_start_matches("anthropic.")),
                        );
                        Value::Object(o)
                    }
                }
            }
        };
        Ok(Some(out))
    }

    fn messages(&mut self, messages: &[Message]) -> Result<Vec<Value>, ProviderError> {
        let mut out: Vec<(String, Vec<Value>)> = Vec::new();
        for m in messages {
            let role = match &m.role {
                Role::Tool => "user",
                other => other.as_str(),
            }
            .to_owned();
            let mut content = Vec::new();
            for b in &m.content {
                if let Some(v) = self.block(b)? {
                    content.push(v);
                }
            }
            match out.last_mut() {
                Some((last, existing)) if *last == role && role != "system" => {
                    existing.extend(content)
                }
                _ => out.push((role, content)),
            }
        }
        Ok(out
            .into_iter()
            .map(|(role, content)| json!({"role": role, "content": content}))
            .collect())
    }

    fn system(&mut self, system: &[SystemBlock]) -> Vec<Value> {
        system
            .iter()
            .map(|SystemBlock::Text(b)| {
                let mut o = Map::new();
                o.insert("type".into(), json!("text"));
                o.insert("text".into(), json!(b.text));
                block_meta(&mut o, &b.meta, &self.ctx.ns, &self.cache_policy);
                Value::Object(o)
            })
            .collect()
    }

    fn tools(&mut self, tools: &[Tool]) -> Result<Vec<Value>, ProviderError> {
        let mut out = Vec::new();
        for t in tools {
            match t {
                Tool::Function(d) => {
                    let mut o = Map::new();
                    o.insert("name".into(), json!(d.name));
                    o.insert("description".into(), json!(d.description));
                    o.insert("input_schema".into(), d.input_schema.clone());
                    if let Some(strict) = d.strict {
                        o.insert("strict".into(), json!(strict));
                    }
                    if let Some(mark) = &d.cache {
                        if !matches!(self.cache_policy, CachePolicy::None) {
                            o.insert("cache_control".into(), ttl(mark));
                        }
                    }
                    apply_vendor(&mut o, d.vendor.as_ref(), &self.ctx.ns);
                    out.push(Value::Object(o));
                }
                Tool::Provider(p) => {
                    let Some(name) = p.provider_tool.strip_prefix("anthropic.") else {
                        self.warn(
                            "provider_tool",
                            format!("`{}` is not an Anthropic tool", p.provider_tool),
                        );
                        continue;
                    };
                    let value = match &p.config {
                        Some(Value::Object(cfg)) => Value::Object(cfg.clone()),
                        _ => match name {
                            "web_search" => {
                                json!({"type": "web_search_20260209", "name": "web_search"})
                            }
                            "web_fetch" => {
                                json!({"type": "web_fetch_20260209", "name": "web_fetch"})
                            }
                            "code_execution" => {
                                json!({"type": "code_execution_20260521", "name": "code_execution"})
                            }
                            other => {
                                let mut e = ProviderError::new(ErrorKind::Unsupported, format!("provider tool `anthropic.{other}` needs an explicit `config`"));
                                e.code = Some("provider_tool".to_owned());
                                return Err(e);
                            }
                        },
                    };
                    out.push(value);
                }
            }
        }
        Ok(out)
    }
}

fn has_explicit_cache(req: &ChatRequest) -> bool {
    let in_blocks = req
        .messages
        .iter()
        .flat_map(|m| &m.content)
        .any(|b| match b {
            Block::Text(x) => x.meta.cache.is_some(),
            Block::Image(x) => x.meta.cache.is_some(),
            Block::Document(x) => x.meta.cache.is_some(),
            Block::ToolResult(x) => x.meta.cache.is_some(),
            _ => false,
        });
    let in_system = req
        .system
        .iter()
        .flatten()
        .any(|SystemBlock::Text(b)| b.meta.cache.is_some());
    let in_tools = req
        .tools
        .iter()
        .flatten()
        .any(|t| matches!(t, Tool::Function(d) if d.cache.is_some()));
    in_blocks || in_system || in_tools
}

fn auto_cache(body: &mut Body) {
    if let Some(Value::Array(system)) = body.map.get_mut("system") {
        if let Some(Value::Object(last)) = system.last_mut() {
            last.insert("cache_control".into(), json!({"type": "ephemeral"}));
            return;
        }
    }
    if let Some(Value::Array(tools)) = body.map.get_mut("tools") {
        if let Some(Value::Object(last)) = tools.last_mut() {
            last.insert("cache_control".into(), json!({"type": "ephemeral"}));
        }
    }
}

fn base_headers(ctx: &CallContext) -> Headers {
    let mut h = Headers::new();
    h.insert("content-type".into(), "application/json".into());
    h.insert(
        "anthropic-version".into(),
        ctx.api_version
            .clone()
            .unwrap_or_else(|| API_VERSION.to_owned()),
    );
    for (k, v) in &ctx.headers {
        h.insert(k.to_ascii_lowercase(), v.clone());
    }
    h
}

impl Wire for AnthropicMessages {
    fn build_chat(
        &self,
        ctx: &CallContext,
        req: &ChatRequest,
        _index_offset: u32,
    ) -> Result<Built, ProviderError> {
        let prof = profile(&ctx.model);
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
        body.lib("messages", Value::Array(enc.messages(&req.messages)?));
        if let Some(system) = &req.system {
            if !system.is_empty() {
                body.user("system", Value::Array(enc.system(system)));
            }
        }
        if let Some(tools) = &req.tools {
            if !tools.is_empty() {
                body.user("tools", Value::Array(enc.tools(tools)?));
            }
        }
        match opts.max_output_tokens {
            Some(n) => body.user("max_tokens", json!(n)),
            None => body.lib("max_tokens", json!(prof.max_output)),
        }
        let summary = opts
            .reasoning_summary
            .clone()
            .unwrap_or(ReasoningSummary::Auto);
        let display = match summary {
            ReasoningSummary::None => "omitted",
            _ => "summarized",
        };
        let level = opts.reasoning.clone().unwrap_or(ReasoningLevel::Auto);
        let user_set_reasoning = opts.reasoning.is_some();
        let set_thinking = |body: &mut Body, value: Value| {
            if user_set_reasoning || opts.reasoning_summary.is_some() {
                body.user("thinking", value)
            } else {
                body.lib("thinking", value)
            }
        };
        match &level {
            ReasoningLevel::Auto => {
                if prof.adaptive {
                    set_thinking(&mut body, json!({"type": "adaptive", "display": display}));
                }
            }
            ReasoningLevel::None => match (prof.adaptive, prof.disable) {
                (false, _) => {}
                (true, Disable::Yes) => set_thinking(&mut body, json!({"type": "disabled"})),
                (true, Disable::BetweenTools) => {
                    set_thinking(&mut body, json!({"type": "between_tools"}))
                }
                (true, Disable::No) => {
                    enc.warn(
                        "reasoning",
                        format!("reasoning cannot be disabled on {}", ctx.model),
                    );
                    set_thinking(&mut body, json!({"type": "adaptive", "display": display}));
                }
            },
            other => {
                let wanted = other.as_str();
                let effort = match wanted {
                    "minimal" => "low",
                    e => e,
                };
                let effort =
                    if prof.known && !prof.efforts.is_empty() && !prof.efforts.contains(&effort) {
                        let nearest = prof
                            .efforts
                            .iter()
                            .rev()
                            .find(|e| rank(e) <= rank(effort))
                            .or(prof.efforts.first())
                            .copied()
                            .unwrap_or("high");
                        enc.warn(
                            "reasoning",
                            format!("`{wanted}` rounded to `{nearest}` on {}", ctx.model),
                        );
                        nearest
                    } else {
                        if effort != wanted {
                            enc.warn("reasoning", format!("`{wanted}` rounded to `{effort}`"));
                        }
                        effort
                    };
                if prof.adaptive {
                    set_thinking(&mut body, json!({"type": "adaptive", "display": display}));
                }
                if prof.known && prof.efforts.is_empty() {
                    enc.warn("reasoning", format!("{} has no effort control", ctx.model));
                } else {
                    body.user("output_config", json!({"effort": effort}));
                }
            }
        }
        if let Some(out) = &opts.output {
            let mut cfg = body
                .map
                .get("output_config")
                .and_then(Value::as_object)
                .cloned()
                .unwrap_or_default();
            cfg.insert(
                "format".into(),
                json!({"type": "json_schema", "schema": out.json_schema}),
            );
            body.user("output_config", Value::Object(cfg));
        }
        let mut choice: Option<Map<String, Value>> = None;
        if let Some(tc) = &opts.tool_choice {
            let m = match tc.mode.as_str() {
                "auto" => json!({"type": "auto"}),
                "none" => json!({"type": "none"}),
                "required" => json!({"type": "any"}),
                "named" => json!({"type": "tool", "name": tc.name.clone().unwrap_or_default()}),
                other => {
                    return Err(argument_error(format!(
                        "unknown tool_choice mode `{other}`"
                    )))
                }
            };
            if matches!(tc.mode.as_str(), "required" | "named")
                && prof.known
                && !prof.forced_tool_choice
            {
                let mut e = ProviderError::new(
                    ErrorKind::Unsupported,
                    format!("{} does not accept forced tool choice", ctx.model),
                );
                e.code = Some("tool_choice".to_owned());
                return Err(e);
            }
            choice = m.as_object().cloned();
        }
        if let Some(parallel) = opts.parallel_tools {
            let mut m = choice.take().unwrap_or_else(|| {
                json!({"type": "auto"})
                    .as_object()
                    .cloned()
                    .unwrap_or_default()
            });
            m.insert("disable_parallel_tool_use".into(), json!(!parallel));
            choice = Some(m);
        }
        if let Some(c) = choice {
            body.user("tool_choice", Value::Object(c));
        }
        if let Some(stops) = &opts.stop_sequences {
            body.user("stop_sequences", json!(stops));
        }
        if let Some(uid) = &opts.user_id {
            body.user("metadata", json!({"user_id": uid}));
        }
        if matches!(cache_policy, CachePolicy::Auto) && !has_explicit_cache(req) {
            auto_cache(&mut body);
        }
        body.warnings.append(&mut enc.warnings);
        body.merge_vendor(vendor_for(req.vendor.as_ref(), &ctx.ns))?;
        body.enforce_strict(ctx.strict)?;
        let mut headers = base_headers(ctx);
        for (k, v) in req.headers.iter().flatten() {
            headers.insert(k.to_ascii_lowercase(), v.clone());
        }
        let request = HttpRequest {
            method: Method::Post,
            url: join_url(&ctx.base_url, "v1/messages"),
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

    fn wants_continuation(&self, response: &Response) -> bool {
        response.stop.raw.as_ref().and_then(Value::as_str) == Some("pause_turn")
    }

    fn build_embed(&self, _ctx: &CallContext, _req: &EmbedRequest) -> Result<Built, ProviderError> {
        let mut e = ProviderError::new(
            ErrorKind::Unsupported,
            "Anthropic has no embeddings endpoint",
        );
        e.code = Some("embed".to_owned());
        Err(e)
    }

    fn decode_embed(
        &self,
        _ctx: &CallContext,
        _body: &[u8],
    ) -> Result<EmbedResponse, ProviderError> {
        Err(ProviderError::new(
            ErrorKind::Unsupported,
            "Anthropic has no embeddings endpoint",
        ))
    }

    fn build_models(&self, ctx: &CallContext) -> Option<HttpRequest> {
        Some(HttpRequest {
            method: Method::Get,
            url: join_url(&ctx.base_url, "v1/models?limit=1000"),
            headers: base_headers(ctx),
            body: None,
            stream: false,
            timeout: ctx.timeout,
            purpose: Purpose::Models,
        })
    }

    fn decode_models(
        &self,
        ctx: &CallContext,
        body: &[u8],
    ) -> Result<(Vec<ModelInfo>, Option<HttpRequest>), ProviderError> {
        let v = parse_json(body)?;
        let data = v
            .get("data")
            .and_then(Value::as_array)
            .ok_or_else(|| malformed("model listing has no `data`"))?;
        let models = data.iter().map(model_info).collect::<Vec<_>>();
        let next = match (
            v.get("has_more").and_then(Value::as_bool),
            v.get("last_id").and_then(Value::as_str),
        ) {
            (Some(true), Some(last)) => Some(HttpRequest {
                method: Method::Get,
                url: join_url(
                    &ctx.base_url,
                    &format!("v1/models?limit=1000&after_id={last}"),
                ),
                headers: base_headers(ctx),
                body: None,
                stream: false,
                timeout: ctx.timeout,
                purpose: Purpose::Models,
            }),
            _ => None,
        };
        Ok((models, next))
    }

    fn build_model_info(&self, ctx: &CallContext, model: &str) -> Option<HttpRequest> {
        Some(HttpRequest {
            method: Method::Get,
            url: join_url(&ctx.base_url, &format!("v1/models/{model}")),
            headers: base_headers(ctx),
            body: None,
            stream: false,
            timeout: ctx.timeout,
            purpose: Purpose::Models,
        })
    }

    fn decode_model_info(
        &self,
        _ctx: &CallContext,
        body: &[u8],
    ) -> Result<ModelInfo, ProviderError> {
        let v = parse_json(body)?;
        if v.get("id").is_none() {
            return Err(malformed("model reply has no `id`"));
        }
        Ok(model_info(&v))
    }

    fn shipped_capabilities(&self, model: &str) -> Option<Capabilities> {
        let p = profile(model);
        if !p.known {
            return None;
        }
        let mut modes = vec![ToolChoiceMode::Auto, ToolChoiceMode::None];
        if p.forced_tool_choice {
            modes.extend([ToolChoiceMode::Required, ToolChoiceMode::Named]);
        }
        let mut levels = vec![ReasoningLevel::Auto];
        if p.disable != Disable::No {
            levels.push(ReasoningLevel::None);
        }
        levels.extend(p.efforts.iter().map(|e| ReasoningLevel::from(*e)));
        Some(Capabilities {
            reasoning: Some(ReasoningCapability {
                levels: Some(Some(levels)),
                can_disable: Some(Some(p.disable != Disable::No)),
            }),
            tool_choice: Some(Some(modes)),
            parallel_tools: Some(ParallelToolsCapability {
                supported: Some(Some(true)),
                toggle: Some(Some(true)),
            }),
            output: Some(OutputCapability {
                json_schema: Some(Some(true)),
            }),
            cache: Some(CacheCapability {
                breakpoints: Some(Some(true)),
                ttls: Some(Some(vec![CacheTtl::Short, CacheTtl::Long])),
            }),
            inputs: Some(Some(vec![
                InputKind::Text,
                InputKind::Image,
                InputKind::Document,
            ])),
            tools: Some(ToolsCapability {
                functions: Some(Some(true)),
                hosted: Some(Some(
                    [
                        "anthropic.web_search",
                        "anthropic.web_fetch",
                        "anthropic.code_execution",
                    ]
                    .map(str::to_owned)
                    .to_vec(),
                )),
            }),
            limits: Some(Limits {
                context_window: Some(p.context_window),
                max_output_tokens: Some(p.max_output),
            }),
            embeddings: Some(Some(false)),
            source: CapabilitySource::Shipped,
        })
    }

    fn classify(&self, status: u16, headers: &Headers, body: &[u8]) -> ProviderError {
        let parsed: Option<Value> = serde_json::from_slice(body).ok();
        let err = parsed.as_ref().and_then(|v| v.get("error"));
        let vendor_type = err
            .and_then(|e| e.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let vendor_message = err
            .and_then(|e| e.get("message"))
            .and_then(Value::as_str)
            .unwrap_or("");
        let retry_after = headers
            .get("retry-after")
            .and_then(|v| parse_retry_after(v));
        let (kind, retryable) = match (status, vendor_type) {
            (401 | 403, _) | (_, "authentication_error" | "permission_error") => {
                (ErrorKind::Auth, false)
            }
            (402, _) | (_, "billing_error") => (ErrorKind::Quota, false),
            (404, _) | (_, "not_found_error") => (ErrorKind::NotFound, false),
            (413, _) | (_, "request_too_large") => (ErrorKind::RequestTooLarge, false),
            (429, _) | (_, "rate_limit_error") => (ErrorKind::RateLimit, true),
            (_, "overloaded_error") | (_, "api_error") => (ErrorKind::Server, true),
            (408, _) | (_, "timeout_error") => (ErrorKind::Timeout, true),
            (s, _) if s >= 500 => (ErrorKind::Server, true),
            (400, _) if vendor_message.contains("prompt is too long") => {
                (ErrorKind::ContextOverflow, false)
            }
            (s, _) if (400..500).contains(&s) => (ErrorKind::BadRequest, false),
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
            code: if vendor_type.is_empty() {
                None
            } else {
                Some(vendor_type.to_owned())
            },
            retryable,
            retry_after,
            attempts: 0,
            body: parsed.clone(),
            request_id: headers.get("request-id").cloned(),
            origin: None,
        }
    }
}

fn rank(effort: &str) -> u8 {
    match effort {
        "low" => 1,
        "medium" => 2,
        "high" => 3,
        "xhigh" => 4,
        "max" => 5,
        _ => 0,
    }
}

fn model_info(v: &Value) -> ModelInfo {
    let id = str_field(v, "id").unwrap_or_default().to_owned();
    let p = profile(&id);
    let context_window = u64_field(v, "max_input_tokens").or(if p.known {
        Some(p.context_window)
    } else {
        None
    });
    let max_output_tokens =
        u64_field(v, "max_tokens").or(if p.known { Some(p.max_output) } else { None });
    ModelInfo {
        id,
        kind: ModelKind::Chat,
        context_window,
        max_output_tokens,
        capabilities: None,
        raw: Some(v.clone()),
    }
}

/// A block being streamed.
#[derive(Debug)]
struct Partial {
    kind: &'static str,
    start: Value,
    text: String,
    json: String,
    signature: Option<String>,
    deltas: Vec<Value>,
}

/// Decoder of one Messages stream.
struct Decoder {
    ctx: CallContext,
    offset: u32,
    sse: SseParser,
    blocks: Vec<(u32, Partial)>,
    content: Vec<Block>,
    raw_blocks: Vec<Value>,
    message: Option<Value>,
    usage: Map<String, Value>,
    stop_reason: Option<String>,
    stop_extra: Map<String, Value>,
    complete: bool,
}

impl Decoder {
    fn new(ctx: CallContext, offset: u32) -> Self {
        Decoder {
            ctx,
            offset,
            sse: SseParser::default(),
            blocks: Vec::new(),
            content: Vec::new(),
            raw_blocks: Vec::new(),
            message: None,
            usage: Map::new(),
            stop_reason: None,
            stop_extra: Map::new(),
            complete: false,
        }
    }

    fn partial_mut(&mut self, index: u32) -> Option<&mut Partial> {
        self.blocks
            .iter_mut()
            .find(|(i, _)| *i == index)
            .map(|(_, p)| p)
    }

    fn event(&mut self, ev: &Value, out: &mut Vec<Event>) -> Result<(), ProviderError> {
        if self.ctx.include_raw {
            out.push(Event::Raw {
                parent_id: None,
                event: ev.clone(),
            });
        }
        let kind = str_field(ev, "type").unwrap_or("");
        match kind {
            "message_start" => {
                let message = ev.get("message").cloned().unwrap_or(Value::Null);
                if let Some(u) = message.get("usage").and_then(Value::as_object) {
                    self.usage = u.clone();
                }
                self.message = Some(message);
            }
            "content_block_start" => {
                let index = u64_field(ev, "index")
                    .ok_or_else(|| malformed("content_block_start without index"))?
                    as u32;
                let block = ev.get("content_block").cloned().unwrap_or(Value::Null);
                let vendor_kind = str_field(&block, "type").unwrap_or("").to_owned();
                let (kind, block_kind, id, name) = match vendor_kind.as_str() {
                    "text" => ("text", BlockKind::Text, None, None),
                    "thinking" | "redacted_thinking" => {
                        ("reasoning", BlockKind::Reasoning, None, None)
                    }
                    "tool_use" => (
                        "tool_call",
                        BlockKind::ToolCall,
                        str_field(&block, "id").map(str::to_owned),
                        str_field(&block, "name").map(str::to_owned),
                    ),
                    "compaction" => ("compaction", BlockKind::Compaction, None, None),
                    _ => (
                        "custom",
                        BlockKind::Custom,
                        str_field(&block, "id").map(str::to_owned),
                        str_field(&block, "name").map(str::to_owned),
                    ),
                };
                let text = match kind {
                    "text" => str_field(&block, "text").unwrap_or("").to_owned(),
                    "reasoning" => str_field(&block, "thinking").unwrap_or("").to_owned(),
                    _ => String::new(),
                };
                let signature = str_field(&block, "signature").map(str::to_owned);
                self.blocks.push((
                    index,
                    Partial {
                        kind,
                        start: block,
                        text,
                        json: String::new(),
                        signature,
                        deltas: Vec::new(),
                    },
                ));
                out.push(Event::BlockStart {
                    parent_id: None,
                    index: index + self.offset,
                    kind: block_kind,
                    id,
                    name,
                });
            }
            "content_block_delta" => {
                let index = u64_field(ev, "index")
                    .ok_or_else(|| malformed("content_block_delta without index"))?
                    as u32;
                let delta = ev.get("delta").cloned().unwrap_or(Value::Null);
                let offset = self.offset;
                let Some(p) = self.partial_mut(index) else {
                    return Err(malformed(format!("delta for unknown block {index}")));
                };
                match str_field(&delta, "type").unwrap_or("") {
                    "text_delta" => {
                        let t = str_field(&delta, "text").unwrap_or("").to_owned();
                        p.text.push_str(&t);
                        out.push(Event::TextDelta {
                            parent_id: None,
                            index: index + offset,
                            text: t,
                        });
                    }
                    "thinking_delta" => {
                        let t = str_field(&delta, "thinking").unwrap_or("").to_owned();
                        p.text.push_str(&t);
                        out.push(Event::ReasoningDelta {
                            parent_id: None,
                            index: index + offset,
                            text: t,
                        });
                    }
                    "signature_delta" => {
                        p.signature = Some(format!(
                            "{}{}",
                            p.signature.take().unwrap_or_default(),
                            str_field(&delta, "signature").unwrap_or("")
                        ));
                    }
                    "input_json_delta" => {
                        let j = str_field(&delta, "partial_json").unwrap_or("").to_owned();
                        p.json.push_str(&j);
                        if p.kind == "tool_call" {
                            out.push(Event::ToolCallDelta {
                                parent_id: None,
                                index: index + offset,
                                arguments_json: j,
                            });
                        }
                    }
                    _ => p.deltas.push(delta),
                }
            }
            "content_block_stop" => {
                let index = u64_field(ev, "index")
                    .ok_or_else(|| malformed("content_block_stop without index"))?
                    as u32;
                let pos = self
                    .blocks
                    .iter()
                    .position(|(i, _)| *i == index)
                    .ok_or_else(|| malformed(format!("stop for unknown block {index}")))?;
                let (_, p) = self.blocks.remove(pos);
                let (block, raw) = self.finish_block(p);
                self.raw_blocks.push(raw);
                self.content.push(block.clone());
                out.push(Event::BlockStop {
                    parent_id: None,
                    index: index + self.offset,
                    block,
                });
            }
            "message_delta" => {
                if let Some(d) = ev.get("delta").and_then(Value::as_object) {
                    for (k, v) in d {
                        if k == "stop_reason" {
                            self.stop_reason = v.as_str().map(str::to_owned);
                        } else if !v.is_null() {
                            self.stop_extra.insert(k.clone(), v.clone());
                        }
                    }
                }
                if let Some(u) = ev.get("usage").and_then(Value::as_object) {
                    for (k, v) in u {
                        if !v.is_null() {
                            self.usage.insert(k.clone(), v.clone());
                        }
                    }
                }
            }
            "message_stop" => self.complete = true,
            "error" => {
                let body = ev.get("error").cloned().unwrap_or(Value::Null);
                let vendor_type = str_field(&body, "type").unwrap_or("");
                let message = str_field(&body, "message").unwrap_or("stream error");
                let kind = match vendor_type {
                    "overloaded_error" | "api_error" => ErrorKind::Server,
                    "rate_limit_error" => ErrorKind::RateLimit,
                    "authentication_error" | "permission_error" => ErrorKind::Auth,
                    _ => ErrorKind::BadRequest,
                };
                let mut e = ProviderError::new(kind, format!("stream error: {message}"));
                e.code = Some(vendor_type.to_owned());
                e.body = Some(ev.clone());
                return Err(e);
            }
            _ => {}
        }
        Ok(())
    }

    fn finish_block(&self, p: Partial) -> (Block, Value) {
        let meta = BlockMeta::default();
        match p.kind {
            "text" => {
                let mut raw = p.start.clone();
                raw["text"] = json!(p.text);
                for d in &p.deltas {
                    if let Some(c) = d.get("citation") {
                        raw["citations"]
                            .as_array_mut()
                            .map(|a| a.push(c.clone()))
                            .unwrap_or_else(|| raw["citations"] = json!([c]));
                    }
                }
                let vendor = raw.get("citations").map(|c| {
                    let mut m = VendorMap::new();
                    m.insert("anthropic".into(), json!({"citations": c}));
                    m
                });
                (
                    Block::Text(TextBlock {
                        meta: BlockMeta { vendor, ..meta },
                        text: p.text,
                    }),
                    raw,
                )
            }
            "reasoning" => {
                let redacted = str_field(&p.start, "type") == Some("redacted_thinking");
                let (state, raw) = if redacted {
                    (
                        json!({"type": "redacted_thinking", "data": p.start.get("data").cloned().unwrap_or(Value::Null)}),
                        p.start.clone(),
                    )
                } else {
                    let mut raw = p.start.clone();
                    raw["thinking"] = json!(p.text);
                    if let Some(s) = &p.signature {
                        raw["signature"] = json!(s);
                    }
                    (json!({"signature": p.signature}), raw)
                };
                (
                    Block::Reasoning(ReasoningBlock {
                        meta,
                        text: p.text,
                        vendor_state: Some(state),
                        origin: Some(self.ctx.origin()),
                    }),
                    raw,
                )
            }
            "tool_call" => {
                let id = str_field(&p.start, "id").unwrap_or("").to_owned();
                let name = str_field(&p.start, "name").unwrap_or("").to_owned();
                let (arguments, truncated, arguments_json) =
                    parse_arguments(&p.json, p.start.get("input"));
                let mut raw = p.start.clone();
                raw["input"] = arguments.clone();
                (
                    Block::ToolCall(ToolCallBlock {
                        meta,
                        id,
                        name,
                        arguments,
                        truncated,
                        arguments_json,
                        status: None,
                    }),
                    raw,
                )
            }
            "compaction" => {
                let mut raw = p.start.clone();
                merge_deltas(&mut raw, &p.deltas, &p.json);
                let text = raw
                    .get("content")
                    .and_then(Value::as_str)
                    .map(str::to_owned);
                (
                    Block::Compaction(CompactionBlock {
                        meta,
                        vendor_state: raw.clone(),
                        text,
                        origin: self.ctx.origin(),
                    }),
                    raw,
                )
            }
            _ => {
                let mut raw = p.start.clone();
                merge_deltas(&mut raw, &p.deltas, &p.json);
                let kind = format!(
                    "anthropic.{}",
                    str_field(&p.start, "type").unwrap_or("unknown")
                );
                let id = str_field(&p.start, "id").map(str::to_owned);
                (
                    Block::Custom(CustomBlock {
                        meta: BlockMeta { id, ..meta },
                        kind,
                        data: Some(raw.clone()),
                        origin: Some(self.ctx.origin()),
                    }),
                    raw,
                )
            }
        }
    }
}

fn merge_deltas(raw: &mut Value, deltas: &[Value], json: &str) {
    if !json.is_empty() {
        raw["input"] = serde_json::from_str(json).unwrap_or(Value::String(json.to_owned()));
    }
    if !deltas.is_empty() {
        raw["deltas"] = Value::Array(deltas.to_vec());
    }
}

fn parse_arguments(
    json: &str,
    start_input: Option<&Value>,
) -> (Value, Option<bool>, Option<String>) {
    if json.trim().is_empty() {
        let v = start_input
            .cloned()
            .filter(|v| v.is_object())
            .unwrap_or_else(|| json!({}));
        return (v, None, None);
    }
    match serde_json::from_str::<Value>(json) {
        Ok(v) if v.is_object() => (v, None, None),
        _ => (json!({}), Some(true), Some(json.to_owned())),
    }
}

fn map_stop(raw: Option<&str>) -> StopReason {
    match raw {
        Some("end_turn") => StopReason::Complete,
        Some("tool_use") => StopReason::ToolCall,
        Some("max_tokens") => StopReason::MaxOutputTokens,
        Some("stop_sequence") => StopReason::StopSequence,
        Some("pause_turn") => StopReason::ContinueRequired,
        Some("refusal") => StopReason::Refusal,
        Some("compaction") => StopReason::Compaction,
        Some("model_context_window_exceeded") => StopReason::ContextWindowExceeded,
        _ => StopReason::OtherReason,
    }
}

impl ChatDecoder for Decoder {
    fn feed(&mut self, bytes: &[u8]) -> Result<Decoded, ProviderError> {
        let mut out = Decoded::default();
        for sse in self.sse.feed(bytes) {
            let v: Value = serde_json::from_str(&sse.data)
                .map_err(|e| malformed(format!("bad SSE JSON: {e}")))?;
            self.event(&v, &mut out.events)?;
        }
        out.complete = self.complete;
        Ok(out)
    }

    fn finish(&mut self) -> Result<Response, ProviderError> {
        if let Some(sse) = self.sse.finish() {
            let v: Value = serde_json::from_str(&sse.data)
                .map_err(|e| malformed(format!("bad SSE JSON: {e}")))?;
            let mut events = Vec::new();
            self.event(&v, &mut events)?;
        }
        if !self.complete {
            return Err(malformed("stream ended before message_stop"));
        }
        let content = std::mem::take(&mut self.content);
        let vendor_stop = map_stop(self.stop_reason.as_deref());
        let reason = derive_stop(&content, vendor_stop);
        let usage = if self.usage.is_empty() {
            None
        } else {
            let get = |k: &str| self.usage.get(k).and_then(Value::as_u64);
            let input = get("input_tokens").unwrap_or(0);
            let cache_read = get("cache_read_input_tokens");
            let cache_write = get("cache_creation_input_tokens");
            Some(Usage {
                input: InputUsage {
                    total: input + cache_read.unwrap_or(0) + cache_write.unwrap_or(0),
                    cache_read,
                    cache_write,
                },
                output: OutputUsage {
                    total: get("output_tokens").unwrap_or(0),
                    reasoning: None,
                },
                raw: Some(Value::Object(self.usage.clone())),
            })
        };
        let raw = if self.ctx.include_raw {
            let mut m = self.message.clone().unwrap_or_else(|| json!({}));
            m["content"] = Value::Array(std::mem::take(&mut self.raw_blocks));
            m["stop_reason"] = json!(self.stop_reason);
            for (k, v) in &self.stop_extra {
                m[k] = v.clone();
            }
            m["usage"] = Value::Object(self.usage.clone());
            Some(m)
        } else {
            None
        };
        Ok(Response {
            content,
            stop: Stop {
                reason,
                raw: self.stop_reason.as_ref().map(|s| json!(s)),
            },
            usage,
            warnings: Vec::new(),
            origin: self.ctx.origin(),
            latency_ms: 0.0,
            first_token_ms: None,
            raw,
        })
    }
}
