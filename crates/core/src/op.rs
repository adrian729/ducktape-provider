//! The effects state machine behind every call.
use crate::auth;
use crate::effects::{
    Clock, Effect, HttpRequest, Input, Method, Notification, ObserverHook, OperationKind,
    OperationResult, Purpose, Resume, Start, Step, TransportErrorKind,
};
use crate::retry;
use crate::secret::SecretStore;
use crate::types::*;
use crate::wire::{self, CallContext, ChatDecoder, Wire};
use serde_json::{json, Value};

/// Default connect timeout in seconds.
pub const DEFAULT_CONNECT_SECS: f64 = 10.0;
/// Default read timeout in seconds.
pub const DEFAULT_READ_SECS: f64 = 60.0;
/// Default retries after the first attempt.
pub const DEFAULT_RETRIES: u32 = 2;
/// Most server-tool continuations the core performs per call.
pub const MAX_CONTINUATIONS: u32 = 10;

/// Default endpoint of each vendor.
pub fn default_base_url(ns: &Ns) -> Option<&'static str> {
    match ns {
        Ns::Anthropic => Some("https://api.anthropic.com"),
        Ns::Openai => Some("https://api.openai.com"),
        Ns::Google => Some("https://generativelanguage.googleapis.com"),
        Ns::Ollama => Some("http://127.0.0.1:11434"),
        Ns::Other(_) => None,
    }
}

/// Codec id of each vendor.
pub fn wire_of(ns: &Ns) -> Option<&'static str> {
    match ns {
        Ns::Anthropic => Some("anthropic.messages"),
        Ns::Openai => Some("openai.responses"),
        Ns::Google => Some("google.interactions"),
        Ns::Ollama => Some("ollama.native"),
        Ns::Other(_) => None,
    }
}

/// Resolved endpoint of a provider.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub wire: &'static str,
    pub base_url: String,
}

impl Endpoint {
    /// Picks wire and base URL for a config.
    pub fn resolve(config: &ProviderConfig) -> Result<Self, Box<ProviderError>> {
        let wire = wire_of(&config.ns).ok_or_else(|| {
            Box::new(ProviderError::new(
                ErrorKind::Unsupported,
                format!("unknown provider namespace {:?}", config.ns.as_str()),
            ))
        })?;
        let base_url = match &config.base_url {
            Some(url) => url.trim_end_matches('/').to_owned(),
            None => default_base_url(&config.ns).unwrap_or_default().to_owned(),
        };
        Ok(Endpoint { wire, base_url })
    }
}

/// What the operation is doing.
#[derive(Debug)]
enum Job {
    Chat {
        request: ChatRequest,
        stream: bool,
        continuations: u32,
        content: Vec<Block>,
        usage: Option<Usage>,
        warnings: Vec<Warning>,
        first_token_ms: Option<f64>,
    },
    Embed {
        request: EmbedRequest,
    },
    Models {
        collected: Vec<ModelInfo>,
    },
    ModelInfo {
        model: String,
    },
    Capabilities {
        model: String,
    },
    AuthStatus,
}

/// A credential chosen for this operation.
#[derive(Debug, Clone)]
struct Chosen {
    secret_ref: String,
    kind: AuthKind,
    placement: Placement,
    label: String,
}

#[derive(Debug)]
enum State {
    Auth {
        sources: Vec<AuthSource>,
        index: usize,
    },
    Sending,
    Receiving,
    ErrorBody {
        status: u16,
        headers: Headers,
        body: Vec<u8>,
    },
    Sleeping,
    Finished,
}

/// One in-flight call, driven by the host's effect loop.
pub struct Operation {
    config: ProviderConfig,
    endpoint: Endpoint,
    secrets: SecretStore,
    wire: Option<Box<dyn Wire>>,
    job: Job,
    state: State,
    started_ms: i64,
    now_ms: i64,
    entropy: Vec<u8>,
    chosen: Option<Chosen>,
    ctx: Option<CallContext>,
    pending: Option<HttpRequest>,
    attempt: u32,
    retries: u32,
    decoder: Option<Box<dyn ChatDecoder>>,
    body: Vec<u8>,
    received_bytes: bool,
    response_headers: Headers,
    fetch_planned: bool,
    fetch_queue: Vec<(usize, usize)>,
    fetching: Option<(usize, usize)>,
}

impl std::fmt::Debug for Operation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Operation")
            .field("state", &self.state)
            .field("attempt", &self.attempt)
            .finish()
    }
}

fn step_done(result: OperationResult) -> Step {
    Step::effect(Effect::Done { result })
}

fn redact_headers(headers: &Headers) -> Value {
    let mut m = serde_json::Map::new();
    for (k, v) in headers {
        let secret = matches!(
            k.as_str(),
            "authorization" | "x-api-key" | "x-goog-api-key" | "api-key" | "cookie"
        );
        m.insert(k.clone(), json!(if secret { "***" } else { v.as_str() }));
    }
    Value::Object(m)
}

impl Operation {
    /// Opens an operation and returns its first step.
    pub fn start(start: Start) -> (Self, Step) {
        let job = match start.operation {
            OperationKind::Chat { request } => Job::Chat {
                request,
                stream: false,
                continuations: 0,
                content: Vec::new(),
                usage: None,
                warnings: Vec::new(),
                first_token_ms: None,
            },
            OperationKind::Stream { request } => Job::Chat {
                request,
                stream: true,
                continuations: 0,
                content: Vec::new(),
                usage: None,
                warnings: Vec::new(),
                first_token_ms: None,
            },
            OperationKind::Embed { request } => Job::Embed { request },
            OperationKind::Models { .. } => Job::Models {
                collected: Vec::new(),
            },
            OperationKind::ModelInfo { model } => Job::ModelInfo { model },
            OperationKind::Capabilities { model } => Job::Capabilities { model },
            OperationKind::AuthStatus => Job::AuthStatus,
            OperationKind::Login { .. } | OperationKind::Logout => {
                let endpoint = Endpoint {
                    wire: "",
                    base_url: String::new(),
                };
                let op = Operation::new(start.provider, endpoint, Job::AuthStatus, &start.clock);
                return (
                    op,
                    Step::failed(ProviderError::new(
                        ErrorKind::Unsupported,
                        "login is not implemented yet",
                    )),
                );
            }
        };
        let endpoint = match Endpoint::resolve(&start.provider) {
            Ok(endpoint) => endpoint,
            Err(error) => {
                let endpoint = Endpoint {
                    wire: "",
                    base_url: String::new(),
                };
                let op = Operation::new(start.provider, endpoint, job, &start.clock);
                return (op, Step::failed(*error));
            }
        };
        let mut op = Operation::new(start.provider, endpoint, job, &start.clock);
        op.wire = wire::for_ns(&op.config.ns);
        let step = op.begin();
        (op, step)
    }

    fn new(config: ProviderConfig, endpoint: Endpoint, job: Job, clock: &Clock) -> Self {
        Operation {
            config,
            endpoint,
            secrets: SecretStore::default(),
            wire: None,
            job,
            state: State::Finished,
            started_ms: clock.now_ms,
            now_ms: clock.now_ms,
            entropy: clock.entropy.clone(),
            chosen: None,
            ctx: None,
            pending: None,
            attempt: 0,
            retries: DEFAULT_RETRIES,
            decoder: None,
            body: Vec::new(),
            received_bytes: false,
            response_headers: Headers::new(),
            fetch_planned: false,
            fetch_queue: Vec::new(),
            fetching: None,
        }
    }

    /// Stores a secret the host resolved for this operation.
    pub fn set_secret(&mut self, secret_ref: &str, bytes: Vec<u8>) {
        self.secrets.set(secret_ref, bytes);
    }

    /// Continues after the host performed the pending effect.
    pub fn resume(&mut self, resume: Resume) -> Step {
        self.now_ms = resume.clock.now_ms;
        self.entropy = resume.clock.entropy;
        if let Input::Cancelled = resume.input {
            self.state = State::Finished;
            return Step::failed(self.error(ErrorKind::Cancelled, "cancelled by the host"));
        }
        let state = std::mem::replace(&mut self.state, State::Finished);
        match (state, resume.input) {
            (State::Auth { sources, index }, Input::Secret { secret_ref, found }) => {
                self.auth_answer(sources, index, &secret_ref, found)
            }
            (State::Sending, Input::HttpResponse { status, headers }) => {
                self.response(status, headers)
            }
            (State::Sending, Input::HttpError { error }) => {
                let kind = match error.kind {
                    TransportErrorKind::Timeout => ErrorKind::Timeout,
                    TransportErrorKind::Cancelled => ErrorKind::Cancelled,
                    _ => ErrorKind::Connection,
                };
                let mut e = self.error(kind.clone(), error.message);
                e.retryable = !matches!(kind, ErrorKind::Cancelled) && !self.received_bytes;
                self.fail_or_retry(e)
            }
            (State::Receiving, Input::HttpChunk { bytes }) => self.chunk(&bytes),
            (State::Receiving, Input::HttpEnd) => self.end(),
            (State::Receiving, Input::HttpError { error }) => {
                let kind = if matches!(error.kind, TransportErrorKind::Timeout) {
                    ErrorKind::Timeout
                } else {
                    ErrorKind::Connection
                };
                Step::failed(self.error(
                    kind,
                    format!("connection lost mid-reply: {}", error.message),
                ))
            }
            (
                State::ErrorBody {
                    status,
                    headers,
                    mut body,
                },
                Input::HttpChunk { bytes },
            ) => {
                body.extend_from_slice(&bytes);
                self.state = State::ErrorBody {
                    status,
                    headers,
                    body,
                };
                Step::effect(Effect::Receive)
            }
            (
                State::ErrorBody {
                    status,
                    headers,
                    body,
                },
                Input::HttpEnd | Input::HttpError { .. },
            ) => {
                if self.fetching.take().is_some() {
                    let mut e = self.error(
                        ErrorKind::BadRequest,
                        format!("image url fetch failed with HTTP {status}"),
                    );
                    e.status = Some(status);
                    e.code = Some("image_fetch".to_owned());
                    return Step::failed(e);
                }
                if let (Job::Chat { .. }, Some(w), Some(ctx)) = (&self.job, &self.wire, &self.ctx) {
                    if let Some(response) = w.error_as_response(ctx, status, &body) {
                        return self.finish_chat(response);
                    }
                }
                let mut e = match &self.wire {
                    Some(w) => w.classify(status, &headers, &body),
                    None => ProviderError::new(ErrorKind::Server, format!("HTTP {status}")),
                };
                e.attempts = self.attempt;
                e.origin = self.ctx.as_ref().map(CallContext::origin);
                self.fail_or_retry(e)
            }
            (State::Sleeping, Input::Slept) => self.send(),
            (State::Finished, _) => {
                Step::failed(self.error(ErrorKind::Unsupported, "operation already finished"))
            }
            (state, input) => {
                let msg = format!(
                    "unexpected input {} in state {:?}",
                    input_name(&input),
                    state
                );
                Step::failed(self.error(ErrorKind::BadRequest, msg))
            }
        }
    }

    /// Delivers raw body bytes of the open response.
    pub fn feed(&mut self, bytes: &[u8], clock: Clock) -> Step {
        self.resume(Resume {
            clock,
            input: Input::HttpChunk {
                bytes: bytes.to_vec(),
            },
        })
    }

    /// The provider this operation runs against.
    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// The resolved endpoint.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    fn elapsed_ms(&self) -> f64 {
        (self.now_ms - self.started_ms) as f64
    }

    fn error(&self, kind: ErrorKind, message: impl Into<String>) -> ProviderError {
        let mut error = ProviderError::new(kind, message);
        error.attempts = self.attempt;
        error.origin = Some(match &self.ctx {
            Some(ctx) => ctx.origin(),
            None => Origin {
                wire: self.endpoint.wire.to_owned(),
                ns: self.config.ns.clone(),
                base_url: self.endpoint.base_url.clone(),
                model: self.model_hint(),
                auth_source: None,
            },
        });
        error
    }

    fn model_hint(&self) -> String {
        match &self.job {
            Job::Chat { request, .. } => request.model.clone(),
            Job::Embed { request } => request.model.clone(),
            Job::ModelInfo { model } | Job::Capabilities { model } => model.clone(),
            _ => self.config.default_model.clone().unwrap_or_default(),
        }
    }

    fn begin(&mut self) -> Step {
        if let Job::AuthStatus = self.job {
            let sources = auth::sources(
                &self.config.ns,
                &self.endpoint.base_url,
                self.config.auth.as_ref(),
            );
            if sources.is_empty() {
                return step_done(OperationResult::AuthStatus {
                    status: AuthStatus {
                        source: None,
                        expires_at: None,
                        session: None,
                    },
                });
            }
            return self.auth_next(sources, 0);
        }
        if self.wire.is_none() {
            return Step::failed(self.error(
                ErrorKind::Unsupported,
                format!("{} is not implemented yet", self.endpoint.wire),
            ));
        }
        if let Err(e) = self.prepare() {
            return Step::failed(e);
        }
        if let Job::Capabilities { model } = &self.job {
            let model = model.clone();
            if let Some(caps) = self.local_capabilities(&model) {
                return step_done(OperationResult::Capabilities {
                    capabilities: Some(caps),
                });
            }
            if self
                .wire
                .as_ref()
                .and_then(|w| w.build_model_info(self.ctx.as_ref().unwrap(), &model))
                .is_none()
            {
                return step_done(OperationResult::Capabilities { capabilities: None });
            }
        }
        let sources = auth::sources(
            &self.config.ns,
            &self.endpoint.base_url,
            self.config.auth.as_ref(),
        );
        if sources.is_empty() {
            if matches!(self.config.ns, Ns::Ollama) {
                return self.first_request();
            }
            let mut e = self.error(
                ErrorKind::Auth,
                "no credential: pass `auth` or set the vendor's environment variable",
            );
            e.code = Some("no_credential".to_owned());
            return Step::failed(e);
        }
        self.auth_next(sources, 0)
    }

    fn prepare(&mut self) -> Result<(), ProviderError> {
        let model = match &mut self.job {
            Job::Chat { request, .. } => {
                if request.model.is_empty() {
                    request.model = self.config.default_model.clone().unwrap_or_default();
                }
                merge_chat_defaults(request, &self.config);
                request.model.clone()
            }
            Job::Embed { request } => {
                if request.model.is_empty() {
                    request.model = self
                        .config
                        .default_embedding_model
                        .clone()
                        .unwrap_or_default();
                }
                if request.timeout.is_none() {
                    request.timeout = self.config.timeout;
                }
                request.model.clone()
            }
            Job::ModelInfo { model } | Job::Capabilities { model } => model.clone(),
            _ => String::new(),
        };
        if model.is_empty()
            && matches!(
                self.job,
                Job::Chat { .. }
                    | Job::Embed { .. }
                    | Job::ModelInfo { .. }
                    | Job::Capabilities { .. }
            )
        {
            return Err(wire::argument_error(
                "no model: pass `model` or set a default model on the provider",
            ));
        }
        let (strict, include_raw, retries) = match &self.job {
            Job::Chat { request, .. } => (
                request.strict.or(self.config.strict).unwrap_or(false),
                request.include_raw.unwrap_or(false),
                request
                    .options
                    .as_ref()
                    .and_then(|o| o.retries)
                    .unwrap_or(DEFAULT_RETRIES),
            ),
            Job::Embed { request } => (
                false,
                request.include_raw.unwrap_or(false),
                request
                    .options
                    .as_ref()
                    .and_then(|o| o.retries)
                    .unwrap_or(DEFAULT_RETRIES),
            ),
            _ => (false, false, DEFAULT_RETRIES),
        };
        self.retries = retries;
        self.ctx = Some(CallContext {
            ns: self.config.ns.clone(),
            wire: self.endpoint.wire,
            base_url: self.endpoint.base_url.clone(),
            model,
            auth_source: None,
            api_version: self
                .config
                .overrides
                .as_ref()
                .and_then(|o| o.api_version.clone()),
            headers: self.config.headers.clone().unwrap_or_default(),
            timeout: Some(with_timeout_defaults(self.config.timeout)),
            strict,
            include_raw,
            overrides: self.config.overrides.clone(),
        });
        Ok(())
    }

    fn local_capabilities(&self, model: &str) -> Option<Capabilities> {
        let shipped = self
            .wire
            .as_ref()
            .and_then(|w| w.shipped_capabilities(model));
        let user = self
            .config
            .overrides
            .as_ref()
            .and_then(|o| o.capabilities.clone());
        match (shipped, user) {
            (Some(s), Some(u)) => Some(overlay_capabilities(s, u)),
            (Some(s), None) => Some(s),
            (None, Some(u)) => Some(Capabilities {
                source: CapabilitySource::User,
                ..u
            }),
            (None, None) => None,
        }
    }

    fn auth_next(&mut self, sources: Vec<AuthSource>, index: usize) -> Step {
        let Some(source) = sources.get(index).cloned() else {
            if let Job::AuthStatus = self.job {
                return step_done(OperationResult::AuthStatus {
                    status: AuthStatus {
                        source: None,
                        expires_at: None,
                        session: None,
                    },
                });
            }
            if self.wire.as_ref().is_some_and(|w| w.auth_optional()) {
                return self.first_request();
            }
            let tried: Vec<String> = sources.iter().map(auth::label).collect();
            let mut e = self.error(
                ErrorKind::Auth,
                format!("no credential found; tried {}", tried.join(", ")),
            );
            e.code = Some("no_credential".to_owned());
            return Step::failed(e);
        };
        match &source {
            AuthSource::Inline { secret_ref, .. } if self.secrets.contains(secret_ref) => {
                let secret_ref = secret_ref.clone();
                self.auth_answer(sources, index, &secret_ref, true)
            }
            AuthSource::Session { .. } | AuthSource::Exchange { .. } => {
                let mut e = self.error(
                    ErrorKind::Unsupported,
                    format!(
                        "auth source {} is not implemented yet",
                        auth::label(&source)
                    ),
                );
                e.code = Some("auth_source".to_owned());
                Step::failed(e)
            }
            _ => {
                let secret_ref = match &source {
                    AuthSource::Inline { secret_ref, .. } => secret_ref.clone(),
                    _ => format!("auth:{index}"),
                };
                self.state = State::Auth { sources, index };
                Step::effect(Effect::NeedSecret { secret_ref, source })
            }
        }
    }

    fn auth_answer(
        &mut self,
        sources: Vec<AuthSource>,
        index: usize,
        secret_ref: &str,
        found: bool,
    ) -> Step {
        if !found || !self.secrets.contains(secret_ref) {
            return self.auth_next(sources, index + 1);
        }
        let source = &sources[index];
        let transport = auth::transport_of(source);
        let kind = transport
            .and_then(|t| t.kind.clone())
            .unwrap_or_else(|| auth::default_kind(&self.config.ns, source));
        let placement = transport
            .and_then(|t| t.placement.clone())
            .unwrap_or_else(|| auth::default_placement(&self.config.ns, &kind));
        let label = auth::label(source);
        if let Job::AuthStatus = self.job {
            return step_done(OperationResult::AuthStatus {
                status: AuthStatus {
                    source: Some(label),
                    expires_at: None,
                    session: None,
                },
            });
        }
        self.chosen = Some(Chosen {
            secret_ref: secret_ref.to_owned(),
            kind,
            placement,
            label: label.clone(),
        });
        if let Some(ctx) = &mut self.ctx {
            ctx.auth_source = Some(label);
        }
        self.first_request()
    }

    fn first_request(&mut self) -> Step {
        let ctx = self.ctx.clone().expect("prepared");
        if !self.fetch_planned {
            self.fetch_planned = true;
            if self.wire.as_ref().is_some_and(|w| w.inline_images()) {
                if let Job::Chat { request, .. } = &self.job {
                    self.fetch_queue = plan_fetches(&request.messages);
                }
            }
        }
        if !self.fetch_queue.is_empty() {
            return self.next_fetch();
        }
        let wire = self.wire.as_ref().expect("wire");
        let built = match &self.job {
            Job::Chat {
                request, content, ..
            } => wire.build_chat(&ctx, request, content.len() as u32),
            Job::Embed { request } => wire.build_embed(&ctx, request),
            Job::Models { .. } => match wire.build_models(&ctx) {
                Some(r) => Ok(wire::Built {
                    request: r,
                    warnings: Vec::new(),
                }),
                None => return step_done(OperationResult::Models { models: Vec::new() }),
            },
            Job::ModelInfo { model } | Job::Capabilities { model } => {
                match wire.build_model_info(&ctx, model) {
                    Some(r) => Ok(wire::Built {
                        request: r,
                        warnings: Vec::new(),
                    }),
                    None => {
                        let mut e = self.error(
                            ErrorKind::Unsupported,
                            format!("{} has no model endpoint", ctx.wire),
                        );
                        e.code = Some("model_info".to_owned());
                        return Step::failed(e);
                    }
                }
            }
            Job::AuthStatus => unreachable!(),
        };
        let built = match built {
            Ok(b) => b,
            Err(mut e) => {
                e.origin = Some(ctx.origin());
                return Step::failed(e);
            }
        };
        if let Job::Chat { warnings, .. } = &mut self.job {
            warnings.extend(built.warnings);
        }
        let mut request = built.request;
        if let Some(chosen) = &self.chosen {
            if let Some(secret) = self.secrets.get(&chosen.secret_ref) {
                let cred = auth::Credential {
                    kind: chosen.kind.clone(),
                    placement: chosen.placement.clone(),
                    secret,
                    label: chosen.label.clone(),
                };
                auth::place(&mut request.headers, &cred);
            }
        }
        request.timeout = Some(with_timeout_defaults(request.timeout.or(ctx.timeout)));
        self.pending = Some(request);
        self.attempt = 0;
        self.send()
    }

    fn next_fetch(&mut self) -> Step {
        let target = self.fetch_queue.remove(0);
        let Job::Chat { request, .. } = &self.job else {
            unreachable!()
        };
        let url = image_at(&request.messages, target)
            .and_then(|b| b.url.clone())
            .unwrap_or_default();
        self.fetching = Some(target);
        self.pending = Some(HttpRequest {
            method: Method::Get,
            url,
            headers: Headers::new(),
            body: None,
            stream: false,
            timeout: self.ctx.as_ref().and_then(|c| c.timeout),
            purpose: Purpose::Fetch,
        });
        self.attempt = 0;
        self.send()
    }

    fn fetched(&mut self, target: (usize, usize)) -> Step {
        use base64::Engine;
        let media_type = self
            .response_headers
            .get("content-type")
            .map(|v| {
                v.split(';')
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_ascii_lowercase()
            })
            .filter(|v| v.starts_with("image/"));
        let data = base64::engine::general_purpose::STANDARD.encode(&self.body);
        let Job::Chat { request, .. } = &mut self.job else {
            unreachable!()
        };
        if let Some(block) = image_at_mut(&mut request.messages, target) {
            block.media_type = media_type.or_else(|| wire::sniff_image(&data).map(str::to_owned));
            block.data = Some(data);
            block.url = None;
        }
        self.first_request()
    }

    fn send(&mut self) -> Step {
        let request = self.pending.clone().expect("pending request");
        self.attempt += 1;
        self.received_bytes = false;
        self.body.clear();
        self.state = State::Sending;
        let data = json!({
            "attempt": self.attempt,
            "method": request.method,
            "url": request.url,
            "headers": redact_headers(&request.headers),
            "body_bytes": request.body.as_ref().map(|b| b.len()).unwrap_or(0),
        });
        Step {
            notify: vec![Notification::Observer {
                hook: ObserverHook::OnRequest,
                data,
            }],
            effect: Effect::NeedHttp { request },
        }
    }

    fn response(&mut self, status: u16, headers: Headers) -> Step {
        self.received_bytes = true;
        let notify = vec![Notification::Observer {
            hook: ObserverHook::OnResponse,
            data: json!({"attempt": self.attempt, "status": status, "elapsed_ms": self.elapsed_ms(), "request_id": headers.get("request-id").or(headers.get("x-request-id"))}),
        }];
        if !(200..300).contains(&status) {
            self.state = State::ErrorBody {
                status,
                headers,
                body: Vec::new(),
            };
            return Step {
                notify,
                effect: Effect::Receive,
            };
        }
        self.response_headers = headers;
        if self.fetching.is_some() {
            self.state = State::Receiving;
            return Step {
                notify,
                effect: Effect::Receive,
            };
        }
        if let Job::Chat { content, .. } = &self.job {
            let ctx = self.ctx.clone().expect("ctx");
            self.decoder = Some(
                self.wire
                    .as_ref()
                    .expect("wire")
                    .chat_decoder(&ctx, content.len() as u32),
            );
        }
        self.state = State::Receiving;
        Step {
            notify,
            effect: Effect::Receive,
        }
    }

    fn chunk(&mut self, bytes: &[u8]) -> Step {
        let Some(decoder) = self.decoder.as_mut() else {
            self.body.extend_from_slice(bytes);
            self.state = State::Receiving;
            return Step::effect(Effect::Receive);
        };
        match decoder.feed(bytes) {
            Ok(decoded) => {
                let elapsed = self.elapsed_ms();
                let Job::Chat {
                    stream,
                    first_token_ms,
                    ..
                } = &mut self.job
                else {
                    unreachable!()
                };
                if first_token_ms.is_none()
                    && decoded.events.iter().any(|e| {
                        matches!(
                            e,
                            Event::TextDelta { .. }
                                | Event::ReasoningDelta { .. }
                                | Event::ToolCallDelta { .. }
                        )
                    })
                {
                    *first_token_ms = Some(elapsed);
                }
                let notify = if *stream {
                    decoded
                        .events
                        .into_iter()
                        .map(|event| Notification::Event {
                            event: Box::new(event),
                        })
                        .collect()
                } else {
                    Vec::new()
                };
                self.state = State::Receiving;
                Step {
                    notify,
                    effect: Effect::Receive,
                }
            }
            Err(mut e) => {
                e.attempts = self.attempt;
                e.origin = self.ctx.as_ref().map(CallContext::origin);
                Step::failed(e)
            }
        }
    }

    fn end(&mut self) -> Step {
        if let Some(target) = self.fetching.take() {
            return self.fetched(target);
        }
        match &mut self.job {
            Job::Chat { .. } => self.end_chat(),
            Job::Embed { .. } => {
                let ctx = self.ctx.clone().expect("ctx");
                match self
                    .wire
                    .as_ref()
                    .expect("wire")
                    .decode_embed(&ctx, &self.body)
                {
                    Ok(mut r) => {
                        r.latency_ms = self.elapsed_ms();
                        step_done(OperationResult::Embed { response: r })
                    }
                    Err(e) => Step::failed(self.with_origin(e)),
                }
            }
            Job::Models { .. } => {
                let ctx = self.ctx.clone().expect("ctx");
                match self
                    .wire
                    .as_ref()
                    .expect("wire")
                    .decode_models(&ctx, &self.body)
                {
                    Ok((page, next)) => {
                        let Job::Models { collected } = &mut self.job else {
                            unreachable!()
                        };
                        collected.extend(page);
                        match next {
                            Some(mut r) => {
                                if let Some(chosen) = &self.chosen {
                                    if let Some(secret) = self.secrets.get(&chosen.secret_ref) {
                                        auth::place(
                                            &mut r.headers,
                                            &auth::Credential {
                                                kind: chosen.kind.clone(),
                                                placement: chosen.placement.clone(),
                                                secret,
                                                label: chosen.label.clone(),
                                            },
                                        );
                                    }
                                }
                                self.pending = Some(r);
                                self.attempt = 0;
                                self.send()
                            }
                            None => step_done(OperationResult::Models {
                                models: std::mem::take(collected),
                            }),
                        }
                    }
                    Err(e) => Step::failed(self.with_origin(e)),
                }
            }
            Job::ModelInfo { .. } => {
                let ctx = self.ctx.clone().expect("ctx");
                match self
                    .wire
                    .as_ref()
                    .expect("wire")
                    .decode_model_info(&ctx, &self.body)
                {
                    Ok(info) => step_done(OperationResult::ModelInfo { info: Some(info) }),
                    Err(e) => Step::failed(self.with_origin(e)),
                }
            }
            Job::Capabilities { model } => {
                let model = model.clone();
                let ctx = self.ctx.clone().expect("ctx");
                match self
                    .wire
                    .as_ref()
                    .expect("wire")
                    .decode_model_info(&ctx, &self.body)
                {
                    Ok(info) => {
                        let mut vendor = info.capabilities.clone().unwrap_or(Capabilities {
                            reasoning: None,
                            tool_choice: None,
                            parallel_tools: None,
                            output: None,
                            cache: None,
                            inputs: None,
                            tools: None,
                            limits: None,
                            embeddings: None,
                            source: CapabilitySource::Vendor,
                        });
                        vendor.source = CapabilitySource::Vendor;
                        if vendor.limits.is_none()
                            && (info.context_window.is_some() || info.max_output_tokens.is_some())
                        {
                            vendor.limits = Some(Limits {
                                context_window: info.context_window,
                                max_output_tokens: info.max_output_tokens,
                            });
                        }
                        let caps = match self
                            .config
                            .overrides
                            .as_ref()
                            .and_then(|o| o.capabilities.clone())
                        {
                            Some(u) => overlay_capabilities(vendor, u),
                            None => vendor,
                        };
                        let _ = model;
                        step_done(OperationResult::Capabilities {
                            capabilities: Some(caps),
                        })
                    }
                    Err(e) => Step::failed(self.with_origin(e)),
                }
            }
            Job::AuthStatus => unreachable!(),
        }
    }

    fn end_chat(&mut self) -> Step {
        let Some(mut decoder) = self.decoder.take() else {
            return Step::failed(self.error(ErrorKind::MalformedResponse, "no decoder"));
        };
        let response = match decoder.finish() {
            Ok(r) => r,
            Err(e) => return Step::failed(self.with_origin(e)),
        };
        self.finish_chat(response)
    }

    fn finish_chat(&mut self, mut response: Response) -> Step {
        let elapsed = self.elapsed_ms();
        let wire = self.wire.as_ref().expect("wire");
        let continue_turn = wire.wants_continuation(&response);
        let Job::Chat {
            request,
            stream,
            continuations,
            content,
            usage,
            warnings,
            first_token_ms,
        } = &mut self.job
        else {
            unreachable!()
        };
        content.append(&mut response.content);
        *usage = merge_usage(usage.take(), response.usage.take());
        if continue_turn && *continuations < MAX_CONTINUATIONS {
            *continuations += 1;
            request.messages.push(Message {
                role: Role::Assistant,
                content: content.clone(),
            });
            return self.first_request();
        }
        if continue_turn {
            warnings.push(Warning {
                kind: WarningKind::Compatibility,
                feature: "continue".into(),
                detail: Some(format!("stopped after {MAX_CONTINUATIONS} continuations")),
            });
        }
        response.content = std::mem::take(content);
        response.stop.reason = wire::derive_stop(&response.content, response.stop.reason.clone());
        response.usage = usage.take();
        response.warnings = std::mem::take(warnings);
        response.latency_ms = elapsed;
        response.first_token_ms = *first_token_ms;
        if *stream {
            Step {
                notify: vec![Notification::Event {
                    event: Box::new(Event::Finish {
                        parent_id: None,
                        response,
                    }),
                }],
                effect: Effect::Done {
                    result: OperationResult::StreamEnd,
                },
            }
        } else {
            step_done(OperationResult::Response { response })
        }
    }

    fn with_origin(&self, mut e: ProviderError) -> ProviderError {
        e.attempts = self.attempt;
        if e.origin.is_none() {
            e.origin = self.ctx.as_ref().map(CallContext::origin);
        }
        e
    }

    fn fail_or_retry(&mut self, error: ProviderError) -> Step {
        let total_ms = self
            .pending
            .as_ref()
            .and_then(|r| r.timeout)
            .and_then(|t| t.total)
            .map(|s| s * 1000.0);
        let wait = retry::backoff_ms(
            self.attempt,
            error.retry_after,
            self.entropy.first().copied().unwrap_or(128),
        );
        let within_total = total_ms.is_none_or(|t| self.elapsed_ms() + wait as f64 <= t);
        if retry::is_retryable(&error) && self.attempt <= self.retries && within_total {
            self.state = State::Sleeping;
            let data = json!({"attempt": self.attempt, "kind": error.kind, "status": error.status, "sleep_ms": wait});
            return Step {
                notify: vec![Notification::Observer {
                    hook: ObserverHook::OnRetry,
                    data,
                }],
                effect: Effect::NeedSleep { ms: wait },
            };
        }
        Step::failed(error)
    }
}

fn input_name(input: &Input) -> &'static str {
    match input {
        Input::HttpResponse { .. } => "http_response",
        Input::HttpChunk { .. } => "http_chunk",
        Input::HttpEnd => "http_end",
        Input::HttpError { .. } => "http_error",
        Input::Slept => "slept",
        Input::Secret { .. } => "secret",
        Input::Read { .. } => "read",
        Input::Written { .. } => "written",
        Input::UserAction { .. } => "user_action",
        Input::Callback { .. } => "callback",
        Input::Cancelled => "cancelled",
    }
}

fn with_timeout_defaults(t: Option<Timeout>) -> Timeout {
    let t = t.unwrap_or_default();
    Timeout {
        connect: t.connect.or(Some(DEFAULT_CONNECT_SECS)),
        read: t.read.or(Some(DEFAULT_READ_SECS)),
        total: t.total,
    }
}

fn merge_usage(a: Option<Usage>, b: Option<Usage>) -> Option<Usage> {
    match (a, b) {
        (None, b) => b,
        (a, None) => a,
        (Some(a), Some(b)) => Some(Usage {
            input: InputUsage {
                total: a.input.total + b.input.total,
                cache_read: add_opt(a.input.cache_read, b.input.cache_read),
                cache_write: add_opt(a.input.cache_write, b.input.cache_write),
            },
            output: OutputUsage {
                total: a.output.total + b.output.total,
                reasoning: add_opt(a.output.reasoning, b.output.reasoning),
            },
            raw: b.raw.or(a.raw),
        }),
    }
}

fn add_opt(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (None, None) => None,
        (a, b) => Some(a.unwrap_or(0) + b.unwrap_or(0)),
    }
}

/// Merges the provider's defaults under a call's request.
pub fn merge_chat_defaults(req: &mut ChatRequest, config: &ProviderConfig) {
    if req.system.is_none() {
        req.system = config.default_system.clone();
    }
    if req.tools.is_none() {
        req.tools = config.default_tools.clone();
    }
    if let Some(defaults) = &config.default_options {
        let o = req.options.get_or_insert_with(Options::default);
        o.max_output_tokens = o.max_output_tokens.or(defaults.max_output_tokens);
        o.reasoning = o.reasoning.clone().or_else(|| defaults.reasoning.clone());
        o.reasoning_summary = o
            .reasoning_summary
            .clone()
            .or_else(|| defaults.reasoning_summary.clone());
        o.tool_choice = o
            .tool_choice
            .clone()
            .or_else(|| defaults.tool_choice.clone());
        o.parallel_tools = o.parallel_tools.or(defaults.parallel_tools);
        o.output = o.output.clone().or_else(|| defaults.output.clone());
        o.stop_sequences = o
            .stop_sequences
            .clone()
            .or_else(|| defaults.stop_sequences.clone());
        o.cache = o.cache.clone().or_else(|| defaults.cache.clone());
        o.retries = o.retries.or(defaults.retries);
        o.user_id = o.user_id.clone().or_else(|| defaults.user_id.clone());
    }
    if let Some(defaults) = &config.default_vendor {
        let v = req.vendor.get_or_insert_with(VendorMap::new);
        for (ns, fields) in defaults {
            match (v.get_mut(ns), fields) {
                (Some(Value::Object(existing)), Value::Object(incoming)) => {
                    for (k, val) in incoming {
                        existing.entry(k.clone()).or_insert_with(|| val.clone());
                    }
                }
                (Some(_), _) => {}
                (None, _) => {
                    v.insert(ns.clone(), fields.clone());
                }
            }
        }
    }
    if req.timeout.is_none() {
        req.timeout = config.timeout;
    }
    if req.strict.is_none() {
        req.strict = config.strict;
    }
}

/// Lays user-declared capabilities over a base.
pub fn overlay_capabilities(base: Capabilities, user: Capabilities) -> Capabilities {
    Capabilities {
        reasoning: user.reasoning.or(base.reasoning),
        tool_choice: user.tool_choice.or(base.tool_choice),
        parallel_tools: user.parallel_tools.or(base.parallel_tools),
        output: user.output.or(base.output),
        cache: user.cache.or(base.cache),
        inputs: user.inputs.or(base.inputs),
        tools: user.tools.or(base.tools),
        limits: user.limits.or(base.limits),
        embeddings: user.embeddings.or(base.embeddings),
        source: CapabilitySource::User,
    }
}

fn plan_fetches(messages: &[Message]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (mi, m) in messages.iter().enumerate() {
        for (bi, b) in m.content.iter().enumerate() {
            if let Block::Image(img) = b {
                if img.data.is_none() && img.url.is_some() {
                    out.push((mi, bi));
                }
            }
        }
    }
    out
}

fn image_at(messages: &[Message], (mi, bi): (usize, usize)) -> Option<&ImageBlock> {
    match messages.get(mi)?.content.get(bi)? {
        Block::Image(img) => Some(img),
        _ => None,
    }
}

fn image_at_mut(messages: &mut [Message], (mi, bi): (usize, usize)) -> Option<&mut ImageBlock> {
    match messages.get_mut(mi)?.content.get_mut(bi)? {
        Block::Image(img) => Some(img),
        _ => None,
    }
}
