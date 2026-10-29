//! The effects state machine behind every call.
use crate::effects::{Clock, Effect, Input, OperationKind, OperationResult, Resume, Start, Step};
use crate::secret::SecretStore;
use crate::types::{AuthStatus, ErrorKind, Ns, Origin, ProviderConfig, ProviderError};

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

    /// Origin stamp for a model served here.
    pub fn origin(&self, ns: &Ns, model: &str) -> Origin {
        Origin {
            wire: self.wire.to_owned(),
            ns: ns.clone(),
            base_url: self.base_url.clone(),
            model: model.to_owned(),
            auth_source: None,
        }
    }
}

#[derive(Debug)]
enum State {
    Finished,
}

/// One in-flight call, driven by the host's effect loop.
#[derive(Debug)]
pub struct Operation {
    config: ProviderConfig,
    endpoint: Endpoint,
    secrets: SecretStore,
    state: State,
    started_ms: i64,
}

impl Operation {
    /// Opens an operation and returns its first step.
    pub fn start(start: Start) -> (Self, Step) {
        let endpoint = match Endpoint::resolve(&start.provider) {
            Ok(endpoint) => endpoint,
            Err(error) => {
                let endpoint = Endpoint {
                    wire: "",
                    base_url: String::new(),
                };
                let op = Operation {
                    config: start.provider,
                    endpoint,
                    secrets: SecretStore::default(),
                    state: State::Finished,
                    started_ms: start.clock.now_ms,
                };
                return (op, Step::failed(*error));
            }
        };
        let mut op = Operation {
            config: start.provider,
            endpoint,
            secrets: SecretStore::default(),
            state: State::Finished,
            started_ms: start.clock.now_ms,
        };
        let step = op.begin(start.operation, &start.clock);
        (op, step)
    }

    /// Stores a secret the host resolved for this operation.
    pub fn set_secret(&mut self, secret_ref: &str, bytes: Vec<u8>) {
        self.secrets.set(secret_ref, bytes);
    }

    /// Continues after the host performed the pending effect.
    pub fn resume(&mut self, resume: Resume) -> Step {
        if let Input::Cancelled = resume.input {
            self.state = State::Finished;
            return Step::failed(self.error(ErrorKind::Cancelled, "cancelled by the host"));
        }
        match self.state {
            State::Finished => {
                Step::failed(self.error(ErrorKind::Unsupported, "operation already finished"))
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

    /// Milliseconds since the operation started.
    pub fn elapsed_ms(&self, clock: &Clock) -> f64 {
        (clock.now_ms - self.started_ms) as f64
    }

    /// The provider this operation runs against.
    pub fn config(&self) -> &ProviderConfig {
        &self.config
    }

    /// The resolved endpoint.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    fn begin(&mut self, kind: OperationKind, _clock: &Clock) -> Step {
        self.state = State::Finished;
        match kind {
            OperationKind::AuthStatus => Step::effect(Effect::Done {
                result: OperationResult::AuthStatus {
                    status: AuthStatus {
                        source: None,
                        expires_at: None,
                        session: None,
                    },
                },
            }),
            other => Step::failed(self.error(
                ErrorKind::Unsupported,
                format!(
                    "{} is not implemented yet for {}",
                    kind_name(&other),
                    self.endpoint.wire
                ),
            )),
        }
    }

    fn error(&self, kind: ErrorKind, message: impl Into<String>) -> ProviderError {
        let mut error = ProviderError::new(kind, message);
        error.origin = Some(self.endpoint.origin(
            &self.config.ns,
            self.config.default_model.as_deref().unwrap_or(""),
        ));
        error
    }
}

fn kind_name(kind: &OperationKind) -> &'static str {
    match kind {
        OperationKind::Chat { .. } => "chat",
        OperationKind::Stream { .. } => "stream",
        OperationKind::Embed { .. } => "embed",
        OperationKind::Models { .. } => "models",
        OperationKind::ModelInfo { .. } => "model_info",
        OperationKind::Capabilities { .. } => "capabilities",
        OperationKind::Login { .. } => "login",
        OperationKind::Logout => "logout",
        OperationKind::AuthStatus => "auth_status",
    }
}
