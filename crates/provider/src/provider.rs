use crate::credential::{Credential, Resolver};
pub use crate::driver::Observer;
use crate::driver::{clock, Driver, Host, LoginHandler};
use crate::error::Error;
use crate::keychain;
use crate::stream::{CancelToken, ChatStream};
use crate::sugar::system_blocks;
use crate::transport::{check_headers, BoxFuture, HttpTransport};
pub use ducktape_core::effects::Callback;
use ducktape_core::effects::{OperationKind, OperationResult, Start};
use ducktape_core::types::{
    Auth, AuthSource, AuthStatus, Capabilities, ChatRequest, EmbedOptions, EmbedRequest,
    EmbedResponse, ErrorKind, Headers, LoginAction, Message, ModelInfo, Ns, Options, Overrides,
    ProviderConfig, Response, SystemBlock, Timeout, Tool, VendorMap,
};
use std::collections::HashMap;
use std::future::IntoFuture;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::mpsc::unbounded_channel;

struct Inner {
    config: ProviderConfig,
    host: Arc<Host>,
    models: Mutex<Option<Vec<ModelInfo>>>,
    infos: Mutex<HashMap<String, ModelInfo>>,
    caps: Mutex<HashMap<String, Capabilities>>,
}

/// One endpoint of one vendor; cheap to clone.
#[derive(Clone)]
pub struct Provider {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Provider")
            .field("ns", &self.inner.config.ns)
            .field("base_url", &self.inner.config.base_url)
            .finish_non_exhaustive()
    }
}

/// Configures a Provider.
#[derive(Default)]
pub struct Builder {
    ns: Option<Ns>,
    base_url: Option<String>,
    auth: Vec<Credential>,
    headers: Option<Headers>,
    timeout: Option<Timeout>,
    default_model: Option<String>,
    default_embedding_model: Option<String>,
    default_system: Option<Vec<SystemBlock>>,
    default_tools: Option<Vec<Tool>>,
    default_options: Option<Options>,
    default_vendor: Option<VendorMap>,
    overrides: Option<Overrides>,
    strict: Option<bool>,
    observer: Option<Arc<dyn Observer>>,
    transport: Option<Arc<dyn HttpTransport>>,
    keychain: Option<keychain::Reader>,
    keychain_write: Option<keychain::Writer>,
}

impl Builder {
    pub fn base_url(mut self, v: impl Into<String>) -> Self {
        self.base_url = Some(v.into());
        self
    }

    /// The credentials to try in order.
    pub fn auth(mut self, v: impl IntoIterator<Item = Credential>) -> Self {
        self.auth = v.into_iter().collect();
        self
    }

    pub fn headers(mut self, v: Headers) -> Self {
        self.headers = Some(v);
        self
    }

    pub fn timeout(mut self, v: Timeout) -> Self {
        self.timeout = Some(v);
        self
    }

    pub fn default_model(mut self, v: impl Into<String>) -> Self {
        self.default_model = Some(v.into());
        self
    }

    pub fn default_embedding_model(mut self, v: impl Into<String>) -> Self {
        self.default_embedding_model = Some(v.into());
        self
    }

    pub fn default_system(mut self, text: impl Into<String>) -> Self {
        self.default_system = Some(system_blocks(text));
        self
    }

    pub fn default_system_blocks(mut self, v: Vec<SystemBlock>) -> Self {
        self.default_system = Some(v);
        self
    }

    pub fn default_tools(mut self, v: Vec<Tool>) -> Self {
        self.default_tools = Some(v);
        self
    }

    pub fn default_options(mut self, v: Options) -> Self {
        self.default_options = Some(v);
        self
    }

    pub fn default_vendor(mut self, v: VendorMap) -> Self {
        self.default_vendor = Some(v);
        self
    }

    pub fn overrides(mut self, v: Overrides) -> Self {
        self.overrides = Some(v);
        self
    }

    pub fn strict(mut self, v: bool) -> Self {
        self.strict = Some(v);
        self
    }

    pub fn observer(mut self, v: impl Observer + 'static) -> Self {
        self.observer = Some(Arc::new(v));
        self
    }

    /// The transport every request goes through.
    pub fn transport(mut self, v: impl HttpTransport + 'static) -> Self {
        self.transport = Some(Arc::new(v));
        self
    }

    /// A reqwest client to send through.
    #[cfg(feature = "reqwest")]
    pub fn client(mut self, v: reqwest::Client) -> Self {
        self.transport = Some(Arc::new(crate::transport::ReqwestTransport::new(v)));
        self
    }

    /// Reads another tool's keychain entry by service and account.
    pub fn keychain<F>(mut self, v: F) -> Self
    where
        F: Fn(&str, &str) -> Option<Vec<u8>> + Send + Sync + 'static,
    {
        self.keychain = Some(Arc::new(v));
        self
    }

    /// Stores a refreshed token under a service and account.
    pub fn keychain_write<F>(mut self, v: F) -> Self
    where
        F: Fn(&str, &str, &[u8]) -> bool + Send + Sync + 'static,
    {
        self.keychain_write = Some(Arc::new(v));
        self
    }

    fn default_transport(&self) -> std::result::Result<Arc<dyn HttpTransport>, Error> {
        #[cfg(feature = "reqwest")]
        {
            Ok(Arc::new(crate::transport::ReqwestTransport::default()))
        }
        #[cfg(not(feature = "reqwest"))]
        {
            Err(Error::argument(
                "no transport: enable the reqwest feature or set one with transport()",
            ))
        }
    }

    pub fn build(self) -> std::result::Result<Provider, Error> {
        let ns = self
            .ns
            .clone()
            .ok_or_else(|| Error::argument("no namespace"))?;
        if let Ns::Other(other) = &ns {
            return Err(Error::argument(format!(
                "unknown namespace {other}; one of anthropic, openai, google, ollama"
            )));
        }
        if let Some(headers) = &self.headers {
            check_headers(headers)?;
        }
        let mut secrets = HashMap::new();
        let mut resolvers: HashMap<String, Arc<dyn Resolver>> = HashMap::new();
        let mut sources = Vec::new();
        for (i, credential) in self.auth.iter().enumerate() {
            if let Some(misuse) = &credential.misuse {
                return Err(Error::argument(misuse.clone()));
            }
            let mut source = credential.source.clone();
            match &mut source {
                AuthSource::Inline { secret_ref, .. } => {
                    *secret_ref = format!("inline:{i}");
                    let secret = credential
                        .secret
                        .clone()
                        .filter(|s| !s.is_empty())
                        .ok_or_else(|| Error::argument("an inline credential needs a key"))?;
                    secrets.insert(secret_ref.clone(), secret);
                }
                AuthSource::Resolver { resolver_id, .. } => {
                    *resolver_id = format!("resolver:{i}");
                    resolvers.insert(
                        resolver_id.clone(),
                        credential.resolver.clone().ok_or_else(|| {
                            Error::argument("a resolver credential needs a function")
                        })?,
                    );
                }
                _ => {}
            }
            sources.push(source);
        }
        let transport = match self.transport {
            Some(t) => t,
            None => self.default_transport()?,
        };
        let reader = self
            .keychain
            .clone()
            .unwrap_or_else(|| Arc::new(keychain::read));
        let writer = match (self.keychain_write, self.keychain.is_none()) {
            (Some(w), _) => Some(w),
            (None, true) => Some(Arc::new(keychain::write) as keychain::Writer),
            (None, false) => None,
        };
        let config = ProviderConfig {
            ns,
            base_url: self.base_url.map(|u| u.trim_end_matches('/').to_owned()),
            auth: (!sources.is_empty()).then(|| Auth {
                sources,
                ..Auth::default()
            }),
            headers: self.headers,
            timeout: self.timeout,
            default_model: self.default_model,
            default_embedding_model: self.default_embedding_model,
            default_system: self.default_system,
            default_tools: self.default_tools,
            default_options: self.default_options,
            default_vendor: self.default_vendor,
            overrides: self.overrides,
            strict: self.strict,
        };
        Ok(Provider {
            inner: Arc::new(Inner {
                config,
                host: Arc::new(Host {
                    secrets,
                    resolvers,
                    keychain: Some(reader),
                    keychain_write: writer,
                    transport,
                    observer: self.observer,
                }),
                models: Mutex::new(None),
                infos: Mutex::new(HashMap::new()),
                caps: Mutex::new(HashMap::new()),
            }),
        })
    }
}

impl Provider {
    /// Starts configuring a provider for a vendor namespace.
    pub fn builder(ns: impl Into<Ns>) -> Builder {
        Builder {
            ns: Some(ns.into()),
            ..Builder::default()
        }
    }

    /// A provider with every default: key from the environment, no default model.
    pub fn new(ns: impl Into<Ns>) -> std::result::Result<Provider, Error> {
        Provider::builder(ns).build()
    }

    /// The vendor namespace.
    pub fn ns(&self) -> &Ns {
        &self.inner.config.ns
    }

    async fn run(
        &self,
        operation: OperationKind,
        events: Option<tokio::sync::mpsc::UnboundedSender<ducktape_core::types::Event>>,
        login: Option<LoginHandler>,
        cancel: Option<CancelToken>,
    ) -> std::result::Result<OperationResult, Error> {
        let start = Start {
            clock: clock(),
            provider: self.inner.config.clone(),
            operation,
        };
        Driver::new(self.inner.host.clone(), events, login, cancel)
            .run(start)
            .await
    }

    /// A chat call; set per-call fields on it, then await it or call `stream()`.
    pub fn chat(&self, messages: impl IntoIterator<Item = Message>) -> ChatCall {
        ChatCall {
            provider: self.clone(),
            messages: messages.into_iter().collect(),
            model: None,
            system: None,
            tools: None,
            options: None,
            vendor: None,
            timeout: None,
            headers: None,
            strict: None,
            include_raw: None,
            cancel: None,
        }
    }

    /// An embed call; set per-call fields on it, then await it.
    pub fn embed<I, S>(&self, input: I) -> EmbedCall
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        EmbedCall {
            provider: self.clone(),
            input: input.into_iter().map(Into::into).collect(),
            model: None,
            dimensions: None,
            retries: None,
            vendor: None,
            timeout: None,
            include_raw: None,
            cancel: None,
        }
    }

    /// Lists the models the endpoint serves; cached after the first call unless refresh is set.
    pub async fn models(&self, refresh: bool) -> std::result::Result<Vec<ModelInfo>, Error> {
        if !refresh {
            if let Some(cached) = self
                .inner
                .models
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone()
            {
                return Ok(cached);
            }
        }
        match self
            .run(OperationKind::Models { refresh }, None, None, None)
            .await?
        {
            OperationResult::Models { models } => {
                *self.inner.models.lock().unwrap_or_else(|e| e.into_inner()) = Some(models.clone());
                Ok(models)
            }
            _ => Err(unexpected("models")),
        }
    }

    /// Describes one model.
    pub async fn model_info(&self, id: &str) -> std::result::Result<ModelInfo, Error> {
        if let Some(cached) = self
            .inner
            .infos
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(id)
        {
            return Ok(cached.clone());
        }
        match self
            .run(
                OperationKind::ModelInfo {
                    model: id.to_owned(),
                },
                None,
                None,
                None,
            )
            .await?
        {
            OperationResult::ModelInfo { info: Some(info) } => {
                self.inner
                    .infos
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(id.to_owned(), info.clone());
                Ok(info)
            }
            OperationResult::ModelInfo { info: None } => Err(Error::provider(
                ErrorKind::NotFound,
                None,
                format!("model {id} not found"),
            )),
            _ => Err(unexpected("model_info")),
        }
    }

    /// Tells what a model supports.
    pub async fn capabilities(&self, model: &str) -> std::result::Result<Capabilities, Error> {
        if let Some(cached) = self
            .inner
            .caps
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(model)
        {
            return Ok(cached.clone());
        }
        match self
            .run(
                OperationKind::Capabilities {
                    model: model.to_owned(),
                },
                None,
                None,
                None,
            )
            .await?
        {
            OperationResult::Capabilities {
                capabilities: Some(caps),
            } => {
                self.inner
                    .caps
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(model.to_owned(), caps.clone());
                Ok(caps)
            }
            OperationResult::Capabilities { capabilities: None } => Err(Error::provider(
                ErrorKind::NotFound,
                None,
                format!("no capabilities known for {model}"),
            )),
            _ => Err(unexpected("capabilities")),
        }
    }

    /// Logs into the provider's session source; the handler shows the user what to do and returns any pasted value.
    pub fn login(
        &self,
        handler: impl Fn(LoginAction) -> std::result::Result<Option<String>, String>
            + Send
            + Sync
            + 'static,
    ) -> LoginCall {
        LoginCall {
            provider: self.clone(),
            handler: Arc::new(handler),
            callback: Callback::default(),
            timeout: None,
        }
    }

    /// Forgets the provider's own session.
    pub async fn logout(&self) -> std::result::Result<(), Error> {
        self.run(OperationKind::Logout, None, None, None)
            .await
            .map(|_| ())
    }

    /// Reports which credential would be used and when it expires.
    pub async fn auth_status(&self) -> std::result::Result<AuthStatus, Error> {
        match self
            .run(OperationKind::AuthStatus, None, None, None)
            .await?
        {
            OperationResult::AuthStatus { status } => Ok(status),
            _ => Err(unexpected("auth_status")),
        }
    }
}

fn unexpected(operation: &str) -> Error {
    Error::provider(
        ErrorKind::MalformedResponse,
        None,
        format!("the core returned another result than {operation} expects"),
    )
}

/// A chat call being configured.
#[must_use = "a call does nothing until awaited or streamed"]
pub struct ChatCall {
    provider: Provider,
    messages: Vec<Message>,
    model: Option<String>,
    system: Option<Vec<SystemBlock>>,
    tools: Option<Vec<Tool>>,
    options: Option<Options>,
    vendor: Option<VendorMap>,
    timeout: Option<Timeout>,
    headers: Option<Headers>,
    strict: Option<bool>,
    include_raw: Option<bool>,
    cancel: Option<CancelToken>,
}

impl std::fmt::Debug for ChatCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatCall")
            .field("provider", &self.provider)
            .field("messages", &self.messages.len())
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl ChatCall {
    /// A token that cancels this call from another task.
    pub fn cancel_token(mut self, v: CancelToken) -> Self {
        self.cancel = Some(v);
        self
    }

    pub fn model(mut self, v: impl Into<String>) -> Self {
        self.model = Some(v.into());
        self
    }

    pub fn system(mut self, text: impl Into<String>) -> Self {
        self.system = Some(system_blocks(text));
        self
    }

    pub fn system_blocks(mut self, v: Vec<SystemBlock>) -> Self {
        self.system = Some(v);
        self
    }

    pub fn tools(mut self, v: impl IntoIterator<Item = Tool>) -> Self {
        self.tools = Some(v.into_iter().collect());
        self
    }

    pub fn options(mut self, v: Options) -> Self {
        self.options = Some(v);
        self
    }

    pub fn vendor(mut self, v: VendorMap) -> Self {
        self.vendor = Some(v);
        self
    }

    pub fn timeout(mut self, v: Timeout) -> Self {
        self.timeout = Some(v);
        self
    }

    pub fn headers(mut self, v: Headers) -> Self {
        self.headers = Some(v);
        self
    }

    pub fn strict(mut self, v: bool) -> Self {
        self.strict = Some(v);
        self
    }

    pub fn include_raw(mut self, v: bool) -> Self {
        self.include_raw = Some(v);
        self
    }

    fn request(self) -> std::result::Result<(Provider, ChatRequest, Option<CancelToken>), Error> {
        if let Some(headers) = &self.headers {
            check_headers(headers)?;
        }
        let model = self
            .model
            .or_else(|| self.provider.inner.config.default_model.clone())
            .filter(|m| !m.is_empty())
            .ok_or_else(|| {
                Error::argument(
                    "no model: set model() on the call or default_model() on the provider",
                )
            })?;
        let request = ChatRequest {
            model,
            messages: self.messages,
            system: self.system,
            tools: self.tools,
            options: self.options,
            vendor: self.vendor,
            timeout: self.timeout,
            headers: self.headers,
            strict: self.strict,
            include_raw: self.include_raw,
        };
        Ok((self.provider, request, self.cancel))
    }

    /// Prepares the stream; the request goes out when the stream is first polled.
    pub fn stream(self) -> std::result::Result<ChatStream, Error> {
        let (provider, request, cancel) = self.request()?;
        let cancel = cancel.unwrap_or_default();
        let (tx, rx) = unbounded_channel();
        let token = cancel.clone();
        let driver: BoxFuture<'static, std::result::Result<OperationResult, Error>> =
            Box::pin(async move {
                provider
                    .run(
                        OperationKind::Stream { request },
                        Some(tx),
                        None,
                        Some(token),
                    )
                    .await
            });
        Ok(ChatStream::new(driver, rx, cancel))
    }
}

impl IntoFuture for ChatCall {
    type Output = std::result::Result<Response, Error>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let (provider, request, cancel) = self.request()?;
            match provider
                .run(OperationKind::Chat { request }, None, None, cancel)
                .await?
            {
                OperationResult::Response { response } => Ok(response),
                _ => Err(unexpected("chat")),
            }
        })
    }
}

/// An embed call being configured.
#[must_use = "a call does nothing until awaited"]
pub struct EmbedCall {
    provider: Provider,
    input: Vec<String>,
    model: Option<String>,
    dimensions: Option<u32>,
    retries: Option<u32>,
    vendor: Option<VendorMap>,
    timeout: Option<Timeout>,
    include_raw: Option<bool>,
    cancel: Option<CancelToken>,
}

impl std::fmt::Debug for EmbedCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EmbedCall")
            .field("provider", &self.provider)
            .field("input", &self.input.len())
            .field("model", &self.model)
            .finish_non_exhaustive()
    }
}

impl EmbedCall {
    /// A token that cancels this call from another task.
    pub fn cancel_token(mut self, v: CancelToken) -> Self {
        self.cancel = Some(v);
        self
    }

    pub fn model(mut self, v: impl Into<String>) -> Self {
        self.model = Some(v.into());
        self
    }

    pub fn dimensions(mut self, v: u32) -> Self {
        self.dimensions = Some(v);
        self
    }

    pub fn retries(mut self, v: u32) -> Self {
        self.retries = Some(v);
        self
    }

    pub fn vendor(mut self, v: VendorMap) -> Self {
        self.vendor = Some(v);
        self
    }

    pub fn timeout(mut self, v: Timeout) -> Self {
        self.timeout = Some(v);
        self
    }

    pub fn include_raw(mut self, v: bool) -> Self {
        self.include_raw = Some(v);
        self
    }
}

impl IntoFuture for EmbedCall {
    type Output = std::result::Result<EmbedResponse, Error>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            if self.input.is_empty() {
                return Err(Error::argument("embed input must not be empty"));
            }
            let model = self.model.or_else(|| self.provider.inner.config.default_embedding_model.clone()).filter(|m| !m.is_empty()).ok_or_else(|| Error::argument("no embedding model: set model() on the call or default_embedding_model() on the provider"))?;
            let request = EmbedRequest {
                model,
                input: self.input,
                options: (self.dimensions.is_some() || self.retries.is_some()).then_some(
                    EmbedOptions {
                        dimensions: self.dimensions,
                        retries: self.retries,
                    },
                ),
                vendor: self.vendor,
                timeout: self.timeout,
                include_raw: self.include_raw,
            };
            let cancel = self.cancel;
            match self
                .provider
                .run(OperationKind::Embed { request }, None, None, cancel)
                .await?
            {
                OperationResult::Embed { response } => Ok(response),
                _ => Err(unexpected("embed")),
            }
        })
    }
}

/// A login call being configured.
#[must_use = "a call does nothing until awaited"]
pub struct LoginCall {
    provider: Provider,
    handler: LoginHandler,
    callback: Callback,
    timeout: Option<Duration>,
}

impl LoginCall {
    /// How the authorization comes back.
    pub fn callback(mut self, v: Callback) -> Self {
        self.callback = v;
        self
    }

    pub fn timeout(mut self, v: Duration) -> Self {
        self.timeout = Some(v);
        self
    }
}

impl IntoFuture for LoginCall {
    type Output = std::result::Result<AuthStatus, Error>;
    type IntoFuture = BoxFuture<'static, Self::Output>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let operation = OperationKind::Login {
                callback: self.callback,
                timeout_ms: self.timeout.map(|t| t.as_millis() as u64),
            };
            match self
                .provider
                .run(operation, None, Some(self.handler), None)
                .await?
            {
                OperationResult::AuthStatus { status } => Ok(status),
                _ => Err(unexpected("login")),
            }
        })
    }
}
