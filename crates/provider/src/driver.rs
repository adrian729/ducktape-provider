use crate::credential::Resolver;
use crate::error::Error;
use crate::keychain;
use crate::stream::CancelToken;
use crate::transport::{HttpBody, HttpTransport, TransportError, TransportErrorKind};
use ducktape_core::effects::{
    Clock, Effect, HttpRequest, Input, Locator, Notification, ObserverHook, OperationResult,
    Resume, Start, Step,
};
use ducktape_core::types::{AuthSource, ErrorKind, Event, LoginAction};
use ducktape_core::Operation;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::future::Future;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc::UnboundedSender;

/// Receives request, response, event and retry notifications as redacted JSON.
pub trait Observer: Send + Sync {
    fn on_request(&self, _data: &Value) {}
    fn on_response(&self, _data: &Value) {}
    fn on_event(&self, _data: &Value) {}
    fn on_retry(&self, _data: &Value) {}
}

pub(crate) type LoginHandler =
    Arc<dyn Fn(LoginAction) -> std::result::Result<Option<String>, String> + Send + Sync>;

pub(crate) struct Host {
    pub secrets: HashMap<String, Vec<u8>>,
    pub resolvers: HashMap<String, Arc<dyn Resolver>>,
    pub keychain: Option<keychain::Reader>,
    pub keychain_write: Option<keychain::Writer>,
    pub transport: Arc<dyn HttpTransport>,
    pub observer: Option<Arc<dyn Observer>>,
}

impl<T: Observer + ?Sized> Observer for Arc<T> {
    fn on_request(&self, data: &Value) {
        (**self).on_request(data)
    }
    fn on_response(&self, data: &Value) {
        (**self).on_response(data)
    }
    fn on_event(&self, data: &Value) {
        (**self).on_event(data)
    }
    fn on_retry(&self, data: &Value) {
        (**self).on_retry(data)
    }
}

const MAX_TIMEOUT_SECS: f64 = 4_294_967.0;

pub(crate) fn clock() -> Clock {
    let mut entropy = vec![0u8; 32];
    let _ = getrandom::fill(&mut entropy);
    Clock {
        now_ms: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or_default(),
        entropy,
    }
}

pub(crate) fn etag(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn transport_error(kind: TransportErrorKind, message: &str) -> TransportError {
    TransportError {
        kind,
        message: message.to_owned(),
    }
}

fn seconds(value: Option<f64>) -> Option<Duration> {
    value
        .filter(|v| v.is_finite() && *v > 0.0)
        .map(|v| Duration::from_secs_f64(v.min(MAX_TIMEOUT_SECS)))
}

async fn until<F: Future>(cancel: &Option<CancelToken>, fut: F) -> Option<F::Output> {
    match cancel {
        Some(token) => tokio::select! {
            biased;
            _ = token.cancelled() => None,
            out = fut => Some(out),
        },
        None => Some(fut.await),
    }
}

fn shorter(a: Option<Duration>, b: Option<Duration>) -> Option<Duration> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, None) => a,
        (None, b) => b,
    }
}

/// The effect loop: answers what the core asks for with the host's HTTP, files, environment and keychain.
pub(crate) struct Driver {
    host: Arc<Host>,
    events: Option<UnboundedSender<Event>>,
    login: Option<LoginHandler>,
    cancel: Option<CancelToken>,
    body: Option<Box<dyn HttpBody>>,
    deadline: Option<Instant>,
    read_timeout: Option<Duration>,
}

impl Driver {
    pub fn new(
        host: Arc<Host>,
        events: Option<UnboundedSender<Event>>,
        login: Option<LoginHandler>,
        cancel: Option<CancelToken>,
    ) -> Self {
        Driver {
            host,
            events,
            login,
            cancel,
            body: None,
            deadline: None,
            read_timeout: None,
        }
    }

    fn is_cancelled(&self) -> bool {
        self.cancel.as_ref().is_some_and(CancelToken::is_cancelled)
    }

    pub async fn run(mut self, start: Start) -> std::result::Result<OperationResult, Error> {
        let (mut op, mut step) = Operation::start(start);
        loop {
            let Step { notify, effect } = step;
            for note in notify {
                match note {
                    Notification::Event { event } => {
                        if let Some(tx) = &self.events {
                            let _ = tx.send(*event);
                        }
                    }
                    Notification::Observer { hook, data } => self.observe(hook, &data),
                }
            }
            let input = match effect {
                Effect::Done { result } => return Ok(result),
                Effect::Failed { error } => return Err(Error::Provider(error)),
                _ if self.is_cancelled() => {
                    self.body = None;
                    Input::Cancelled
                }
                Effect::NeedSecret { secret_ref, source } => {
                    self.secret(&mut op, secret_ref, source).await?
                }
                Effect::NeedHttp { request } => {
                    self.body = None;
                    self.http(request).await
                }
                Effect::Receive => match self.receive(&mut op).await {
                    Ok(next) => {
                        step = next;
                        continue;
                    }
                    Err(input) => input,
                },
                Effect::NeedSleep { ms } => {
                    self.body = None;
                    match until(&self.cancel, tokio::time::sleep(Duration::from_millis(ms))).await {
                        Some(()) => Input::Slept,
                        None => Input::Cancelled,
                    }
                }
                Effect::NeedRead { locator } => self.read(locator).await,
                Effect::NeedWrite {
                    locator,
                    bytes,
                    expect,
                    private,
                } => self.write(locator, bytes, expect, private).await,
                Effect::NeedUserAction { action } => self.user_action(action).await?,
                Effect::NeedCallback {
                    redirect_uri,
                    timeout_ms,
                    action,
                } => self.callback(redirect_uri, timeout_ms, action).await?,
            };
            step = op.resume(Resume {
                clock: clock(),
                input,
            });
        }
    }

    fn observe(&self, hook: ObserverHook, data: &Value) {
        if let Some(o) = &self.host.observer {
            match hook {
                ObserverHook::OnRequest => o.on_request(data),
                ObserverHook::OnResponse => o.on_response(data),
                ObserverHook::OnEvent => o.on_event(data),
                ObserverHook::OnRetry => o.on_retry(data),
            }
        }
    }

    async fn secret(
        &mut self,
        op: &mut Operation,
        secret_ref: String,
        source: AuthSource,
    ) -> std::result::Result<Input, Error> {
        let value = match source {
            AuthSource::Inline { .. } => self.host.secrets.get(&secret_ref).cloned(),
            AuthSource::Env { names, .. } => names
                .iter()
                .find_map(|n| std::env::var(n).ok().filter(|v| !v.is_empty()))
                .map(String::into_bytes),
            AuthSource::Resolver { resolver_id, .. } => match self.host.resolvers.get(&resolver_id)
            {
                Some(r) => match until(&self.cancel, r.resolve()).await {
                    None => return Ok(Input::Cancelled),
                    Some(resolved) => Some(
                        resolved
                            .map_err(|m| {
                                Error::provider(
                                    ErrorKind::Auth,
                                    Some("resolver_failed"),
                                    format!("the credential resolver failed: {m}"),
                                )
                            })?
                            .into_bytes(),
                    ),
                },
                None => None,
            },
            _ => None,
        };
        let value = value.filter(|v| !v.is_empty());
        let found = value.is_some();
        if let Some(v) = value {
            op.set_secret(&secret_ref, v);
        }
        Ok(Input::Secret { secret_ref, found })
    }

    async fn http(&mut self, request: HttpRequest) -> Input {
        let timeout = request.timeout.unwrap_or_default();
        let total = seconds(timeout.total);
        self.read_timeout = seconds(timeout.read);
        self.deadline = total.and_then(|t| Instant::now().checked_add(t));
        let headers_wait = seconds(timeout.connect)
            .map(|c| c.saturating_add(self.read_timeout.unwrap_or_default()));
        let send = self.host.transport.send(request);
        let sent = match shorter(headers_wait, total) {
            Some(wait) => until(&self.cancel, tokio::time::timeout(wait, send)).await,
            None => until(&self.cancel, send).await.map(Ok),
        };
        let reply = match sent {
            None => return Input::Cancelled,
            Some(Err(_)) => {
                return Input::HttpError {
                    error: transport_error(
                        TransportErrorKind::Timeout,
                        "no response headers within the connect and read timeouts",
                    ),
                }
            }
            Some(Ok(reply)) => reply,
        };
        match reply {
            Ok(reply) => {
                self.body = Some(reply.body);
                Input::HttpResponse {
                    status: reply.status,
                    headers: reply.headers,
                }
            }
            Err(error) => Input::HttpError { error },
        }
    }

    async fn receive(&mut self, op: &mut Operation) -> std::result::Result<Step, Input> {
        let read_timeout = self.read_timeout;
        let deadline = self.deadline;
        let cancel = self.cancel.clone();
        loop {
            let Some(body) = self.body.as_mut() else {
                return Err(Input::HttpEnd);
            };
            let left = deadline.map(|d| d.saturating_duration_since(Instant::now()));
            let next = match shorter(read_timeout, left) {
                Some(wait) => until(&cancel, tokio::time::timeout(wait, body.next()))
                    .await
                    .map(|timed| {
                        timed.unwrap_or_else(|_| {
                            Err(transport_error(
                                TransportErrorKind::Timeout,
                                "read timed out",
                            ))
                        })
                    }),
                None => until(&cancel, body.next()).await,
            };
            let Some(next) = next else {
                self.body = None;
                return Err(Input::Cancelled);
            };
            match next {
                Ok(Some(bytes)) if bytes.is_empty() => continue,
                Ok(Some(bytes)) => return Ok(op.feed(&bytes, clock())),
                Ok(None) => {
                    self.body = None;
                    return Err(Input::HttpEnd);
                }
                Err(error) => {
                    self.body = None;
                    return Err(Input::HttpError { error });
                }
            }
        }
    }

    async fn read(&mut self, locator: Locator) -> Input {
        let missing = Input::Read {
            found: false,
            bytes: None,
            etag: None,
        };
        let (bytes, etag) = match locator {
            Locator::File { path } => {
                match tokio::task::spawn_blocking(move || std::fs::read(&path)).await {
                    Ok(Ok(b)) => {
                        let tag = etag(&b);
                        (b, Some(tag))
                    }
                    _ => return missing,
                }
            }
            Locator::Env { name } => match std::env::var(&name) {
                Ok(v) => (v.into_bytes(), None),
                Err(_) => return missing,
            },
            Locator::Keychain { service, account } => {
                let Some(reader) = self.host.keychain.clone() else {
                    return missing;
                };
                match tokio::task::spawn_blocking(move || reader(&service, &account))
                    .await
                    .ok()
                    .flatten()
                {
                    Some(b) if !b.is_empty() => (b, None),
                    _ => return missing,
                }
            }
        };
        Input::Read {
            found: true,
            bytes: Some(bytes),
            etag,
        }
    }

    async fn write(
        &mut self,
        locator: Locator,
        bytes: Vec<u8>,
        expect: Option<String>,
        private: bool,
    ) -> Input {
        let failed = Input::Written {
            ok: false,
            conflict: false,
        };
        match locator {
            Locator::Keychain { service, account } => {
                let Some(writer) = self.host.keychain_write.clone() else {
                    return failed;
                };
                let ok = tokio::task::spawn_blocking(move || writer(&service, &account, &bytes))
                    .await
                    .unwrap_or(false);
                Input::Written {
                    ok,
                    conflict: false,
                }
            }
            Locator::File { path } => {
                let written = tokio::task::spawn_blocking(move || {
                    let path = Path::new(&path);
                    if let Some(expect) = expect {
                        let current = std::fs::read(path).ok().map(|b| etag(&b));
                        if current.as_deref() != Some(expect.as_str()) {
                            return Input::Written {
                                ok: false,
                                conflict: true,
                            };
                        }
                    }
                    Input::Written {
                        ok: write_file(path, &bytes, private).is_ok(),
                        conflict: false,
                    }
                })
                .await;
                written.unwrap_or(failed)
            }
            Locator::Env { .. } => failed,
        }
    }

    async fn user_action(&mut self, action: LoginAction) -> std::result::Result<Input, Error> {
        let Some(handler) = self.login.clone() else {
            return Err(Error::provider(
                ErrorKind::Unsupported,
                None,
                "this call cannot ask the user for anything",
            ));
        };
        let answered = tokio::task::spawn_blocking(move || handler(action));
        let Some(joined) = until(&self.cancel, answered).await else {
            return Ok(Input::Cancelled);
        };
        let value = joined
            .map_err(|e| e.to_string())
            .and_then(|r| r)
            .map_err(|m| {
                Error::provider(
                    ErrorKind::Auth,
                    Some("user_action_failed"),
                    format!("the login handler failed: {m}"),
                )
            })?;
        Ok(Input::UserAction { value })
    }

    async fn callback(
        &mut self,
        redirect_uri: String,
        timeout_ms: Option<u64>,
        action: LoginAction,
    ) -> std::result::Result<Input, Error> {
        let (origin, path, addr) = loopback_target(&redirect_uri)?;
        let listener = TcpListener::bind(&addr).await.map_err(|_| {
            Error::provider(
                ErrorKind::Connection,
                Some("callback_port_busy"),
                format!("{addr} is in use; close the other login or use Callback::Paste"),
            )
        })?;
        if let Input::Cancelled = self.user_action(action).await? {
            return Ok(Input::Cancelled);
        }
        let wait = Duration::from_millis(timeout_ms.unwrap_or(DEFAULT_LOGIN_TIMEOUT_MS));
        let waited = until(
            &self.cancel,
            tokio::time::timeout(wait, accept_callback(&listener, &path)),
        )
        .await;
        drop(listener);
        match waited {
            None => Ok(Input::Cancelled),
            Some(Err(_)) => Err(Error::provider(
                ErrorKind::Timeout,
                Some("login_timeout"),
                format!(
                    "no sign-in callback arrived within {} seconds",
                    wait.as_secs().max(1)
                ),
            )),
            Some(Ok(target)) => Ok(Input::Callback {
                url: format!("{origin}{target}"),
            }),
        }
    }
}

const DEFAULT_LOGIN_TIMEOUT_MS: u64 = 300_000;
const CALLBACK_PAGE: &str =
    "<!doctype html><title>Signed in</title><p>The sign-in finished. You can close this window.</p>";

fn loopback_target(redirect_uri: &str) -> std::result::Result<(String, String, String), Error> {
    let (scheme, rest) = redirect_uri.split_once("://").unwrap_or(("", redirect_uri));
    let (authority, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let host = authority
        .rsplit_once(':')
        .map(|(h, _)| h)
        .unwrap_or(authority);
    if scheme != "http" || (host != "127.0.0.1" && host != "[::1]") {
        return Err(Error::provider(
            ErrorKind::Unsupported,
            None,
            format!("the login callback must be a loopback address, not {redirect_uri}"),
        ));
    }
    let path = path.split(['?', '#']).next().unwrap_or("/").to_owned();
    Ok((
        format!("{scheme}://{authority}"),
        path,
        authority.to_owned(),
    ))
}

async fn accept_callback(listener: &TcpListener, path: &str) -> String {
    loop {
        let Ok((mut stream, _)) = listener.accept().await else {
            tokio::time::sleep(Duration::from_millis(100)).await;
            continue;
        };
        let served =
            tokio::time::timeout(Duration::from_secs(5), serve_callback(&mut stream, path)).await;
        if let Ok(Ok(Some(target))) = served {
            return target;
        }
    }
}

async fn serve_callback(stream: &mut TcpStream, path: &str) -> std::io::Result<Option<String>> {
    let mut head = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            break;
        }
        head.extend_from_slice(&chunk[..n]);
        if head.windows(4).any(|w| w == b"\r\n\r\n") || head.len() >= 8192 {
            break;
        }
    }
    let text = String::from_utf8_lossy(&head);
    let target = text
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("")
        .to_owned();
    let hit = target.split('?').next() == Some(path);
    let (status, body) = if hit {
        ("200 OK", CALLBACK_PAGE)
    } else {
        ("404 Not Found", "")
    };
    let reply = format!(
        "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(reply.as_bytes()).await?;
    let _ = stream.shutdown().await;
    Ok(hit.then_some(target))
}

fn write_file(path: &Path, bytes: &[u8], private: bool) -> std::io::Result<()> {
    let dir = path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    create_private_dirs(&dir)?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.subsec_nanos())
        .unwrap_or_default();
    let tmp = dir.join(format!("{name}.tmp{}{nanos}", std::process::id()));
    let result = (|| {
        use std::io::Write;
        let mut file = create_with_mode(&tmp, if private { 0o600 } else { 0o644 })?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

#[cfg(unix)]
fn create_private_dirs(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::DirBuilderExt;
    std::fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(dir)
}

#[cfg(not(unix))]
fn create_private_dirs(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)
}

#[cfg(unix)]
fn create_with_mode(path: &Path, mode: u32) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(mode)
        .open(path)
}

#[cfg(not(unix))]
fn create_with_mode(path: &Path, _mode: u32) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
}
