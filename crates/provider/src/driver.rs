use crate::credential::Resolver;
use crate::error::Error;
use crate::keychain;
use crate::transport::{Body, Transport, TransportError, TransportErrorKind};
use ducktape_core::effects::{
    Clock, Effect, HttpRequest, Input, Locator, Notification, ObserverHook, OperationResult,
    Resume, Start, Step,
};
use ducktape_core::types::{AuthSource, ErrorKind, Event, LoginAction};
use ducktape_core::Operation;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
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
    pub transport: Arc<dyn Transport>,
    pub observer: Option<Arc<dyn Observer>>,
}

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
        .filter(|v| v.is_finite() && *v >= 0.0)
        .map(Duration::from_secs_f64)
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
    body: Option<Box<dyn Body>>,
    deadline: Option<Instant>,
    read_timeout: Option<Duration>,
}

impl Driver {
    pub fn new(
        host: Arc<Host>,
        events: Option<UnboundedSender<Event>>,
        login: Option<LoginHandler>,
    ) -> Self {
        Driver {
            host,
            events,
            login,
            body: None,
            deadline: None,
            read_timeout: None,
        }
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
                    tokio::time::sleep(Duration::from_millis(ms)).await;
                    Input::Slept
                }
                Effect::NeedRead { locator } => self.read(locator).await,
                Effect::NeedWrite {
                    locator,
                    bytes,
                    expect,
                    private,
                } => self.write(locator, bytes, expect, private).await,
                Effect::NeedUserAction { action } => self.user_action(action)?,
                Effect::NeedCallback { .. } => {
                    return Err(Error::provider(
                        ErrorKind::Unsupported,
                        None,
                        "the loopback login callback is not implemented in the Rust crate yet",
                    ))
                }
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
                Some(r) => Some(
                    r.resolve()
                        .await
                        .map_err(|m| {
                            Error::provider(
                                ErrorKind::Auth,
                                Some("resolver_failed"),
                                format!("the credential resolver failed: {m}"),
                            )
                        })?
                        .into_bytes(),
                ),
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
        self.deadline = total.map(|t| Instant::now() + t);
        let send = self.host.transport.send(request);
        let reply = match shorter(seconds(timeout.connect), total) {
            Some(wait) => match tokio::time::timeout(wait, send).await {
                Ok(reply) => reply,
                Err(_) => {
                    return Input::HttpError {
                        error: transport_error(
                            TransportErrorKind::Timeout,
                            "no response headers within the connect timeout",
                        ),
                    }
                }
            },
            None => send.await,
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
        loop {
            let Some(body) = self.body.as_mut() else {
                return Err(Input::HttpEnd);
            };
            let left = deadline.map(|d| d.saturating_duration_since(Instant::now()));
            let next = match shorter(read_timeout, left) {
                Some(wait) => tokio::time::timeout(wait, body.next())
                    .await
                    .unwrap_or_else(|_| {
                        Err(transport_error(
                            TransportErrorKind::Timeout,
                            "read timed out",
                        ))
                    }),
                None => body.next().await,
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
            Locator::File { path } => match std::fs::read(&path) {
                Ok(b) => {
                    let tag = etag(&b);
                    (b, Some(tag))
                }
                Err(_) => return missing,
            },
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
            }
            Locator::Env { .. } => failed,
        }
    }

    fn user_action(&mut self, action: LoginAction) -> std::result::Result<Input, Error> {
        let Some(handler) = &self.login else {
            return Err(Error::provider(
                ErrorKind::Unsupported,
                None,
                "this call cannot ask the user for anything",
            ));
        };
        let value = handler(action).map_err(|m| {
            Error::provider(
                ErrorKind::Auth,
                Some("user_action_failed"),
                format!("the login handler failed: {m}"),
            )
        })?;
        Ok(Input::UserAction { value })
    }
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
        std::fs::write(&tmp, bytes)?;
        set_mode(&tmp, if private { 0o600 } else { 0o644 })?;
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
fn set_mode(path: &Path, mode: u32) -> std::io::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
}

#[cfg(not(unix))]
fn set_mode(_path: &Path, _mode: u32) -> std::io::Result<()> {
    Ok(())
}
