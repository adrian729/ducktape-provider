//! Sign in with ChatGPT: registration, PKCE, code exchange, our own store and sign-out.
use crate::effects::{HttpRequest, Locator, Method, Purpose};
use crate::session::{jwt_claims, parse_object, rfc3339, str_of};

use crate::types::{
    AuthSource, AuthStatus, ErrorKind, Headers, LoginAction, ProviderConfig, ProviderError,
    SessionPreset, Timeout,
};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

pub const AUTHORIZE_URL: &str = "https://auth.openai.com/api/accounts/authorize";
pub const TOKEN_URL: &str = "https://auth.openai.com/api/accounts/oauth/token";
pub const DISCOVERY_URL: &str = "https://auth.openai.com/.well-known/openid-configuration";
pub const ISSUER: &str = "https://auth.openai.com";
pub const RESOURCE: &str = "https://api.openai.com/v1";
pub const REDIRECT_URI: &str = "http://127.0.0.1:1455/auth/callback";
pub const SCOPES: &str =
    "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct";
pub const PLAN_SCOPE: &str = "chatgpt.tokens.use.direct";
pub const DYNAMIC_CLIENT: &str = "dynamic_agent_client";
pub const DEFAULT_APP_NAME: &str = "ducktape-provider";
pub const DEFAULT_TIMEOUT_MS: u64 = 300_000;
const TOKEN_TIMEOUT_S: f64 = 30.0;
const REVOKE_BACKOFF_MS: u64 = 5_000;

/// What the operation must do next for a login or a logout.
#[derive(Debug)]
pub enum Action {
    Read(Locator),
    Http(HttpRequest),
    Sleep(u64),
    Write {
        locator: Locator,
        bytes: Vec<u8>,
    },
    UserAction(LoginAction),
    Callback {
        redirect_uri: String,
        timeout_ms: u64,
        action: LoginAction,
    },
    Done(AuthStatus),
    Failed(ProviderError),
}

#[derive(Debug, Clone, PartialEq)]
enum Stage {
    Env(Vec<&'static str>),
    Store,
    Browser,
    Paste,
    Exchange,
    Writing,
    Discovery,
    Revoking,
    Clearing,
    Done,
}

/// The login or logout state machine of one operation.
#[derive(Debug)]
pub struct LoginFlow {
    login: bool,
    paste: bool,
    timeout_ms: u64,
    app_name: String,
    store: Option<String>,
    env: Vec<(&'static str, Option<String>)>,
    stage: Stage,
    path: Option<String>,
    record: Map<String, Value>,
    state: String,
    nonce: String,
    verifier: String,
    client_id: Option<String>,
    registering: bool,
    pending: Option<HttpRequest>,
    retried: bool,
    outcome: Option<Result<AuthStatus, ProviderError>>,
}

fn auth_error(code: &str, message: impl Into<String>) -> ProviderError {
    let mut e = ProviderError::new(ErrorKind::Auth, message);
    e.code = Some(code.to_owned());
    e
}

fn unsupported(code: &str, message: impl Into<String>) -> ProviderError {
    let mut e = ProviderError::new(ErrorKind::Unsupported, message);
    e.code = Some(code.to_owned());
    e
}

/// The session preset a login or logout acts on.
pub fn preset_of(config: &ProviderConfig) -> Option<SessionPreset> {
    config.auth.as_ref().and_then(|a| {
        a.sources.iter().find_map(|s| match s {
            AuthSource::Session { preset, .. } => Some(preset.clone()),
            _ => None,
        })
    })
}

fn store_of(config: &ProviderConfig) -> Option<String> {
    config.auth.as_ref().and_then(|a| {
        a.sources.iter().find_map(|s| match s {
            AuthSource::Session { store, .. } => store.clone(),
            _ => None,
        })
    })
}

/// Environment variables that locate our own store.
pub fn env_candidates() -> Vec<&'static str> {
    vec!["XDG_CONFIG_HOME", "HOME", "APPDATA", "USERPROFILE"]
}

/// Our own store for a preset, from the environment the host reported.
pub fn store_path(
    store: Option<&str>,
    env: &dyn Fn(&str) -> Option<String>,
    preset: &SessionPreset,
) -> Option<String> {
    if let Some(store) = store {
        return Some(store.to_owned());
    }
    let file = format!("{}.json", preset.as_str());
    if let Some(xdg) = env("XDG_CONFIG_HOME").filter(|v| !v.is_empty()) {
        return Some(crate::session::join(&xdg, &["ducktape", &file]));
    }
    if let Some(home) = env("HOME").filter(|v| !v.is_empty()) {
        return Some(crate::session::join(&home, &[".config", "ducktape", &file]));
    }
    if let Some(appdata) = env("APPDATA").filter(|v| !v.is_empty()) {
        return Some(crate::session::join(&appdata, &["ducktape", &file]));
    }
    if let Some(up) = env("USERPROFILE").filter(|v| !v.is_empty()) {
        return Some(crate::session::join(
            &up,
            &["AppData", "Roaming", "ducktape", &file],
        ));
    }
    None
}

fn derive(entropy: &[u8], label: &str) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(entropy);
    h.update(label.as_bytes());
    h.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn uuid_urn(bytes: &[u8; 32]) -> String {
    let mut b = [0u8; 16];
    b.copy_from_slice(&bytes[..16]);
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let h = hex(&b);
    format!(
        "urn:uuid:{}-{}-{}-{}-{}",
        &h[0..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..32]
    )
}

/// Percent-encodes a query value.
pub fn encode(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for b in value.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

fn decode(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => match u8::from_str_radix(&value[i + 1..i + 3], 16) {
                Ok(v) => {
                    out.push(v);
                    i += 2;
                }
                Err(_) => out.push(b'%'),
            },
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Encodes form fields in the given order.
pub fn form(fields: &[(&str, &str)]) -> String {
    fields
        .iter()
        .map(|(k, v)| format!("{}={}", encode(k), encode(v)))
        .collect::<Vec<_>>()
        .join("&")
}

/// The query parameters of a URL, decoded, first value wins.
pub fn query_of(url: &str) -> Map<String, Value> {
    let mut out = Map::new();
    let query = url
        .split_once('?')
        .map(|(_, q)| q)
        .unwrap_or("")
        .split('#')
        .next()
        .unwrap_or("");
    for pair in query.split('&').filter(|p| !p.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        out.entry(decode(k)).or_insert_with(|| json!(decode(v)));
    }
    out
}

pub(crate) fn post_form(url: &str, fields: &[(&str, &str)]) -> HttpRequest {
    let mut headers = Headers::new();
    headers.insert(
        "content-type".to_owned(),
        "application/x-www-form-urlencoded".to_owned(),
    );
    headers.insert("accept".to_owned(), "application/json".to_owned());
    HttpRequest {
        method: Method::Post,
        url: url.to_owned(),
        headers,
        body: Some(form(fields)),
        stream: false,
        timeout: Some(Timeout {
            connect: Some(10.0),
            read: Some(TOKEN_TIMEOUT_S),
            total: Some(TOKEN_TIMEOUT_S),
        }),
        purpose: Purpose::Auth,
    }
}

fn get_json(url: &str) -> HttpRequest {
    let mut headers = Headers::new();
    headers.insert("accept".to_owned(), "application/json".to_owned());
    HttpRequest {
        method: Method::Get,
        url: url.to_owned(),
        headers,
        body: None,
        stream: false,
        timeout: Some(Timeout {
            connect: Some(10.0),
            read: Some(TOKEN_TIMEOUT_S),
            total: Some(TOKEN_TIMEOUT_S),
        }),
        purpose: Purpose::Auth,
    }
}

fn aud_matches(claims: &Map<String, Value>, client_id: &str) -> bool {
    match claims.get("aud") {
        Some(Value::String(s)) => s == client_id,
        Some(Value::Array(items)) => items.iter().any(|v| v.as_str() == Some(client_id)),
        _ => false,
    }
}

impl LoginFlow {
    /// Prepares a login or a logout for the provider's session preset.
    pub fn new(
        config: &ProviderConfig,
        login: bool,
        paste: bool,
        timeout_ms: Option<u64>,
        app_name: Option<String>,
        entropy: &[u8],
    ) -> Result<Self, ProviderError> {
        let preset = match preset_of(config) {
            Some(p) => p,
            None => {
                return Err(unsupported(
                    "auth_source",
                    "no session source configured: pass `auth=session(...)`",
                ))
            }
        };
        match (&preset, login) {
            (SessionPreset::ChatgptPlan, _) => {}
            (p, true) => {
                return Err(match crate::session::logout_hint(p) {
                    Some(_) => unsupported(
                        "auth_source",
                        format!(
                            "{} is owned by another tool; {}",
                            p.as_str(),
                            crate::session::login_hint(p)
                        ),
                    ),
                    None => unsupported(
                        "auth_source",
                        format!("login for {} is not available yet", p.as_str()),
                    ),
                })
            }
            (p, false) => {
                return Err(match crate::session::logout_hint(p) {
                    Some(hint) => unsupported(
                        "auth_source",
                        format!(
                            "{} is owned by another tool; log out with `{hint}`",
                            p.as_str()
                        ),
                    ),
                    None => unsupported(
                        "auth_source",
                        format!("logout for {} is not available yet", p.as_str()),
                    ),
                })
            }
        }
        let app_name = app_name
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_APP_NAME.to_owned());
        Ok(LoginFlow {
            login,
            paste,
            timeout_ms: timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS),
            app_name,
            store: store_of(config),
            env: Vec::new(),
            stage: Stage::Env(env_candidates()),
            path: None,
            record: Map::new(),
            state: hex(&derive(entropy, "state")[..16]),
            nonce: hex(&derive(entropy, "nonce")[..16]),
            verifier: URL_SAFE_NO_PAD.encode(derive(entropy, "verifier")),
            client_id: None,
            registering: false,
            pending: None,
            retried: false,
            outcome: None,
        })
    }

    /// Whether this is a login rather than a logout.
    pub fn is_login(&self) -> bool {
        self.login
    }

    /// Begins the flow.
    pub fn start(&mut self) -> Action {
        self.advance()
    }

    fn env_value(&self, name: &str) -> Option<String> {
        self.env
            .iter()
            .find(|(n, _)| *n == name)
            .and_then(|(_, v)| v.clone())
    }

    fn advance(&mut self) -> Action {
        if let Stage::Env(pending) = &self.stage {
            if let Some(name) = pending.first().copied() {
                return Action::Read(Locator::Env {
                    name: name.to_owned(),
                });
            }
        }
        let env = |name: &str| self.env_value(name);
        let Some(path) = store_path(self.store.as_deref(), &env, &SessionPreset::ChatgptPlan)
        else {
            return Action::Failed(auth_error(
                "store_unavailable",
                "no home directory to keep the ChatGPT login in; pass `store=` on the session",
            ));
        };
        self.path = Some(path.clone());
        self.stage = Stage::Store;
        Action::Read(Locator::File { path })
    }

    /// Continues after the host answered a read.
    pub fn on_read(&mut self, found: bool, bytes: Option<Vec<u8>>, now_ms: i64) -> Action {
        let bytes = if found { bytes } else { None };
        match self.stage.clone() {
            Stage::Env(mut pending) => {
                let name = pending.remove(0);
                let value = bytes
                    .and_then(|b| String::from_utf8(b).ok())
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty());
                self.env.push((name, value));
                self.stage = Stage::Env(pending);
                self.advance()
            }
            Stage::Store => {
                if let Some(b) = bytes {
                    match parse_object(&b) {
                        Some(map) => self.record = map,
                        None => {
                            return Action::Failed(auth_error(
                                "store_malformed",
                                "the ChatGPT login store is not a JSON object",
                            ))
                        }
                    }
                }
                if self.login {
                    self.authorize(now_ms)
                } else {
                    self.revoke()
                }
            }
            _ => Action::Failed(auth_error("auth_source", "unexpected read during a login")),
        }
    }

    fn authorize(&mut self, _now_ms: i64) -> Action {
        let saved = str_of(&self.record, "client_id").filter(|c| c != DYNAMIC_CLIENT);
        self.registering = saved.is_none();
        let client_id = saved.unwrap_or_else(|| DYNAMIC_CLIENT.to_owned());
        let host_id = str_of(&self.record, "ext_agent_host_id")
            .unwrap_or_else(|| uuid_urn(&derive(self.verifier.as_bytes(), "host")));
        self.record
            .insert("ext_agent_host_id".to_owned(), json!(host_id));
        let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(self.verifier.as_bytes()));
        let mut params: Vec<(&str, String)> = vec![("client_id", client_id.clone())];
        if self.registering {
            params.push(("agent_name_hint", self.app_name.clone()));
        }
        params.push(("ext_agent_host_id", host_id));
        if !self.registering {
            if let Some(hint) = str_of(&self.record, "id_token") {
                params.push(("id_token_hint", hint));
            }
            if let Some(email) = str_of(&self.record, "email") {
                params.push(("login_hint", email));
            }
        }
        params.extend([
            ("response_type", "code".to_owned()),
            ("redirect_uri", REDIRECT_URI.to_owned()),
            ("scope", SCOPES.to_owned()),
            ("resource", RESOURCE.to_owned()),
            ("state", self.state.clone()),
            ("nonce", self.nonce.clone()),
            ("code_challenge_method", "S256".to_owned()),
            ("code_challenge", challenge),
        ]);
        let query: Vec<String> = params
            .iter()
            .map(|(k, v)| format!("{k}={}", encode(v)))
            .collect();
        let url = format!("{AUTHORIZE_URL}?{}", query.join("&"));
        self.client_id = Some(client_id);
        let action = LoginAction::OpenUrl {
            url,
            instructions: "Open this link in your browser and sign in with ChatGPT".to_owned(),
        };
        if self.paste {
            self.stage = Stage::Browser;
            Action::UserAction(action)
        } else {
            self.stage = Stage::Exchange;
            Action::Callback {
                redirect_uri: REDIRECT_URI.to_owned(),
                timeout_ms: self.timeout_ms,
                action,
            }
        }
    }

    /// Continues after the host showed an action; the value is what the user pasted.
    pub fn on_user_action(&mut self, value: Option<String>, now_ms: i64) -> Action {
        match self.stage {
            Stage::Browser => {
                self.stage = Stage::Paste;
                Action::UserAction(LoginAction::Paste {
                    prompt: "Paste the full address the browser was sent to after signing in"
                        .to_owned(),
                })
            }
            Stage::Paste => match value.map(|v| v.trim().to_owned()).filter(|v| !v.is_empty()) {
                Some(v) => {
                    self.stage = Stage::Exchange;
                    self.on_callback(&v, now_ms)
                }
                None => Action::Failed(auth_error(
                    "login_failed",
                    "nothing was pasted; the login did not complete",
                )),
            },
            _ => Action::Failed(auth_error(
                "auth_source",
                "unexpected user action during a login",
            )),
        }
    }

    /// Continues with the redirect the browser was sent to.
    pub fn on_callback(&mut self, url: &str, _now_ms: i64) -> Action {
        if self.stage != Stage::Exchange {
            return Action::Failed(auth_error(
                "auth_source",
                "unexpected callback during a login",
            ));
        }
        let query = if url.contains('?') || url.contains('=') {
            let pasted = if url.contains('?') {
                url.to_owned()
            } else {
                format!("?{url}")
            };
            query_of(&pasted)
        } else {
            let mut m = Map::new();
            m.insert("code".to_owned(), json!(url));
            m
        };
        if let Some(error) = str_of(&query, "error") {
            let description = str_of(&query, "error_description").unwrap_or_default();
            return Action::Failed(if error == "access_denied" {
                auth_error("login_declined", "the ChatGPT sign-in was declined")
            } else {
                auth_error(
                    "login_failed",
                    format!("the ChatGPT sign-in failed: {error} {description}")
                        .trim()
                        .to_owned(),
                )
            });
        }
        if query.contains_key("state") && str_of(&query, "state").as_deref() != Some(&self.state) {
            return Action::Failed(auth_error(
                "state_mismatch",
                "the login callback did not match this login attempt",
            ));
        }
        let Some(code) = str_of(&query, "code").filter(|c| !c.is_empty()) else {
            return Action::Failed(auth_error(
                "login_failed",
                "the login callback carried no authorization code",
            ));
        };
        let issued = str_of(&query, "client_id").filter(|c| c != DYNAMIC_CLIENT);
        let client_id =
            match (self.registering, issued, self.client_id.clone()) {
                (true, Some(issued), _) => issued,
                (true, None, _) => return Action::Failed(auth_error(
                    "registration_incomplete",
                    "the ChatGPT sign-in returned no client id; the registration did not complete",
                )),
                (false, Some(issued), Some(saved)) if issued != saved => {
                    return Action::Failed(auth_error(
                        "client_mismatch",
                        "the login callback named a different client than the saved one",
                    ))
                }
                (false, _, Some(saved)) => saved,
                (false, _, None) => {
                    return Action::Failed(auth_error("auth_source", "no client id for the login"))
                }
            };
        self.client_id = Some(client_id.clone());
        let request = post_form(
            TOKEN_URL,
            &[
                ("grant_type", "authorization_code"),
                ("client_id", &client_id),
                ("code", &code),
                ("code_verifier", &self.verifier),
                ("redirect_uri", REDIRECT_URI),
                ("resource", RESOURCE),
            ],
        );
        self.pending = Some(request.clone());
        Action::Http(request)
    }

    /// Continues after an HTTP reply.
    pub fn on_http(&mut self, status: u16, body: &[u8], now_ms: i64) -> Action {
        match self.stage {
            Stage::Exchange => self.exchanged(status, body, now_ms),
            Stage::Discovery => self.discovered(status, body),
            Stage::Revoking => self.revoked(status),
            _ => Action::Failed(auth_error(
                "auth_source",
                "unexpected HTTP reply during a login",
            )),
        }
    }

    /// Continues after a request could not be made.
    pub fn on_http_error(&mut self, message: String) -> Action {
        match self.stage {
            Stage::Exchange => {
                let mut e = ProviderError::new(
                    ErrorKind::Connection,
                    format!("the token endpoint could not be reached: {message}"),
                );
                e.code = Some("login_failed".to_owned());
                Action::Failed(e)
            }
            Stage::Discovery | Stage::Revoking if !self.retried => {
                self.retried = true;
                Action::Sleep(REVOKE_BACKOFF_MS)
            }
            Stage::Discovery | Stage::Revoking => self.clear(Some(format!(
                "the ChatGPT session could not be revoked remotely: {message}"
            ))),
            _ => Action::Failed(auth_error(
                "auth_source",
                "unexpected transport error during a login",
            )),
        }
    }

    /// Continues after the backoff before a second revocation attempt.
    pub fn on_slept(&mut self) -> Action {
        match (&self.stage, self.pending.clone()) {
            (Stage::Discovery | Stage::Revoking, Some(request)) => Action::Http(request),
            _ => Action::Failed(auth_error("auth_source", "unexpected sleep during a login")),
        }
    }

    fn exchanged(&mut self, status: u16, body: &[u8], now_ms: i64) -> Action {
        let payload = parse_object(body);
        if status != 200 {
            let code = payload
                .as_ref()
                .and_then(|p| str_of(p, "error"))
                .unwrap_or_default();
            return Action::Failed(auth_error(
                "login_failed",
                if code == "invalid_grant" {
                    "the authorization code was rejected; start the login again".to_owned()
                } else {
                    format!("the token endpoint answered HTTP {status} {code}")
                        .trim()
                        .to_owned()
                },
            ));
        }
        let Some(payload) = payload else {
            return Action::Failed(auth_error(
                "login_failed",
                "the token endpoint answered with no JSON object",
            ));
        };
        let (Some(access), Some(id_token)) = (
            str_of(&payload, "access_token"),
            str_of(&payload, "id_token"),
        ) else {
            return Action::Failed(auth_error(
                "login_failed",
                "the token endpoint answered without an access token and an id token",
            ));
        };
        let client_id = self.client_id.clone().unwrap_or_default();
        let Some(claims) = jwt_claims(&id_token) else {
            return Action::Failed(auth_error(
                "id_token_invalid",
                "the id token could not be decoded",
            ));
        };
        let now_s = now_ms.div_euclid(1000);
        let valid = str_of(&claims, "iss").as_deref() == Some(ISSUER)
            && aud_matches(&claims, &client_id)
            && claims
                .get("exp")
                .and_then(Value::as_i64)
                .is_some_and(|e| e > now_s)
            && str_of(&claims, "nonce").as_deref() == Some(&self.nonce);
        if !valid {
            return Action::Failed(auth_error(
                "id_token_invalid",
                "the id token was not issued for this login attempt",
            ));
        }
        let Some(subject) = str_of(&claims, "sub") else {
            return Action::Failed(auth_error(
                "id_token_invalid",
                "the id token names no subject",
            ));
        };
        let expires_in = payload
            .get("expires_in")
            .and_then(Value::as_i64)
            .unwrap_or(3600);
        let scopes: Vec<String> = str_of(&payload, "scope")
            .unwrap_or_default()
            .split_whitespace()
            .map(str::to_owned)
            .collect();
        let expires_at = now_s + expires_in;
        let record = &mut self.record;
        record.insert("version".to_owned(), json!(1));
        record.insert("client_id".to_owned(), json!(client_id));
        record.insert("issuer".to_owned(), json!(ISSUER));
        record.insert("subject".to_owned(), json!(subject));
        match str_of(&claims, "email") {
            Some(email) => {
                record.insert("email".to_owned(), json!(email));
            }
            None => {
                record.remove("email");
            }
        }
        record.insert("id_token".to_owned(), json!(id_token));
        record.insert("access_token".to_owned(), json!(access));
        match str_of(&payload, "refresh_token") {
            Some(r) => {
                record.insert("refresh_token".to_owned(), json!(r));
            }
            None => {
                record.remove("refresh_token");
            }
        }
        record.insert(
            "token_type".to_owned(),
            json!(str_of(&payload, "token_type").unwrap_or_else(|| "Bearer".to_owned())),
        );
        record.insert("expires_in".to_owned(), json!(expires_in));
        record.insert("expires_at".to_owned(), json!(expires_at));
        record.insert("scopes".to_owned(), json!(scopes));
        record.insert("saved_at".to_owned(), json!(rfc3339(now_ms)));
        let granted = scopes.iter().any(|s| s == PLAN_SCOPE);
        self.outcome = Some(if granted {
            Ok(AuthStatus {
                source: Some("session:chatgpt_plan".to_owned()),
                expires_at: Some(rfc3339(expires_at * 1000)),
                session: Some("chatgpt_plan".to_owned()),
            })
        } else {
            Err(auth_error(
                "plan_unavailable",
                "the sign-in completed without ChatGPT plan usage; log in again and allow it",
            ))
        });
        self.write()
    }

    fn write(&mut self) -> Action {
        self.stage = Stage::Writing;
        let mut bytes =
            serde_json::to_vec_pretty(&Value::Object(self.record.clone())).unwrap_or_default();
        bytes.push(b'\n');
        Action::Write {
            locator: Locator::File {
                path: self.path.clone().unwrap_or_default(),
            },
            bytes,
        }
    }

    /// Continues after the host wrote the store.
    pub fn on_written(&mut self, ok: bool) -> Action {
        let clearing = self.stage == Stage::Clearing;
        self.stage = Stage::Done;
        if !ok {
            return Action::Failed(auth_error(
                "store_unwritable",
                format!(
                    "the ChatGPT login could not be {} {}",
                    if clearing { "cleared from" } else { "saved to" },
                    self.path.clone().unwrap_or_default()
                ),
            ));
        }
        match self.outcome.take() {
            Some(Ok(status)) => Action::Done(status),
            Some(Err(e)) => Action::Failed(e),
            None => Action::Done(AuthStatus {
                source: None,
                expires_at: None,
                session: None,
            }),
        }
    }

    fn revoke(&mut self) -> Action {
        let has_tokens = self.record.contains_key("refresh_token")
            || self.record.contains_key("access_token")
            || self.record.contains_key("id_token");
        if !has_tokens {
            return Action::Done(AuthStatus {
                source: None,
                expires_at: None,
                session: None,
            });
        }
        if str_of(&self.record, "refresh_token").is_none() {
            return self.clear(None);
        }
        self.stage = Stage::Discovery;
        let request = get_json(DISCOVERY_URL);
        self.pending = Some(request.clone());
        Action::Http(request)
    }

    fn discovered(&mut self, status: u16, body: &[u8]) -> Action {
        let endpoint = parse_object(body).and_then(|d| str_of(&d, "revocation_endpoint"));
        match (status, endpoint) {
            (200, Some(endpoint)) if endpoint.starts_with("https://") => {
                self.stage = Stage::Revoking;
                self.retried = false;
                let token = str_of(&self.record, "refresh_token").unwrap_or_default();
                let client_id = str_of(&self.record, "client_id").unwrap_or_default();
                let request = post_form(
                    &endpoint,
                    &[
                        ("token", &token),
                        ("token_type_hint", "refresh_token"),
                        ("client_id", &client_id),
                    ],
                );
                self.pending = Some(request.clone());
                Action::Http(request)
            }
            (s, _) if (500..600).contains(&s) && !self.retried => {
                self.retried = true;
                Action::Sleep(REVOKE_BACKOFF_MS)
            }
            (s, _) => self.clear(Some(format!(
                "the ChatGPT session could not be revoked remotely: discovery answered HTTP {s}"
            ))),
        }
    }

    fn revoked(&mut self, status: u16) -> Action {
        match status {
            200..=299 => self.clear(None),
            500..=599 if !self.retried => {
                self.retried = true;
                Action::Sleep(REVOKE_BACKOFF_MS)
            }
            s => self.clear(Some(format!(
                "the ChatGPT session could not be revoked remotely: HTTP {s}"
            ))),
        }
    }

    fn clear(&mut self, unconfirmed: Option<String>) -> Action {
        for key in [
            "access_token",
            "refresh_token",
            "id_token",
            "expires_in",
            "expires_at",
            "scopes",
            "token_type",
        ] {
            self.record.remove(key);
        }
        self.outcome = Some(match unconfirmed {
            None => Ok(AuthStatus {
                source: None,
                expires_at: None,
                session: None,
            }),
            Some(message) => {
                let mut e = ProviderError::new(ErrorKind::Connection, message);
                e.code = Some("revoke_unconfirmed".to_owned());
                Err(e)
            }
        });
        self.stage = Stage::Clearing;
        let mut bytes =
            serde_json::to_vec_pretty(&Value::Object(self.record.clone())).unwrap_or_default();
        bytes.push(b'\n');
        Action::Write {
            locator: Locator::File {
                path: self.path.clone().unwrap_or_default(),
            },
            bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_decoding() {
        let q = query_of("http://127.0.0.1:1455/auth/callback?code=a%20b&state=s&x=1#frag");
        assert_eq!(str_of(&q, "code").as_deref(), Some("a b"));
        assert_eq!(str_of(&q, "state").as_deref(), Some("s"));
        assert_eq!(str_of(&q, "x").as_deref(), Some("1"));
    }

    #[test]
    fn form_and_encode() {
        assert_eq!(encode("a b/c~"), "a%20b%2Fc~");
        assert_eq!(form(&[("a", "1 2"), ("b", "x&y")]), "a=1%202&b=x%26y");
    }

    #[test]
    fn uuid_has_version_and_variant_bits() {
        let u = uuid_urn(&[0u8; 32]);
        assert!(u.starts_with("urn:uuid:00000000-0000-4000-8000-"));
    }
}
