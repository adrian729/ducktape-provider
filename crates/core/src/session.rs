//! Login sessions: store layouts, token parsing, refresh and write-back per preset.
use crate::effects::{HttpRequest, Locator, Method, Purpose};
use crate::types::{
    AuthKind, ErrorKind, Headers, Placement, ProviderError, RefreshMode, SessionPreset, Vendor,
};
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

const OAUTH_BETA: &str = "oauth-2025-04-20";
const ANTHROPIC_DEFAULT_BASE: &str = "https://api.anthropic.com";
const CODEX_BASE: &str = "https://chatgpt.com/backend-api/codex";
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";
const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const CODEX_CLIENT_VERSION: &str = "0.166.0";
const CLAUDE_CODE_IDENTITY_DIALECT: &str = "claude_code";
const ADVISORY_MS: i64 = 120_000;
const MANDATORY_MS: i64 = 30_000;
const TOKEN_TIMEOUT_S: f64 = 30.0;
const REFRESH_BACKOFF_MS: u64 = 5_000;

/// A credential a session produced.
#[derive(Debug, Clone)]
pub struct SessionCredential {
    pub token: String,
    pub kind: AuthKind,
    pub placement: Placement,
    pub headers: Headers,
    pub base_url: Option<String>,
    pub api_version: Option<String>,
    pub expires_at_ms: Option<i64>,
    pub label: String,
    pub session: String,
    pub dialect: Option<String>,
    pub client_version: Option<String>,
}

/// What the operation must do next for a session.
#[derive(Debug)]
pub enum Action {
    Read(Locator),
    Http(HttpRequest),
    Sleep(u64),
    Write {
        locator: Locator,
        bytes: Vec<u8>,
        expect: Option<String>,
    },
    Ready(SessionCredential),
    NotFound,
    Failed(ProviderError),
}

#[derive(Debug, Clone, PartialEq)]
enum Stage {
    Env(Vec<&'static str>),
    ActiveConfig,
    Config,
    Credentials,
    Keychain,
    Refreshing,
    Writing,
    Version(Vec<String>),
    Done,
}

/// The session state machine of one operation.
#[derive(Debug)]
pub struct SessionFlow {
    preset: SessionPreset,
    vendor: Vendor,
    store: Option<String>,
    profile: Option<String>,
    refresh: RefreshMode,
    stage: Stage,
    env: Vec<(&'static str, Option<String>)>,
    dir: Option<String>,
    source: Option<Locator>,
    etag: Option<String>,
    config: Option<Map<String, Value>>,
    creds: Option<Map<String, Value>>,
    ready: Option<SessionCredential>,
    pending: Option<HttpRequest>,
    retried: bool,
    forced: bool,
    expired_ms: Option<i64>,
    want_version: bool,
    held: Option<SessionCredential>,
}

/// The CLI command that logs out of a borrowed session.
pub fn logout_hint(preset: &SessionPreset) -> Option<&'static str> {
    match preset {
        SessionPreset::AnthropicProfile => Some("ant auth logout"),
        SessionPreset::ClaudeCode => Some("claude /logout"),
        SessionPreset::Codex => Some("codex logout"),
        _ => None,
    }
}

pub(crate) fn login_hint(preset: &SessionPreset) -> &'static str {
    match preset {
        SessionPreset::AnthropicProfile => "run `ant auth login`",
        SessionPreset::ClaudeCode => "run `claude` and log in",
        SessionPreset::ClaudeCodeTokenEnv => {
            "run `claude setup-token` and export CLAUDE_CODE_OAUTH_TOKEN"
        }
        SessionPreset::Codex => "run `codex login`",
        SessionPreset::ChatgptPlan => "call `login()` to sign in with ChatGPT",
        _ => "log in again",
    }
}

fn auth_error(code: &str, message: impl Into<String>) -> ProviderError {
    let mut e = ProviderError::new(ErrorKind::Auth, message);
    e.code = Some(code.to_owned());
    e
}

fn env_candidates(preset: &SessionPreset) -> Vec<&'static str> {
    match preset {
        SessionPreset::AnthropicProfile => vec![
            "ANTHROPIC_CONFIG_DIR",
            "ANTHROPIC_PROFILE",
            "HOME",
            "APPDATA",
            "USERPROFILE",
        ],
        SessionPreset::ClaudeCode => vec!["CLAUDE_CONFIG_DIR", "HOME", "USERPROFILE"],
        SessionPreset::ClaudeCodeTokenEnv => vec!["CLAUDE_CODE_OAUTH_TOKEN"],
        SessionPreset::Codex => vec!["CODEX_HOME", "HOME", "USERPROFILE"],
        SessionPreset::ChatgptPlan => crate::login::env_candidates(),
        _ => Vec::new(),
    }
}

pub(crate) fn join(dir: &str, parts: &[&str]) -> String {
    let sep = if dir.contains('\\') && !dir.contains('/') {
        '\\'
    } else {
        '/'
    };
    let mut out = dir.trim_end_matches(['/', '\\']).to_owned();
    for p in parts {
        out.push(sep);
        out.push_str(p);
    }
    out
}

fn file(path: &str) -> Action {
    Action::Read(Locator::File {
        path: path.to_owned(),
    })
}

fn plain_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(['/', '\\']) && !name.starts_with('.')
}

pub(crate) fn parse_object(bytes: &[u8]) -> Option<Map<String, Value>> {
    match serde_json::from_slice::<Value>(bytes) {
        Ok(Value::Object(m)) => Some(m),
        _ => None,
    }
}

pub(crate) fn str_of(map: &Map<String, Value>, key: &str) -> Option<String> {
    map.get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
}

pub(crate) fn jwt_claims(token: &str) -> Option<Map<String, Value>> {
    let payload = token.split('.').nth(1)?;
    let bytes = URL_SAFE_NO_PAD.decode(payload.trim_end_matches('=')).ok()?;
    parse_object(&bytes)
}

fn jwt_exp_ms(token: &str) -> Option<i64> {
    jwt_claims(token)?.get("exp")?.as_i64().map(|s| s * 1000)
}

/// RFC 3339 text of a unix millisecond timestamp.
pub fn rfc3339(ms: i64) -> String {
    let secs = ms.div_euclid(1000);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn token_request(url: &str, body: Value, extra: &[(&str, &str)]) -> HttpRequest {
    let mut headers = Headers::new();
    headers.insert("content-type".to_owned(), "application/json".to_owned());
    for (k, v) in extra {
        headers.insert((*k).to_owned(), (*v).to_owned());
    }
    HttpRequest {
        method: Method::Post,
        url: url.to_owned(),
        headers,
        body: Some(body.to_string()),
        stream: false,
        timeout: Some(crate::types::Timeout {
            connect: Some(10.0),
            read: Some(TOKEN_TIMEOUT_S),
            total: Some(TOKEN_TIMEOUT_S),
        }),
        purpose: Purpose::Auth,
    }
}

impl SessionFlow {
    /// Prepares the flow for one session source; unknown presets fail.
    pub fn new(
        vendor: &Vendor,
        preset: &SessionPreset,
        store: Option<String>,
        profile: Option<String>,
        refresh: Option<RefreshMode>,
    ) -> Result<Self, ProviderError> {
        let default_refresh = match preset {
            SessionPreset::AnthropicProfile | SessionPreset::ChatgptPlan => RefreshMode::WriteBack,
            _ => RefreshMode::Never,
        };
        let refresh = match preset {
            SessionPreset::ChatgptPlan => Some(RefreshMode::WriteBack),
            _ => refresh,
        };
        match preset {
            SessionPreset::AnthropicProfile
            | SessionPreset::ClaudeCode
            | SessionPreset::ClaudeCodeTokenEnv
            | SessionPreset::Codex
            | SessionPreset::ChatgptPlan => {}
            other => {
                return Err(auth_error(
                    "auth_source",
                    format!(
                        "session preset {} is not available yet; it ships once a recorded fixture proves the flow",
                        other.as_str()
                    ),
                ))
            }
        }
        Ok(SessionFlow {
            preset: preset.clone(),
            vendor: vendor.clone(),
            store,
            profile,
            refresh: refresh.unwrap_or(default_refresh),
            stage: Stage::Env(env_candidates(preset)),
            env: Vec::new(),
            dir: None,
            source: None,
            etag: None,
            config: None,
            creds: None,
            ready: None,
            pending: None,
            retried: false,
            forced: false,
            expired_ms: None,
            want_version: false,
            held: None,
        })
    }

    /// Also finds the client version the Codex backend asks for when listing models.
    pub fn with_client_version(mut self) -> Self {
        self.want_version = true;
        self
    }

    /// The preset name.
    pub fn preset(&self) -> &SessionPreset {
        &self.preset
    }

    /// Expiry of the stored credential when it was found expired or unrefreshable.
    pub fn expired_at_ms(&self) -> Option<i64> {
        self.expired_ms
    }

    /// Whether a 401 has already forced a re-read of the store.
    pub fn forced(&self) -> bool {
        self.forced
    }

    /// Begins resolving the session.
    pub fn start(&mut self) -> Action {
        let action = self.advance();
        self.deliver(action)
    }

    fn env_value(&self, name: &str) -> Option<&str> {
        self.env
            .iter()
            .find(|(n, _)| *n == name)
            .and_then(|(_, v)| v.as_deref())
    }

    fn advance(&mut self) -> Action {
        if let Stage::Env(pending) = &mut self.stage {
            if let Some(name) = pending.first().copied() {
                return Action::Read(Locator::Env {
                    name: name.to_owned(),
                });
            }
            self.stage = Stage::Config;
        }
        match self.preset {
            SessionPreset::ClaudeCodeTokenEnv => self.token_env(),
            SessionPreset::ClaudeCode => self.claude_code_locate(),
            SessionPreset::AnthropicProfile => self.profile_locate(),
            SessionPreset::Codex => self.codex_locate(),
            SessionPreset::ChatgptPlan => self.plan_locate(),
            _ => Action::NotFound,
        }
    }

    /// Continues after the host answered a read.
    fn on_read_step(
        &mut self,
        found: bool,
        bytes: Option<Vec<u8>>,
        etag: Option<String>,
        now_ms: i64,
    ) -> Action {
        let bytes = if found { bytes } else { None };
        if matches!(self.stage, Stage::Credentials | Stage::Keychain) {
            self.etag = if found { etag } else { None };
        }
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
            Stage::ActiveConfig => {
                let name = bytes
                    .and_then(|b| String::from_utf8(b).ok())
                    .map(|s| s.trim().to_owned())
                    .filter(|s| plain_name(s));
                self.profile = Some(name.unwrap_or_else(|| "default".to_owned()));
                self.profile_config()
            }
            Stage::Config => match bytes {
                None => Action::NotFound,
                Some(b) => self.profile_parse_config(&b),
            },
            Stage::Credentials => match self.preset {
                SessionPreset::ClaudeCode => self.claude_code_parse(bytes, now_ms),
                SessionPreset::AnthropicProfile => self.profile_credentials(bytes, now_ms),
                SessionPreset::Codex => match bytes {
                    Some(b) => self.codex_parse(&b, now_ms),
                    None if self.store.is_some() || self.forced => Action::NotFound,
                    None => self.codex_keychain(),
                },
                SessionPreset::ChatgptPlan => self.plan_parse(bytes, now_ms),
                _ => Action::NotFound,
            },
            Stage::Keychain => match bytes {
                Some(b) => self.codex_parse(&b, now_ms),
                None => Action::NotFound,
            },
            _ => Action::Failed(auth_error("auth_source", "unexpected read for a session")),
        }
    }

    /// Continues after the host answered a read.
    pub fn on_read(
        &mut self,
        found: bool,
        bytes: Option<Vec<u8>>,
        etag: Option<String>,
        now_ms: i64,
    ) -> Action {
        if let Stage::Version(pending) = self.stage.clone() {
            return self.on_version(pending, if found { bytes } else { None });
        }
        let action = self.on_read_step(found, bytes, etag, now_ms);
        self.deliver(action)
    }

    /// Continues after the token endpoint replied.
    pub fn on_http(&mut self, status: u16, body: &[u8], now_ms: i64) -> Action {
        let action = self.on_http_step(status, body, now_ms);
        self.deliver(action)
    }

    /// Continues after the backoff before a second token request.
    pub fn on_slept(&mut self) -> Action {
        let action = self.on_slept_step();
        self.deliver(action)
    }

    /// Continues after the host answered the write; the refreshed token is used either way.
    pub fn on_written(&mut self) -> Action {
        let action = self.on_written_step();
        self.deliver(action)
    }

    /// Re-reads the store after the vendor rejected the token; refreshes when allowed.
    pub fn force_refresh(&mut self) -> Action {
        self.held = None;
        let action = self.force_refresh_step();
        self.deliver(action)
    }

    fn deliver(&mut self, action: Action) -> Action {
        match action {
            Action::Ready(cred)
                if self.want_version
                    && cred.client_version.is_none()
                    && cred.base_url.as_deref() == Some(CODEX_BASE) =>
            {
                let paths: Vec<String> = match self.codex_home() {
                    Some(home) => vec![
                        join(&home, &["models_cache.json"]),
                        join(&home, &["version.json"]),
                    ],
                    None => Vec::new(),
                };
                self.held = Some(cred);
                self.next_version(paths)
            }
            other => other,
        }
    }

    fn next_version(&mut self, paths: Vec<String>) -> Action {
        match paths.first().cloned() {
            Some(path) => {
                self.stage = Stage::Version(paths);
                file(&path)
            }
            None => self.release(None),
        }
    }

    fn on_version(&mut self, mut pending: Vec<String>, bytes: Option<Vec<u8>>) -> Action {
        pending.remove(0);
        let found = bytes.as_deref().and_then(parse_object).and_then(|m| {
            ["client_version", "latest_version"]
                .iter()
                .find_map(|k| str_of(&m, k))
                .filter(|v| {
                    !v.is_empty()
                        && v.len() <= 32
                        && v.chars()
                            .all(|c| c.is_ascii_alphanumeric() || ".-+".contains(c))
                })
        });
        match found {
            Some(version) => self.release(Some(version)),
            None => self.next_version(pending),
        }
    }

    fn release(&mut self, version: Option<String>) -> Action {
        self.stage = Stage::Done;
        match self.held.take() {
            Some(mut cred) => {
                cred.client_version =
                    Some(version.unwrap_or_else(|| CODEX_CLIENT_VERSION.to_owned()));
                Action::Ready(cred)
            }
            None => Action::NotFound,
        }
    }

    fn token_env(&mut self) -> Action {
        self.stage = Stage::Done;
        match self.env_value("CLAUDE_CODE_OAUTH_TOKEN") {
            Some(token) => Action::Ready(self.credential(token.to_owned(), None, None, None)),
            None => Action::NotFound,
        }
    }

    fn home(&self) -> Option<String> {
        self.env_value("HOME")
            .or_else(|| self.env_value("USERPROFILE"))
            .map(str::to_owned)
    }

    fn claude_code_locate(&mut self) -> Action {
        let path = match (&self.store, self.env_value("CLAUDE_CONFIG_DIR")) {
            (Some(store), _) => store.clone(),
            (None, Some(dir)) => join(dir, &[".credentials.json"]),
            (None, None) => match self.home() {
                Some(home) => join(&home, &[".claude", ".credentials.json"]),
                None => return Action::NotFound,
            },
        };
        self.source = Some(Locator::File { path: path.clone() });
        self.stage = Stage::Credentials;
        file(&path)
    }

    fn claude_code_parse(&mut self, bytes: Option<Vec<u8>>, now_ms: i64) -> Action {
        let Some(bytes) = bytes else {
            return Action::NotFound;
        };
        let Some(map) = parse_object(&bytes) else {
            return Action::Failed(auth_error(
                "store_malformed",
                "the Claude Code credentials file is not a JSON object",
            ));
        };
        let Some(oauth) = map.get("claudeAiOauth").and_then(Value::as_object) else {
            return Action::NotFound;
        };
        let Some(token) = str_of(oauth, "accessToken") else {
            return Action::NotFound;
        };
        let expires = oauth.get("expiresAt").and_then(Value::as_i64);
        self.stage = Stage::Done;
        if expires.is_some_and(|e| e <= now_ms + MANDATORY_MS) {
            self.expired_ms = expires;
            return Action::Failed(auth_error(
                "credential_expired",
                format!(
                    "the Claude Code login has expired; {}",
                    login_hint(&self.preset)
                ),
            ));
        }
        Action::Ready(self.credential(token, expires, None, None))
    }

    fn profile_locate(&mut self) -> Action {
        if let Some(name) = self
            .profile
            .as_deref()
            .or_else(|| self.env_value("ANTHROPIC_PROFILE"))
        {
            if !plain_name(name) {
                return Action::Failed(auth_error(
                    "auth_source",
                    format!("Anthropic profile name {name:?} must be a plain name without path separators"),
                ));
            }
        }
        let dir = match (&self.store, self.env_value("ANTHROPIC_CONFIG_DIR")) {
            (Some(store), _) => store.clone(),
            (None, Some(dir)) => dir.to_owned(),
            (None, None) => match (
                self.env_value("HOME"),
                self.env_value("APPDATA"),
                self.env_value("USERPROFILE"),
            ) {
                (Some(home), _, _) => join(home, &[".config", "anthropic"]),
                (None, Some(appdata), _) => join(appdata, &["Anthropic"]),
                (None, None, Some(up)) => join(up, &["AppData", "Roaming", "Anthropic"]),
                _ => return Action::NotFound,
            },
        };
        self.dir = Some(dir.clone());
        if self.profile.is_none() {
            if let Some(p) = self.env_value("ANTHROPIC_PROFILE") {
                self.profile = Some(p.to_owned());
            } else {
                self.stage = Stage::ActiveConfig;
                return file(&join(&dir, &["active_config"]));
            }
        }
        self.profile_config()
    }

    fn profile_config(&mut self) -> Action {
        let dir = self.dir.clone().unwrap_or_default();
        let profile = self.profile.clone().unwrap_or_else(|| "default".to_owned());
        let path = join(&dir, &["configs", &format!("{profile}.json")]);
        self.stage = Stage::Config;
        file(&path)
    }

    fn profile_parse_config(&mut self, bytes: &[u8]) -> Action {
        let Some(config) = parse_object(bytes) else {
            return Action::Failed(auth_error(
                "store_malformed",
                "the Anthropic profile config is not a JSON object",
            ));
        };
        let Some(auth) = config.get("authentication").and_then(Value::as_object) else {
            return Action::Failed(auth_error(
                "store_malformed",
                "the Anthropic profile config has no `authentication` object",
            ));
        };
        match auth.get("type").and_then(Value::as_str) {
            Some("user_oauth") => {}
            Some(other) => {
                return Action::Failed(auth_error(
                    "auth_source",
                    format!("Anthropic profile authentication type {other:?} is not supported yet; only user_oauth profiles are"),
                ))
            }
            None => {
                return Action::Failed(auth_error(
                    "store_malformed",
                    "the Anthropic profile config has no authentication type",
                ))
            }
        }
        let dir = self.dir.clone().unwrap_or_default();
        let profile = self.profile.clone().unwrap_or_else(|| "default".to_owned());
        let creds_path = str_of(auth, "credentials_path")
            .unwrap_or_else(|| join(&dir, &["credentials", &format!("{profile}.json")]));
        self.source = Some(Locator::File {
            path: creds_path.clone(),
        });
        self.config = Some(config);
        self.stage = Stage::Credentials;
        file(&creds_path)
    }

    fn profile_base_url(&self) -> String {
        self.config
            .as_ref()
            .and_then(|c| str_of(c, "base_url"))
            .map(|u| u.trim_end_matches('/').to_owned())
            .unwrap_or_else(|| ANTHROPIC_DEFAULT_BASE.to_owned())
    }

    fn profile_credentials(&mut self, bytes: Option<Vec<u8>>, now_ms: i64) -> Action {
        let Some(bytes) = bytes else {
            return Action::Failed(auth_error(
                "credential_expired",
                format!(
                    "the Anthropic profile {:?} has no credentials file; {}",
                    self.profile.clone().unwrap_or_default(),
                    login_hint(&self.preset)
                ),
            ));
        };
        let Some(creds) = parse_object(&bytes) else {
            return Action::Failed(auth_error(
                "store_malformed",
                "the Anthropic profile credentials file is not a JSON object",
            ));
        };
        let Some(token) = str_of(&creds, "access_token") else {
            return Action::Failed(auth_error(
                "store_malformed",
                "the Anthropic profile credentials file has no access_token",
            ));
        };
        let expires_ms = creds
            .get("expires_at")
            .and_then(Value::as_i64)
            .map(|s| s * 1000);
        let client_id = self
            .config
            .as_ref()
            .and_then(|c| c.get("authentication"))
            .and_then(Value::as_object)
            .and_then(|a| str_of(a, "client_id"));
        let refresh_token = str_of(&creds, "refresh_token");
        self.creds = Some(creds);
        let stale = self.forced || expires_ms.is_some_and(|e| e <= now_ms + ADVISORY_MS);
        if stale && self.refresh != RefreshMode::Never {
            if let (Some(client_id), Some(refresh_token)) = (client_id, refresh_token) {
                let url = format!("{}/v1/oauth/token", self.profile_base_url());
                return self.refresh_with(token_request(
                    &url,
                    json!({"grant_type": "refresh_token", "refresh_token": refresh_token, "client_id": client_id}),
                    &[("anthropic-beta", OAUTH_BETA)],
                ));
            }
        }
        self.finish_with(token, expires_ms, now_ms)
    }

    fn refresh_with(&mut self, request: HttpRequest) -> Action {
        self.stage = Stage::Refreshing;
        self.pending = Some(request.clone());
        Action::Http(request)
    }

    fn finish_with(&mut self, token: String, expires_ms: Option<i64>, now_ms: i64) -> Action {
        self.stage = Stage::Done;
        if expires_ms.is_some_and(|e| e <= now_ms + MANDATORY_MS) {
            self.expired_ms = expires_ms;
            return Action::Failed(auth_error(
                "credential_expired",
                format!(
                    "the {} login has expired; {}",
                    self.preset.as_str(),
                    login_hint(&self.preset)
                ),
            ));
        }
        let cred = match self.preset {
            SessionPreset::AnthropicProfile => {
                let mut headers = Headers::new();
                if let Some(ws) = self.config.as_ref().and_then(|c| str_of(c, "workspace_id")) {
                    headers.insert("anthropic-workspace-id".to_owned(), ws);
                }
                let base = self.profile_base_url();
                let pin = (base != ANTHROPIC_DEFAULT_BASE).then_some(base);
                self.credential(token, expires_ms, pin, Some(headers))
            }
            SessionPreset::Codex => {
                let mut headers = Headers::new();
                if let Some(account) = self.codex_account_id() {
                    headers.insert("chatgpt-account-id".to_owned(), account);
                }
                let mut cred = self.credential(
                    token,
                    expires_ms,
                    Some(CODEX_BASE.to_owned()),
                    Some(headers),
                );
                cred.api_version = Some(String::new());
                cred
            }
            _ => self.credential(token, expires_ms, None, None),
        };
        Action::Ready(cred)
    }

    fn plan_locate(&mut self) -> Action {
        let path = {
            let env = |name: &str| self.env_value(name).map(str::to_owned);
            crate::login::store_path(self.store.as_deref(), &env, &self.preset)
        };
        let Some(path) = path else {
            return Action::NotFound;
        };
        self.source = Some(Locator::File { path: path.clone() });
        self.stage = Stage::Credentials;
        file(&path)
    }

    fn plan_parse(&mut self, bytes: Option<Vec<u8>>, now_ms: i64) -> Action {
        let Some(bytes) = bytes else {
            return Action::NotFound;
        };
        let Some(record) = parse_object(&bytes) else {
            return Action::Failed(auth_error(
                "store_malformed",
                "the ChatGPT login store is not a JSON object",
            ));
        };
        let Some(token) = str_of(&record, "access_token") else {
            return Action::NotFound;
        };
        let granted = record
            .get("scopes")
            .and_then(Value::as_array)
            .is_some_and(|s| {
                s.iter()
                    .any(|v| v.as_str() == Some(crate::login::PLAN_SCOPE))
            });
        if !granted {
            self.stage = Stage::Done;
            return Action::Failed(auth_error(
                "plan_unavailable",
                "the ChatGPT sign-in has no plan usage; log in again and allow it",
            ));
        }
        let expires_ms = record
            .get("expires_at")
            .and_then(Value::as_i64)
            .map(|s| s * 1000);
        let refresh_token = str_of(&record, "refresh_token");
        let client_id = str_of(&record, "client_id");
        self.creds = Some(record);
        let stale = self.forced || expires_ms.is_some_and(|e| e <= now_ms + ADVISORY_MS);
        if stale {
            if let (Some(client_id), Some(refresh_token)) = (client_id, refresh_token) {
                return self.refresh_with(crate::login::post_form(
                    crate::login::TOKEN_URL,
                    &[
                        ("grant_type", "refresh_token"),
                        ("client_id", &client_id),
                        ("refresh_token", &refresh_token),
                        ("resource", crate::login::RESOURCE),
                    ],
                ));
            }
        }
        self.finish_with(token, expires_ms, now_ms)
    }

    fn credential(
        &self,
        token: String,
        expires_at_ms: Option<i64>,
        base_url: Option<String>,
        headers: Option<Headers>,
    ) -> SessionCredential {
        let mut headers = headers.unwrap_or_default();
        if matches!(self.vendor, Vendor::Anthropic) {
            headers.insert("anthropic-beta".to_owned(), OAUTH_BETA.to_owned());
        }
        SessionCredential {
            token,
            kind: AuthKind::Bearer,
            placement: Placement::Authorization,
            headers,
            base_url,
            api_version: None,
            expires_at_ms,
            label: format!("session:{}", self.preset.as_str()),
            session: self.preset.as_str().to_owned(),
            dialect: match self.preset {
                SessionPreset::ClaudeCode | SessionPreset::ClaudeCodeTokenEnv => {
                    Some(CLAUDE_CODE_IDENTITY_DIALECT.to_owned())
                }
                SessionPreset::ChatgptPlan => Some("chatgpt_plan".to_owned()),
                _ => None,
            },
            client_version: None,
        }
    }

    fn codex_home(&self) -> Option<String> {
        if let Some(dir) = self.env_value("CODEX_HOME") {
            return Some(dir.to_owned());
        }
        self.home().map(|h| join(&h, &[".codex"]))
    }

    fn codex_locate(&mut self) -> Action {
        let path = match &self.store {
            Some(store) => store.clone(),
            None => match self.codex_home() {
                Some(home) => join(&home, &["auth.json"]),
                None => return Action::NotFound,
            },
        };
        self.source = Some(Locator::File { path: path.clone() });
        self.stage = Stage::Credentials;
        file(&path)
    }

    fn codex_keychain(&mut self) -> Action {
        let Some(home) = self.codex_home() else {
            return Action::NotFound;
        };
        let digest = Sha256::digest(home.as_bytes());
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        let locator = Locator::Keychain {
            service: "Codex Auth".to_owned(),
            account: format!("cli|{}", &hex[..16]),
        };
        self.source = Some(locator.clone());
        self.stage = Stage::Keychain;
        Action::Read(locator)
    }

    fn codex_account_id(&self) -> Option<String> {
        let tokens = self.creds.as_ref()?.get("tokens")?.as_object()?;
        if let Some(id) = str_of(tokens, "account_id") {
            return Some(id);
        }
        let claims = jwt_claims(&str_of(tokens, "id_token")?)?;
        claims
            .get("https://api.openai.com/auth")?
            .as_object()
            .and_then(|a| str_of(a, "chatgpt_account_id"))
    }

    fn codex_parse(&mut self, bytes: &[u8], now_ms: i64) -> Action {
        let Some(map) = parse_object(bytes) else {
            return Action::Failed(auth_error(
                "store_malformed",
                "the Codex auth store is not a JSON object",
            ));
        };
        let tokens = map.get("tokens").and_then(Value::as_object).cloned();
        let access = tokens.as_ref().and_then(|t| str_of(t, "access_token"));
        match access {
            Some(token) => {
                let expires_ms = jwt_exp_ms(&token);
                let refresh_token = tokens.as_ref().and_then(|t| str_of(t, "refresh_token"));
                self.creds = Some(map);
                let stale = self.forced || expires_ms.is_some_and(|e| e <= now_ms + ADVISORY_MS);
                if stale && self.refresh != RefreshMode::Never {
                    if let Some(refresh_token) = refresh_token {
                        return self.refresh_with(token_request(
                            CODEX_TOKEN_URL,
                            json!({"client_id": CODEX_CLIENT_ID, "grant_type": "refresh_token", "refresh_token": refresh_token}),
                            &[],
                        ));
                    }
                }
                self.finish_with(token, expires_ms, now_ms)
            }
            None => match str_of(&map, "OPENAI_API_KEY") {
                Some(key) => {
                    self.stage = Stage::Done;
                    Action::Ready(self.credential(key, None, None, None))
                }
                None => Action::NotFound,
            },
        }
    }

    /// Continues after the token endpoint replied.
    fn on_http_step(&mut self, status: u16, body: &[u8], now_ms: i64) -> Action {
        let payload = parse_object(body);
        if status != 200 {
            let code = payload
                .as_ref()
                .and_then(|p| str_of(p, "error"))
                .unwrap_or_default();
            let revoked = status == 401
                || status == 403
                || matches!(
                    code.as_str(),
                    "invalid_grant"
                        | "invalid_refresh_token"
                        | "token_expired"
                        | "refresh_token_expired"
                        | "refresh_token_reused"
                        | "refresh_token_invalidated"
                );
            let rejected = matches!(
                code.as_str(),
                "invalid_client"
                    | "unauthorized_client"
                    | "unsupported_grant_type"
                    | "invalid_request"
                    | "invalid_scope"
            );
            if rejected {
                self.retried = true;
                return self.refresh_failed(
                    format!("token endpoint rejected the request (HTTP {status} {code})"),
                    now_ms,
                );
            }
            if revoked {
                self.stage = Stage::Done;
                self.expired_ms = self.current_token().1;
                return Action::Failed(auth_error(
                    "credential_expired",
                    format!(
                        "the {} session could not be refreshed (HTTP {status}{}); {}",
                        self.preset.as_str(),
                        if code.is_empty() {
                            String::new()
                        } else {
                            format!(" {code}")
                        },
                        login_hint(&self.preset)
                    ),
                ));
            }
            return self.refresh_failed(format!("token endpoint answered HTTP {status}"), now_ms);
        }
        let Some(payload) = payload else {
            return self.refresh_failed(
                "token endpoint answered with no JSON object".to_owned(),
                now_ms,
            );
        };
        let Some(access) = str_of(&payload, "access_token") else {
            return self.refresh_failed(
                "token endpoint answered without an access_token".to_owned(),
                now_ms,
            );
        };
        match self.preset {
            SessionPreset::AnthropicProfile => {
                let expires_in = payload
                    .get("expires_in")
                    .and_then(Value::as_i64)
                    .unwrap_or(3600);
                let expires_s = now_ms.div_euclid(1000) + expires_in;
                let mut creds = self.creds.clone().unwrap_or_default();
                let refresh =
                    str_of(&payload, "refresh_token").or_else(|| str_of(&creds, "refresh_token"));
                creds.insert("version".to_owned(), json!("1.0"));
                creds.insert("type".to_owned(), json!("oauth_token"));
                creds.insert("access_token".to_owned(), json!(access));
                creds.insert("expires_at".to_owned(), json!(expires_s));
                if let Some(r) = refresh {
                    creds.insert("refresh_token".to_owned(), json!(r));
                }
                self.creds = Some(creds.clone());
                self.after_refresh(creds, access, Some(expires_s * 1000), now_ms)
            }
            SessionPreset::Codex => {
                let mut creds = self.creds.clone().unwrap_or_default();
                let mut tokens = creds
                    .get("tokens")
                    .and_then(Value::as_object)
                    .cloned()
                    .unwrap_or_default();
                tokens.insert("access_token".to_owned(), json!(access));
                if let Some(r) = str_of(&payload, "refresh_token") {
                    tokens.insert("refresh_token".to_owned(), json!(r));
                }
                if let Some(id) = str_of(&payload, "id_token") {
                    tokens.insert("id_token".to_owned(), json!(id));
                }
                creds.insert("tokens".to_owned(), Value::Object(tokens));
                creds.insert("last_refresh".to_owned(), json!(rfc3339(now_ms)));
                self.creds = Some(creds.clone());
                let expires = jwt_exp_ms(&access);
                self.after_refresh(creds, access, expires, now_ms)
            }
            SessionPreset::ChatgptPlan => {
                let expires_in = payload
                    .get("expires_in")
                    .and_then(Value::as_i64)
                    .unwrap_or(3600);
                let expires_s = now_ms.div_euclid(1000) + expires_in;
                let mut creds = self.creds.clone().unwrap_or_default();
                creds.insert("access_token".to_owned(), json!(access));
                if let Some(r) = str_of(&payload, "refresh_token") {
                    creds.insert("refresh_token".to_owned(), json!(r));
                }
                if let Some(id) = str_of(&payload, "id_token") {
                    creds.insert("id_token".to_owned(), json!(id));
                }
                if let Some(scope) = str_of(&payload, "scope") {
                    let scopes: Vec<&str> = scope.split_whitespace().collect();
                    creds.insert("scopes".to_owned(), json!(scopes));
                }
                creds.insert("expires_in".to_owned(), json!(expires_in));
                creds.insert("expires_at".to_owned(), json!(expires_s));
                creds.insert("saved_at".to_owned(), json!(rfc3339(now_ms)));
                self.creds = Some(creds.clone());
                self.after_refresh(creds, access, Some(expires_s * 1000), now_ms)
            }
            _ => Action::NotFound,
        }
    }

    /// Continues after the token endpoint could not be reached.
    pub fn on_http_error(&mut self, message: String, now_ms: i64) -> Action {
        let action = self.refresh_failed(message, now_ms);
        self.deliver(action)
    }

    /// Continues after the backoff before a second token request.
    fn on_slept_step(&mut self) -> Action {
        match self.pending.clone() {
            Some(request) if self.stage == Stage::Refreshing => Action::Http(request),
            _ => Action::Failed(auth_error("auth_source", "unexpected sleep for a session")),
        }
    }

    fn refresh_failed(&mut self, detail: String, now_ms: i64) -> Action {
        let (token, expires) = self.current_token();
        if let Some(token) = token {
            if !self.forced && expires.is_none_or(|e| e > now_ms + MANDATORY_MS) {
                return self.finish_with(token, expires, now_ms);
            }
        }
        if !self.retried && self.pending.is_some() {
            self.retried = true;
            return Action::Sleep(REFRESH_BACKOFF_MS);
        }
        self.stage = Stage::Done;
        self.expired_ms = expires;
        Action::Failed(auth_error(
            "refresh_failed",
            format!(
                "the {} session could not be refreshed: {detail}",
                self.preset.as_str()
            ),
        ))
    }

    fn current_token(&self) -> (Option<String>, Option<i64>) {
        let Some(creds) = &self.creds else {
            return (None, None);
        };
        match self.preset {
            SessionPreset::Codex => {
                let token = creds
                    .get("tokens")
                    .and_then(Value::as_object)
                    .and_then(|t| str_of(t, "access_token"));
                let exp = token.as_deref().and_then(jwt_exp_ms);
                (token, exp)
            }
            _ => (
                str_of(creds, "access_token"),
                creds
                    .get("expires_at")
                    .and_then(Value::as_i64)
                    .map(|s| s * 1000),
            ),
        }
    }

    fn after_refresh(
        &mut self,
        creds: Map<String, Value>,
        token: String,
        expires_ms: Option<i64>,
        now_ms: i64,
    ) -> Action {
        if self.refresh == RefreshMode::WriteBack {
            if let Some(locator) = self.source.clone() {
                self.ready = match self.finish_with(token, expires_ms, now_ms) {
                    Action::Ready(c) => Some(c),
                    other => return other,
                };
                self.stage = Stage::Writing;
                let mut bytes =
                    serde_json::to_vec_pretty(&Value::Object(creds)).unwrap_or_default();
                bytes.push(b'\n');
                return Action::Write {
                    locator,
                    bytes,
                    expect: self.etag.clone(),
                };
            }
        }
        self.finish_with(token, expires_ms, now_ms)
    }

    /// Continues after the host answered the write; the refreshed token is used either way.
    fn on_written_step(&mut self) -> Action {
        self.stage = Stage::Done;
        match self.ready.take() {
            Some(cred) => Action::Ready(cred),
            None => Action::NotFound,
        }
    }

    /// Re-reads the store after the vendor rejected the token; refreshes when allowed.
    fn force_refresh_step(&mut self) -> Action {
        self.forced = true;
        self.etag = None;
        if self.preset == SessionPreset::ClaudeCodeTokenEnv {
            self.env.clear();
            self.stage = Stage::Env(env_candidates(&self.preset));
            return self.advance();
        }
        match self.source.clone() {
            Some(locator) => {
                self.stage = match locator {
                    Locator::Keychain { .. } => Stage::Keychain,
                    _ => Stage::Credentials,
                };
                Action::Read(locator)
            }
            None => Action::NotFound,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc3339_formats_known_dates() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_761_696_000_000), "2025-10-29T00:00:00Z");
        assert_eq!(rfc3339(951_782_400_000), "2000-02-29T00:00:00Z");
    }

    #[test]
    fn jwt_exp_is_read_without_verification() {
        let payload = URL_SAFE_NO_PAD.encode(br#"{"exp": 1700000000}"#);
        assert_eq!(
            jwt_exp_ms(&format!("h.{payload}.s")),
            Some(1_700_000_000_000)
        );
        assert_eq!(jwt_exp_ms("not-a-jwt"), None);
    }
}
