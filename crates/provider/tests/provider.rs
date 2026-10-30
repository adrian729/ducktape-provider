use ducktape_conformance::replay::{ScriptedHttp, WireFixture};
use ducktape_provider::transport::{
    BoxFuture, HttpBody, HttpRequest, HttpTransport, Reply, TransportError, TransportErrorKind,
};
use ducktape_provider::*;
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

type Hook = Box<dyn Fn(&str) + Send + Sync>;

struct Seen {
    url: String,
    headers: Headers,
    body: Option<Value>,
}

struct Scripted {
    script: Mutex<VecDeque<ScriptedHttp>>,
    seen: Mutex<Vec<Seen>>,
    delay: Duration,
    on_request: Option<Hook>,
    bodies_dropped: Arc<AtomicUsize>,
}

impl Scripted {
    fn new(fixture: &WireFixture) -> Arc<Self> {
        Arc::new(Scripted::with_delay(fixture, Duration::ZERO))
    }

    fn with_delay(fixture: &WireFixture, delay: Duration) -> Self {
        Scripted {
            script: Mutex::new(fixture.http.iter().cloned().collect()),
            seen: Mutex::new(Vec::new()),
            delay,
            on_request: None,
            bodies_dropped: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn requests(&self) -> usize {
        self.seen.lock().unwrap().len()
    }

    fn bodies_dropped(&self) -> usize {
        self.bodies_dropped.load(Ordering::SeqCst)
    }

    fn header(&self, index: usize, name: &str) -> Option<String> {
        self.seen
            .lock()
            .unwrap()
            .get(index)
            .and_then(|s| s.headers.get(name).cloned())
    }

    fn url(&self, index: usize) -> String {
        self.seen.lock().unwrap()[index].url.clone()
    }
}

struct Chunks {
    chunks: VecDeque<Vec<u8>>,
    delay: Duration,
    dropped: Arc<AtomicUsize>,
}

impl Drop for Chunks {
    fn drop(&mut self) {
        self.dropped.fetch_add(1, Ordering::SeqCst);
    }
}

impl HttpBody for Chunks {
    fn next(&mut self) -> BoxFuture<'_, std::result::Result<Option<Vec<u8>>, TransportError>> {
        Box::pin(async move {
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            Ok(self.chunks.pop_front())
        })
    }
}

impl HttpTransport for Scripted {
    fn send(
        &self,
        request: HttpRequest,
    ) -> BoxFuture<'_, std::result::Result<Reply, TransportError>> {
        Box::pin(async move {
            let headers = request
                .headers
                .iter()
                .map(|(k, v)| (k.to_lowercase(), v.clone()))
                .collect();
            self.seen.lock().unwrap().push(Seen {
                url: request.url.clone(),
                headers,
                body: request
                    .body
                    .as_deref()
                    .and_then(|b| serde_json::from_str(b).ok()),
            });
            if let Some(hook) = &self.on_request {
                hook(&request.url);
            }
            let Some(h) = self.script.lock().unwrap().pop_front() else {
                return Err(TransportError {
                    kind: TransportErrorKind::Connect,
                    message: "script exhausted".into(),
                });
            };
            if let Some(error) = h.error {
                return Err(error);
            }
            let chunks: VecDeque<Vec<u8>> = if h.chunks.is_empty() {
                h.body.into_iter().map(String::into_bytes).collect()
            } else {
                h.chunks.into_iter().map(String::into_bytes).collect()
            };
            Ok(Reply {
                status: h.status.unwrap_or(200),
                headers: h.headers,
                body: Box::new(Chunks {
                    chunks,
                    delay: self.delay,
                    dropped: self.bodies_dropped.clone(),
                }),
            })
        })
    }
}

fn load(wire: &str, name: &str) -> WireFixture {
    let path = ducktape_conformance::fixtures_dir()
        .join("wire")
        .join(wire)
        .join(format!("{name}.json"));
    serde_json::from_slice(
        &std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap()
}

fn anthropic(name: &str) -> WireFixture {
    load("anthropic.messages", name)
}

fn openai(name: &str) -> WireFixture {
    load("openai.responses", name)
}

fn request_of(fixture: &WireFixture) -> ChatRequest {
    match &fixture.start.operation {
        ducktape_core::effects::OperationKind::Chat { request }
        | ducktape_core::effects::OperationKind::Stream { request } => request.clone(),
        other => panic!("not a chat fixture: {other:?}"),
    }
}

fn done_response(fixture: &WireFixture) -> Response {
    match &fixture.exchange.last().unwrap().step.effect {
        ducktape_core::effects::Effect::Done {
            result: ducktape_core::effects::OperationResult::Response { response },
        } => response.clone(),
        other => panic!("not a done response: {other:?}"),
    }
}

fn last_text(fixture: &WireFixture) -> String {
    match done_response(fixture).content.last().unwrap() {
        Block::Text(t) => t.text.clone(),
        other => panic!("not text: {other:?}"),
    }
}

fn provider(fixture: &WireFixture, server: Arc<Scripted>) -> Provider {
    let mut b = Provider::builder(fixture.start.provider.vendor.clone())
        .transport(server)
        .auth([Credential::key("sk-ant-test-key")]);
    if let Some(url) = &fixture.start.provider.base_url {
        b = b.base_url(url);
    }
    b.build().unwrap()
}

static TEMP_DIRS: AtomicUsize = AtomicUsize::new(0);

struct TempDir(std::path::PathBuf);

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

impl std::ops::Deref for TempDir {
    type Target = std::path::Path;
    fn deref(&self) -> &std::path::Path {
        &self.0
    }
}

fn temp_dir() -> TempDir {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "ducktape-rs-{}-{}-{nanos}",
        std::process::id(),
        TEMP_DIRS.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    TempDir(dir)
}

fn session_store(fixture: &WireFixture, dir: &std::path::Path, prefix: &str) {
    for (path, content) in &fixture.files {
        let target = dir.join(&path[prefix.len() + 1..]);
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::write(target, content).unwrap();
    }
}

fn jwt(exp: i64) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    format!(
        "{}.{}.sig",
        URL_SAFE_NO_PAD.encode(b"{\"alg\":\"RS256\"}"),
        URL_SAFE_NO_PAD.encode(format!("{{\"exp\":{exp}}}"))
    )
}

#[tokio::test]
async fn chat_matches_fixture() {
    let fixture = anthropic("chat_text_reasoning");
    let server = Scripted::new(&fixture);
    let p = provider(&fixture, server.clone());
    let req = request_of(&fixture);
    let r = p
        .chat(req.messages.clone())
        .model(&req.model)
        .options(req.options.clone().unwrap_or_default())
        .await
        .unwrap();
    let expected = done_response(&fixture);
    assert_eq!(expected.content.len(), r.content.len());
    assert_eq!(expected.text(), r.text());
    assert_eq!(expected.stop.reason, r.stop.reason);
    assert_eq!(expected.usage, r.usage);
    assert!(!r.text().is_empty());
    assert_eq!(
        server.header(0, "x-api-key").as_deref(),
        Some("sk-ant-test-key")
    );
    assert_eq!(
        server.seen.lock().unwrap()[0].body.as_ref().unwrap()["model"],
        req.model
    );
    assert_eq!(server.bodies_dropped(), 1);
}

#[tokio::test]
async fn stream_ends_with_finish() {
    let fixture = anthropic("stream_text_reasoning_chunked");
    let p = provider(&fixture, Scripted::new(&fixture));
    let req = request_of(&fixture);
    let mut s = p
        .chat(req.messages.clone())
        .model(&req.model)
        .stream()
        .unwrap();
    let mut events = Vec::new();
    while let Some(ev) = s.next().await {
        events.push(ev.unwrap());
    }
    assert!(matches!(events[0], Event::BlockStart { .. }));
    assert!(matches!(events.last().unwrap(), Event::Finish { .. }));
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            Event::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(!text.is_empty());
    assert_eq!(text, s.response().unwrap().text());
    let p = provider(&fixture, Scripted::new(&fixture));
    let r = p
        .chat(req.messages.clone())
        .model(&req.model)
        .stream()
        .unwrap()
        .finish()
        .await
        .unwrap();
    assert_eq!(r.text(), text);
}

#[tokio::test]
async fn tool_round_trip_builds_history() {
    let fixture = anthropic("chat_tool_call");
    let p = provider(&fixture, Scripted::new(&fixture));
    let req = request_of(&fixture);
    let r = p
        .chat(req.messages.clone())
        .model(&req.model)
        .tools(req.tools.clone().unwrap())
        .await
        .unwrap();
    assert_eq!(r.stop.reason, StopReason::ToolCall);
    let call = r.tool_calls()[0].clone();
    let mut history = req.messages.clone();
    history.push(assistant(&r));
    history.push(tool(tool_result(&call.id, "4")));
    assert_eq!(history[history.len() - 2].role, Role::Assistant);
    match &history.last().unwrap().content[0] {
        Block::ToolResult(t) => assert_eq!(t.tool_call_id, call.id),
        other => panic!("{other:?}"),
    }
}

#[tokio::test]
async fn errors_carry_kinds() {
    let fixture = anthropic("chat_context_overflow");
    let p = provider(&fixture, Scripted::new(&fixture));
    let req = request_of(&fixture);
    let e = p
        .chat(req.messages.clone())
        .model(&req.model)
        .await
        .unwrap_err();
    assert!(e.is(ErrorKind::ContextOverflow));
    assert_eq!(e.provider_error().unwrap().status, Some(400));
    assert!(e.to_string().contains("context_overflow"));
}

#[tokio::test]
async fn no_credential_is_auth_error() {
    let fixture = anthropic("chat_no_credential");
    let server = Scripted::new(&fixture);
    let p = Provider::builder("anthropic")
        .transport(server.clone())
        .auth([Credential::env(["DUCKTAPE_TEST_UNSET_KEY"])])
        .build()
        .unwrap();
    let e = p.chat([user("hi")]).model("m").await.unwrap_err();
    assert!(e.is(ErrorKind::Auth));
    assert_eq!(server.requests(), 0);
    let empty_inline = Credential::source(AuthSource::Inline {
        secret_ref: String::new(),
        transport: Default::default(),
    });
    assert!(matches!(
        Provider::builder("anthropic")
            .transport(server.clone())
            .auth([empty_inline])
            .build(),
        Err(Error::Argument(_))
    ));
    assert!(matches!(
        Provider::builder("anthropic")
            .transport(server.clone())
            .auth([Credential::key("k").profile("work")])
            .build(),
        Err(Error::Argument(_))
    ));
}

#[tokio::test]
async fn resolver_credential() {
    let fixture = anthropic("chat_text_reasoning");
    let req = request_of(&fixture);
    let server = Scripted::new(&fixture);
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let p = Provider::builder("anthropic")
        .transport(server.clone())
        .auth([Credential::resolver(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            async { Ok::<_, String>("sk-resolved".to_owned()) }
        })])
        .build()
        .unwrap();
    p.chat(req.messages.clone())
        .model(&req.model)
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        server.header(0, "x-api-key").as_deref(),
        Some("sk-resolved")
    );
    let failing = Provider::builder("anthropic")
        .transport(Scripted::new(&fixture))
        .auth([Credential::resolver(|| async {
            Err::<String, _>("vault down")
        })])
        .build()
        .unwrap();
    let e = failing
        .chat(req.messages.clone())
        .model(&req.model)
        .await
        .unwrap_err();
    assert!(e.is(ErrorKind::Auth) && e.to_string().contains("vault down"));
}

#[tokio::test]
async fn argument_errors_come_first() {
    let fixture = anthropic("chat_text_reasoning");
    let server = Scripted::new(&fixture);
    let p = provider(&fixture, server.clone());
    assert!(matches!(
        p.chat([user("hi")]).await,
        Err(Error::Argument(_))
    ));
    assert!(matches!(
        Provider::builder("nope").transport(server.clone()).build(),
        Err(Error::Argument(_))
    ));
    assert!(matches!(
        p.embed(Vec::<String>::new()).await,
        Err(Error::Argument(_))
    ));
    let bad = Headers::from([("x-bad".to_owned(), "a\nb".to_owned())]);
    assert!(matches!(
        p.chat([user("hi")]).model("m").headers(bad.clone()).await,
        Err(Error::Argument(_))
    ));
    assert!(matches!(
        Provider::builder("anthropic")
            .transport(server.clone())
            .headers(bad)
            .build(),
        Err(Error::Argument(_))
    ));
    let bad_name = Headers::from([("bad name".to_owned(), "v".to_owned())]);
    assert!(matches!(
        Provider::builder("anthropic")
            .transport(server.clone())
            .headers(bad_name)
            .build(),
        Err(Error::Argument(_))
    ));
    assert_eq!(server.requests(), 0);
}

#[tokio::test]
async fn connection_failure_retries_then_succeeds() {
    let fixture = anthropic("chat_connection_reset_retried");
    let server = Scripted::new(&fixture);
    let p = provider(&fixture, server.clone());
    let req = request_of(&fixture);
    let r = p
        .chat(req.messages.clone())
        .model(&req.model)
        .await
        .unwrap();
    assert_eq!(server.requests(), 2);
    assert!(!r.text().is_empty());
}

#[tokio::test]
async fn models_are_cached() {
    let fixture = anthropic("models_two_pages");
    let server = Scripted::new(&fixture);
    let p = provider(&fixture, server.clone());
    let models = p.models(false).await.unwrap();
    let again = p.models(false).await.unwrap();
    assert!(!models.is_empty());
    assert_eq!(models.len(), again.len());
    assert_eq!(server.requests(), 2);
}

#[tokio::test]
async fn capabilities_shipped() {
    let fixture = anthropic("capabilities_shipped");
    let server = Scripted::new(&fixture);
    let p = provider(&fixture, server.clone());
    let model = match &fixture.start.operation {
        ducktape_core::effects::OperationKind::Capabilities { model } => model.clone(),
        other => panic!("{other:?}"),
    };
    let caps = p.capabilities(&model).await.unwrap();
    assert_eq!(serde_json::to_value(&caps).unwrap()["source"], "shipped");
    assert_eq!(server.requests(), 0);
}

#[tokio::test]
async fn dropping_the_stream_stops_it() {
    let fixture = anthropic("stream_text_reasoning_chunked");
    let server = Arc::new(Scripted::with_delay(&fixture, Duration::from_millis(30)));
    let p = provider(&fixture, server.clone());
    let req = request_of(&fixture);
    let mut s = p
        .chat(req.messages.clone())
        .model(&req.model)
        .stream()
        .unwrap();
    assert!(s.next().await.unwrap().is_ok());
    assert_eq!(server.bodies_dropped(), 0);
    drop(s);
    assert_eq!(server.bodies_dropped(), 1);
    assert_eq!(server.requests(), 1);
}

#[tokio::test]
async fn cancel_token_ends_the_stream_with_cancelled() {
    let fixture = anthropic("stream_text_reasoning_chunked");
    let server = Arc::new(Scripted::with_delay(&fixture, Duration::from_millis(30)));
    let p = provider(&fixture, server.clone());
    let req = request_of(&fixture);
    let mut s = p
        .chat(req.messages.clone())
        .model(&req.model)
        .stream()
        .unwrap();
    let token = s.cancel_token();
    assert!(s.next().await.unwrap().is_ok());
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(10)).await;
        token.cancel();
    });
    let mut last = None;
    while let Some(ev) = s.next().await {
        last = Some(ev);
    }
    assert!(last.unwrap().unwrap_err().is(ErrorKind::Cancelled));
    assert_eq!(server.bodies_dropped(), 1);
    assert!(s.response().is_none());
    let s = p
        .chat(req.messages.clone())
        .model(&req.model)
        .stream()
        .unwrap();
    s.cancel();
    assert!(s.finish().await.unwrap_err().is(ErrorKind::Cancelled));
}

#[tokio::test]
async fn cancel_token_ends_a_chat_with_cancelled() {
    let fixture = anthropic("chat_text_reasoning");
    let server = Arc::new(Scripted::with_delay(&fixture, Duration::from_millis(300)));
    let p = provider(&fixture, server.clone());
    let req = request_of(&fixture);
    let token = CancelToken::new();
    let stop = token.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        stop.cancel();
    });
    let e = p
        .chat(req.messages.clone())
        .model(&req.model)
        .cancel_token(token)
        .await
        .unwrap_err();
    assert!(e.is(ErrorKind::Cancelled), "{e}");
    assert_eq!(server.bodies_dropped(), 1);
}

#[tokio::test]
async fn dropping_the_call_during_the_resolver_sends_nothing() {
    let fixture = anthropic("chat_text_reasoning");
    let server = Scripted::new(&fixture);
    let p = Provider::builder("anthropic")
        .transport(server.clone())
        .auth([Credential::resolver(|| async {
            tokio::time::sleep(Duration::from_secs(60)).await;
            Ok::<_, String>("late".to_owned())
        })])
        .build()
        .unwrap();
    let req = request_of(&fixture);
    let call = p.chat(req.messages.clone()).model(&req.model);
    assert!(tokio::time::timeout(Duration::from_millis(50), call)
        .await
        .is_err());
    assert_eq!(server.requests(), 0);
    let token = CancelToken::new();
    token.cancel();
    let e = p
        .chat(req.messages.clone())
        .model(&req.model)
        .cancel_token(token)
        .await
        .unwrap_err();
    assert!(e.is(ErrorKind::Cancelled));
    assert_eq!(server.requests(), 0);
}

#[tokio::test]
async fn odd_timeouts_are_ignored() {
    let fixture = anthropic("chat_text_reasoning");
    let req = request_of(&fixture);
    for total in [1e12, 1e30, f64::INFINITY, f64::NAN] {
        let p = provider(&fixture, Scripted::new(&fixture));
        let r = p
            .chat(req.messages.clone())
            .model(&req.model)
            .timeout(Timeout {
                read: Some(0.0),
                connect: Some(-1.0),
                total: Some(total),
            })
            .await
            .unwrap();
        assert!(!r.text().is_empty());
    }
}

#[tokio::test]
async fn read_timeout_maps_to_timeout() {
    let fixture = anthropic("chat_text_reasoning");
    let server = Arc::new(Scripted::with_delay(&fixture, Duration::from_millis(400)));
    let p = provider(&fixture, server);
    let req = request_of(&fixture);
    let e = p
        .chat(req.messages.clone())
        .model(&req.model)
        .timeout(Timeout {
            read: Some(0.05),
            ..Timeout::default()
        })
        .options(Options {
            retries: Some(0),
            ..Options::default()
        })
        .await
        .unwrap_err();
    assert!(e.is(ErrorKind::Timeout), "{e}");
}

#[tokio::test]
async fn embed() {
    let fixture = openai("embed_two_inputs");
    let p = Provider::builder("openai")
        .transport(Scripted::new(&fixture))
        .auth([Credential::key("sk-test")])
        .build()
        .unwrap();
    let req = match &fixture.start.operation {
        ducktape_core::effects::OperationKind::Embed { request } => request.clone(),
        other => panic!("{other:?}"),
    };
    let r = p.embed(req.input.clone()).model(&req.model).await.unwrap();
    assert_eq!(r.embeddings.len(), 2);
    assert_eq!(r.embeddings[0][0], 0.1);
}

#[test]
fn image_bytes_become_base64() {
    let png = [0x89u8, 0x50, 0x4e, 0x47];
    let m = user([text("what is this?"), image(&png, "image/png")]);
    match &m.content[1] {
        Block::Image(i) => assert_eq!(i.data.as_deref(), Some("iVBORw==")),
        other => panic!("{other:?}"),
    }
    assert_eq!(m.role, Role::User);
}

#[tokio::test]
async fn profile_refreshes_and_writes_back() {
    let fixture = anthropic("session_profile_refresh_write_back");
    let dir = temp_dir();
    session_store(&fixture, &dir, "/home/u/.config/anthropic");
    let server = Scripted::new(&fixture);
    let p = Provider::builder("anthropic")
        .transport(server.clone())
        .auth([Credential::session("anthropic_profile")
            .store(dir.to_string_lossy())
            .profile("default")
            .refresh("write_back")])
        .build()
        .unwrap();
    let req = request_of(&fixture);
    let r = p
        .chat(req.messages.clone())
        .model(&req.model)
        .await
        .unwrap();
    assert_eq!(r.text(), last_text(&fixture));
    assert_eq!(server.url(0), "https://api.anthropic.com/v1/oauth/token");
    assert_eq!(
        server.header(1, "authorization").as_deref(),
        Some("Bearer sk-ant-oat01-new")
    );
    let creds = dir.join("credentials").join("default.json");
    let written: Value = serde_json::from_slice(&std::fs::read(&creds).unwrap()).unwrap();
    assert_eq!(written["access_token"], "sk-ant-oat01-new");
    assert_eq!(written["extra"], "kept");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&creds).unwrap().permissions().mode();
        assert_eq!(mode & 0o777, 0o600);
    }
    assert_eq!(
        p.auth_status().await.unwrap().source.as_deref(),
        Some("session:anthropic_profile")
    );
}

#[tokio::test]
async fn write_conflict_keeps_the_refreshed_token() {
    let fixture = anthropic("session_profile_write_conflict_keeps_token");
    let dir = temp_dir();
    session_store(&fixture, &dir, "/home/u/.config/anthropic");
    let creds = dir.join("credentials").join("default.json");
    let mut server = Scripted::with_delay(&fixture, Duration::ZERO);
    let target = creds.clone();
    server.on_request = Some(Box::new(move |url: &str| {
        if url.ends_with("/v1/oauth/token") {
            std::fs::write(
                &target,
                "{\"access_token\": \"sk-ant-oat01-other-process\", \"expires_at\": 4102444800}",
            )
            .unwrap();
        }
    }));
    let p = Provider::builder("anthropic")
        .transport(Arc::new(server))
        .auth([Credential::session("anthropic_profile")
            .store(dir.to_string_lossy())
            .profile("default")
            .refresh("write_back")])
        .build()
        .unwrap();
    let req = request_of(&fixture);
    let r = p
        .chat(req.messages.clone())
        .model(&req.model)
        .await
        .unwrap();
    assert_eq!(r.text(), last_text(&fixture));
    assert!(std::fs::read_to_string(&creds)
        .unwrap()
        .contains("other-process"));
}

#[tokio::test]
async fn expired_borrowed_session_names_the_owner() {
    let fixture = anthropic("session_claude_code_expired");
    let dir = temp_dir();
    session_store(&fixture, &dir, "/home/u/.claude");
    let p = Provider::builder("anthropic")
        .transport(Scripted::new(&fixture))
        .auth([Credential::session("claude_code")
            .store(dir.join(".credentials.json").to_string_lossy())])
        .build()
        .unwrap();
    let e = p.chat([user("hi")]).model("m").await.unwrap_err();
    assert_eq!(
        e.provider_error().unwrap().code.as_deref(),
        Some("credential_expired")
    );
    assert!(e.to_string().contains("claude"));
    let status = p.auth_status().await.unwrap();
    assert_eq!(status.source.as_deref(), Some("session:claude_code"));
    assert!(status.expires_at.is_some());
}

#[tokio::test]
async fn codex_keychain_write_back_uses_the_writer() {
    let codex_home = temp_dir();
    std::env::set_var("CODEX_HOME", &*codex_home);
    let mut fixture = openai("session_codex_keychain_write_back");
    let (key, stored) = fixture
        .keychain
        .iter()
        .next()
        .map(|(k, v)| (k.clone(), v.clone()))
        .unwrap();
    let service = key.split('|').next().unwrap().to_owned();
    let mut doc: Value = serde_json::from_str(&stored).unwrap();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;
    doc["tokens"]["access_token"] = Value::String(jwt(now + 60));
    let mut fresh: Value = serde_json::from_str(fixture.http[0].body.as_deref().unwrap()).unwrap();
    fresh["access_token"] = Value::String(jwt(now + 86400));
    fixture.http[0].body = Some(fresh.to_string());
    let written: Arc<Mutex<Vec<(String, String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let doc_text = doc.to_string();
    let reader = move |s: &str, _a: &str| (s == service).then(|| doc_text.clone().into_bytes());
    let sink = written.clone();
    let writer = move |s: &str, a: &str, v: &[u8]| {
        sink.lock().unwrap().push((
            s.to_owned(),
            a.to_owned(),
            String::from_utf8_lossy(v).into_owned(),
        ));
        true
    };
    let req = request_of(&fixture);
    let server = Scripted::new(&fixture);
    let p = Provider::builder("openai")
        .transport(server.clone())
        .auth([Credential::session("codex").refresh("write_back")])
        .keychain(reader.clone())
        .keychain_write(writer)
        .build()
        .unwrap();
    let r = p
        .chat(req.messages.clone())
        .model(&req.model)
        .await
        .unwrap();
    assert_eq!(r.text(), last_text(&fixture));
    let w = written.lock().unwrap().clone();
    assert_eq!(w[0].0, "Codex Auth");
    assert!(w[0].1.starts_with("cli|"));
    assert!(w[0].2.contains("rt_d"));
    assert_eq!(
        server.header(1, "chatgpt-account-id").as_deref(),
        Some("acct_1")
    );
    let reader_only = Provider::builder("openai")
        .transport(Scripted::new(&fixture))
        .auth([Credential::session("codex").refresh("write_back")])
        .keychain(reader)
        .build()
        .unwrap();
    assert_eq!(
        reader_only
            .chat(req.messages.clone())
            .model(&req.model)
            .await
            .unwrap()
            .text(),
        last_text(&fixture)
    );
}

#[test]
fn default_keychain_reader_never_panics() {
    assert!(keychain::read("ducktape-provider-test", "nobody").is_none());
    assert!(!keychain::write("ducktape-provider-test\n", "nobody", b"x"));
}

#[tokio::test]
async fn observer_sees_hooks() {
    struct Hooks(Mutex<Vec<&'static str>>);
    impl Observer for Hooks {
        fn on_request(&self, _: &Value) {
            self.0.lock().unwrap().push("request");
        }
        fn on_response(&self, _: &Value) {
            self.0.lock().unwrap().push("response");
        }
    }
    let fixture = anthropic("chat_text_reasoning");
    let hooks = Arc::new(Hooks(Mutex::new(Vec::new())));
    let p = Provider::builder("anthropic")
        .transport(Scripted::new(&fixture))
        .auth([Credential::key("k")])
        .observer(hooks.clone())
        .build()
        .unwrap();
    let req = request_of(&fixture);
    p.chat(req.messages.clone())
        .model(&req.model)
        .await
        .unwrap();
    assert!(hooks.0.lock().unwrap().contains(&"request"));
}

#[cfg(feature = "reqwest")]
mod reqwest_transport {
    use super::*;
    use ducktape_provider::transport::{Method, Purpose, ReqwestTransport};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    async fn serve(reply: &'static [u8]) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut socket, _)) = listener.accept().await {
                let mut buf = vec![0u8; 65536];
                let _ = socket.read(&mut buf).await;
                let _ = socket.write_all(reply).await;
                let _ = socket.shutdown().await;
            }
        });
        format!("http://{addr}")
    }

    fn get(url: String) -> HttpRequest {
        HttpRequest {
            method: Method::Get,
            url,
            headers: Headers::new(),
            body: None,
            stream: false,
            timeout: None,
            purpose: Purpose::Fetch,
        }
    }

    #[tokio::test]
    async fn redirects_are_not_followed_and_headers_are_joined() {
        let url = serve(b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:9/elsewhere\r\nSet-Cookie: a=1\r\nSet-Cookie: b=2\r\nContent-Length: 0\r\n\r\n").await;
        let reply = ReqwestTransport::default().send(get(url)).await.unwrap();
        assert_eq!(reply.status, 302);
        assert_eq!(
            reply.headers.get("set-cookie").map(String::as_str),
            Some("a=1, b=2")
        );
    }

    #[tokio::test]
    async fn a_truncated_body_is_a_reset() {
        let url = serve(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nshort").await;
        let mut reply = ReqwestTransport::default().send(get(url)).await.unwrap();
        let error = loop {
            match reply.body.next().await {
                Ok(Some(_)) => continue,
                Ok(None) => panic!("the truncated body did not fail"),
                Err(e) => break e,
            }
        };
        assert_eq!(error.kind, TransportErrorKind::Reset, "{}", error.message);
        assert!(error.message.contains(':'), "{}", error.message);
    }

    #[tokio::test]
    async fn a_refused_connection_is_a_connect_error() {
        let e = ReqwestTransport::default()
            .send(get("http://127.0.0.1:9/".to_owned()))
            .await
            .unwrap_err();
        assert_eq!(e.kind, TransportErrorKind::Connect, "{}", e.message);
        assert!(
            e.message.to_lowercase().contains("refused"),
            "{}",
            e.message
        );
    }
}
