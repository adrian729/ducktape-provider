//! Scripted host that replays a wire fixture through the core.
use ducktape_core::effects::{Clock, Effect, Input, Locator, Resume, Start, Step, TransportError};
use ducktape_core::types::Headers;
use ducktape_core::Operation;
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One scripted HTTP exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptedHttp {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<u16>,
    #[serde(default)]
    pub headers: Headers,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub chunks: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<TransportError>,
}

/// A wire fixture: the script the host follows and the golden exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WireFixture {
    pub name: String,
    pub source: String,
    pub start: Start,
    #[serde(default)]
    pub secrets: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub env: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub files: std::collections::BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub keychain: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub http: Vec<ScriptedHttp>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub writes: Vec<ScriptedWrite>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub user_actions: Vec<Option<String>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub callbacks: Vec<String>,
    #[serde(default)]
    pub exchange: Vec<Exchange>,
}

/// One scripted answer to a write.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScriptedWrite {
    pub ok: bool,
    #[serde(default)]
    pub conflict: bool,
}

/// The etag replay hosts report for file reads.
pub fn etag(bytes: &[u8]) -> String {
    use sha2::Digest;
    let digest = sha2::Sha256::digest(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// One step of the golden exchange.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Exchange {
    pub step: Step,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Input>,
}

fn clock(tick: i64) -> Clock {
    Clock {
        now_ms: 1_761_696_000_000 + tick * 100,
        entropy: vec![128; 32],
    }
}

/// Runs the script and returns the exchange the core produced.
pub fn run(fixture: &WireFixture) -> Vec<Exchange> {
    let mut start = fixture.start.clone();
    start.clock = clock(0);
    let (mut op, mut step) = Operation::start(start);
    let mut out = Vec::new();
    let mut http = fixture.http.iter();
    let mut writes = fixture.writes.iter();
    let mut user_actions = fixture.user_actions.iter();
    let mut callbacks = fixture.callbacks.iter();
    let mut tick = 0;
    let mut queue: Vec<Input> = Vec::new();
    for _ in 0..10_000 {
        let input = match &step.effect {
            Effect::Done { .. } | Effect::Failed { .. } => {
                out.push(Exchange { step, input: None });
                return out;
            }
            Effect::NeedSecret { secret_ref, .. } => match fixture.secrets.get(secret_ref) {
                Some(value) => {
                    op.set_secret(secret_ref, value.clone().into_bytes());
                    Input::Secret {
                        secret_ref: secret_ref.clone(),
                        found: true,
                    }
                }
                None => Input::Secret {
                    secret_ref: secret_ref.clone(),
                    found: false,
                },
            },
            Effect::NeedHttp { .. } => {
                let Some(h) = http.next() else {
                    panic!("{}: script has no HTTP reply for request", fixture.name)
                };
                match &h.error {
                    Some(e) => Input::HttpError { error: e.clone() },
                    None => {
                        let chunks: Vec<String> = if h.chunks.is_empty() {
                            h.body.iter().cloned().collect()
                        } else {
                            h.chunks.clone()
                        };
                        queue = chunks
                            .into_iter()
                            .map(|c| Input::HttpChunk {
                                bytes: c.into_bytes(),
                            })
                            .collect();
                        queue.push(Input::HttpEnd);
                        queue.reverse();
                        Input::HttpResponse {
                            status: h.status.unwrap_or(200),
                            headers: h.headers.clone(),
                        }
                    }
                }
            }
            Effect::Receive => queue.pop().unwrap_or(Input::HttpEnd),
            Effect::NeedSleep { .. } => Input::Slept,
            Effect::NeedRead { locator } => {
                let found = match locator {
                    Locator::File { path } => fixture.files.get(path),
                    Locator::Env { name } => fixture.env.get(name),
                    Locator::Keychain { service, account } => {
                        fixture.keychain.get(&format!("{service}|{account}"))
                    }
                };
                let tagged = matches!(locator, Locator::File { .. });
                Input::Read {
                    found: found.is_some(),
                    bytes: found.map(|v| v.clone().into_bytes()),
                    etag: found.filter(|_| tagged).map(|v| etag(v.as_bytes())),
                }
            }
            Effect::NeedWrite { .. } => match writes.next() {
                Some(w) => Input::Written {
                    ok: w.ok,
                    conflict: w.conflict,
                },
                None => Input::Written {
                    ok: true,
                    conflict: false,
                },
            },
            Effect::NeedUserAction { .. } => Input::UserAction {
                value: user_actions.next().cloned().flatten(),
            },
            Effect::NeedCallback { .. } => match callbacks.next() {
                Some(url) => Input::Callback { url: url.clone() },
                None => panic!("{}: script has no callback for the login", fixture.name),
            },
        };
        out.push(Exchange {
            step,
            input: Some(input.clone()),
        });
        tick += 1;
        step = op.resume(Resume {
            clock: clock(tick),
            input,
        });
    }
    panic!("{}: replay did not finish", fixture.name)
}

/// A step with request bodies parsed, so JSON key order does not matter.
pub fn normalize(step: &Step) -> Value {
    let mut v = serde_json::to_value(step).expect("step");
    if let Some(body) = v.pointer_mut("/effect/request/body") {
        if let Some(text) = body.as_str() {
            if let Ok(parsed) = serde_json::from_str::<Value>(text) {
                *body = parsed;
            }
        }
    }
    v
}
