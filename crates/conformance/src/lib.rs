//! Runs the shared fixtures against the core and the ABI.
use ducktape_core::effects::{Effect, Input, Notification, Resume, Start, Step};
use ducktape_core::types::*;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;
use std::path::{Path, PathBuf};

pub mod replay;

/// A type sample: a schema definition name and a value that must round-trip.
#[derive(Debug, serde::Deserialize)]
pub struct TypeSample {
    pub def: String,
    pub value: Value,
}

/// The repo's `fixtures/` directory.
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

fn roundtrip<T: DeserializeOwned + Serialize>(value: &Value) -> Result<Value, String> {
    let typed: T =
        serde_json::from_value(value.clone()).map_err(|e| format!("deserialize: {e}"))?;
    serde_json::to_value(&typed).map_err(|e| format!("serialize: {e}"))
}

/// Round-trips a value through the Rust type named by `def`.
pub fn roundtrip_def(def: &str, value: &Value) -> Result<Value, String> {
    match def {
        "Message" => roundtrip::<Message>(value),
        "Block" => roundtrip::<Block>(value),
        "Tool" => roundtrip::<Tool>(value),
        "ChatRequest" => roundtrip::<ChatRequest>(value),
        "Response" => roundtrip::<Response>(value),
        "Event" => roundtrip::<Event>(value),
        "ProviderError" => roundtrip::<ProviderError>(value),
        "EmbedRequest" => roundtrip::<EmbedRequest>(value),
        "EmbedResponse" => roundtrip::<EmbedResponse>(value),
        "Capabilities" => roundtrip::<Capabilities>(value),
        "ModelInfo" => roundtrip::<ModelInfo>(value),
        "AuthStatus" => roundtrip::<AuthStatus>(value),
        "LoginAction" => roundtrip::<LoginAction>(value),
        "AuthSource" => roundtrip::<AuthSource>(value),
        "ProviderConfig" => roundtrip::<ProviderConfig>(value),
        "Start" => roundtrip::<Start>(value),
        "Resume" => roundtrip::<Resume>(value),
        "Step" => roundtrip::<Step>(value),
        "Input" => roundtrip::<Input>(value),
        "Effect" => roundtrip::<Effect>(value),
        "Notification" => roundtrip::<Notification>(value),
        other => Err(format!("no Rust type for {other}")),
    }
}

/// Loads every `*.json` sample under a directory, sorted by name.
pub fn load_samples(dir: &Path) -> Vec<(String, TypeSample)> {
    let mut entries: Vec<_> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("read {}: {e}", dir.display()))
        .map(|e| e.expect("dir entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    entries.sort();
    entries
        .into_iter()
        .map(|p| {
            let name = p.file_name().unwrap().to_string_lossy().into_owned();
            let text = std::fs::read_to_string(&p).expect("read sample");
            let sample = serde_json::from_str(&text).unwrap_or_else(|e| panic!("{name}: {e}"));
            (name, sample)
        })
        .collect()
}
