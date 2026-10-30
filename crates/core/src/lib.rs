//! Sans-I/O core: normalized types, codecs and the effects state machine.
#![forbid(unsafe_code)]
#![allow(clippy::result_large_err, clippy::large_enum_variant)]

pub mod auth;
pub mod effects;
pub mod login;
pub mod op;
pub mod retry;
pub mod secret;
pub mod session;
pub mod sse;
pub mod types;
pub mod wire;

pub use op::Operation;
pub use secret::Secret;

/// Version of the JSON contract this core implements.
pub const SCHEMA_VERSION: &str = "1.0.0";
