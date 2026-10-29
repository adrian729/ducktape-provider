//! Sans-I/O core: normalized types, codecs and the effects state machine.
#![forbid(unsafe_code)]

pub mod effects;
pub mod op;
pub mod secret;
pub mod types;

pub use op::Operation;
pub use secret::Secret;

/// Version of the JSON contract this core implements.
pub const SCHEMA_VERSION: &str = "1.0.0";
