//! One normalized interface over model vendors, on the sans-I/O core with an async effect loop; needs a tokio runtime with time enabled.
#![forbid(unsafe_code)]
#![allow(clippy::result_large_err, clippy::large_enum_variant)]

mod credential;
mod driver;
mod error;
pub mod keychain;
mod provider;
mod stream;
mod sugar;
pub mod transport;

pub use credential::{Credential, Resolver};
pub use ducktape_core::types::*;
pub use error::Error;
pub use provider::{Builder, Callback, ChatCall, EmbedCall, LoginCall, Observer, Provider};
pub use stream::{CancelToken, ChatStream};
pub use sugar::*;

/// Result of every call.
pub type Result<T> = std::result::Result<T, Error>;
