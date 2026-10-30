use ducktape_core::types::{ErrorKind, ProviderError};
use std::fmt;

/// Why a call failed: a mistake in the arguments, or an error from the provider or the host.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    Argument(String),
    Provider(ProviderError),
}

impl Error {
    pub(crate) fn argument(message: impl Into<String>) -> Self {
        Error::Argument(message.into())
    }

    pub(crate) fn provider(
        kind: ErrorKind,
        code: Option<&str>,
        message: impl Into<String>,
    ) -> Self {
        let mut error = ProviderError::new(kind, message);
        error.code = code.map(str::to_owned);
        Error::Provider(error)
    }

    /// The error kind when the provider failed, none for an argument mistake.
    pub fn kind(&self) -> Option<&ErrorKind> {
        match self {
            Error::Argument(_) => None,
            Error::Provider(e) => Some(&e.kind),
        }
    }

    /// Whether this is a provider error of the given kind.
    pub fn is(&self, kind: ErrorKind) -> bool {
        self.kind() == Some(&kind)
    }

    /// The normalized error when the provider failed.
    pub fn provider_error(&self) -> Option<&ProviderError> {
        match self {
            Error::Argument(_) => None,
            Error::Provider(e) => Some(e),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Argument(m) => write!(f, "argument error: {m}"),
            Error::Provider(e) => match &e.code {
                Some(code) => write!(f, "[{} {}] {}", e.kind.as_str(), code, e.message),
                None => write!(f, "[{}] {}", e.kind.as_str(), e.message),
            },
        }
    }
}

impl std::error::Error for Error {}

impl From<ProviderError> for Error {
    fn from(e: ProviderError) -> Self {
        Error::Provider(e)
    }
}
