use std::collections::HashMap;
use std::fmt;
use zeroize::Zeroizing;

/// A credential value that is wiped on drop and never printed.
pub struct Secret(Zeroizing<Vec<u8>>);

impl Secret {
    /// Wraps the bytes, taking ownership.
    pub fn new(bytes: Vec<u8>) -> Self {
        Secret(Zeroizing::new(bytes))
    }

    /// The raw bytes, for building a request header.
    pub fn expose(&self) -> &[u8] {
        &self.0
    }

    /// The bytes as text when they are valid UTF-8.
    pub fn expose_str(&self) -> Option<&str> {
        std::str::from_utf8(&self.0).ok()
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("***")
    }
}

impl fmt::Display for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("***")
    }
}

/// Secrets of one operation, keyed by `secret_ref`.
#[derive(Debug, Default)]
pub struct SecretStore {
    slots: HashMap<String, Secret>,
}

impl SecretStore {
    /// Stores or replaces the value for a reference.
    pub fn set(&mut self, secret_ref: &str, bytes: Vec<u8>) {
        self.slots.insert(secret_ref.to_owned(), Secret::new(bytes));
    }

    /// Looks a reference up.
    pub fn get(&self, secret_ref: &str) -> Option<&Secret> {
        self.slots.get(secret_ref)
    }

    /// Whether a reference has a value.
    pub fn contains(&self, secret_ref: &str) -> bool {
        self.slots.contains_key(secret_ref)
    }
}
