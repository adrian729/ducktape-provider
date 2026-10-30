use crate::transport::BoxFuture;
use ducktape_core::types::{
    AuthKind, AuthSource, Placement, RefreshMode, SessionPreset, Transport,
};
use std::sync::Arc;

/// Fetches a token at request time.
pub trait Resolver: Send + Sync {
    fn resolve(&self) -> BoxFuture<'_, std::result::Result<String, String>>;
}

impl<F, Fut, E> Resolver for F
where
    F: Fn() -> Fut + Send + Sync,
    Fut: std::future::Future<Output = std::result::Result<String, E>> + Send + 'static,
    E: std::fmt::Display,
{
    fn resolve(&self) -> BoxFuture<'_, std::result::Result<String, String>> {
        let fut = self();
        Box::pin(async move { fut.await.map_err(|e| e.to_string()) })
    }
}

/// One place a credential comes from.
#[derive(Clone)]
pub struct Credential {
    pub(crate) source: AuthSource,
    pub(crate) secret: Option<Vec<u8>>,
    pub(crate) resolver: Option<Arc<dyn Resolver>>,
}

impl std::fmt::Debug for Credential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credential")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

impl Credential {
    fn new(source: AuthSource) -> Self {
        Credential {
            source,
            secret: None,
            resolver: None,
        }
    }

    /// A key given in code.
    pub fn key(secret: impl Into<String>) -> Self {
        let mut c = Credential::new(AuthSource::Inline {
            secret_ref: String::new(),
            transport: Transport::default(),
        });
        c.secret = Some(secret.into().into_bytes());
        c
    }

    /// The first non-empty environment variable among the names.
    pub fn env<I, S>(names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Credential::new(AuthSource::Env {
            names: names.into_iter().map(Into::into).collect(),
            transport: Transport::default(),
        })
    }

    /// An existing login of another tool, or the library's own session.
    pub fn session(preset: impl Into<SessionPreset>) -> Self {
        Credential::new(AuthSource::Session {
            preset: preset.into(),
            store: None,
            client_id: None,
            profile: None,
            refresh: None,
        })
    }

    /// A token fetched at request time.
    pub fn resolver(resolver: impl Resolver + 'static) -> Self {
        let mut c = Credential::new(AuthSource::Resolver {
            resolver_id: String::new(),
            transport: Transport::default(),
        });
        c.resolver = Some(Arc::new(resolver));
        c
    }

    /// A source in the schema's own form.
    pub fn source(source: AuthSource) -> Self {
        Credential::new(source)
    }

    /// The file or directory a session preset reads instead of its default location.
    pub fn store(mut self, store: impl Into<String>) -> Self {
        if let AuthSource::Session { store: s, .. } = &mut self.source {
            *s = Some(store.into());
        }
        self
    }

    /// The OAuth client id of a session preset.
    pub fn client_id(mut self, client_id: impl Into<String>) -> Self {
        if let AuthSource::Session { client_id: c, .. } = &mut self.source {
            *c = Some(client_id.into());
        }
        self
    }

    /// The profile of a session preset.
    pub fn profile(mut self, profile: impl Into<String>) -> Self {
        if let AuthSource::Session { profile: p, .. } = &mut self.source {
            *p = Some(profile.into());
        }
        self
    }

    /// What to do with a refreshed session token.
    pub fn refresh(mut self, refresh: impl Into<RefreshMode>) -> Self {
        if let AuthSource::Session { refresh: r, .. } = &mut self.source {
            *r = Some(refresh.into());
        }
        self
    }

    /// The same credential sent as another header kind.
    pub fn kind(mut self, kind: impl Into<AuthKind>, placement: Option<Placement>) -> Self {
        let transport = match &mut self.source {
            AuthSource::Inline { transport, .. }
            | AuthSource::Env { transport, .. }
            | AuthSource::Resolver { transport, .. }
            | AuthSource::Exchange { transport, .. } => transport,
            AuthSource::Session { .. } => return self,
        };
        transport.kind = Some(kind.into());
        if placement.is_some() {
            transport.placement = placement;
        }
        self
    }
}
