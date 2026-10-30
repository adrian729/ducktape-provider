//! Credential sources and header placement.
use crate::secret::Secret;
use crate::types::{Auth, AuthKind, AuthSource, Headers, Placement, Transport, Vendor};

/// A credential ready to be sent.
#[derive(Debug)]
pub struct Credential<'a> {
    pub kind: AuthKind,
    pub placement: Placement,
    pub secret: &'a Secret,
    pub label: String,
}

/// The vendor's own env sources, used only on its own base URL.
pub fn default_sources(vendor: &Vendor) -> Vec<AuthSource> {
    let env = |name: &str, kind: AuthKind, placement: Placement| AuthSource::Env {
        names: vec![name.to_owned()],
        transport: Transport {
            kind: Some(kind),
            placement: Some(placement),
        },
    };
    match vendor {
        Vendor::Anthropic => vec![
            env("ANTHROPIC_API_KEY", AuthKind::ApiKey, Placement::XApiKey),
            env(
                "ANTHROPIC_AUTH_TOKEN",
                AuthKind::Bearer,
                Placement::Authorization,
            ),
        ],
        Vendor::Openai => vec![env(
            "OPENAI_API_KEY",
            AuthKind::Bearer,
            Placement::Authorization,
        )],
        Vendor::Google => vec![
            env("GEMINI_API_KEY", AuthKind::ApiKey, Placement::XGoogApiKey),
            env("GOOGLE_API_KEY", AuthKind::ApiKey, Placement::XGoogApiKey),
        ],
        Vendor::Ollama => vec![AuthSource::Env {
            names: vec!["OLLAMA_API_KEY".to_owned()],
            transport: Transport {
                kind: Some(AuthKind::Bearer),
                placement: Some(Placement::Authorization),
            },
        }],
        Vendor::Other(_) => Vec::new(),
    }
}

/// Whether a provider without `auth` may fall back to the vendor's env sources.
pub fn env_fallback_allowed(vendor: &Vendor, base_url: &str) -> bool {
    match crate::op::default_base_url(vendor) {
        Some(default) => same_origin(default, base_url),
        None => false,
    }
}

/// The sources to try, in order.
pub fn sources(vendor: &Vendor, base_url: &str, auth: Option<&Auth>) -> Vec<AuthSource> {
    match auth {
        Some(a) if !a.sources.is_empty() => a.sources.clone(),
        _ if env_fallback_allowed(vendor, base_url) => default_sources(vendor),
        _ => Vec::new(),
    }
}

/// Default placement for a kind on a wire.
pub fn default_placement(vendor: &Vendor, kind: &AuthKind) -> Placement {
    match (vendor, kind) {
        (_, AuthKind::Bearer) => Placement::Authorization,
        (Vendor::Anthropic, _) => Placement::XApiKey,
        (Vendor::Google, _) => Placement::XGoogApiKey,
        _ => Placement::Authorization,
    }
}

/// Default kind for a source without one.
pub fn default_kind(vendor: &Vendor, source: &AuthSource) -> AuthKind {
    match source {
        AuthSource::Session { .. } | AuthSource::Exchange { .. } => AuthKind::Bearer,
        _ => match vendor {
            Vendor::Anthropic | Vendor::Google => AuthKind::ApiKey,
            _ => AuthKind::Bearer,
        },
    }
}

/// The transport settings a source declares.
pub fn transport_of(source: &AuthSource) -> Option<&Transport> {
    match source {
        AuthSource::Inline { transport, .. }
        | AuthSource::Env { transport, .. }
        | AuthSource::Resolver { transport, .. }
        | AuthSource::Exchange { transport, .. } => Some(transport),
        AuthSource::Session { .. } => None,
    }
}

/// Human label of a source, also reported as `origin.auth_source`.
pub fn label(source: &AuthSource) -> String {
    match source {
        AuthSource::Inline { .. } => "inline".to_owned(),
        AuthSource::Env { names, .. } => format!("env:{}", names.join(",")),
        AuthSource::Resolver { resolver_id, .. } => format!("resolver:{resolver_id}"),
        AuthSource::Session { preset, .. } => format!("session:{}", preset.as_str()),
        AuthSource::Exchange { grant, .. } => format!("exchange:{}", grant.as_str()),
    }
}

/// Writes the credential header(s) into `headers`.
pub fn place(headers: &mut Headers, cred: &Credential<'_>) {
    let Some(value) = cred.secret.expose_str() else {
        return;
    };
    match cred.kind {
        AuthKind::None => {}
        AuthKind::Bearer => {
            headers.insert("authorization".to_owned(), format!("Bearer {value}"));
        }
        _ => match &cred.placement {
            Placement::Authorization => {
                headers.insert("authorization".to_owned(), format!("Bearer {value}"));
            }
            other => {
                headers.insert(other.as_str().to_owned(), value.to_owned());
            }
        },
    }
}

fn same_origin(a: &str, b: &str) -> bool {
    origin(a) == origin(b)
}

fn origin(url: &str) -> (String, String) {
    let (scheme, rest) = url.split_once("://").unwrap_or(("", url));
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    (scheme.to_ascii_lowercase(), host.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_fallback_only_on_vendor_origin() {
        assert!(env_fallback_allowed(
            &Vendor::Openai,
            "https://api.openai.com/v1"
        ));
        assert!(!env_fallback_allowed(
            &Vendor::Openai,
            "https://api.groq.com/openai"
        ));
        assert!(env_fallback_allowed(
            &Vendor::Ollama,
            "http://127.0.0.1:11434"
        ));
    }
}
