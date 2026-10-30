//! The HTTP transport the effect loop sends through, and the reqwest default.
pub use ducktape_core::effects::{
    HttpRequest, Method, Purpose, TransportError, TransportErrorKind,
};
pub use ducktape_core::types::Headers;
use std::future::Future;
use std::pin::Pin;

/// A boxed future.
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// The body of a reply, pulled chunk by chunk; `None` at the end.
pub trait Body: Send {
    fn next(&mut self) -> BoxFuture<'_, std::result::Result<Option<Vec<u8>>, TransportError>>;
}

/// Status, headers and body of a reply once the headers arrived.
pub struct Reply {
    pub status: u16,
    pub headers: Headers,
    pub body: Box<dyn Body>,
}

/// Sends one request as the core built it; the loop applies the timeouts around it.
pub trait Transport: Send + Sync {
    fn send(
        &self,
        request: HttpRequest,
    ) -> BoxFuture<'_, std::result::Result<Reply, TransportError>>;
}

#[cfg(feature = "reqwest")]
pub use with_reqwest::ReqwestTransport;

#[cfg(feature = "reqwest")]
mod with_reqwest {
    use super::{
        Body, BoxFuture, Headers, HttpRequest, Method, Reply, Transport, TransportError,
        TransportErrorKind,
    };

    /// A transport over a reqwest client.
    pub struct ReqwestTransport {
        client: reqwest::Client,
    }

    impl ReqwestTransport {
        /// Wraps a client; redirects are not followed and no client-level timeout is set by default.
        pub fn new(client: reqwest::Client) -> Self {
            ReqwestTransport { client }
        }

        /// The default client.
        pub fn default_client() -> reqwest::Client {
            reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_default()
        }
    }

    impl Default for ReqwestTransport {
        fn default() -> Self {
            ReqwestTransport::new(ReqwestTransport::default_client())
        }
    }

    struct ReqwestBody(reqwest::Response);

    impl Body for ReqwestBody {
        fn next(&mut self) -> BoxFuture<'_, Result<Option<Vec<u8>>, TransportError>> {
            Box::pin(async move {
                self.0
                    .chunk()
                    .await
                    .map(|chunk| chunk.map(|b| b.to_vec()))
                    .map_err(classify)
            })
        }
    }

    pub(crate) fn classify(e: reqwest::Error) -> TransportError {
        let message = e.to_string();
        let mut chain = String::new();
        let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&e);
        while let Some(s) = source {
            chain.push_str(&s.to_string().to_lowercase());
            chain.push(' ');
            source = s.source();
        }
        let kind = if e.is_timeout() {
            TransportErrorKind::Timeout
        } else if chain.contains("dns")
            || chain.contains("resolve")
            || chain.contains("name or service not known")
        {
            TransportErrorKind::Dns
        } else if chain.contains("certificate")
            || chain.contains("tls")
            || chain.contains("ssl")
            || chain.contains("handshake")
        {
            TransportErrorKind::Tls
        } else if e.is_connect() {
            TransportErrorKind::Connect
        } else if chain.contains("reset")
            || chain.contains("broken pipe")
            || chain.contains("closed")
            || chain.contains("eof")
            || chain.contains("incomplete")
        {
            TransportErrorKind::Reset
        } else {
            TransportErrorKind::Other
        };
        TransportError { kind, message }
    }

    impl Transport for ReqwestTransport {
        fn send(&self, request: HttpRequest) -> BoxFuture<'_, Result<Reply, TransportError>> {
            Box::pin(async move {
                let method = match request.method {
                    Method::Get => reqwest::Method::GET,
                    Method::Post => reqwest::Method::POST,
                    Method::Put => reqwest::Method::PUT,
                    Method::Delete => reqwest::Method::DELETE,
                };
                let mut builder = self.client.request(method, &request.url);
                for (name, value) in &request.headers {
                    builder = builder.header(name.as_str(), value.as_str());
                }
                if let Some(body) = request.body {
                    builder = builder.body(body);
                }
                let response = builder.send().await.map_err(classify)?;
                let mut headers = Headers::new();
                for (name, value) in response.headers() {
                    let text = value.to_str().unwrap_or_default().to_owned();
                    let key = name.as_str().to_lowercase();
                    match headers.get_mut(&key) {
                        Some(existing) => {
                            existing.push_str(", ");
                            existing.push_str(&text);
                        }
                        None => {
                            headers.insert(key, text);
                        }
                    }
                }
                Ok(Reply {
                    status: response.status().as_u16(),
                    headers,
                    body: Box::new(ReqwestBody(response)),
                })
            })
        }
    }
}
