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
pub trait HttpBody: Send {
    fn next(&mut self) -> BoxFuture<'_, std::result::Result<Option<Vec<u8>>, TransportError>>;
}

/// Status, headers and body of a reply once the headers arrived.
pub struct Reply {
    pub status: u16,
    pub headers: Headers,
    pub body: Box<dyn HttpBody>,
}

impl std::fmt::Debug for Reply {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reply")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .finish_non_exhaustive()
    }
}

/// Sends one request as the core built it; the loop applies the timeouts around it.
pub trait HttpTransport: Send + Sync {
    fn send(
        &self,
        request: HttpRequest,
    ) -> BoxFuture<'_, std::result::Result<Reply, TransportError>>;
}

impl<T: HttpTransport + ?Sized> HttpTransport for std::sync::Arc<T> {
    fn send(
        &self,
        request: HttpRequest,
    ) -> BoxFuture<'_, std::result::Result<Reply, TransportError>> {
        (**self).send(request)
    }
}

/// Rejects header names and values the wire would refuse.
pub(crate) fn check_headers(headers: &Headers) -> std::result::Result<(), crate::error::Error> {
    for (name, value) in headers {
        let token = !name.is_empty()
            && name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b));
        if !token {
            return Err(crate::error::Error::argument(format!(
                "invalid header name {name:?}"
            )));
        }
        if value.bytes().any(|b| b == b'\r' || b == b'\n' || b == 0) {
            return Err(crate::error::Error::argument(format!(
                "invalid value for header {name}"
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "reqwest")]
pub use with_reqwest::ReqwestTransport;

#[cfg(feature = "reqwest")]
mod with_reqwest {
    use super::{
        BoxFuture, Headers, HttpBody, HttpRequest, HttpTransport, Method, Reply, TransportError,
        TransportErrorKind,
    };

    /// A transport over a reqwest client.
    pub struct ReqwestTransport {
        client: reqwest::Client,
    }

    impl ReqwestTransport {
        /// Wraps a client as configured by the caller, redirect policy and timeouts included.
        pub fn new(client: reqwest::Client) -> Self {
            ReqwestTransport { client }
        }

        /// The default client: no redirects, no client-level timeout.
        pub fn default_client() -> reqwest::Client {
            reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .expect("the default reqwest client builds")
        }
    }

    impl Default for ReqwestTransport {
        fn default() -> Self {
            ReqwestTransport::new(ReqwestTransport::default_client())
        }
    }

    struct ReqwestBody(reqwest::Response);

    impl HttpBody for ReqwestBody {
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
        let mut parts = Vec::new();
        let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&e);
        while let Some(s) = source {
            let text = s.to_string();
            if parts.last() != Some(&text) {
                parts.push(text);
            }
            source = s.source();
        }
        let message = parts.join(": ");
        let chain = message.to_lowercase();
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
        } else if e.is_body()
            || e.is_decode()
            || chain.contains("reset")
            || chain.contains("broken pipe")
            || chain.contains("closed")
            || chain.contains("eof")
            || chain.contains("end of file")
            || chain.contains("incomplete")
        {
            TransportErrorKind::Reset
        } else {
            TransportErrorKind::Other
        };
        TransportError { kind, message }
    }

    impl HttpTransport for ReqwestTransport {
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
                    let text = String::from_utf8_lossy(value.as_bytes()).into_owned();
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
