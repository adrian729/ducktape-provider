use crate::error::Error;
use crate::transport::BoxFuture;
use ducktape_core::effects::OperationResult;
use ducktape_core::types::{Event, Response};
use futures_core::Stream;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::Notify;

#[derive(Default)]
struct CancelInner {
    flag: AtomicBool,
    notify: Notify,
}

/// A cloneable handle that cancels a call from anywhere; the call then fails with `cancelled`.
#[derive(Clone, Default)]
pub struct CancelToken(Arc<CancelInner>);

impl std::fmt::Debug for CancelToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CancelToken")
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

impl CancelToken {
    /// A token nobody has cancelled yet.
    pub fn new() -> Self {
        CancelToken::default()
    }

    /// Cancels every call holding this token.
    pub fn cancel(&self) {
        self.0.flag.store(true, Ordering::SeqCst);
        self.0.notify.notify_waiters();
    }

    /// Whether `cancel` was called.
    pub fn is_cancelled(&self) -> bool {
        self.0.flag.load(Ordering::SeqCst)
    }

    pub(crate) async fn cancelled(&self) {
        let notified = self.0.notify.notified();
        if self.is_cancelled() {
            return;
        }
        notified.await;
    }
}

/// A streamed chat reply: a stream of events, then the final response; dropping it cancels the call.
pub struct ChatStream {
    driver: Option<BoxFuture<'static, std::result::Result<OperationResult, Error>>>,
    events: UnboundedReceiver<Event>,
    cancel: CancelToken,
    response: Option<Response>,
    failure: Option<Error>,
}

impl std::fmt::Debug for ChatStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatStream")
            .field("running", &self.driver.is_some())
            .field("cancelled", &self.cancel.is_cancelled())
            .field("finished", &self.response.is_some())
            .finish()
    }
}

impl ChatStream {
    pub(crate) fn new(
        driver: BoxFuture<'static, std::result::Result<OperationResult, Error>>,
        events: UnboundedReceiver<Event>,
        cancel: CancelToken,
    ) -> Self {
        ChatStream {
            driver: Some(driver),
            events,
            cancel,
            response: None,
            failure: None,
        }
    }

    /// A handle that cancels this stream from another task.
    pub fn cancel_token(&self) -> CancelToken {
        self.cancel.clone()
    }

    /// The next event, or none once the stream ended.
    pub async fn next(&mut self) -> Option<std::result::Result<Event, Error>> {
        std::future::poll_fn(|cx| Pin::new(&mut *self).poll_next(cx)).await
    }

    /// The final response once the `finish` event went by.
    pub fn response(&self) -> Option<&Response> {
        self.response.as_ref()
    }

    /// Drains the remaining events and returns the final response.
    pub async fn finish(mut self) -> std::result::Result<Response, Error> {
        while let Some(event) = self.next().await {
            event?;
        }
        self.response.take().ok_or_else(|| {
            Error::provider(
                ducktape_core::types::ErrorKind::MalformedResponse,
                None,
                "the stream ended without a finish event",
            )
        })
    }

    /// Stops the stream; the next event is a `cancelled` error, and dropping the stream releases the connection.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

impl futures_core::Stream for ChatStream {
    type Item = std::result::Result<Event, Error>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        loop {
            match this.events.poll_recv(cx) {
                Poll::Ready(Some(event)) => {
                    if let Event::Finish { response, .. } = &event {
                        this.response = Some(response.clone());
                    }
                    return Poll::Ready(Some(Ok(event)));
                }
                Poll::Ready(None) => return Poll::Ready(this.failure.take().map(Err)),
                Poll::Pending => {}
            }
            match this.driver.as_mut() {
                None => return Poll::Ready(this.failure.take().map(Err)),
                Some(driver) => match driver.as_mut().poll(cx) {
                    Poll::Ready(result) => {
                        if let Err(e) = result {
                            this.failure = Some(e);
                        }
                        this.driver = None;
                    }
                    Poll::Pending => return Poll::Pending,
                },
            }
        }
    }
}
