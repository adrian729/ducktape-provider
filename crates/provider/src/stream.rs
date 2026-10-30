use crate::error::Error;
use crate::transport::BoxFuture;
use ducktape_core::effects::OperationResult;
use ducktape_core::types::{Event, Response};
use futures_core::Stream;
use std::pin::Pin;
use std::task::{Context, Poll};
use tokio::sync::mpsc::UnboundedReceiver;

/// A streamed chat reply: a stream of events, then the final response; dropping it cancels the call.
pub struct ChatStream {
    driver: Option<BoxFuture<'static, std::result::Result<OperationResult, Error>>>,
    events: UnboundedReceiver<Event>,
    response: Option<Response>,
    failure: Option<Error>,
}

impl ChatStream {
    pub(crate) fn new(
        driver: BoxFuture<'static, std::result::Result<OperationResult, Error>>,
        events: UnboundedReceiver<Event>,
    ) -> Self {
        ChatStream {
            driver: Some(driver),
            events,
            response: None,
            failure: None,
        }
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

    /// Stops the stream and releases the connection.
    pub fn cancel(self) {}
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
