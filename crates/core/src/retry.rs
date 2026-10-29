//! Backoff policy shared by every wire.
use crate::types::{ErrorKind, ProviderError};

/// Longest `retry-after` the core will honor, in seconds.
pub const MAX_RETRY_AFTER_SECS: f64 = 60.0;

/// Whether an error may be retried before any response byte was consumed.
pub fn is_retryable(error: &ProviderError) -> bool {
    error.retryable
        && matches!(
            error.kind,
            ErrorKind::RateLimit | ErrorKind::Server | ErrorKind::Connection | ErrorKind::Timeout
        )
        && error.retry_after.is_none_or(|s| s <= MAX_RETRY_AFTER_SECS)
}

/// Milliseconds to wait before attempt `attempt` (1 = first retry), with jitter from one entropy byte.
pub fn backoff_ms(attempt: u32, retry_after: Option<f64>, entropy: u8) -> u64 {
    if let Some(secs) = retry_after {
        return (secs.max(0.0) * 1000.0) as u64;
    }
    let base = 500.0 * 2f64.powi(attempt.saturating_sub(1) as i32);
    let capped = base.min(8000.0);
    let jitter = 0.75 + 0.5 * (entropy as f64 / 255.0);
    (capped * jitter) as u64
}

/// Parses a `retry-after` header value in seconds.
pub fn parse_retry_after(value: &str) -> Option<f64> {
    value.trim().parse::<f64>().ok().filter(|s| *s >= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backoff_grows_and_caps() {
        assert_eq!(backoff_ms(1, None, 128), 500);
        assert!(backoff_ms(2, None, 0) >= 750);
        assert!(backoff_ms(10, None, 255) <= 10000);
        assert_eq!(backoff_ms(1, Some(2.5), 0), 2500);
    }
}
