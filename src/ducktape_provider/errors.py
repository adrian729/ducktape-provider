import http.client
import math
import urllib.error
from typing import NoReturn

from .types import (
    APIError,
    AuthError,
    ContextOverflowError,
    DucktapeError,
    MalformedResponseError,
    RateLimitError,
    RequestTimeoutError,
    ServerError,
    UnsupportedBlockError,
)

# Re-exported so existing `errors.X` references keep working.
__all__ = [
    "APIError",
    "AuthError",
    "ContextOverflowError",
    "DucktapeError",
    "MalformedResponseError",
    "RateLimitError",
    "RequestTimeoutError",
    "ServerError",
    "UnsupportedBlockError",
]

_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "context length",
    "maximum context length",
    "prompt is too long",
    "too many tokens",
)

# An error body only needs to be long enough to classify and explain the failure;
# a misbehaving server must not make us buffer an unbounded one.
_MAX_ERROR_BODY_BYTES = 64 * 1024
# How much of a body or vendor message is quoted in an exception's message, so a
# log line stays readable; the full (capped) body is still on `APIError.body`.
_MAX_MESSAGE_DETAIL_CHARS = 2048


def _parse_retry_after(e: urllib.error.HTTPError) -> float | None:
    header_value = e.headers.get("Retry-After") if e.headers else None
    if header_value is None:
        return None
    try:
        value = float(header_value)
    except ValueError:
        return None
    # nan/inf/negative would reach a caller's time.sleep(retry_after) and raise there.
    if not math.isfinite(value) or value < 0:
        return None
    return value


def _classify(
    message: str, status: int | None, body: str, retry_after: float | None = None
) -> APIError:
    if status in (401, 403):
        return AuthError(message, status=status, body=body)
    if status == 429:
        return RateLimitError(
            message, status=status, body=body, retry_after=retry_after
        )
    if status is not None and status >= 500:
        return ServerError(message, status=status, body=body, retry_after=retry_after)
    if status in (400, None) and any(
        m in body.lower() for m in _CONTEXT_OVERFLOW_MARKERS
    ):
        return ContextOverflowError(message, status=status, body=body)
    return APIError(message, status=status, body=body)


def _truncate(text: str) -> str:
    if len(text) <= _MAX_MESSAGE_DETAIL_CHARS:
        return text
    return f"{text[:_MAX_MESSAGE_DETAIL_CHARS]}... [truncated]"


def _cap_body(body: str) -> str:
    # Same cap as an HTTP error body (raise_for_http_error reads at most this many
    # bytes); a vendor error's body is already str, so bound length instead of bytes.
    return body[:_MAX_ERROR_BODY_BYTES]


def raise_for_http_error(vendor: str, e: urllib.error.HTTPError) -> NoReturn:
    # The status alone still classifies the error, so a body that can't be read
    # (timeout, connection dropped) must not replace it with a raw socket error.
    try:
        body = e.read(_MAX_ERROR_BODY_BYTES).decode(errors="replace")
    except (OSError, http.client.HTTPException, ValueError):
        body = ""
    finally:
        e.close()
    raise _classify(
        f"{vendor} chat failed: {e.code} {_truncate(body)}",
        e.code,
        body,
        _parse_retry_after(e),
    ) from e


def raise_for_vendor_error(
    vendor: str, message: str, *, status: int | None = None, body: str = ""
) -> NoReturn:
    """Raises for an error the vendor reported inside a 200 response body or stream.

    `status` is the HTTP code the vendor documents for the same error type when
    returned outside a stream, so both paths land on the same exception class.
    """
    raise _classify(
        f"{vendor} chat failed: {_truncate(message)}",
        status,
        _cap_body(body or message),
    )


def raise_for_connection_error(vendor: str, e: BaseException) -> NoReturn:
    if isinstance(e, TimeoutError) or (
        isinstance(e, urllib.error.URLError) and isinstance(e.reason, TimeoutError)
    ):
        raise RequestTimeoutError(f"{vendor} chat timed out") from e
    if isinstance(e, http.client.IncompleteRead):
        raise APIError(f"{vendor} chat failed: connection closed mid-response") from e
    reason = e.reason if isinstance(e, urllib.error.URLError) else e
    raise APIError(
        f"{vendor} chat failed: {str(reason) or type(reason).__name__}"
    ) from e


def raise_for_malformed_response(vendor: str, e: BaseException) -> NoReturn:
    # A bare KeyError/IndexError message is just the key, e.g. "'index'".
    detail = f"{type(e).__name__}: {e}" if isinstance(e, LookupError) else str(e)
    raise MalformedResponseError(
        f"{vendor} chat failed: malformed response: {_truncate(detail)}"
    ) from e


def raise_for_truncated_stream(vendor: str, terminal: str) -> NoReturn:
    # A plain APIError, not MalformedResponseError: what did arrive parsed fine,
    # so this is the connection ending early, which a retry can fix.
    raise APIError(f"{vendor} chat failed: stream ended before {terminal}")
