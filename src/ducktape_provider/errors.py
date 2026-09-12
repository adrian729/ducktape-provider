import urllib.error
from typing import NoReturn

_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "context length",
    "maximum context length",
    "prompt is too long",
    "too many tokens",
)


class DucktapeError(Exception):
    pass


class APIError(DucktapeError):
    def __init__(self, message: str, *, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class AuthError(APIError):
    pass


class RateLimitError(APIError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        retry_after: float | None = None,
    ):
        super().__init__(message, status=status, body=body)
        self.retry_after = retry_after


class ServerError(APIError):
    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        body: str = "",
        retry_after: float | None = None,
    ):
        super().__init__(message, status=status, body=body)
        self.retry_after = retry_after


class ContextOverflowError(APIError):
    pass


class RequestTimeoutError(APIError):
    pass


class UnsupportedBlockError(DucktapeError):
    pass


def _parse_retry_after(e: urllib.error.HTTPError) -> float | None:
    header_value = e.headers.get("Retry-After") if e.headers else None
    if header_value is None:
        return None
    try:
        return float(header_value)
    except ValueError:
        return None


def raise_for_http_error(vendor: str, e: urllib.error.HTTPError) -> NoReturn:
    body = e.read().decode(errors="replace")
    message = f"{vendor} chat failed: {e.code} {body}"
    if e.code in (401, 403):
        raise AuthError(message, status=e.code, body=body) from e
    if e.code == 429:
        raise RateLimitError(
            message, status=e.code, body=body, retry_after=_parse_retry_after(e)
        ) from e
    if e.code >= 500:
        raise ServerError(
            message, status=e.code, body=body, retry_after=_parse_retry_after(e)
        ) from e
    if e.code == 400 and any(m in body.lower() for m in _CONTEXT_OVERFLOW_MARKERS):
        raise ContextOverflowError(message, status=e.code, body=body) from e
    raise APIError(message, status=e.code, body=body) from e


def raise_for_connection_error(vendor: str, e: BaseException) -> NoReturn:
    if isinstance(e, TimeoutError) or (
        isinstance(e, urllib.error.URLError) and isinstance(e.reason, TimeoutError)
    ):
        raise RequestTimeoutError(f"{vendor} chat timed out") from e
    reason = e.reason if isinstance(e, urllib.error.URLError) else e
    raise APIError(f"{vendor} chat failed: {reason}") from e
