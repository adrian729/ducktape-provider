"""Errors raised by every call."""
from __future__ import annotations

import builtins
from typing import Any, Optional

from ._types import Origin


class ProviderError(Exception):
    """Base of every error the library raises; `kind` says which."""

    kind: str = "other"

    def __init__(
        self,
        message: str,
        *,
        kind: Optional[str] = None,
        status: Optional[int] = None,
        code: Optional[str] = None,
        retryable: bool = False,
        retry_after: Optional[float] = None,
        attempts: int = 1,
        body: Any = None,
        request_id: Optional[str] = None,
        origin: Optional[Origin] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if kind is not None:
            self.kind = kind
        self.status = status
        self.code = code
        self.retryable = retryable
        self.retry_after = retry_after
        self.attempts = attempts
        self.body = body
        self.request_id = request_id
        self.origin = origin

    def __str__(self) -> str:
        parts = [self.kind]
        if self.code:
            parts.append(self.code)
        if self.status:
            parts.append(str(self.status))
        return f"[{' '.join(parts)}] {self.message}"


class AuthError(ProviderError):
    """No credential, or the vendor rejected it."""

    kind = "auth"


class RateLimitError(ProviderError):
    """Rate limited with a retry window."""

    kind = "rate_limit"


class QuotaError(ProviderError):
    """Billing or spend limit reached."""

    kind = "quota"


class ServerError(ProviderError):
    """The vendor failed."""

    kind = "server"


class NetworkError(ProviderError):
    """DNS, connection, TLS or a dropped reply."""

    kind = "connection"


class TimeoutError(ProviderError, builtins.TimeoutError):
    """A timeout was exceeded."""

    kind = "timeout"


class CancelledError(ProviderError):
    """The call was cancelled."""

    kind = "cancelled"


class NotFoundError(ProviderError):
    """Unknown model or endpoint."""

    kind = "not_found"


class ContextOverflowError(ProviderError):
    """The input is too long for the model."""

    kind = "context_overflow"


class RequestTooLargeError(ProviderError):
    """The request body is over the vendor's limit."""

    kind = "request_too_large"


class BadRequestError(ProviderError):
    """The vendor rejected the request."""

    kind = "bad_request"


class MalformedResponseError(ProviderError):
    """The vendor's reply could not be parsed."""

    kind = "malformed_response"


class UnsupportedError(ProviderError):
    """The provider cannot do what was asked."""

    kind = "unsupported"


_BY_KIND: dict[str, type[ProviderError]] = {
    cls.kind: cls
    for cls in (
        AuthError,
        RateLimitError,
        QuotaError,
        ServerError,
        NetworkError,
        TimeoutError,
        CancelledError,
        NotFoundError,
        ContextOverflowError,
        RequestTooLargeError,
        BadRequestError,
        MalformedResponseError,
        UnsupportedError,
    )
}


def error_from_dict(data: dict[str, Any]) -> ProviderError:
    """Builds the matching error class from the core's error data."""
    kind = data.get("kind", "other")
    cls = _BY_KIND.get(kind, ProviderError)
    origin = data.get("origin")
    return cls(
        data.get("message", ""),
        kind=kind,
        status=data.get("status"),
        code=data.get("code"),
        retryable=bool(data.get("retryable", False)),
        retry_after=data.get("retry_after"),
        attempts=int(data.get("attempts", 1)),
        body=data.get("body"),
        request_id=data.get("request_id"),
        origin=Origin.from_dict(origin) if origin else None,
    )
