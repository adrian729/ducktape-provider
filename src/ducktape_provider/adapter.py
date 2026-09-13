"""The abstract per-vendor adapter interface every concrete adapter implements."""

import re
from typing import Any

from .types import Adapter

__all__ = ["Adapter"]

# A socket timeout must fit CPython's internal nanosecond clock (about 9.2e9 s), or
# urlopen fails with a raw OverflowError; inf and 1e300 would otherwise pass `> 0`.
# Anything near that is effectively "no timeout", which None already expresses.
_MAX_TIMEOUT = 1e9


def _validate_timeout(owner: str, timeout: object) -> None:
    """Raises ValueError unless `timeout` is None or a usable number of seconds."""
    # bool is an int subclass but never meant as a duration; NaN fails the chained
    # comparison.
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not 0 < timeout <= _MAX_TIMEOUT
    ):
        raise ValueError(
            f"{owner} timeout must be a positive number of seconds up to "
            f"{_MAX_TIMEOUT:,.0f}, or None for no timeout, got {timeout!r}"
        )


def _merge_config(
    vendor: str,
    payload: dict[str, Any],
    config: dict[str, Any] | None,
    reserved: frozenset[str],
    default_timeout: float,
) -> tuple[float | None, dict[str, str]]:
    """Merges `config` into the request body in place; returns its timeout and headers.

    `timeout` and `headers` configure the HTTP call itself, so they are split out
    instead of being sent to the vendor. Reserved keys are the ones the adapter's
    own parsing depends on (e.g. `stream`), where an override would break the call.
    An explicit `timeout: None` means no timeout, so only an absent key falls back
    to `default_timeout`.
    """
    config = dict(config or {})
    if clash := sorted(reserved & config.keys()):
        raise ValueError(
            f"{vendor} config cannot set {', '.join(clash)}: these come from the "
            "chat()/stream_chat() call itself (reserved: "
            f"{', '.join(sorted(reserved))})"
        )
    timeout = config.pop("timeout", default_timeout)
    _validate_timeout(f"{vendor} config", timeout)
    headers = config.pop("headers", None) or {}
    payload.update(config)
    return timeout, headers


# Mirror http.client's own checks, which would otherwise fail inside urlopen with
# the offending value (often an API key) quoted in the message.
_LEGAL_HEADER_NAME = re.compile(r"[^:\s][^:\r\n]*", re.ASCII)
_ILLEGAL_HEADER_VALUE = re.compile(r"\n(?![ \t])|\r(?![ \t\n])")


def _validate_headers(vendor: str, headers: dict[str, Any]) -> None:
    """Raises ValueError naming the bad header's key, never its value."""
    for key, value in headers.items():
        if (
            not isinstance(key, str)
            or not key.isascii()
            or not _LEGAL_HEADER_NAME.fullmatch(key)
        ):
            raise ValueError(f"{vendor} request header name {key!r} is invalid")
        if isinstance(value, int) and not isinstance(value, bool):
            continue
        if isinstance(value, bytes):
            value = value.decode("latin-1")
        elif not isinstance(value, str):
            raise TypeError(
                f"{vendor} request header {key!r} must be a string, "
                f"got {type(value).__name__}"
            )
        # A test rather than catching UnicodeEncodeError, whose context would
        # carry the value along with the ValueError raised here.
        if any(ord(c) > 0xFF for c in value):
            raise ValueError(
                f"{vendor} request header {key!r} contains a non-latin-1 character"
            )
        if _ILLEGAL_HEADER_VALUE.search(value):
            raise ValueError(f"{vendor} request header {key!r} contains a line break")
