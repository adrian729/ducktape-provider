"""The abstract per-vendor adapter interface every concrete adapter implements."""

import functools
import ipaddress
import os
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Collection, Mapping
from typing import Any, NoReturn, Self, cast

from .types import Adapter, AuthError

__all__ = ["Adapter"]

_API_KEY = re.compile(r"[\x21-\x7e]+")

_REDACTED = "<redacted>"


class _Secret:
    """An API key, or a function returning one, that never shows up in reprs,
    formatted strings, pickles or copies.

    The raw key is only ever read inline, as the argument that puts it on a
    request, so no frame of ours holds it in a local that a traceback, debugger
    or error reporter could capture.
    """

    __slots__ = ("_value",)

    _value: object

    def __init__(self, value: object) -> None:
        self._value = value

    def __repr__(self) -> str:
        return _REDACTED

    def __str__(self) -> str:
        return _REDACTED

    def __format__(self, format_spec: str) -> str:
        return _REDACTED

    def __reduce__(self) -> NoReturn:
        raise TypeError("an API key cannot be pickled")

    def __reduce_ex__(self, protocol: Any) -> NoReturn:
        raise TypeError("an API key cannot be pickled")

    def __getstate__(self) -> NoReturn:
        raise TypeError("an API key cannot be pickled")

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        return self

    def among(self, names: Collection[str]) -> str | None:
        """This value if it is one of `names`, which aren't secret; otherwise None."""
        if type(self._value) is str and self._value in names:
            return self._value
        return None

    def bind(self, *args: object) -> "_Secret":
        """A function source that calls this one with `args`."""
        return _Secret(
            functools.partial(cast(Callable[..., object], self._value), *args)
        )

    def resolve(self) -> "_Secret":
        """The key for one request: a function source is called, a str is itself."""
        if callable(self._value):
            return _Secret(self._value())
        return self

    def validate(self, vendor: str) -> None:
        """Checks a configured source: a function, or a usable str key."""
        if callable(self._value):
            return
        if not isinstance(self._value, str):
            raise TypeError(
                f"{vendor} API key must be a str or a function returning one, "
                f"got {type(self._value).__name__}"
            )
        if not self._value:
            raise ValueError(f"{vendor} API key is empty")
        self.check(vendor)

    def check(self, vendor: str) -> None:
        """Raises unless this is a key that can be sent: AuthError when missing."""
        if self._value is None or (isinstance(self._value, str) and not self._value):
            raise AuthError(f"{vendor} API key is missing")
        if not isinstance(self._value, str):
            raise TypeError(
                f"{vendor} API key must be a str, got {type(self._value).__name__}"
            )
        if not _API_KEY.fullmatch(self._value):
            raise ValueError(
                f"{vendor} API key contains whitespace, control or non-ASCII "
                "characters; check for a trailing newline"
            )

    def reveal(self) -> str:
        """The raw key. Only its type is checked here, so call `check` first."""
        if not isinstance(self._value, str):
            raise TypeError("API key was not checked before use")
        return self._value


def _key_url_allowed(url: str) -> bool:
    """Whether an API key may be sent to `url`.

    Only over https, or plain http to a literal loopback IP (a local gateway or
    test server) that no proxy sits in front of. `localhost` is excluded since
    name resolution can point it elsewhere, and userinfo since `a@b` is easily
    misread as host `a`.
    """
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
    except ValueError:
        return False
    if not host or "@" in parts.netloc:
        return False
    if parts.scheme == "https":
        return True
    if parts.scheme != "http":
        return False
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
    return loopback and (
        "http" not in urllib.request.getproxies()
        or bool(urllib.request.proxy_bypass(parts.netloc))
    )


def _current_key(source: _Secret | None, env_var: str) -> _Secret:
    """The key for one request; an explicit source means the env var is never read."""
    if source is not None:
        return source.resolve()
    return _Secret(os.environ.get(env_var))


def _request_key(
    vendor: str, url: str, source: _Secret | None, env_var: str
) -> _Secret:
    """The checked key to send to `url`; its source is only called if `url` is allowed."""
    if not _key_url_allowed(url):
        raise ValueError(
            f"{vendor} API key is only sent over https, or over http to a loopback "
            "IP address with no proxy; refusing to send it to the configured URL"
        )
    key = _current_key(source, env_var)
    key.check(vendor)
    return key


def _new_request(
    url: str,
    data: bytes | None = None,
    defaults: Mapping[str, str] | None = None,
    headers: Mapping[str, Any] | None = None,
    auth: tuple[str, str, Callable[[], _Secret]] | None = None,
) -> urllib.request.Request:
    """A request with `defaults`, then `headers`, then `auth` as unredirected headers.

    urllib copies `Request.headers` onto a redirected request to any host, even
    https to http, and merges the two header dicts with a different precedence
    on 3.12 than on 3.13+; keeping every header in one dict avoids both. `auth`
    is (header name, value prefix, key getter), skipped without calling the
    getter when `headers` already sets that header.
    """
    req = urllib.request.Request(url, data=data)
    for name, value in (defaults or {}).items():
        req.add_unredirected_header(name, value)
    for name, value in (headers or {}).items():
        req.add_unredirected_header(name, value)
    if auth is not None and not any(
        name.lower() == auth[0].lower() for name in headers or {}
    ):
        req.add_unredirected_header(auth[0], auth[1] + auth[2]().reveal())
    return req


_MAX_TIMEOUT = 1e9


def _validate_timeout(owner: str, timeout: object) -> None:
    """Raises ValueError unless `timeout` is None or a usable number of seconds."""
    if timeout is not None and (
        isinstance(timeout, bool)
        or not isinstance(timeout, int | float)
        or not 0 < timeout <= _MAX_TIMEOUT
    ):
        raise ValueError(
            f"{owner} timeout must be a positive number of seconds up to "
            f"{_MAX_TIMEOUT:,.0f}, or None for no timeout, got {timeout!r}"
        )


def _is_key_like(name: str) -> bool:
    """Whether a config key looks like an API key being passed the wrong way,
    e.g. `api_key`, `apiKey`, `OPENAI_API_KEY`, `x-goog-api-key`."""
    normalized = re.sub(r"[^a-z]", "", name.lower())
    return normalized.endswith(("apikey", "apikeys")) or normalized == "authorization"


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
    for key in config:
        if isinstance(key, str) and _is_key_like(key):
            raise ValueError(
                f"{vendor} config cannot set {key!r}: it would be sent in the request "
                "body; pass the key as api_key= on the adapter or "
                "Provider(api_keys=...)"
            )
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


_LEGAL_HEADER_NAME = re.compile(r"[^:\s][^:\r\n]*", re.ASCII)
_ILLEGAL_HEADER_VALUE = re.compile(r"\n(?![ \t])|\r(?![ \t\n])")


def _validate_headers(vendor: str, headers: dict[str, Any]) -> None:
    """Raises ValueError naming the bad header's key, never its value."""
    for key, value in headers.items():
        if (
            not isinstance(key, str)
            or not key.isascii()
            or key != key.strip()
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
        if any(ord(c) > 0xFF for c in value):
            raise ValueError(
                f"{vendor} request header {key!r} contains a non-latin-1 character"
            )
        if _ILLEGAL_HEADER_VALUE.search(value):
            raise ValueError(f"{vendor} request header {key!r} contains a line break")
