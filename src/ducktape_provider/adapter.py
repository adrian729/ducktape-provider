"""The abstract per-vendor adapter interface every concrete adapter implements."""

import functools
import ipaddress
import logging
import os
import re
import sys
import urllib.parse
import urllib.request
from collections.abc import Callable, Collection, Mapping
from typing import Any, NoReturn, Self, cast

from .streaming import _clear_tracebacks
from .types import Adapter, AuthError, SystemBlock

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

    def sets(self, lowered_names: Collection[str]) -> bool:
        """Whether this configured header's name, lowercased, is one of
        `lowered_names` (already-lowercased ASCII names). Used both to check a
        call header override and to check a Provider auth header. Reads only
        the name out of the tuple — never binds the value to a local, unlike
        `header`/`validate_header`, which do and clean up after themselves."""
        return cast(tuple[str, object], self._value)[0].lower() in lowered_names

    def header(self, vendor: str) -> "_Secret":
        """The header for one request: a function source is called (its own
        exceptions propagate unchanged); the resulting value is then checked
        (non-empty str, latin-1, no CR/LF/NUL) and returned as a fresh, resolved
        `_Secret((name, value))`. Raises `_BadHeaderType`/`_BadHeaderValue`
        without the header's name or value — never a plain `TypeError`/
        `ValueError`, so a caller can tell a bad configured value apart from
        whatever its own source function raised. `value` is deleted before any
        raise, in `finally`, so this frame never holds it as a local on the
        exception's traceback.
        """
        name, value = cast(tuple[str, object], self._value)
        if callable(value):
            value = value()
        try:
            if not isinstance(value, str):
                raise _BadHeaderType(
                    f"{vendor} configured header value must be a str or a "
                    f"function returning one, got {type(value).__name__}"
                )
            if not value or any(ord(c) > 0xFF or c in "\r\n\x00" for c in value):
                raise _BadHeaderValue(f"{vendor} configured header value is invalid")
            result = _Secret((name, value))
        finally:
            del value
        return result

    def pair(self) -> tuple[str, str]:
        """The resolved (name, value) pair, only ever unpacked inline as
        `*secret.pair()` — never bound to a `name, value` loop variable, which
        would leave the value in a frame local."""
        return cast(tuple[str, str], self._value)

    def validate_header(self, owner: str) -> None:
        """Checks a configured header source at construction: a legal str name,
        and a str-or-function value that is valid on its face (a function's
        return value is checked later, per request, by `header`). Raises
        without the header's name or value; `value` is deleted before any
        raise, in `finally`, so this frame never holds it as a local on the
        exception's traceback.
        """
        name, value = cast(tuple[object, object], self._value)
        if type(name) is not str:
            raise TypeError(
                f"{owner} has a configured header whose name must be a str, "
                f"got {type(name).__name__}"
            )
        if not _header_name_ok(name):
            raise ValueError(f"{owner} has a configured header with an invalid name")
        try:
            if callable(value):
                return
            if not isinstance(value, str):
                raise TypeError(
                    f"{owner} has a configured header whose value must be a "
                    f"str or a function returning one, got {type(value).__name__}"
                )
            if not value or any(ord(c) > 0xFF or c in "\r\n\x00" for c in value):
                raise ValueError(
                    f"{owner} has a configured header with an invalid value"
                )
        finally:
            del value


class _BadConfiguredHeader(Exception):
    """Base for a configured header failing `_Secret.header`'s own check, as
    opposed to an exception raised by the header's own source function. A
    probe (`models()`, Ollama `is_available()`) catches only these, so a
    header function's own `TypeError`/`ValueError` still propagates."""


class _BadHeaderType(_BadConfiguredHeader, TypeError):
    pass


class _BadHeaderValue(_BadConfiguredHeader, ValueError):
    pass


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
    resolved: Collection[_Secret] | None = None,
    auth: tuple[str, str, Callable[[], _Secret]] | None = None,
) -> urllib.request.Request:
    """A request with, in order, `defaults`, `resolved`, `headers` and `auth` as
    unredirected headers.

    urllib copies `Request.headers` onto a redirected request to any host, even
    https to http, and merges the two header dicts with a different precedence
    on 3.12 than on 3.13+; keeping every header in one dict avoids both.
    `resolved` is the Provider's configured headers, already resolved and
    checked by `_resolve_headers`; each is only ever unpacked inline as
    `*secret.pair()`, never bound to a `name, value` loop variable, which would
    leave the value in a frame local. `auth` is (header name, value prefix, key
    getter), skipped without calling the getter when `resolved` or `headers`
    already sets that header. The whole body is wrapped so a caller interrupted
    (e.g. Ctrl-C) partway never gets back a half-built `req` holding some but
    not all of its secrets, with no `req` local of our own left over either;
    `stop` is captured first, so an interruption here doesn't wipe the
    traceback of an exception the caller is already handling.
    """
    stop = sys.exception()
    try:
        req = urllib.request.Request(url, data=data)
        for name, value in (defaults or {}).items():
            req.add_unredirected_header(name, value)
        for secret in resolved or ():
            req.add_unredirected_header(*secret.pair())
        for name, value in (headers or {}).items():
            req.add_unredirected_header(name, value)
        if auth is not None and not any(
            name.lower() == auth[0].lower()
            for name in (*(headers or {}), *(s.pair()[0] for s in resolved or ()))
        ):
            req.add_unredirected_header(auth[0], auth[1] + auth[2]().reveal())
        return req
    except BaseException as e:
        _clear_tracebacks(e, stop)
        raise


def _resolve_headers(
    vendor: str,
    url: str,
    provider_headers: Collection[_Secret],
    call_header_names: Collection[str],
) -> list[_Secret]:
    """The Provider's configured headers to send with one request to `url`.

    Drops any a call header overrides (case-insensitive, ASCII names), without
    calling its function. If any remain and `url` isn't allowed to carry them
    (see `_key_url_allowed`), raises `ValueError`. Otherwise resolves and checks
    each remaining header (see `_Secret.header`); called once per request, by
    `_build_request` for chat/stream and once per `models()`/`is_available()`
    call (covering every page of Claude's `models()`).
    """
    lowered_call_names = {n.lower() for n in call_header_names if isinstance(n, str)}
    remaining = [h for h in provider_headers if not h.sets(lowered_call_names)]
    if remaining and not _key_url_allowed(url):
        raise ValueError(
            f"{vendor} configured headers are only sent over https, or over http "
            "to a loopback IP address with no proxy; refusing to send them to "
            "the configured URL"
        )
    return [h.header(vendor) for h in remaining]


def _resolve_probe_auth(
    vendor: str,
    url: str,
    key_source: _Secret | None,
    env_var: str,
    auth_header: str,
    provider_headers: Collection[_Secret],
) -> tuple[_Secret | None, list[_Secret]] | None:
    """The (key, resolved headers) to probe `url` with, or `None` when the
    caller should report the vendor unavailable/empty instead of raising: `url`
    can't carry a key or configured headers, the key is missing or invalid
    (unless a configured auth header already covers it), or a configured
    header is bad.

    Shared by `ClaudeAdapter` and `OpenAIAdapter`'s `models()`, so a change to
    this precedence (a configured auth header replacing the key check) can't
    be applied to one and not the other.
    """
    if not _key_url_allowed(url):
        return None
    auth_configured = any(h.sets({auth_header}) for h in provider_headers)
    key: _Secret | None = None
    if not auth_configured:
        key = _current_key(key_source, env_var)
        try:
            key.check(vendor)
        except (AuthError, TypeError, ValueError):
            return None
    try:
        headers = _resolve_headers(vendor, url, provider_headers, ())
    except _BadConfiguredHeader:
        return None
    return key, headers


def _warn_headers_refused(adapter: Any, logger_: logging.Logger, vendor: str) -> None:
    """Logs once per adapter instance that its configured headers can't be sent
    to the resolved URL, naming the provider (or `vendor`) only — never the URL,
    which may carry userinfo, and never a header's name or value."""
    if adapter._warned_transport:
        return
    adapter._warned_transport = True
    logger_.warning(
        "provider %r has configured headers, but its request URL is not https "
        "or an unproxied loopback address; not sending them",
        adapter._provider_name or vendor,
    )


_MAX_TIMEOUT = 1e9


def _system_text(system: str | list[SystemBlock]) -> str:
    """Flattens a system prompt to text for vendors without block caching."""
    if isinstance(system, str):
        return system
    return "\n".join(block["text"] for block in system)


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
    *,
    operation: str = "chat",
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
        caller = "chat()/stream_chat()" if operation == "chat" else f"{operation}()"
        raise ValueError(
            f"{vendor} config cannot set {', '.join(clash)}: these come from the "
            f"{caller} call itself (reserved: {', '.join(sorted(reserved))})"
        )
    timeout = config.pop("timeout", default_timeout)
    _validate_timeout(f"{vendor} config", timeout)
    headers = config.pop("headers", None) or {}
    payload.update(config)
    return timeout, headers


_LEGAL_HEADER_NAME = re.compile(r"[^:\s]+", re.ASCII)
_ILLEGAL_HEADER_VALUE = re.compile(r"\n(?![ \t])|\r(?![ \t\n])")


def _header_name_ok(name: str) -> bool:
    """Whether `name` is a legal, ASCII HTTP header name: non-empty, with no
    colon or whitespace (leading, trailing or internal)."""
    return name.isascii() and bool(_LEGAL_HEADER_NAME.fullmatch(name))


def _validate_headers(vendor: str, headers: dict[str, Any]) -> None:
    """Raises ValueError naming the bad header's key, never its value."""
    for key, value in headers.items():
        if not isinstance(key, str) or not _header_name_ok(key):
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
