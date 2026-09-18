import asyncio
import concurrent.futures
import contextvars
import copy
import enum
import functools
import importlib.metadata
import logging
import re
import sys
import threading
import urllib.error
from collections.abc import AsyncGenerator, Callable, Collection, Iterator, Mapping
from typing import Any, Literal, TypeVar

from .adapter import Adapter, _is_key_like, _Secret, _validate_timeout
from .adapters.claude import ClaudeAdapter
from .adapters.ollama import OllamaLocalAdapter
from .adapters.openai import OpenAIAdapter
from .errors import APIError, AuthError
from .streaming import _clear_tracebacks
from .types import Config, Message, ModelInfo, Response, StreamEvent, ToolDef

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "ducktape_provider.adapters"

_STREAM_BUFFER_SIZE = 64
_READER_POLL_SECONDS = 1.0

_EVENT = "event"
_ERROR = "error"
_DONE = "done"

_T = TypeVar("_T")


class _Default(enum.Enum):
    TIMEOUT = enum.auto()

    def __repr__(self) -> str:
        return "<adapter default>"


def _normalize_dist(name: str) -> str:
    """PEP 503 form, so `Typing_Extensions` and `typing-extensions` name the same distribution."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _dist_name(entry_point: importlib.metadata.EntryPoint) -> str | None:
    dist = entry_point.dist
    return _normalize_dist(dist.name) if dist is not None else None


def _normalize_allowlist_entry(entry: str) -> str:
    dist, sep, name = entry.partition(":")
    return f"{_normalize_dist(dist)}:{name}" if sep else entry


def _probe_available(name: str, adapter: Adapter) -> bool:
    """adapter.is_available(), treating a raise the same as a False return."""
    try:
        return adapter.is_available()
    except Exception:
        logger.warning("provider %r raised from is_available()", name, exc_info=True)
        return False


def _usable_models(name: str, adapter: Adapter) -> list[str] | None:
    """Sorted adapter.models(), or None if the adapter is unavailable or raises."""
    if not _probe_available(name, adapter):
        return None
    try:
        return sorted(adapter.models())
    except Exception:
        logger.warning("provider %r raised while listing models", name, exc_info=True)
        return None


def _should_evict_cache(exc: APIError) -> bool:
    """Whether an auto-matched call's failure means the cached provider is bad,
    not just the request. A 404 means the model moved off that provider; an
    AuthError means the provider itself rejected us. A plain status-None
    APIError also covers truncated streams, mid-response IncompleteRead, and
    unmapped vendor error events — all cases where the provider was reached and
    answered, just badly — so those evict only when the underlying failure was
    a connect/DNS/TLS-handshake URLError (raise_for_connection_error's `from e`),
    meaning the provider was never actually reached.
    """
    if exc.status == 404:
        return True
    if isinstance(exc, AuthError):
        return True
    return (
        type(exc) is APIError
        and exc.status is None
        and isinstance(exc.__cause__, urllib.error.URLError)
    )


@functools.cache
def _scan_entry_points() -> tuple[importlib.metadata.EntryPoint, ...]:
    """Adapter entry points in a stable order, so same-name plugins always resolve the same way."""
    return tuple(
        sorted(
            importlib.metadata.entry_points(group=ENTRY_POINT_GROUP),
            key=lambda ep: (ep.name, _dist_name(ep) or "", ep.value),
        )
    )


_loaded_adapter_classes: dict[importlib.metadata.EntryPoint, type[Adapter]] = {}


def _load_adapter_class(
    entry_point: importlib.metadata.EntryPoint,
) -> type[Adapter] | None:
    """The entry point's Adapter subclass, or None (warned) if it can't be used."""
    if (cached := _loaded_adapter_classes.get(entry_point)) is not None:
        return cached
    dist = _dist_name(entry_point)
    try:
        loaded = entry_point.load()
    except Exception:
        logger.warning(
            "Skipping third-party adapter %r from %s: failed to load",
            entry_point.name,
            dist,
            exc_info=True,
        )
        return None
    if not (isinstance(loaded, type) and issubclass(loaded, Adapter)):
        logger.warning(
            "Skipping third-party adapter %r from %s: %r is not an Adapter subclass",
            entry_point.name,
            dist,
            loaded,
        )
        return None
    _loaded_adapter_classes[entry_point] = loaded
    return loaded


def _clear_discovery_cache() -> None:
    _scan_entry_points.cache_clear()
    _loaded_adapter_classes.clear()


_HEADERED_ADAPTER_CLASSES = (ClaudeAdapter, OpenAIAdapter, OllamaLocalAdapter)


def _snapshot_config(config: Config | Mapping[str, Any] | None) -> dict[str, Any]:
    """A frame-safe snapshot of `Provider(config=...)`, read with one `items()`
    pass per container.

    A header (legal only under `providers[name]["headers"]`, since it must
    never reach another vendor or plugin) is wrapped as a `_Secret((name,
    value))` the moment it is read off its mapping, then checked through
    `_Secret.validate_header` — so no free function here ever takes a header's
    name or value as a parameter, and neither ever sits in a frame local as
    plain text. Every other key is `copy.deepcopy`d. Runs inside
    `Provider.__init__`'s guarded step.
    """
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError(
            f"Provider config must be a Mapping, not a {type(config).__name__}"
        )
    top = dict(config.items())
    if "headers" in top:
        raise ValueError(
            "Provider config cannot set headers at the top level; set them per "
            "provider instead, as config['providers'][name]['headers'], so a "
            "header never reaches another vendor or plugin"
        )
    providers_in = top.pop("providers", None)
    snapshot = {k: copy.deepcopy(v) for k, v in top.items()}
    if providers_in is not None:
        snapshot["providers"] = _snapshot_providers(providers_in)
    return snapshot


def _snapshot_providers(providers_in: object) -> dict[str, dict[str, Any]]:
    if not isinstance(providers_in, Mapping):
        raise TypeError(
            "Provider config['providers'] must be a Mapping, not a "
            f"{type(providers_in).__name__}"
        )
    out: dict[str, dict[str, Any]] = {}
    for name, entry in dict(providers_in.items()).items():
        if type(name) is not str:
            raise TypeError(
                "Provider config['providers'] names must be str, got "
                f"{type(name).__name__}"
            )
        out[name] = _snapshot_provider_entry(name, entry)
    return out


def _snapshot_provider_entry(name: str, entry: object) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise TypeError(
            f"Provider config['providers'][{name!r}] must be a Mapping, not a "
            f"{type(entry).__name__}"
        )
    items = dict(entry.items())
    headers_in = items.pop("headers", None)
    out = {k: copy.deepcopy(v) for k, v in items.items()}
    if headers_in is not None:
        out["headers"] = _snapshot_headers(name, headers_in)
    return out


def _snapshot_headers(provider: str, headers_in: object) -> tuple[_Secret, ...]:
    if not isinstance(headers_in, Mapping):
        raise TypeError(
            f"Provider config['providers'][{provider!r}]['headers'] must be a "
            f"Mapping, not a {type(headers_in).__name__}"
        )
    secrets = [_Secret(pair) for pair in dict(headers_in.items()).items()]
    for secret in secrets:
        secret.validate_header(provider)
    return tuple(secrets)


def _check_non_header_keys(
    owner: str, entry: Mapping[str, Any], reserved: frozenset[str]
) -> None:
    """Raises for a top-level or per-provider config key that isn't a plain
    vendor field: `timeout` is validated, `providers`/`headers` are skipped
    (handled elsewhere), a key-like key or one reserved by the adapter raises
    ValueError."""
    for key in entry:
        if key in ("providers", "headers"):
            continue
        if key == "timeout":
            _validate_timeout(owner, entry[key])
            continue
        if isinstance(key, str) and _is_key_like(key):
            raise ValueError(
                f"{owner} cannot set {key!r}: it looks like an API key; pass it "
                "as api_key= on the adapter or Provider(api_keys=...) instead"
            )
        if key in reserved:
            raise ValueError(
                f"{owner} cannot set {key!r}: it is reserved for the "
                "chat()/stream_chat() call itself"
            )


def _validate_provider_config(
    adapters: Mapping[str, Adapter], snapshot: Mapping[str, Any]
) -> None:
    """Checks a config snapshot against the registered adapters: provider
    names, reserved/key-like config keys, and (for headers) that the target is
    a built-in adapter or a subclass of one. Touches provider names and
    non-header keys only — headers were already checked while the snapshot
    was taken, in `_snapshot_config`.
    """
    reserved_union: frozenset[str] = frozenset[str]().union(
        *(getattr(a, "_RESERVED_CONFIG", frozenset()) for a in adapters.values())
    )
    _check_non_header_keys("Provider config", snapshot, reserved_union)
    providers = snapshot.get("providers") or {}
    if unknown := sorted(set(providers) - set(adapters)):
        registered = ", ".join(repr(n) for n in adapters) or "none"
        names = ", ".join(repr(n) for n in unknown)
        verb = "is" if len(unknown) == 1 else "are"
        raise ValueError(
            f"Provider config['providers'] names {names}, which {verb} not a "
            f"registered provider; registered providers: {registered}"
        )
    for name, entry in providers.items():
        adapter = adapters[name]
        if entry.get("headers") and not isinstance(adapter, _HEADERED_ADAPTER_CLASSES):
            raise ValueError(
                f"provider {name!r} cannot take configured headers: only "
                "built-in adapters (ClaudeAdapter, OpenAIAdapter, "
                "OllamaLocalAdapter, or a subclass) support them; pass them "
                "per call instead"
            )
        _check_non_header_keys(
            f"provider {name!r} config",
            entry,
            getattr(adapter, "_RESERVED_CONFIG", frozenset()),
        )


class _EvictingStream:
    """Wraps an auto-matched stream so an `APIError` that reaches the consumer can
    evict the cache entry — and forwards `close`/`cancel` to the wrapped stream
    unchanged, so wrapping it here doesn't hide either from whatever holds this
    object (a direct sync consumer, or `_stream_off_loop`'s reader thread). A
    generator can't carry those methods, which is why this is a class."""

    __slots__ = ("_inner", "_model", "_provider", "_resolved_name", "cancel")

    def __init__(
        self,
        inner: Iterator[StreamEvent],
        provider: "Provider",
        model: str,
        resolved_name: str,
    ) -> None:
        self._inner = inner
        self._provider = provider
        self._model = model
        self._resolved_name = resolved_name
        cancel = getattr(inner, "cancel", None)
        if cancel is not None:
            self.cancel = cancel

    def __iter__(self) -> "_EvictingStream":
        return self

    def __next__(self) -> StreamEvent:
        try:
            return next(self._inner)
        except APIError as exc:
            self._provider._maybe_evict_auto_match(
                exc, self._model, self._resolved_name
            )
            raise

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()

    def throw(self, *args: Any, **kwargs: Any) -> StreamEvent:
        throw = getattr(self._inner, "throw", None)
        if throw is None:
            raise AttributeError("wrapped stream has no throw()")
        try:
            return throw(*args, **kwargs)
        except APIError as exc:
            self._provider._maybe_evict_auto_match(
                exc, self._model, self._resolved_name
            )
            raise

    def send(self, value: None) -> StreamEvent:
        send = getattr(self._inner, "send", None)
        if send is None:
            raise AttributeError("wrapped stream has no send()")
        try:
            return send(value)
        except APIError as exc:
            self._provider._maybe_evict_auto_match(
                exc, self._model, self._resolved_name
            )
            raise


class Provider:
    def __init__(
        self,
        adapters: Mapping[str, Adapter] | None = None,
        timeout: float | None | Literal[_Default.TIMEOUT] = _Default.TIMEOUT,
        autodiscover: bool | Collection[str] = False,
        executor: concurrent.futures.Executor | None = None,
        api_keys: Mapping[str, str | Callable[[], str | None]]
        | Callable[[str], str | None]
        | None = None,
        config: Config | Mapping[str, Any] | None = None,
    ):
        """`timeout=None` disables timeouts for every call; leaving it out keeps each adapter's default.

        `api_keys` gives the adapters that take an API key (`ClaudeAdapter`,
        `OpenAIAdapter` and their subclasses, built in or passed in `adapters`)
        their key by registered name, before plugins are discovered: a mapping of
        name to key or zero-argument function, or one function called with the
        name. It works like each adapter's `api_key=`: the env var is no longer
        read, and a function is called on every request, possibly from several
        threads at once. A passed adapter is never modified; a shallow copy gets
        the key, so a subclass's own mutable state (locks, dicts) stays shared
        with the original.

        `config` sets defaults for every call, with the same shape as a call's
        own `config`; see the README's Configuration section. `headers` are only
        allowed per provider, as `config["providers"][name]["headers"]`, never
        at the top level: a header must never reach another vendor or plugin.
        They are only allowed for built-in adapters (or a subclass, built in or
        passed in `adapters`), get the same guarantees as `api_keys` (redacted,
        never pickled, never in a frame local of ours, sent only over https or
        to an unproxied loopback address), and are also sent when checking
        availability and listing models. A configured auth header (`x-api-key`
        for Claude, `authorization` for OpenAI) replaces the key.
        """
        stop = sys.exception()
        try:
            if isinstance(api_keys, Mapping):
                credentials: list[tuple[_Secret, _Secret]] | _Secret | None = [
                    (_Secret(name), _Secret(value))
                    for name, value in dict(api_keys.items()).items()
                ]
            elif callable(api_keys):
                credentials = _Secret(api_keys)
            elif api_keys is not None:
                raise TypeError(
                    "api_keys must be a mapping of provider name to key or a "
                    f"function, not a {type(api_keys).__name__}"
                )
            else:
                credentials = None
            provider_config = _snapshot_config(config)
        except BaseException as e:
            _clear_tracebacks(e, stop)
            raise
        finally:
            api_keys = None
            config = None
        if timeout is not _Default.TIMEOUT:
            _validate_timeout("Provider", timeout)
        if timeout is not _Default.TIMEOUT and "timeout" in provider_config:
            raise ValueError(
                "Provider got a timeout both as timeout= and in config['timeout']; "
                "pass just one"
            )
        self._adapters: dict[str, Adapter] = (
            dict(adapters)
            if adapters is not None
            else {
                "claude": ClaudeAdapter(),
                "openai": OpenAIAdapter(),
                "ollama-local": OllamaLocalAdapter(),
            }
        )
        if credentials is not None:
            self._apply_api_keys(credentials)
        if isinstance(autodiscover, (str, bytes, bytearray)):
            raise TypeError(
                "autodiscover must be a bool or a collection of str names, not a"
                f" {type(autodiscover).__name__}"
            )
        if not isinstance(autodiscover, bool):
            autodiscover = tuple(autodiscover)
            for entry in autodiscover:
                if not isinstance(entry, str):
                    raise TypeError(
                        f"autodiscover entries must be str, got {type(entry).__name__}"
                    )
        self._plain_name_alternatives: dict[str, list[str]] = {}
        if autodiscover:
            self._register_plugins(
                None if autodiscover is True else frozenset(autodiscover)
            )
        _validate_provider_config(self._adapters, provider_config)
        self._config: dict[str, Any] = provider_config
        if timeout is not _Default.TIMEOUT:
            self._config["timeout"] = timeout
        self._apply_provider_headers()
        self._executor = executor
        self._auto_match_cache: dict[str, str] = {}
        self._auto_match_lock = threading.Lock()

    def _apply_provider_headers(self) -> None:
        """Copies each targeted adapter with its configured Provider headers,
        the same way `_apply_api_keys` copies a keyed adapter — validated by
        `_validate_provider_config` first, so every target here is a built-in
        adapter or a subclass of one."""
        headered = {
            name: adapter
            for name, adapter in self._adapters.items()
            if isinstance(adapter, _HEADERED_ADAPTER_CLASSES)
        }
        for name, entry in (self._config.get("providers") or {}).items():
            headers = entry.get("headers")
            if not headers:
                continue
            adapter = headered[name]
            copied = copy.copy(adapter)
            if copied is adapter:
                raise TypeError(
                    f"cannot configure headers for {name!r}: copying its "
                    "adapter returned the same instance, which would change "
                    "the caller's"
                )
            copied._models_cache = None
            copied._cache_time = 0.0
            copied._warned_transport = False
            copied._provider_headers = headers
            copied._provider_name = name
            self._adapters[name] = copied

    def _apply_api_keys(
        self, credentials: list[tuple[_Secret, _Secret]] | _Secret
    ) -> None:
        """Replaces each targeted keyed adapter with a copy using its key source."""
        keyed = {
            name: adapter
            for name, adapter in self._adapters.items()
            if isinstance(adapter, (ClaudeAdapter, OpenAIAdapter))
        }
        if isinstance(credentials, _Secret):
            if not keyed:
                raise ValueError(
                    "api_keys is a function, but no provider that takes an API key "
                    "is registered"
                )
            targets = [(name, credentials.bind(name)) for name in keyed]
        else:
            targets = []
            for wrapped_name, source in credentials:
                if (name := wrapped_name.among(keyed)) is None:
                    raise ValueError(
                        "api_keys names a provider that is not registered or takes "
                        "no API key; providers that take one: "
                        f"{', '.join(repr(n) for n in keyed) or 'none'}"
                    )
                targets.append((name, source))
        for name, source in targets:
            source.validate(name)
            adapter = keyed[name]
            if adapter._key_source is not None:
                raise ValueError(
                    f"api_keys sets a key for {name!r}, whose adapter already has "
                    "its own api_key"
                )
            copied = copy.copy(adapter)
            if copied is adapter:
                raise TypeError(
                    f"api_keys cannot set a key for {name!r}: copying its adapter "
                    "returned the same instance, which would change the caller's"
                )
            copied._models_cache = None
            copied._cache_time = 0.0
            copied._key_source = source
            self._adapters[name] = copied

    def _register_plugins(self, allowed: frozenset[str] | None) -> None:
        unmatched = set(allowed) if allowed is not None else set()
        normalized_allowed = (
            {entry: _normalize_allowlist_entry(entry) for entry in allowed}
            if allowed is not None
            else {}
        )

        candidates: list[
            tuple[importlib.metadata.EntryPoint, str | None, str | None, bool]
        ] = []
        for entry_point in _scan_entry_points():
            dist = _dist_name(entry_point)
            qualified = f"{dist}:{entry_point.name}" if dist is not None else None
            if allowed is not None:
                matches = {
                    entry
                    for entry, normalized in normalized_allowed.items()
                    if entry == entry_point.name or normalized == qualified
                }
                if not matches:
                    continue
                unmatched -= matches
                qualified_match = any(":" in entry for entry in matches)
            else:
                qualified_match = False
            candidates.append((entry_point, dist, qualified, qualified_match))

        plain_name_counts: dict[str, int] = {}
        for entry_point, _dist, _qualified, qualified_match in candidates:
            if not qualified_match:
                plain_name_counts[entry_point.name] = (
                    plain_name_counts.get(entry_point.name, 0) + 1
                )
        contested = {name for name, count in plain_name_counts.items() if count > 1}
        for name in sorted(contested):
            dists = sorted(
                (_dist_name(ep) or "?")
                for ep, _dist, _qualified, qm in candidates
                if not qm and ep.name == name
            )
            logger.warning(
                "Multiple third-party adapters named %r discovered (from %s);"
                " none will use the plain name",
                name,
                ", ".join(dists),
            )

        for entry_point, dist, qualified, qualified_match in candidates:
            collision_reason: str | None = None
            if qualified_match:
                name = qualified
                if name is None or name in self._adapters:
                    logger.warning(
                        "Skipping third-party adapter %r from %s: requested"
                        " name %r is unavailable",
                        entry_point.name,
                        dist,
                        qualified,
                    )
                    continue
                if entry_point.name in self._adapters:
                    collision_reason = "collides with an existing provider"
            else:
                name = entry_point.name
                if name in self._adapters:
                    collision_reason = "collides with an existing provider"
                elif name in contested:
                    collision_reason = "shares its name with another discovered adapter"
                if collision_reason is not None:
                    if qualified is None or qualified in self._adapters:
                        logger.warning(
                            "Skipping third-party adapter %r from %s: name already"
                            " taken and no free distribution-prefixed name",
                            entry_point.name,
                            dist,
                        )
                        continue
                    name = qualified

            adapter_class = _load_adapter_class(entry_point)
            if adapter_class is None:
                continue
            try:
                adapter = adapter_class()
            except Exception:
                logger.warning(
                    "Skipping third-party adapter %r from %s: failed to instantiate",
                    entry_point.name,
                    dist,
                    exc_info=True,
                )
                continue

            if collision_reason is not None:
                logger.warning(
                    "Third-party adapter %r from %s %s; registered as %r",
                    entry_point.name,
                    dist,
                    collision_reason,
                    name,
                )
                self._plain_name_alternatives.setdefault(entry_point.name, []).append(
                    name
                )
            self._adapters[name] = adapter

        if unmatched:
            logger.warning(
                "autodiscover entries matched no installed third-party adapter: %s",
                ", ".join(repr(entry) for entry in sorted(unmatched)),
            )

    def _adapter(self, provider: str) -> Adapter:
        try:
            return self._adapters[provider]
        except KeyError:
            configured = ", ".join(repr(name) for name in self._adapters) or "none"
            message = (
                f"unknown provider {provider!r}; configured providers: {configured}"
            )
            alternatives = self._plain_name_alternatives.get(provider)
            if alternatives:
                message += (
                    f"; {provider!r} is ambiguous among multiple third-party"
                    f" adapters, registered as: {', '.join(repr(n) for n in alternatives)}"
                )
            raise KeyError(message) from None

    def _resolve_provider(
        self, provider: str | None, model: str
    ) -> tuple[str, Adapter]:
        """An explicit provider resolves eagerly (unknown name raises); omitted,
        the first available adapter (in registration order) whose models()
        contains `model` is matched, and the match is cached on this instance
        so a repeat call skips the is_available()/models() I/O. Auto-match (on
        a cache miss) does that I/O, so callers on the event loop must run this
        off it.
        """
        if provider is not None:
            return provider, self._adapter(provider)
        with self._auto_match_lock:
            cached_name = self._auto_match_cache.get(model)
            if cached_name is not None:
                return cached_name, self._adapters[cached_name]
        checked: list[str] = []
        for name, adapter in self._adapters.items():
            if not _probe_available(name, adapter):
                continue
            checked.append(name)
            try:
                models = adapter.models()
            except Exception:
                logger.warning(
                    "provider %r raised while listing models during auto-match",
                    name,
                    exc_info=True,
                )
                continue
            if model in models:
                with self._auto_match_lock:
                    first_fill = model not in self._auto_match_cache
                    self._auto_match_cache[model] = name
                if first_fill:
                    logger.warning(
                        "no provider given; matched model %r to provider %r",
                        model,
                        name,
                    )
                return name, adapter
        available = ", ".join(repr(name) for name in checked) or "none"
        raise KeyError(
            f"no provider serves model {model!r}; available providers checked: {available}"
        )

    def _evict_auto_match(self, model: str, resolved_name: str) -> None:
        """Drops a stale cache entry after a call through it fails in a way that
        implicates the provider itself, not just the request (see `_should_evict_cache`)."""
        with self._auto_match_lock:
            if self._auto_match_cache.get(model) == resolved_name:
                del self._auto_match_cache[model]

    def _maybe_evict_auto_match(
        self, exc: APIError, model: str, resolved_name: str
    ) -> None:
        if _should_evict_cache(exc):
            self._evict_auto_match(model, resolved_name)

    async def _run_off_loop(self, func: Callable[[], _T]) -> _T:
        """Runs func on self._executor (or the loop's default) with the caller's contextvars."""
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(self._executor, ctx.run, func)

    def providers(self) -> dict[str, bool]:
        """Configured providers, available or not."""
        return {
            name: _probe_available(name, adapter)
            for name, adapter in self._adapters.items()
        }

    def models(self) -> dict[str, list[str]]:
        """Usable model ids by provider; nothing from unreachable vendors."""
        return {
            name: models
            for name, adapter in self._adapters.items()
            if (models := _usable_models(name, adapter)) is not None
        }

    async def async_providers(self) -> dict[str, bool]:
        """providers(), probing every adapter concurrently off the event loop."""
        adapters = list(self._adapters.items())
        results = await asyncio.gather(
            *(
                self._run_off_loop(functools.partial(_probe_available, name, adapter))
                for name, adapter in adapters
            )
        )
        return {name: result for (name, _), result in zip(adapters, results)}

    async def async_models(self) -> dict[str, list[str]]:
        """models(), querying every adapter concurrently off the event loop."""
        adapters = list(self._adapters.items())
        results = await asyncio.gather(
            *(
                self._run_off_loop(functools.partial(_usable_models, name, adapter))
                for name, adapter in adapters
            )
        )
        return {
            name: result
            for (name, _), result in zip(adapters, results)
            if result is not None
        }

    def model_info(
        self, model: str, *, provider: str | None = None
    ) -> ModelInfo | None:
        """Context window and max output for `model`, when the provider knows it.
        `None` if unknown, or if the resolved provider doesn't expose it at all
        (e.g. `openai`). Resolves the provider the same way `chat` does — an
        explicit `provider` short-circuits; omitted, the same auto-match/cache
        used by `chat` and `models`. An adapter's `model_info` raising is not
        swallowed here, same as `chat`/`stream_chat`.
        """
        _, adapter = self._resolve_provider(provider, model)
        return adapter.model_info(model)

    async def async_model_info(
        self, model: str, *, provider: str | None = None
    ) -> ModelInfo | None:
        """`model_info`, off the event loop — resolution included, the same way
        `async_chat` resolves an omitted `provider` off the loop rather than on it.
        """

        def call() -> ModelInfo | None:
            _, adapter = self._resolve_provider(provider, model)
            return adapter.model_info(model)

        return await self._run_off_loop(call)

    def _resolve_config(
        self, provider: str, config: Config | Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Layers Provider top-level config, Provider `config["providers"][provider]`,
        call top-level config, call `config["providers"][provider]` — each
        overriding the previous.

        `headers` merge key by key instead, so a later layer's header adds to
        the earlier layers' rather than dropping them. The Provider's own
        configured headers never enter this dict — they are secrets, sent
        straight from each adapter's `_provider_headers`, not through `config`.
        """
        config = config or {}
        provider_top = {k: v for k, v in self._config.items() if k != "providers"}
        provider_per = {
            k: v
            for k, v in (
                (self._config.get("providers") or {}).get(provider) or {}
            ).items()
            if k != "headers"
        }
        call_top = {k: v for k, v in config.items() if k != "providers"}
        call_per = (config.get("providers") or {}).get(provider) or {}
        merged: dict[str, Any] = {}
        for layer in (provider_top, provider_per, call_top, call_per):
            for key, value in layer.items():
                if key == "headers" and isinstance(value, Mapping):
                    current = merged.get(key)
                    base = current if isinstance(current, Mapping) else {}
                    merged[key] = {**base, **value}
                else:
                    merged[key] = value
        return merged

    def chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: Config | Mapping[str, Any] | None = None,
        *,
        provider: str | None = None,
    ) -> Response:
        resolved_name, adapter = self._resolve_provider(provider, model)
        try:
            return adapter.chat(
                model,
                messages,
                system,
                tools,
                self._resolve_config(resolved_name, config),
            )
        except APIError as exc:
            if provider is None:
                self._maybe_evict_auto_match(exc, model, resolved_name)
            raise

    def stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: Config | Mapping[str, Any] | None = None,
        *,
        provider: str | None = None,
    ) -> Iterator[StreamEvent]:
        resolved_name, adapter = self._resolve_provider(provider, model)
        stream = adapter.stream_chat(
            model, messages, system, tools, self._resolve_config(resolved_name, config)
        )
        if provider is not None:
            return stream
        return _EvictingStream(stream, self, model, resolved_name)

    async def async_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: Config | Mapping[str, Any] | None = None,
        *,
        provider: str | None = None,
    ) -> Response:
        def call() -> Response:
            resolved_name, adapter = self._resolve_provider(provider, model)
            try:
                return adapter.chat(
                    model,
                    messages,
                    system,
                    tools,
                    self._resolve_config(resolved_name, config),
                )
            except APIError as exc:
                if provider is None:
                    self._maybe_evict_auto_match(exc, model, resolved_name)
                raise

        return await self._run_off_loop(call)

    def async_stream_chat(
        self,
        model: str,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolDef] | None = None,
        config: Config | Mapping[str, Any] | None = None,
        *,
        provider: str | None = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        if provider is not None:
            adapter = self._adapter(provider)
            open_stream = functools.partial(
                adapter.stream_chat,
                model,
                messages,
                system,
                tools,
                self._resolve_config(provider, config),
            )
        else:

            def open_stream() -> Iterator[StreamEvent]:
                resolved_name, adapter = self._resolve_provider(None, model)
                inner = adapter.stream_chat(
                    model,
                    messages,
                    system,
                    tools,
                    self._resolve_config(resolved_name, config),
                )
                return _EvictingStream(inner, self, model, resolved_name)

        return self._stream_off_loop(open_stream)

    async def _stream_off_loop(
        self, open_stream: Callable[[], Iterator[StreamEvent]]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Drives open_stream on a dedicated reader thread, handing events to the loop through a bounded buffer.

        The reader creates, iterates, and closes the sync iterator, so no
        adapter code ever runs on the event loop. When the consumer stops early
        (break, exception, cancellation) the reader is told to stop, and — for a
        stream that supports `cancel()` (the built-in adapters' streams do) — its
        socket is force-closed immediately rather than waiting for the current
        blocked read to return on its own; a stream that doesn't support it
        degrades to the old wait-for-it behavior.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        slots = threading.Semaphore(_STREAM_BUFFER_SIZE)
        stopped = threading.Event()
        live: list[Iterator[StreamEvent]] = []

        def put(item: tuple[str, Any]) -> bool:
            while not slots.acquire(timeout=_READER_POLL_SECONDS):
                if stopped.is_set() or loop.is_closed():
                    return False
            if stopped.is_set() or loop.is_closed():
                return False
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            except RuntimeError:
                return False
            return True

        def read() -> None:
            outcome: tuple[str, Any] = (_DONE, None)
            consumer_gone = False
            iterator: Iterator[StreamEvent] | None = None
            try:
                iterator = iter(open_stream())
                live.append(iterator)
                if stopped.is_set():
                    cancel = getattr(iterator, "cancel", None)
                    if cancel is not None:
                        cancel()
                for event in iterator:
                    if not put((_EVENT, event)):
                        consumer_gone = True
                        break
            except BaseException as exc:  # noqa: BLE001
                outcome = (_ERROR, exc)
            close = getattr(iterator, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as exc:
                    if outcome[0] == _DONE and not consumer_gone:
                        outcome = (_ERROR, exc)
                    else:
                        logger.warning("Error closing stream", exc_info=True)
            if not consumer_gone:
                put(outcome)

        threading.Thread(
            target=contextvars.copy_context().run,
            args=(read,),
            name="ducktape-stream-reader",
            daemon=True,
        ).start()
        try:
            while True:
                kind, payload = await queue.get()
                slots.release()
                if kind == _EVENT:
                    yield payload
                    continue
                if kind == _ERROR:
                    raise payload
                return
        finally:
            stopped.set()
            if live:
                cancel = getattr(live[0], "cancel", None)
                if cancel is not None:
                    cancel()
            slots.release()
