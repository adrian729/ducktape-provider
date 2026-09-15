import asyncio
import concurrent.futures
import contextvars
import copy
import enum
import functools
import importlib.metadata
import logging
import re
import threading
import urllib.error
from collections.abc import AsyncGenerator, Callable, Collection, Iterator, Mapping
from typing import Any, Literal, TypeVar

from .adapter import Adapter, _Secret, _validate_timeout
from .adapters.claude import ClaudeAdapter
from .adapters.ollama import OllamaLocalAdapter
from .adapters.openai import OpenAIAdapter
from .errors import APIError, AuthError
from .types import Config, Message, Response, StreamEvent, ToolDef

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "ducktape_provider.adapters"

# Caps events buffered between a stream's reader thread and a slow consumer;
# one HTTP chunk can decode into several events, so leave room for bursts.
_STREAM_BUFFER_SIZE = 64
# How often a reader blocked on a full buffer re-checks whether its loop died
# without ever closing the async generator (so nothing will release a slot).
_READER_POLL_SECONDS = 1.0

_EVENT = "event"
_ERROR = "error"
_DONE = "done"

_T = TypeVar("_T")


class _Default(enum.Enum):
    # An enum member rather than object() so type checkers can narrow it away.
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


# Only successes are cached: a failed import may be transient (e.g. a package
# mid-upgrade), so later Providers retry it.
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
        """
        # Wrapped before anything can raise, so no raw key sits in a local of
        # this frame on a traceback. Names too, since an inverted mapping puts
        # the key there. Mapping.items() is also read only once.
        if isinstance(api_keys, Mapping):
            credentials: list[tuple[_Secret, _Secret]] | _Secret | None = [
                (_Secret(name), _Secret(value)) for name, value in api_keys.items()
            ]
            api_keys = None
        elif callable(api_keys):
            credentials = _Secret(api_keys)
            api_keys = None
        elif api_keys is not None:
            api_keys_type = type(api_keys).__name__
            api_keys = None
            raise TypeError(
                "api_keys must be a mapping of provider name to key or a function, "
                f"not a {api_keys_type}"
            )
        else:
            credentials = None
        if timeout is not _Default.TIMEOUT:
            _validate_timeout("Provider", timeout)
        # Copied so discovery never registers plugins into the caller's mapping.
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
            # Materialized once: a generator would be exhausted by this validation
            # pass, leaving nothing for frozenset() below to register.
            autodiscover = tuple(autodiscover)
            for entry in autodiscover:
                if not isinstance(entry, str):
                    raise TypeError(
                        f"autodiscover entries must be str, got {type(entry).__name__}"
                    )
        # A plain name a plugin lost to a collision, mapped to the qualified
        # name(s) it registered under instead — used to make an unknown-provider
        # KeyError on that plain name point at its qualified alternatives.
        self._plain_name_alternatives: dict[str, list[str]] = {}
        if autodiscover:
            self._register_plugins(
                None if autodiscover is True else frozenset(autodiscover)
            )
        self._executor = executor
        self._config: dict[str, Any] = {}
        if timeout is not _Default.TIMEOUT:
            self._config["timeout"] = timeout
        # Auto-match results only (explicit `provider=` calls never touch this),
        # so a repeat call for the same model skips is_available()/models() I/O.
        # Guarded by a lock since async resolution runs on worker threads.
        self._auto_match_cache: dict[str, str] = {}
        self._auto_match_lock = threading.Lock()

    def _apply_api_keys(
        self, credentials: list[tuple[_Secret, _Secret]] | _Secret
    ) -> None:
        """Replaces each targeted keyed adapter with a copy using its key source."""
        # Error messages name registered providers and types only: an unknown
        # name may be a key itself, e.g. from an inverted mapping.
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
            # partial, not a lambda closing over the loop variable, which would
            # hand every adapter the last provider's name.
            targets = [(name, credentials.bind(name)) for name in keyed]
        else:
            targets = []
            for wrapped_name, source in credentials:
                # Unwrapped only once known to be a registered provider name.
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

        # (entry_point, dist, qualified) for every candidate, plus whether it was
        # selected via a dist-qualified allowlist entry (e.g. "beta-dist:shared")
        # rather than a plain one — a qualified match always registers under
        # that exact qualified name, so it never competes for the plain one.
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

        # A name two or more plugins would otherwise both plainly want is
        # ambiguous — which one gets it would depend on install order or dist
        # sort, and a new install could silently reroute an existing plain-name
        # call to a different vendor. So none of them gets it.
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
                # The plain name was asked for in qualified form, so it never
                # competes for the plain name — but if a built-in or explicit
                # adapter already holds the plain name, still say so.
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
                # self._adapters never changes after __init__, so a cached name
                # is always still in it.
                return cached_name, self._adapters[cached_name]
        checked: list[str] = []
        for name, adapter in self._adapters.items():
            if not _probe_available(name, adapter):
                continue
            checked.append(name)
            try:
                models = adapter.models()
            except Exception:
                # One adapter's probe failing shouldn't stop matching against
                # the rest — models() is documented to return set() rather
                # than raise, but a misbehaving adapter shouldn't wedge this.
                logger.warning(
                    "provider %r raised while listing models during auto-match",
                    name,
                    exc_info=True,
                )
                continue
            if model in models:
                with self._auto_match_lock:
                    # The lock makes two concurrent first-fills log only once; a
                    # second log happens only if the entry was evicted between
                    # one fill and the next.
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

    def _evict_on_error(
        self, stream: Iterator[StreamEvent], model: str, resolved_name: str
    ) -> Iterator[StreamEvent]:
        """Wraps an auto-matched stream so an error that reaches the consumer can evict the cache entry."""
        try:
            yield from stream
        except APIError as exc:
            self._maybe_evict_auto_match(exc, model, resolved_name)
            raise

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

    def _resolve_config(
        self, provider: str, config: Config | Mapping[str, Any] | None
    ) -> dict[str, Any]:
        """Layers self._config, config, config["providers"][provider] — each overrides the previous.

        `headers` merge key by key instead, so a per-provider header adds to the
        call's headers rather than dropping them.
        """
        config = config or {}
        call_level = {k: v for k, v in config.items() if k != "providers"}
        per_provider = (config.get("providers") or {}).get(provider) or {}
        merged: dict[str, Any] = {}
        for layer in (self._config, call_level, per_provider):
            for key, value in layer.items():
                if key == "headers" and isinstance(value, Mapping):
                    current = merged.get(key)
                    base = current if isinstance(current, Mapping) else {}
                    # Always a fresh dict, so adapters never share the caller's.
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
        return self._evict_on_error(stream, model, resolved_name)

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
        # Resolution (when provider is omitted) can hit the network, so it runs
        # inside the off-loop call rather than eagerly on the caller's thread.
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

    # A plain def (not an async generator): with an explicit provider, an
    # unknown name still raises at call time rather than on first iteration.
    # With provider omitted, resolution itself may do network I/O, so it's
    # deferred into open_stream and runs on the stream's reader thread instead
    # — a no-match KeyError there surfaces on first iteration, not call time.
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
                return self._evict_on_error(inner, model, resolved_name)

        return self._stream_off_loop(open_stream)

    async def _stream_off_loop(
        self, open_stream: Callable[[], Iterator[StreamEvent]]
    ) -> AsyncGenerator[StreamEvent, None]:
        """Drives open_stream on a dedicated reader thread, handing events to the loop through a bounded buffer.

        The reader creates, iterates, and closes the sync iterator, so no
        adapter code ever runs on the event loop. When the consumer stops early
        (break, exception, cancellation) the reader is told to stop and closes
        the iterator as soon as its current blocking read returns.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue()
        # Bounds the queue from the producer side: a thread can't await a full
        # asyncio.Queue, but it can block on a semaphore the consumer releases.
        slots = threading.Semaphore(_STREAM_BUFFER_SIZE)
        stopped = threading.Event()

        def put(item: tuple[str, Any]) -> bool:
            while not slots.acquire(timeout=_READER_POLL_SECONDS):
                if stopped.is_set() or loop.is_closed():
                    return False
            if stopped.is_set() or loop.is_closed():
                return False
            try:
                loop.call_soon_threadsafe(queue.put_nowait, item)
            # The loop can still close between the check above and this call.
            except RuntimeError:
                return False
            return True

        def read() -> None:
            outcome: tuple[str, Any] = (_DONE, None)
            consumer_gone = False
            iterator: Iterator[StreamEvent] | None = None
            try:
                iterator = iter(open_stream())
                for event in iterator:
                    if not put((_EVENT, event)):
                        consumer_gone = True
                        break
            # Everything, not just Exception: the consumer re-raises it on the loop.
            except BaseException as exc:  # noqa: BLE001
                outcome = (_ERROR, exc)
            # Close before signalling the end, so the adapter's cleanup has run
            # by the time the consumer sees the stream finish.
            close = getattr(iterator, "close", None)
            if close is not None:
                try:
                    close()
                except BaseException as exc:
                    if outcome[0] == _DONE and not consumer_gone:
                        outcome = (_ERROR, exc)
                    else:
                        # Nobody will see this error otherwise: the consumer
                        # left, or gets the stream's own (root-cause) error.
                        logger.warning("Error closing stream", exc_info=True)
            # A consumer that left will never free a slot for the outcome, so
            # putting it would only stall this thread for a poll interval.
            if not consumer_gone:
                put(outcome)

        # Its own daemon thread, not an executor worker: a reader stays blocked
        # for as long as the consumer is slow, and holding a shared worker that
        # long deadlocks consumers that await the same executor per event
        # (async_chat, asyncio.to_thread, getaddrinfo).
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
            # Wakes a reader blocked on a full buffer so it can see the stop.
            slots.release()
