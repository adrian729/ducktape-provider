import asyncio
import concurrent.futures
import contextvars
import enum
import functools
import importlib.metadata
import logging
import re
import threading
from collections.abc import AsyncGenerator, Callable, Collection, Iterator, Mapping
from typing import Any, Literal, TypeVar

from .adapter import Adapter, _validate_timeout
from .adapters.claude import ClaudeAdapter
from .adapters.ollama import OllamaLocalAdapter
from .adapters.openai import OpenAIAdapter
from .errors import APIError
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
    ):
        """`timeout=None` disables timeouts for every call; leaving it out keeps each adapter's default."""
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
        if isinstance(autodiscover, str):
            raise TypeError(
                "autodiscover must be a bool or a collection of names, not a str"
            )
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

    def _register_plugins(self, allowed: frozenset[str] | None) -> None:
        unmatched = set(allowed) if allowed is not None else set()
        normalized_allowed = (
            {entry: _normalize_allowlist_entry(entry) for entry in allowed}
            if allowed is not None
            else {}
        )
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

            name = entry_point.name
            if name in self._adapters:
                if qualified is None or qualified in self._adapters:
                    logger.warning(
                        "Skipping third-party adapter %r from %s: name already taken"
                        " and no free distribution-prefixed name",
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

            if name != entry_point.name:
                logger.warning(
                    "Third-party adapter %r from %s collides with an existing"
                    " provider; registered as %r",
                    entry_point.name,
                    dist,
                    name,
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
            raise KeyError(
                f"unknown provider {provider!r}; configured providers: {configured}"
            ) from None

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
                cached_adapter = self._adapters.get(cached_name)
                if cached_adapter is not None:
                    return cached_name, cached_adapter
                # The cached provider was since removed from self._adapters.
                del self._auto_match_cache[model]
        checked: list[str] = []
        for name, adapter in self._adapters.items():
            if not adapter.is_available():
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
                    # Only the fill that wins the race logs, so two concurrent
                    # first calls don't double-warn (not a hard guarantee, just
                    # what the lock happens to buy us for free).
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
        """Drops a stale cache entry after a 404 through it (model gone from that provider)."""
        with self._auto_match_lock:
            if self._auto_match_cache.get(model) == resolved_name:
                del self._auto_match_cache[model]

    def _evict_on_error(
        self, stream: Iterator[StreamEvent], model: str, resolved_name: str
    ) -> Iterator[StreamEvent]:
        """Wraps an auto-matched stream so a 404 that reaches the consumer evicts the cache entry."""
        try:
            yield from stream
        except APIError as exc:
            if exc.status == 404:
                self._evict_auto_match(model, resolved_name)
            raise

    async def _run_off_loop(self, func: Callable[[], _T]) -> _T:
        """Runs func on self._executor (or the loop's default) with the caller's contextvars."""
        loop = asyncio.get_running_loop()
        ctx = contextvars.copy_context()
        return await loop.run_in_executor(self._executor, ctx.run, func)

    def providers(self) -> dict[str, bool]:
        """Configured providers, available or not."""
        return {
            name: adapter.is_available() for name, adapter in self._adapters.items()
        }

    def models(self) -> dict[str, list[str]]:
        """Usable model ids by provider; nothing from unreachable vendors."""
        return {
            name: sorted(adapter.models())
            for name, adapter in self._adapters.items()
            if adapter.is_available()
        }

    async def async_providers(self) -> dict[str, bool]:
        """providers(), probing every adapter concurrently off the event loop."""
        adapters = list(self._adapters.items())
        results = await asyncio.gather(
            *(self._run_off_loop(adapter.is_available) for _, adapter in adapters)
        )
        return {name: result for (name, _), result in zip(adapters, results)}

    async def async_models(self) -> dict[str, list[str]]:
        """models(), querying every adapter concurrently off the event loop."""

        def usable_models(adapter: Adapter) -> list[str] | None:
            return sorted(adapter.models()) if adapter.is_available() else None

        adapters = list(self._adapters.items())
        results = await asyncio.gather(
            *(
                self._run_off_loop(functools.partial(usable_models, adapter))
                for _, adapter in adapters
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
            if provider is None and exc.status == 404:
                self._evict_auto_match(model, resolved_name)
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
                if provider is None and exc.status == 404:
                    self._evict_auto_match(model, resolved_name)
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
