"""The Provider classes and their streams."""
from __future__ import annotations

import dataclasses
import threading
from typing import Any, AsyncIterator, Callable, Iterator, Optional, Sequence, Union

import httpx

from ._driver import Cancel, Host, drive
from ._errors import CancelledError, NotFoundError
from ._sugar import EnvAuth, ResolverAuth, SessionAuth, blocks
from ._transport import Run, run_async, run_sync
from ._types import (
    AuthStatus,
    Capabilities,
    EmbedResponse,
    Event,
    LoginAction,
    Message,
    Model,
    ModelInfo,
    Options,
    Response,
    Tool,
    decode,
    encode,
)

VENDORS = ("anthropic", "openai", "google", "ollama")
_OPTION_KEYS = {f.name for f in dataclasses.fields(Options)}
_TIMEOUT_KEYS = {"connect", "read", "total"}
_CALLBACKS = ("loopback", "paste")


def _timeout(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return {"total": float(value)}
    if isinstance(value, Model):
        return value.to_dict()
    if isinstance(value, dict):
        unknown = set(value) - _TIMEOUT_KEYS
        if unknown:
            raise ValueError(f"unknown timeout keys: {sorted(unknown)}")
        return {k: v for k, v in value.items() if v is not None}
    raise TypeError("timeout must be seconds or {connect, read, total}")


def _tool_choice(value: Any) -> Any:
    if isinstance(value, str):
        return {"mode": value}
    if isinstance(value, dict) and "mode" not in value and "name" in value:
        return {"mode": "named", "name": value["name"]}
    return value


def _options(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, Model):
        value = value.to_dict()
    if not isinstance(value, dict):
        raise TypeError("options must be a dict or Options")
    unknown = set(value) - _OPTION_KEYS
    if unknown:
        raise ValueError(f"unknown options: {sorted(unknown)}")
    out = dict(value)
    if "tool_choice" in out:
        out["tool_choice"] = _tool_choice(out["tool_choice"])
    return encode(out)


def _system(value: Any) -> Optional[list[dict[str, Any]]]:
    if value is None:
        return None
    if isinstance(value, str):
        return [{"type": "text", "text": value}]
    return encode(blocks(tuple(value)))


def _tools(value: Any) -> Optional[list[dict[str, Any]]]:
    if value is None:
        return None
    out = []
    for item in value:
        if isinstance(item, Model):
            out.append(item.to_dict())
        elif isinstance(item, dict):
            out.append(decode(Tool, item).to_dict())
        else:
            raise TypeError(f"a tool must be a ToolDef, ProviderTool or dict, not {type(item).__name__}")
    return out


def _messages(value: Sequence[Any]) -> list[dict[str, Any]]:
    out = []
    for item in value:
        if isinstance(item, Message):
            out.append(item.to_dict())
        elif isinstance(item, dict):
            out.append(Message.from_dict(item).to_dict())
        else:
            raise TypeError(f"a message must be a Message or dict, not {type(item).__name__}")
    if not out:
        raise ValueError("messages must not be empty")
    return out


def _drop_nulls(mapping: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if mapping is None:
        return None
    return {k: v for k, v in mapping.items() if v is not None}


def _without(defaults: Optional[dict[str, Any]], call: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if defaults is None or not call:
        return defaults
    nulls = {k for k, v in call.items() if v is None}
    return {k: v for k, v in defaults.items() if k not in nulls} or None


def _without_vendor(defaults: Optional[dict[str, Any]], call: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if defaults is None or not call:
        return defaults
    out: dict[str, Any] = {}
    for key, fields in defaults.items():
        incoming = call.get(key)
        if incoming is None and key in call:
            continue
        if isinstance(fields, dict) and isinstance(incoming, dict):
            nulls = {k for k, v in incoming.items() if v is None}
            fields = {k: v for k, v in fields.items() if k not in nulls}
        out[key] = fields
    return out or None


def _vendor(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("vendor must be a dict keyed by vendor")
    return {key: _drop_nulls(fields) if isinstance(fields, dict) else fields for key, fields in value.items()}


class _Base:
    """Configuration shared by the blocking and the asyncio providers."""

    def __init__(
        self,
        vendor: str,
        *,
        base_url: Optional[str] = None,
        auth: Any = None,
        headers: Optional[dict[str, str]] = None,
        timeout: Any = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        default_system: Any = None,
        default_tools: Any = None,
        default_options: Any = None,
        default_vendor: Any = None,
        overrides: Any = None,
        strict: Optional[bool] = None,
        observer: Any = None,
        dangerously_allow_browser: Optional[bool] = None,
        keychain: Optional[Callable[[str, str], Any]] = None,
        keychain_write: Optional[Callable[[str, str, bytes], Any]] = None,
    ) -> None:
        if vendor not in VENDORS:
            raise ValueError(f"unknown vendor {vendor!r}; one of {', '.join(VENDORS)}")
        self.vendor = vendor
        self.host = Host()
        if keychain is not None:
            if not callable(keychain):
                raise TypeError("keychain must be a callable taking (service, account)")
            self.host.keychain = keychain
            self.host.keychain_write = None
        if keychain_write is not None:
            if not callable(keychain_write):
                raise TypeError("keychain_write must be a callable taking (service, account, value)")
            self.host.keychain_write = keychain_write
        self.observer = observer
        self._config: dict[str, Any] = {"vendor": vendor}
        if base_url is not None:
            self._config["base_url"] = base_url.rstrip("/")
        if auth is not None or dangerously_allow_browser is not None:
            self._config["auth"] = {"sources": self._sources(auth)}
            if dangerously_allow_browser is not None:
                self._config["auth"]["dangerously_allow_browser"] = dangerously_allow_browser
        if headers:
            self._config["headers"] = dict(headers)
        if timeout is not None:
            self._config["timeout"] = _timeout(timeout)
        if default_model is not None:
            self._config["default_model"] = default_model
        if default_embedding_model is not None:
            self._config["default_embedding_model"] = default_embedding_model
        if default_system is not None:
            self._config["default_system"] = _system(default_system)
        if default_tools is not None:
            self._config["default_tools"] = _tools(default_tools)
        if default_options is not None:
            self._config["default_options"] = _drop_nulls(_options(default_options))
        if default_vendor is not None:
            self._config["default_vendor"] = _vendor(default_vendor)
        if overrides is not None:
            self._config["overrides"] = encode(overrides)
        if strict is not None:
            self._config["strict"] = bool(strict)
        self._models: Optional[list[ModelInfo]] = None
        self._model_info: dict[str, ModelInfo] = {}
        self._capabilities: dict[str, Capabilities] = {}
        self._lock = threading.Lock()

    def _sources(self, auth: Any) -> list[dict[str, Any]]:
        items = list(auth) if isinstance(auth, (list, tuple)) else [auth]
        sources: list[dict[str, Any]] = []
        for i, item in enumerate(items):
            if item is None:
                continue
            if isinstance(item, str):
                ref = f"inline:{i}"
                self.host.secrets[ref] = item
                sources.append({"type": "inline", "secret_ref": ref})
            elif isinstance(item, EnvAuth):
                source: dict[str, Any] = {"type": "env", "names": list(item.names)}
                if item.kind:
                    source["kind"] = item.kind
                if item.placement:
                    source["placement"] = item.placement
                sources.append(source)
            elif isinstance(item, SessionAuth):
                source = {"type": "session", "preset": item.preset}
                for key in ("store", "client_id", "profile", "refresh"):
                    value = getattr(item, key)
                    if value:
                        source[key] = value
                sources.append(source)
            elif isinstance(item, ResolverAuth):
                rid = f"resolver:{i}"
                self.host.resolvers[rid] = item.fn
                source = {"type": "resolver", "resolver_id": rid}
                if item.kind:
                    source["kind"] = item.kind
                if item.placement:
                    source["placement"] = item.placement
                sources.append(source)
            elif isinstance(item, dict):
                sources.append(dict(item))
            else:
                raise TypeError(f"auth must be a string, env(), session(), resolver() or a list of those, not {type(item).__name__}")
        return sources

    def _chat_start(
        self,
        stream: bool,
        messages: Sequence[Any],
        model: Optional[str],
        system: Any,
        tools: Any,
        options: Any,
        vendor: Any,
        timeout: Any,
        headers: Optional[dict[str, str]],
        strict: Optional[bool],
        include_raw: Optional[bool],
    ) -> dict[str, Any]:
        model = model or self._config.get("default_model")
        if not model:
            raise ValueError("no model: pass model= or set default_model")
        request: dict[str, Any] = {"model": model, "messages": _messages(messages)}
        call_options = _options(options)
        call_vendor = _vendor(vendor)
        if system is not None:
            request["system"] = _system(system)
        if tools is not None:
            request["tools"] = _tools(tools)
        if call_options:
            request["options"] = _drop_nulls(call_options)
        if call_vendor:
            request["vendor"] = call_vendor
        if timeout is not None:
            request["timeout"] = _timeout(timeout)
        if headers:
            request["headers"] = dict(headers)
        if strict is not None:
            request["strict"] = bool(strict)
        if include_raw is not None:
            request["include_raw"] = bool(include_raw)
        config = dict(self._config)
        config["default_options"] = _without(config.get("default_options"), call_options)
        config["default_vendor"] = _without_vendor(config.get("default_vendor"), call_vendor)
        config = {k: v for k, v in config.items() if v is not None}
        return {
            "provider": config,
            "operation": {"type": "stream" if stream else "chat", "request": request},
        }

    def _embed_start(
        self,
        input: Union[str, Sequence[str]],
        model: Optional[str],
        options: Any,
        vendor: Any,
        timeout: Any,
        include_raw: Optional[bool],
    ) -> dict[str, Any]:
        texts = [input] if isinstance(input, str) else list(input)
        if not texts:
            raise ValueError("embed input must not be empty")
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("embed input must be a string or a list of strings")
        model = model or self._config.get("default_embedding_model")
        if not model:
            raise ValueError("no embedding model: pass model= or set default_embedding_model")
        request: dict[str, Any] = {"model": model, "input": texts}
        if options is not None:
            if isinstance(options, Model):
                options = options.to_dict()
            unknown = set(options) - {"dimensions", "retries"}
            if unknown:
                raise ValueError(f"unknown embed options: {sorted(unknown)}")
            request["options"] = _drop_nulls(options)
        if vendor is not None:
            request["vendor"] = _vendor(vendor)
        if timeout is not None:
            request["timeout"] = _timeout(timeout)
        if include_raw is not None:
            request["include_raw"] = bool(include_raw)
        return {"provider": self._config, "operation": {"type": "embed", "request": request}}

    def _start(self, operation: dict[str, Any]) -> dict[str, Any]:
        return {"provider": self._config, "operation": operation}

    def _login_start(self, callback: str, timeout: Optional[float]) -> dict[str, Any]:
        if callback not in _CALLBACKS:
            raise ValueError(f"callback must be one of {', '.join(_CALLBACKS)}")
        operation: dict[str, Any] = {"type": "login", "callback": callback}
        if timeout is not None:
            operation["timeout_ms"] = int(timeout * 1000)
        return self._start(operation)

    @staticmethod
    def _model_info_result(model: str, result: dict[str, Any]) -> ModelInfo:
        info = result.get("info")
        if info is None:
            raise NotFoundError(f"unknown model {model!r}", code="model_not_found")
        return ModelInfo.from_dict(info)

    @staticmethod
    def _capabilities_result(model: str, result: dict[str, Any]) -> Capabilities:
        caps = result.get("capabilities")
        if caps is None:
            raise NotFoundError(f"unknown model {model!r}", code="model_not_found")
        return Capabilities.from_dict(caps)


class Stream:
    """A streamed chat reply: iterate the events, `cancel()` to stop."""

    def __init__(self, events: Iterator[dict[str, Any]], cancel: Cancel, run: Run) -> None:
        self._events = events
        self._cancel = cancel
        self._run = run
        self.response: Optional[Response] = None

    def __iter__(self) -> Iterator[Event]:
        return self

    def __next__(self) -> Event:
        event = decode(Event, next(self._events))
        if event.type == "finish":
            self.response = event.response
        return event

    def cancel(self) -> None:
        self._cancel.set()

    def close(self) -> None:
        self._events.close()

    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncStream:
    """An asyncio streamed chat reply: iterate the events, `cancel()` to stop."""

    def __init__(self, events: AsyncIterator[dict[str, Any]], cancel: Cancel, run: Run) -> None:
        self._events = events
        self._cancel = cancel
        self._run = run
        self.response: Optional[Response] = None

    def __aiter__(self) -> AsyncIterator[Event]:
        return self

    async def __anext__(self) -> Event:
        event = decode(Event, await self._events.__anext__())
        if event.type == "finish":
            self.response = event.response
        return event

    def cancel(self) -> None:
        self._cancel.set()

    async def aclose(self) -> None:
        await self._events.aclose()

    async def __aenter__(self) -> AsyncStream:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


class Provider(_Base):
    """One endpoint of one vendor, blocking API."""

    def __init__(
        self,
        vendor: str,
        *,
        base_url: Optional[str] = None,
        auth: Any = None,
        headers: Optional[dict[str, str]] = None,
        timeout: Any = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        default_system: Any = None,
        default_tools: Any = None,
        default_options: Any = None,
        default_vendor: Any = None,
        overrides: Any = None,
        strict: Optional[bool] = None,
        observer: Any = None,
        dangerously_allow_browser: Optional[bool] = None,
        keychain: Optional[Callable[[str, str], Any]] = None,
        keychain_write: Optional[Callable[[str, str, bytes], Any]] = None,
        client: Optional[httpx.Client] = None,
    ) -> None:
        super().__init__(
            vendor,
            base_url=base_url,
            auth=auth,
            headers=headers,
            timeout=timeout,
            default_model=default_model,
            default_embedding_model=default_embedding_model,
            default_system=default_system,
            default_tools=default_tools,
            default_options=default_options,
            default_vendor=default_vendor,
            overrides=overrides,
            strict=strict,
            observer=observer,
            dangerously_allow_browser=dangerously_allow_browser,
            keychain=keychain,
            keychain_write=keychain_write,
        )
        self._client = client

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()

    def __enter__(self) -> Provider:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _run(self, start: dict[str, Any], handle_action: Any = None) -> dict[str, Any]:
        cancel = Cancel()
        run = Run()
        for _ in run_sync(drive(start, self.host, cancel, handle_action), self.client, cancel, self.observer, run):
            pass
        assert run.result is not None
        return run.result

    def chat(
        self,
        messages: Sequence[Any],
        *,
        model: Optional[str] = None,
        system: Any = None,
        tools: Any = None,
        options: Any = None,
        vendor: Any = None,
        timeout: Any = None,
        headers: Optional[dict[str, str]] = None,
        strict: Optional[bool] = None,
        include_raw: Optional[bool] = None,
    ) -> Response:
        start = self._chat_start(False, messages, model, system, tools, options, vendor, timeout, headers, strict, include_raw)
        return Response.from_dict(self._run(start)["response"])

    def stream(
        self,
        messages: Sequence[Any],
        *,
        model: Optional[str] = None,
        system: Any = None,
        tools: Any = None,
        options: Any = None,
        vendor: Any = None,
        timeout: Any = None,
        headers: Optional[dict[str, str]] = None,
        strict: Optional[bool] = None,
        include_raw: Optional[bool] = None,
    ) -> Stream:
        start = self._chat_start(True, messages, model, system, tools, options, vendor, timeout, headers, strict, include_raw)
        cancel = Cancel()
        run = Run()
        events = run_sync(drive(start, self.host, cancel), self.client, cancel, self.observer, run)
        return Stream(events, cancel, run)

    def embed(
        self,
        input: Union[str, Sequence[str]],
        *,
        model: Optional[str] = None,
        options: Any = None,
        vendor: Any = None,
        timeout: Any = None,
        include_raw: Optional[bool] = None,
    ) -> EmbedResponse:
        start = self._embed_start(input, model, options, vendor, timeout, include_raw)
        return EmbedResponse.from_dict(self._run(start)["response"])

    def models(self, refresh: bool = False) -> list[ModelInfo]:
        with self._lock:
            if self._models is None or refresh:
                result = self._run(self._start({"type": "models", "refresh": refresh}))
                self._models = [ModelInfo.from_dict(m) for m in result["models"]]
            return list(self._models)

    def model_info(self, id: str) -> ModelInfo:
        with self._lock:
            if id not in self._model_info:
                self._model_info[id] = self._model_info_result(id, self._run(self._start({"type": "model_info", "model": id})))
            return self._model_info[id]

    def capabilities(self, model: str) -> Capabilities:
        with self._lock:
            if model not in self._capabilities:
                self._capabilities[model] = self._capabilities_result(
                    model, self._run(self._start({"type": "capabilities", "model": model}))
                )
            return self._capabilities[model]

    def login(
        self,
        handle_action: Callable[[LoginAction], Optional[str]],
        *,
        callback: str = "loopback",
        timeout: Optional[float] = None,
    ) -> AuthStatus:
        def handler(action: dict[str, Any]) -> Optional[str]:
            return handle_action(decode(LoginAction, action))

        return AuthStatus.from_dict(self._run(self._login_start(callback, timeout), handler)["status"])

    def logout(self) -> None:
        self._run(self._start({"type": "logout"}))

    def auth_status(self) -> AuthStatus:
        return AuthStatus.from_dict(self._run(self._start({"type": "auth_status"}))["status"])


class AsyncProvider(_Base):
    """One endpoint of one vendor, asyncio API."""

    def __init__(
        self,
        vendor: str,
        *,
        base_url: Optional[str] = None,
        auth: Any = None,
        headers: Optional[dict[str, str]] = None,
        timeout: Any = None,
        default_model: Optional[str] = None,
        default_embedding_model: Optional[str] = None,
        default_system: Any = None,
        default_tools: Any = None,
        default_options: Any = None,
        default_vendor: Any = None,
        overrides: Any = None,
        strict: Optional[bool] = None,
        observer: Any = None,
        dangerously_allow_browser: Optional[bool] = None,
        keychain: Optional[Callable[[str, str], Any]] = None,
        keychain_write: Optional[Callable[[str, str, bytes], Any]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        super().__init__(
            vendor,
            base_url=base_url,
            auth=auth,
            headers=headers,
            timeout=timeout,
            default_model=default_model,
            default_embedding_model=default_embedding_model,
            default_system=default_system,
            default_tools=default_tools,
            default_options=default_options,
            default_vendor=default_vendor,
            overrides=overrides,
            strict=strict,
            observer=observer,
            dangerously_allow_browser=dangerously_allow_browser,
            keychain=keychain,
            keychain_write=keychain_write,
        )
        self._client = client

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncProvider:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    async def _run(self, start: dict[str, Any], handle_action: Any = None) -> dict[str, Any]:
        cancel = Cancel()
        run = Run()
        async for _ in run_async(drive(start, self.host, cancel, handle_action), self.client, cancel, self.observer, run):
            pass
        assert run.result is not None
        return run.result

    async def chat(
        self,
        messages: Sequence[Any],
        *,
        model: Optional[str] = None,
        system: Any = None,
        tools: Any = None,
        options: Any = None,
        vendor: Any = None,
        timeout: Any = None,
        headers: Optional[dict[str, str]] = None,
        strict: Optional[bool] = None,
        include_raw: Optional[bool] = None,
    ) -> Response:
        start = self._chat_start(False, messages, model, system, tools, options, vendor, timeout, headers, strict, include_raw)
        return Response.from_dict((await self._run(start))["response"])

    def stream(
        self,
        messages: Sequence[Any],
        *,
        model: Optional[str] = None,
        system: Any = None,
        tools: Any = None,
        options: Any = None,
        vendor: Any = None,
        timeout: Any = None,
        headers: Optional[dict[str, str]] = None,
        strict: Optional[bool] = None,
        include_raw: Optional[bool] = None,
    ) -> AsyncStream:
        start = self._chat_start(True, messages, model, system, tools, options, vendor, timeout, headers, strict, include_raw)
        cancel = Cancel()
        run = Run()
        events = run_async(drive(start, self.host, cancel), self.client, cancel, self.observer, run)
        return AsyncStream(events, cancel, run)

    async def embed(
        self,
        input: Union[str, Sequence[str]],
        *,
        model: Optional[str] = None,
        options: Any = None,
        vendor: Any = None,
        timeout: Any = None,
        include_raw: Optional[bool] = None,
    ) -> EmbedResponse:
        start = self._embed_start(input, model, options, vendor, timeout, include_raw)
        return EmbedResponse.from_dict((await self._run(start))["response"])

    async def models(self, refresh: bool = False) -> list[ModelInfo]:
        if self._models is None or refresh:
            result = await self._run(self._start({"type": "models", "refresh": refresh}))
            self._models = [ModelInfo.from_dict(m) for m in result["models"]]
        return list(self._models)

    async def model_info(self, id: str) -> ModelInfo:
        if id not in self._model_info:
            self._model_info[id] = self._model_info_result(id, await self._run(self._start({"type": "model_info", "model": id})))
        return self._model_info[id]

    async def capabilities(self, model: str) -> Capabilities:
        if model not in self._capabilities:
            self._capabilities[model] = self._capabilities_result(
                model, await self._run(self._start({"type": "capabilities", "model": model}))
            )
        return self._capabilities[model]

    async def login(
        self,
        handle_action: Callable[[LoginAction], Any],
        *,
        callback: str = "loopback",
        timeout: Optional[float] = None,
    ) -> AuthStatus:
        def handler(action: dict[str, Any]) -> Any:
            return handle_action(decode(LoginAction, action))

        return AuthStatus.from_dict((await self._run(self._login_start(callback, timeout), handler))["status"])

    async def logout(self) -> None:
        await self._run(self._start({"type": "logout"}))

    async def auth_status(self) -> AuthStatus:
        return AuthStatus.from_dict((await self._run(self._start({"type": "auth_status"})))["status"])
