"""httpx runners of the effect loop, one blocking and one asyncio."""
from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any, AsyncIterator, Callable, Generator, Iterator, Optional

import httpx

from ._driver import Action, Call, Cancel, Chunk, Failed, Http, Notify, Observe, Sleep


def classify(error: Exception) -> tuple[str, str]:
    """Maps an httpx failure to a transport error kind and message."""
    message = str(error) or type(error).__name__
    if isinstance(error, httpx.TimeoutException):
        return "timeout", message
    if isinstance(error, httpx.ConnectError):
        lowered = message.lower()
        if "ssl" in lowered or "certificate" in lowered or "tls" in lowered:
            return "tls", message
        if "getaddrinfo" in lowered or "name or service" in lowered or "nodename" in lowered or "resolve" in lowered:
            return "dns", message
        return "connect", message
    if isinstance(error, (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError, httpx.CloseError)):
        return "reset", message
    return "other", message


def timeout_of(request: dict[str, Any]) -> httpx.Timeout:
    """The httpx timeout for one request."""
    t = request.get("timeout") or {}
    connect = t.get("connect")
    read = t.get("read")
    return httpx.Timeout(connect=connect, read=read, write=read, pool=connect)


def build(client: Any, request: dict[str, Any]) -> httpx.Request:
    """Builds the httpx request the core asked for."""
    body = request.get("body")
    return client.build_request(
        request["method"],
        request["url"],
        headers=request.get("headers") or {},
        content=body.encode() if isinstance(body, str) else body,
        timeout=timeout_of(request),
    )


def observe(observer: Any, hook: str, data: Any) -> None:
    """Calls one observer hook when the observer has it."""
    if observer is None:
        return
    fn = observer.get(hook) if isinstance(observer, dict) else getattr(observer, hook, None)
    if fn is not None:
        fn(data)


def sync_body(response: httpx.Response) -> Iterator[bytes]:
    """The body chunks, also for responses whose content was given upfront."""
    if response.is_stream_consumed:
        return iter([response.content] if response.content else [])
    return response.iter_raw()


async def _once(content: bytes) -> AsyncIterator[bytes]:
    if content:
        yield content


def async_body(response: httpx.Response) -> AsyncIterator[bytes]:
    """The body chunks, also for responses whose content was given upfront."""
    if response.is_stream_consumed:
        return _once(response.content)
    return response.aiter_raw()


class Run:
    """The result slot of one run."""

    def __init__(self) -> None:
        self.result: Optional[dict[str, Any]] = None


def run_sync(
    gen: Generator[Action, Any, dict[str, Any]],
    client: httpx.Client,
    cancel: Cancel,
    observer: Any,
    run: Run,
) -> Iterator[dict[str, Any]]:
    """Drives a sync operation, yielding stream events and storing the result."""
    reply: Any = None
    response: Optional[httpx.Response] = None
    body: Optional[Iterator[bytes]] = None

    def close() -> None:
        nonlocal response, body
        if response is not None:
            try:
                response.close()
            except Exception:
                pass
        response = body = None

    try:
        while True:
            try:
                action = gen.send(reply)
            except StopIteration as stop:
                run.result = stop.value
                return
            reply = None
            if isinstance(action, Http):
                close()
                try:
                    response = client.send(build(client, action.request), stream=True)
                    body = sync_body(response)
                    cancel.hook = response.close
                    reply = ("response", response.status_code, {k.lower(): v for k, v in response.headers.items()})
                except httpx.HTTPError as e:
                    reply = ("error",) + classify(e)
            elif isinstance(action, Chunk):
                try:
                    reply = next(body) if body is not None else None
                except StopIteration:
                    reply = None
                    close()
                except (httpx.HTTPError, httpx.StreamError) as e:
                    reply = ("error",) + classify(e)
                    close()
            elif isinstance(action, Sleep):
                close()
                time.sleep(action.ms / 1000)
            elif isinstance(action, Call):
                try:
                    reply = action.fn(*action.args)
                except Exception as e:
                    reply = Failed(e)
            elif isinstance(action, Notify):
                yield action.event
            elif isinstance(action, Observe):
                observe(observer, action.hook, action.data)
    finally:
        close()
        gen.close()


async def run_async(
    gen: Generator[Action, Any, dict[str, Any]],
    client: httpx.AsyncClient,
    cancel: Cancel,
    observer: Any,
    run: Run,
) -> AsyncIterator[dict[str, Any]]:
    """Drives an async operation, yielding stream events and storing the result."""
    reply: Any = None
    response: Optional[httpx.Response] = None
    body: Optional[AsyncIterator[bytes]] = None

    async def close() -> None:
        nonlocal response, body
        if response is not None:
            try:
                await response.aclose()
            except Exception:
                pass
        response = body = None

    try:
        while True:
            try:
                action = gen.send(reply)
            except StopIteration as stop:
                run.result = stop.value
                return
            reply = None
            if isinstance(action, Http):
                await close()
                try:
                    response = await client.send(build(client, action.request), stream=True)
                    body = async_body(response)
                    reply = ("response", response.status_code, {k.lower(): v for k, v in response.headers.items()})
                except httpx.HTTPError as e:
                    reply = ("error",) + classify(e)
            elif isinstance(action, Chunk):
                task = asyncio.ensure_future(body.__anext__()) if body is not None else None
                cancel.hook = task.cancel if task is not None else None
                try:
                    reply = await task if task is not None else None
                except StopAsyncIteration:
                    reply = None
                    await close()
                except asyncio.CancelledError:
                    if not cancel.flag:
                        raise
                    reply = ("error", "cancelled", "cancelled")
                    await close()
                except (httpx.HTTPError, httpx.StreamError) as e:
                    reply = ("error",) + classify(e)
                    await close()
                finally:
                    cancel.hook = None
            elif isinstance(action, Sleep):
                await close()
                await asyncio.sleep(action.ms / 1000)
            elif isinstance(action, Call):
                try:
                    if action.blocking:
                        reply = await asyncio.to_thread(action.fn, *action.args)
                    else:
                        reply = action.fn(*action.args)
                    if inspect.isawaitable(reply):
                        reply = await reply
                except Exception as e:
                    reply = Failed(e)
            elif isinstance(action, Notify):
                yield action.event
            elif isinstance(action, Observe):
                observe(observer, action.hook, action.data)
    finally:
        await close()
        gen.close()
