"""The effect loop: turns the core's steps into actions a transport performs."""
from __future__ import annotations

import os
import select
import socket
import time
from urllib.parse import urlsplit
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional

from . import _core
from ._errors import AuthError, NetworkError, ProviderError, TimeoutError, UnsupportedError, error_from_dict
from ._keychain import read_keychain, write_keychain


@dataclass
class Http:
    """Send this request; reply with `("response", status, headers)` or `("error", kind, message)`."""

    request: dict[str, Any]


@dataclass
class Chunk:
    """Read the next body chunk; reply with bytes, `None` at the end, or an error tuple."""


@dataclass
class Sleep:
    """Wait this many milliseconds."""

    ms: int


@dataclass
class Call:
    """Run a host callback and reply with its value."""

    fn: Callable[..., Any]
    args: tuple[Any, ...] = ()
    blocking: bool = False


@dataclass
class Notify:
    """Deliver a stream event."""

    event: dict[str, Any]


@dataclass
class Observe:
    """Fire an observer hook."""

    hook: str
    data: Any


@dataclass
class Failed:
    """A callback raised."""

    error: BaseException


Action = Any


class Cancel:
    """A cancellation flag shared between a call and its driver."""

    def __init__(self) -> None:
        self.flag = False
        self.hook: Optional[Callable[[], Any]] = None

    def set(self) -> None:
        self.flag = True
        hook, self.hook = self.hook, None
        if hook is not None:
            try:
                hook()
            except Exception:
                pass


class Host:
    """What the driver needs from the provider besides the transport."""

    def __init__(self) -> None:
        self.secrets: dict[str, str] = {}
        self.resolvers: dict[str, Callable[[], Any]] = {}
        self.keychain: Callable[[str, str], Any] = read_keychain
        self.keychain_write: Optional[Callable[[str, str, bytes], Any]] = write_keychain


def clock() -> dict[str, Any]:
    """The current time and fresh entropy."""
    return {"now_ms": int(time.time() * 1000), "entropy": os.urandom(32)}


def _secret_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("secret")
    if isinstance(value, bytes):
        return value.decode()
    if not isinstance(value, str):
        raise TypeError("a resolver must return a string or {'secret': ...}")
    return value


def _read(locator: dict[str, Any], host: Host) -> dict[str, Any]:
    kind = locator.get("type")
    if kind == "keychain":
        try:
            value = host.keychain(locator["service"], locator["account"])
        except Exception:
            value = None
        if isinstance(value, str):
            value = value.encode()
        if not value:
            return {"type": "read", "found": False}
        return {"type": "read", "found": True, "bytes": bytes(value)}
    if kind == "file":
        try:
            with open(locator["path"], "rb") as f:
                data = f.read()
        except OSError:
            return {"type": "read", "found": False}
        return {"type": "read", "found": True, "bytes": data, "etag": _etag(data)}
    if kind == "env":
        value = os.environ.get(locator["name"])
        if value is None:
            return {"type": "read", "found": False}
        return {"type": "read", "found": True, "bytes": value.encode()}
    return {"type": "read", "found": False}


def _etag(data: bytes) -> str:
    import hashlib

    return hashlib.sha256(data).hexdigest()


def _write(effect: dict[str, Any], host: Host) -> dict[str, Any]:
    import base64

    locator = effect["locator"]
    data = base64.b64decode(effect["bytes"])
    if locator.get("type") == "keychain":
        if host.keychain_write is None:
            return {"type": "written", "ok": False}
        try:
            ok = bool(host.keychain_write(locator["service"], locator["account"], data))
        except Exception:
            ok = False
        return {"type": "written", "ok": ok}
    if locator.get("type") != "file":
        return {"type": "written", "ok": False}
    path = locator["path"]
    expect = effect.get("expect")
    if expect is not None:
        try:
            with open(path, "rb") as f:
                current = _etag(f.read())
        except OSError:
            current = None
        if current != expect:
            return {"type": "written", "ok": False, "conflict": True}
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(data)
        if effect.get("private", True):
            os.chmod(tmp, 0o600)
        os.replace(tmp, path)
        return {"type": "written", "ok": True}
    except OSError:
        return {"type": "written", "ok": False}


_PAGE = b"<!doctype html><title>Signed in</title><p>Sign-in finished. You can close this window.</p>"
_MAX_REQUEST = 8192


def _reply(conn: socket.socket, status: str, body: bytes) -> None:
    head = f"HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
    try:
        conn.sendall(head.encode() + body)
    except OSError:
        pass


def _request_target(conn: socket.socket) -> Optional[str]:
    conn.settimeout(3)
    data = b""
    try:
        while b"\r\n\r\n" not in data and len(data) < _MAX_REQUEST:
            chunk = conn.recv(1024)
            if not chunk:
                break
            data += chunk
    except OSError:
        return None
    parts = data.split(b"\r\n", 1)[0].decode("latin-1").split(" ")
    return parts[1] if len(parts) >= 2 else None


def _bind_callback(redirect_uri: str) -> tuple[socket.socket, str, str]:
    url = urlsplit(redirect_uri)
    host = url.hostname
    if host not in ("127.0.0.1", "::1"):
        raise UnsupportedError("the login callback must be a loopback address")
    port = url.port or 80
    shown = f"[{host}]" if ":" in host else host
    listener = socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM)
    try:
        if os.name == "posix":
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((host, port))
        listener.listen(8)
    except OSError as e:
        listener.close()
        raise NetworkError(
            f"{shown}:{port} is in use; close the other login or use callback='paste'", code="callback_port_busy"
        ) from e
    return listener, f"{url.scheme}://{shown}:{port}", url.path or "/"


def _await_callback(listener: socket.socket, origin: str, path: str, timeout_ms: int, cancel: Cancel) -> Optional[str]:
    deadline = time.monotonic() + timeout_ms / 1000
    try:
        while not cancel.flag:
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"no sign-in callback arrived within {max(1, round(timeout_ms / 1000))} seconds", code="login_timeout")
            if not select.select([listener], [], [], min(left, 0.1))[0]:
                continue
            try:
                conn, _ = listener.accept()
            except OSError:
                continue
            with conn:
                target = _request_target(conn)
                if target is not None and urlsplit(target).path == path:
                    _reply(conn, "200 OK", _PAGE)
                    return origin + target
                _reply(conn, "404 Not Found", b"not found")
        return None
    finally:
        listener.close()


def drive(
    start: dict[str, Any],
    host: Host,
    cancel: Cancel,
    handle_action: Optional[Callable[[dict[str, Any]], Any]] = None,
) -> Generator[Action, Any, dict[str, Any]]:
    """Runs one operation, yielding actions and returning the operation result."""
    op, step = _core.start({**start, "clock": clock()})
    while True:
        for note in step.get("notify") or ():
            if note["type"] == "event":
                yield Notify(note["event"])
            elif note["type"] == "observer":
                yield Observe(note["hook"], note["data"])
        effect = step["effect"]
        kind = effect["type"]
        if kind == "done":
            return effect["result"]
        if kind == "failed":
            raise error_from_dict(effect["error"])
        if cancel.flag:
            step = op.resume({"clock": clock(), "input": {"type": "cancelled"}})
            continue
        if kind == "need_http":
            reply = yield Http(effect["request"])
            if reply[0] == "response":
                data = {"type": "http_response", "status": reply[1], "headers": reply[2]}
            else:
                data = {"type": "http_error", "error": {"kind": reply[1], "message": reply[2]}}
        elif kind == "receive":
            reply = yield Chunk()
            if isinstance(reply, (bytes, bytearray)):
                step = op.feed(bytes(reply), clock())
                continue
            if reply is None:
                data = {"type": "http_end"}
            else:
                data = {"type": "http_error", "error": {"kind": reply[1], "message": reply[2]}}
        elif kind == "need_sleep":
            yield Sleep(int(effect["ms"]))
            data = {"type": "slept"}
        elif kind == "need_secret":
            ref = effect["secret_ref"]
            source = effect["source"]
            value: Optional[str] = None
            if source["type"] == "inline":
                value = host.secrets.get(ref)
            elif source["type"] == "env":
                for name in source.get("names", ()):
                    value = os.environ.get(name)
                    if value:
                        break
            elif source["type"] == "resolver":
                fn = host.resolvers.get(source["resolver_id"])
                reply = yield Call(fn) if fn else None
                if isinstance(reply, Failed):
                    raise AuthError("the credential resolver failed", code="resolver_failed") from reply.error
                value = _secret_value(reply)
            if value:
                op.set_secret(ref, value.encode())
            data = {"type": "secret", "secret_ref": ref, "found": bool(value)}
        elif kind == "need_read":
            data = _read(effect["locator"], host)
        elif kind == "need_write":
            data = _write(effect, host)
        elif kind == "need_user_action":
            if handle_action is None:
                raise UnsupportedError("this call cannot ask the user for anything")
            reply = yield Call(handle_action, (effect["action"],))
            if isinstance(reply, Failed):
                raise ProviderError("the login handler failed", kind="auth", code="user_action_failed") from reply.error
            data = {"type": "user_action", "value": reply if isinstance(reply, str) else None}
        elif kind == "need_callback":
            listener, origin, path = _bind_callback(effect["redirect_uri"])
            try:
                if handle_action is None:
                    raise UnsupportedError("this call cannot ask the user for anything")
                reply = yield Call(handle_action, (effect["action"],))
                if isinstance(reply, Failed):
                    raise ProviderError("the login handler failed", kind="auth", code="user_action_failed") from reply.error
            except BaseException:
                listener.close()
                raise
            url = yield Call(_await_callback, (listener, origin, path, int(effect.get("timeout_ms") or 300000), cancel), blocking=True)
            if isinstance(url, Failed):
                raise url.error
            if url is None:
                step = op.resume({"clock": clock(), "input": {"type": "cancelled"}})
                continue
            data = {"type": "callback", "url": url}
        else:
            raise UnsupportedError(f"the core asked for an unknown effect {kind!r}")
        step = op.resume({"clock": clock(), "input": data})
