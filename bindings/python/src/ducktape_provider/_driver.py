"""The effect loop: turns the core's steps into actions a transport performs."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Generator, Optional

from . import _core
from ._errors import AuthError, ProviderError, UnsupportedError, error_from_dict
from ._keychain import read_keychain


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
                return {"type": "read", "found": True, "bytes": f.read()}
        except OSError:
            return {"type": "read", "found": False}
    if kind == "env":
        value = os.environ.get(locator["name"])
        if value is None:
            return {"type": "read", "found": False}
        return {"type": "read", "found": True, "bytes": value.encode()}
    return {"type": "read", "found": False}


def _write(effect: dict[str, Any]) -> dict[str, Any]:
    import base64

    locator = effect["locator"]
    if locator.get("type") != "file":
        return {"type": "written", "ok": False}
    path = locator["path"]
    data = base64.b64decode(effect["bytes"])
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
            data = _write(effect)
        elif kind == "need_user_action":
            if handle_action is None:
                raise UnsupportedError("this call cannot ask the user for anything")
            reply = yield Call(handle_action, (effect["action"],))
            if isinstance(reply, Failed):
                raise ProviderError("the login handler failed", kind="auth", code="user_action_failed") from reply.error
            data = {"type": "user_action", "value": reply if isinstance(reply, str) else None}
        elif kind == "need_callback":
            raise UnsupportedError("the loopback login callback is not implemented in the Python binding yet")
        else:
            raise UnsupportedError(f"the core asked for an unknown effect {kind!r}")
        step = op.resume({"clock": clock(), "input": data})
