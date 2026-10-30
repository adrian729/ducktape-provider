"""Runs the Provider end to end over httpx mock transports built from the wire fixtures."""
import asyncio
import base64
import json
import os
import socket
import stat
import threading
import urllib.request
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from conftest import FIXTURES, load

import ducktape_provider as dp
from ducktape_provider import _driver
from ducktape_provider import AsyncProvider, Provider, assistant, tool, tool_result, user

ANTHROPIC = FIXTURES / "wire" / "anthropic.messages"


class Chunks(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = [c.encode() for c in chunks]

    def __iter__(self):
        yield from self.chunks

    async def __aiter__(self):
        for c in self.chunks:
            yield c


def scripted(fixture, seen=None):
    script = iter(fixture["http"])

    def handler(request):
        if seen is not None:
            seen.append(request)
        h = next(script)
        if h.get("error"):
            raise httpx.ConnectError(h["error"]["message"], request=request)
        chunks = h.get("chunks") or ([h["body"]] if h.get("body") is not None else [])
        return httpx.Response(h.get("status", 200), headers=h.get("headers", {}), stream=Chunks(chunks))

    return handler


def provider(name, **kwargs):
    fixture = load(ANTHROPIC / f"{name}.json")
    seen = []
    client = httpx.Client(transport=httpx.MockTransport(scripted(fixture, seen)))
    config = fixture["start"]["provider"]
    kwargs.setdefault("auth", "sk-ant-test-key")
    p = Provider(config["vendor"], client=client, base_url=config.get("base_url"), **kwargs)
    return p, fixture, seen


def done_response(fixture):
    return fixture["exchange"][-1]["step"]["effect"]["result"]["response"]


def without_auth_source(content):
    out = json.loads(json.dumps(content))
    for block in out:
        if isinstance(block.get("origin"), dict):
            block["origin"].pop("auth_source", None)
    return out


def request_of(fixture):
    return fixture["start"]["operation"]["request"]


def test_chat_matches_fixture():
    p, fixture, seen = provider("chat_text_reasoning")
    req = request_of(fixture)
    r = p.chat(req["messages"], model=req["model"], options=req.get("options"))
    expected = done_response(fixture)
    assert without_auth_source(r.to_dict()["content"]) == without_auth_source(expected["content"])
    assert r.stop.reason == expected["stop"]["reason"]
    assert r.usage.to_dict() == expected["usage"]
    assert r.text
    assert seen[0].headers["x-api-key"] == "sk-ant-test-key"
    assert json.loads(seen[0].content)["model"] == req["model"]


def test_stream_ends_with_finish():
    p, fixture, _ = provider("stream_text_reasoning_chunked")
    req = request_of(fixture)
    events = list(p.stream(req["messages"], model=req["model"]))
    assert events[-1].type == "finish"
    text = "".join(e.text for e in events if e.type == "text_delta")
    assert text == events[-1].response.text
    kinds = [e.type for e in events]
    assert kinds[0] == "block_start"
    assert "block_stop" in kinds


def test_tool_round_trip_builds_history():
    p, fixture, seen = provider("chat_tool_call")
    req = request_of(fixture)
    r = p.chat(req["messages"], model=req["model"], tools=req.get("tools"))
    assert r.stop.reason == "tool_call"
    call = r.tool_calls[0]
    history = list(req["messages"]) + [assistant(r), tool(tool_result(call.id, "4"))]
    assert history[-2].role == "assistant"
    assert history[-1].content[0].tool_call_id == call.id


def test_errors_map_to_classes():
    p, fixture, _ = provider("chat_context_overflow")
    req = request_of(fixture)
    with pytest.raises(dp.ContextOverflowError) as info:
        p.chat(req["messages"], model=req["model"])
    assert info.value.kind == "context_overflow"
    assert isinstance(info.value, dp.ProviderError)


def test_no_credential_is_auth_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    p = Provider("anthropic", default_model="m", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    with pytest.raises(dp.AuthError) as info:
        p.chat([user("hi")])
    assert info.value.code == "no_credential"


def test_env_credential_is_used(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    fixture = load(ANTHROPIC / "chat_text_reasoning.json")
    seen = []
    p = Provider("anthropic", client=httpx.Client(transport=httpx.MockTransport(scripted(fixture, seen))))
    req = request_of(fixture)
    p.chat(req["messages"], model=req["model"])
    assert seen[0].headers["x-api-key"] == "sk-from-env"
    assert p.auth_status().source == "env:ANTHROPIC_API_KEY"


def test_resolver_credential():
    fixture = load(ANTHROPIC / "chat_text_reasoning.json")
    seen = []
    p = Provider(
        "anthropic",
        auth=dp.resolver(lambda: {"secret": "sk-resolved"}),
        client=httpx.Client(transport=httpx.MockTransport(scripted(fixture, seen))),
    )
    req = request_of(fixture)
    p.chat(req["messages"], model=req["model"])
    assert seen[0].headers["x-api-key"] == "sk-resolved"


def test_argument_errors_before_any_request():
    p = Provider("anthropic", auth="k", client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500))))
    with pytest.raises(ValueError):
        p.chat([user("hi")])
    with pytest.raises(ValueError):
        p.chat([user("hi")], model="m", options={"temperature": 1})
    with pytest.raises(ValueError):
        p.embed([], model="e")
    with pytest.raises(ValueError):
        Provider("mistral")


def test_connection_error_is_retried_then_fails():
    calls = []

    def handler(request):
        calls.append(1)
        raise httpx.ConnectError("refused", request=request)

    p = Provider("anthropic", auth="k", client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(dp.NetworkError) as info:
        p.chat([user("hi")], model="m", options={"retries": 0})
    assert info.value.kind == "connection"
    assert len(calls) == 1


def test_models_are_cached():
    p, fixture, seen = provider("models_two_pages")
    first = p.models()
    second = p.models()
    assert [m.id for m in first] == [m.id for m in second]
    assert len(seen) == len(fixture["http"])


def test_capabilities_shipped():
    p, fixture, _ = provider("capabilities_shipped")
    caps = p.capabilities(fixture["start"]["operation"]["model"])
    assert caps.source == "shipped"
    assert caps.to_dict() == fixture["exchange"][-1]["step"]["effect"]["result"]["capabilities"]


def test_async_chat_and_stream():
    fixture = load(ANTHROPIC / "stream_text_reasoning_chunked.json")
    req = request_of(fixture)

    async def go():
        async with AsyncProvider(
            "anthropic", auth="k", client=httpx.AsyncClient(transport=httpx.MockTransport(scripted(fixture)))
        ) as p:
            events = [e async for e in p.stream(req["messages"], model=req["model"])]
        assert events[-1].type == "finish"
        chat = load(ANTHROPIC / "chat_text_reasoning.json")
        async with AsyncProvider(
            "anthropic", auth="k", client=httpx.AsyncClient(transport=httpx.MockTransport(scripted(chat)))
        ) as p:
            r = await p.chat(request_of(chat)["messages"], model=request_of(chat)["model"])
        assert without_auth_source(r.to_dict()["content"]) == without_auth_source(done_response(chat)["content"])

    asyncio.run(go())


def test_stream_cancel_raises_cancelled():
    p, fixture, _ = provider("stream_text_reasoning_chunked")
    req = request_of(fixture)
    s = p.stream(req["messages"], model=req["model"])
    first = next(s)
    assert first.type == "block_start"
    s.cancel()
    with pytest.raises(dp.CancelledError):
        for _ in s:
            pass


def test_image_bytes_become_base64():
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    m = user("what is this?", dp.image(png))
    assert m.content[1].data == base64.b64encode(png).decode()
    assert m.to_dict()["content"][0] == {"type": "text", "text": "what is this?"}


def session_store(fixture, tmp_path, prefix):
    for path, text in fixture.get("files", {}).items():
        target = tmp_path / path.removeprefix(prefix).lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


def test_session_profile_refreshes_and_writes_back(monkeypatch, tmp_path):
    fixture = load(ANTHROPIC / "session_profile_refresh_write_back.json")
    session_store(fixture, tmp_path, "/home/u/.config/anthropic")
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_PROFILE", raising=False)
    seen = []
    p = Provider(
        "anthropic", auth=dp.session("anthropic_profile"), client=httpx.Client(transport=httpx.MockTransport(scripted(fixture, seen)))
    )
    req = request_of(fixture)
    r = p.chat(req["messages"], model=req["model"])
    assert r.text == done_response(fixture)["content"][-1]["text"]
    assert seen[0].url.path == "/v1/oauth/token"
    assert seen[1].headers["authorization"] == "Bearer sk-ant-oat01-new"
    assert seen[1].headers["anthropic-beta"] == "oauth-2025-04-20"
    creds = tmp_path / "credentials" / "default.json"
    written = json.loads(creds.read_text())
    assert written["access_token"] == "sk-ant-oat01-new" and written["extra"] == "kept"
    assert oct(creds.stat().st_mode & 0o777) == "0o600"
    assert dp.session("codex", refresh="memory").refresh == "memory"
    with pytest.raises(ValueError):
        dp.session("codex", refresh="sometimes")


def test_session_expired_names_the_owner(monkeypatch, tmp_path):
    fixture = load(ANTHROPIC / "session_claude_code_expired.json")
    session_store(fixture, tmp_path, "/home/u/.claude")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))
    p = Provider("anthropic", auth=dp.session("claude_code"), client=httpx.Client(transport=httpx.MockTransport(scripted(fixture))))
    with pytest.raises(dp.AuthError) as info:
        p.chat([user("hi")], model="m")
    assert info.value.code == "credential_expired" and "claude" in str(info.value)
    status = p.auth_status()
    assert status.source == "session:claude_code" and status.expires_at == "2025-10-28T23:59:55Z"


def test_keychain_reader_uses_platform_tools(monkeypatch):
    from ducktape_provider import _keychain

    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        import subprocess

        if argv[0] == "security":
            return subprocess.CompletedProcess(argv, 0, stdout=b"from-mac\n", stderr=b"")
        if argv[0] == "secret-tool":
            return subprocess.CompletedProcess(argv, 1, stdout=b"", stderr=b"")
        raise AssertionError(argv)

    monkeypatch.setattr(_keychain.subprocess, "run", fake_run)
    monkeypatch.setattr(_keychain.sys, "platform", "darwin")
    assert _keychain.read_keychain("Codex Auth", "cli|abc") == b"from-mac"
    assert calls[-1] == ["security", "find-generic-password", "-s", "Codex Auth", "-a", "cli|abc", "-w"]
    monkeypatch.setattr(_keychain.sys, "platform", "linux")
    assert _keychain.read_keychain("Codex Auth", "cli|abc") is None
    assert calls[-1] == ["secret-tool", "lookup", "service", "Codex Auth", "username", "cli|abc"]
    monkeypatch.setattr(_keychain.sys, "platform", "sunos5")
    assert _keychain.read_keychain("Codex Auth", "cli|abc") is None


def test_codex_keychain_fallback(monkeypatch, tmp_path):
    fixture = load(FIXTURES / "wire" / "openai.responses" / "session_codex_keychain_fallback.json")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (service_account, stored), = fixture["keychain"].items()
    service, account = service_account.split("|", 1)
    import time

    def jwt(payload):
        enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
        return f"{enc({'alg': 'RS256'})}.{enc(payload)}.sig"

    doc = json.loads(stored)
    doc["tokens"]["access_token"] = jwt({"exp": int(time.time()) + 3600})
    asked = []

    def reader(s, a):
        asked.append((s, a))
        return json.dumps(doc) if s == service else None

    seen = []
    p = Provider("openai", auth=dp.session("codex"), keychain=reader, client=httpx.Client(transport=httpx.MockTransport(scripted(fixture, seen))))
    req = request_of(fixture)
    r = p.chat(req["messages"], model=req["model"])
    assert r.text == done_response(fixture)["content"][-1]["text"]
    assert asked[0][0] == service == "Codex Auth" and asked[0][1].startswith("cli|") and asked[0][1] != account
    assert seen[0].headers["chatgpt-account-id"] == "acct_1"
    with pytest.raises(TypeError):
        Provider("openai", keychain="not callable")


def test_write_conflict_keeps_the_refreshed_token(monkeypatch, tmp_path):
    fixture = load(ANTHROPIC / "session_profile_write_conflict_keeps_token.json")
    session_store(fixture, tmp_path, "/home/u/.config/anthropic")
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("ANTHROPIC_PROFILE", raising=False)
    creds = tmp_path / "credentials" / "default.json"
    inner = scripted(fixture)

    def handler(request):
        if request.url.path == "/v1/oauth/token":
            creds.write_text(json.dumps({"access_token": "sk-ant-oat01-other-process", "expires_at": 4102444800}))
        return inner(request)

    p = Provider("anthropic", auth=dp.session("anthropic_profile"), client=httpx.Client(transport=httpx.MockTransport(handler)))
    req = request_of(fixture)
    r = p.chat(req["messages"], model=req["model"])
    assert r.text == done_response(fixture)["content"][-1]["text"]
    assert json.loads(creds.read_text())["access_token"] == "sk-ant-oat01-other-process"


def test_codex_keychain_write_back_uses_the_writer(monkeypatch, tmp_path):
    fixture = load(FIXTURES / "wire" / "openai.responses" / "session_codex_keychain_write_back.json")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path))
    (service_account, stored), = fixture["keychain"].items()
    service, _ = service_account.split("|", 1)
    import time

    def jwt(payload):
        enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()
        return f"{enc({'alg': 'RS256'})}.{enc(payload)}.sig"

    doc = json.loads(stored)
    doc["tokens"]["access_token"] = jwt({"exp": int(time.time()) + 60})
    fresh = json.loads(fixture["http"][0]["body"])
    fresh["access_token"] = jwt({"exp": int(time.time()) + 86400})
    fixture["http"][0]["body"] = json.dumps(fresh)
    written = []
    reader = lambda s, a: json.dumps(doc) if s == service else None

    def writer(s, a, value):
        written.append((s, a, json.loads(value)))
        return True

    p = Provider(
        "openai",
        auth=dp.session("codex", refresh="write_back"),
        keychain=reader,
        keychain_write=writer,
        client=httpx.Client(transport=httpx.MockTransport(scripted(fixture))),
    )
    req = request_of(fixture)
    r = p.chat(req["messages"], model=req["model"])
    assert r.text == done_response(fixture)["content"][-1]["text"]
    assert written[0][0] == "Codex Auth" and written[0][1].startswith("cli|") and written[0][2]["tokens"]["refresh_token"] == "rt_d"
    reader_only = Provider("openai", auth=dp.session("codex", refresh="write_back"), keychain=reader, client=httpx.Client(transport=httpx.MockTransport(scripted(fixture))))
    assert reader_only.host.keychain_write is None
    assert reader_only.chat(req["messages"], model=req["model"]).text == r.text
    with pytest.raises(TypeError):
        Provider("openai", keychain_write="nope")


def test_login_loopback_callback_completes_and_times_out(monkeypatch, tmp_path):
    monkeypatch.setattr(_driver, "clock", lambda: {"now_ms": 0, "entropy": b"\x80" * 32})
    fixture = load(FIXTURES / "wire" / "openai.responses" / "login_chatgpt_plan_new_registration.json")
    store = tmp_path / "chatgpt_plan.json"
    p = Provider(
        "openai",
        client=httpx.Client(transport=httpx.MockTransport(scripted(fixture))),
        auth=dp.session("chatgpt_plan", store=str(store)),
    )
    statuses = []

    def visit(url):
        query = parse_qs(urlsplit(url).query)
        redirect = query["redirect_uri"][0]
        target = f"{redirect}?code=code_1&state={query['state'][0]}&client_id=oaiapp_test"

        def get():
            with urllib.request.urlopen(target, timeout=5) as reply:
                statuses.append(reply.status)

        thread = threading.Thread(target=get)
        thread.start()
        return thread

    threads = []

    def handler(action):
        if action.type == "open_url":
            threads.append(visit(action.url))

    status = p.login(handler, app_name="Example App")
    for t in threads:
        t.join()
    assert statuses == [200]
    assert status.source == "session:chatgpt_plan" and status.session == "chatgpt_plan"
    assert "oaiapp_test" in store.read_text()
    if os.name == "posix":
        assert stat.S_IMODE(store.stat().st_mode) == 0o600

    other = Provider(
        "openai",
        client=httpx.Client(transport=httpx.MockTransport(scripted(fixture))),
        auth=dp.session("chatgpt_plan", store=str(tmp_path / "other.json")),
    )
    with pytest.raises(dp.TimeoutError) as info:
        other.login(lambda action: None, timeout=0.3)
    assert info.value.code == "login_timeout"
    free = socket.socket()
    free.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    free.bind(("127.0.0.1", 1455))
    free.close()
