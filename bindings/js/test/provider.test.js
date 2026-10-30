import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { test } from "node:test";

import * as dp from "../src/index.js";
import { Provider, assistant, tool, toolResult, user } from "../src/index.js";
import { doneResponse, fixtures, load, requestOf, scripted, withoutAuthSource } from "./helpers.js";

const anthropic = (name) => load(join(fixtures, "wire", "anthropic.messages", `${name}.json`));

function provider(name, options = {}) {
  const fixture = anthropic(name);
  const seen = [];
  const config = fixture.start.provider;
  const p = new Provider(config.ns, { base_url: config.base_url, auth: "sk-ant-test-key", fetch: scripted(fixture, seen), ...options });
  return { p, fixture, seen };
}

test("chat matches the fixture", async () => {
  const { p, fixture, seen } = provider("chat_text_reasoning");
  const req = requestOf(fixture);
  const r = await p.chat(req.messages, { model: req.model, options: req.options });
  const expected = doneResponse(fixture);
  assert.deepEqual(withoutAuthSource(r.content), withoutAuthSource(expected.content));
  assert.equal(r.stop.reason, expected.stop.reason);
  assert.deepEqual(r.usage, expected.usage);
  assert.ok(r.text.length > 0);
  assert.equal(JSON.stringify(r).includes('"text"'), true);
  assert.equal(seen[0].init.headers["x-api-key"], "sk-ant-test-key");
  assert.equal(seen[0].body.model, req.model);
});

test("stream ends with finish and keeps text", async () => {
  const { p, fixture } = provider("stream_text_reasoning_chunked");
  const req = requestOf(fixture);
  const events = [];
  const s = p.stream(req.messages, { model: req.model });
  for await (const ev of s) events.push(ev);
  assert.equal(events.at(-1).type, "finish");
  const text = events.filter((e) => e.type === "text_delta").map((e) => e.text).join("");
  assert.equal(text, s.response.text);
  assert.equal(events[0].type, "block_start");
});

test("tool round trip builds history", async () => {
  const { p, fixture } = provider("chat_tool_call");
  const req = requestOf(fixture);
  const r = await p.chat(req.messages, { model: req.model, tools: req.tools });
  assert.equal(r.stop.reason, "tool_call");
  const call = r.toolCalls[0];
  const history = [...req.messages, assistant(r), tool(toolResult(call.id, "4"))];
  assert.equal(history.at(-2).role, "assistant");
  assert.equal(history.at(-1).content[0].tool_call_id, call.id);
});

test("errors map to classes", async () => {
  const { p, fixture } = provider("chat_context_overflow");
  const req = requestOf(fixture);
  await assert.rejects(p.chat(req.messages, { model: req.model }), (e) => {
    assert.ok(e instanceof dp.ContextOverflowError);
    assert.ok(e instanceof dp.ProviderError);
    assert.equal(e.kind, "context_overflow");
    return true;
  });
});

test("no credential is an auth error", async () => {
  delete process.env.ANTHROPIC_API_KEY;
  delete process.env.ANTHROPIC_AUTH_TOKEN;
  const p = new Provider("anthropic", { default_model: "m", fetch: async () => new Response("", { status: 500 }) });
  await assert.rejects(p.chat([user("hi")]), (e) => e instanceof dp.AuthError && e.code === "no_credential");
});

test("env credential is used", async () => {
  process.env.ANTHROPIC_API_KEY = "sk-from-env";
  try {
    const fixture = anthropic("chat_text_reasoning");
    const seen = [];
    const p = new Provider("anthropic", { fetch: scripted(fixture, seen) });
    const req = requestOf(fixture);
    await p.chat(req.messages, { model: req.model });
    assert.equal(seen[0].init.headers["x-api-key"], "sk-from-env");
    assert.equal((await p.authStatus()).source, "env:ANTHROPIC_API_KEY");
  } finally {
    delete process.env.ANTHROPIC_API_KEY;
  }
});

test("resolver credential, sync or async", async () => {
  const fixture = anthropic("chat_text_reasoning");
  const seen = [];
  const p = new Provider("anthropic", { auth: dp.resolver(async () => ({ secret: "sk-resolved" })), fetch: scripted(fixture, seen) });
  const req = requestOf(fixture);
  await p.chat(req.messages, { model: req.model });
  assert.equal(seen[0].init.headers["x-api-key"], "sk-resolved");
});

test("argument errors come before any request", async () => {
  const p = new Provider("anthropic", { auth: "k", fetch: async () => new Response("", { status: 500 }) });
  await assert.rejects(p.chat([user("hi")]), TypeError);
  await assert.rejects(p.chat([user("hi")], { model: "m", options: { temperature: 1 } }), TypeError);
  await assert.rejects(p.embed([], { model: "e" }), TypeError);
  assert.throws(() => new Provider("mistral"), TypeError);
});

test("connection failure maps to a network error", async () => {
  let calls = 0;
  const p = new Provider("anthropic", {
    auth: "k",
    fetch: async () => {
      calls += 1;
      throw new TypeError("fetch failed", { cause: { code: "ECONNREFUSED" } });
    },
  });
  await assert.rejects(p.chat([user("hi")], { model: "m", options: { retries: 0 } }), (e) => e instanceof dp.NetworkError);
  assert.equal(calls, 1);
});

test("models are cached", async () => {
  const { p, fixture, seen } = provider("models_two_pages");
  const first = await p.models();
  const second = await p.models();
  assert.deepEqual(first.map((m) => m.id), second.map((m) => m.id));
  assert.equal(seen.length, fixture.http.length);
});

test("capabilities come from the shipped table", async () => {
  const { p, fixture } = provider("capabilities_shipped");
  const caps = await p.capabilities(fixture.start.operation.model);
  assert.deepEqual(caps, fixture.exchange.at(-1).step.effect.result.capabilities);
});

test("stream cancel raises cancelled", async () => {
  const { p, fixture } = provider("stream_text_reasoning_chunked");
  const req = requestOf(fixture);
  const s = p.stream(req.messages, { model: req.model });
  const first = await s.next();
  assert.equal(first.value.type, "block_start");
  s.cancel();
  await assert.rejects(
    (async () => {
      for await (const _ of s) void _;
    })(),
    (e) => e instanceof dp.CancelledError,
  );
});

test("abort signal cancels a chat", async () => {
  const fixture = anthropic("chat_text_reasoning");
  const controller = new AbortController();
  const p = new Provider("anthropic", {
    auth: "k",
    fetch: (url, init) => {
      controller.abort();
      return scripted(fixture)(url, init);
    },
  });
  const req = requestOf(fixture);
  await assert.rejects(p.chat(req.messages, { model: req.model, signal: controller.signal }), (e) => e instanceof dp.CancelledError);
});

test("image bytes become base64", () => {
  const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0, 0, 0, 0]);
  const m = user("what is this?", dp.image(png));
  assert.equal(m.content[1].data, Buffer.from(png).toString("base64"));
  assert.deepEqual(m.content[0], { type: "text", text: "what is this?" });
});

function sessionStore(fixture, dir, prefix) {
  for (const [path, text] of Object.entries(fixture.files ?? {})) {
    const target = join(dir, path.slice(prefix.length));
    mkdirSync(dirname(target), { recursive: true });
    writeFileSync(target, text);
  }
}

test("codex session refreshes and writes back", async () => {
  const fixture = load(join(fixtures, "wire", "openai.responses", "session_codex_refresh_write_back.json"));
  const b64url = (o) => Buffer.from(JSON.stringify(o)).toString("base64url");
  const fresh = `${b64url({ alg: "RS256" })}.${b64url({ exp: Math.floor(Date.now() / 1000) + 86400 })}.sig`;
  fixture.http[0].body = JSON.stringify({ ...JSON.parse(fixture.http[0].body), access_token: fresh });
  const dir = mkdtempSync(join(tmpdir(), "ducktape-codex-"));
  sessionStore(fixture, dir, "/home/u/.codex");
  process.env.CODEX_HOME = dir;
  try {
    const seen = [];
    const p = new Provider("openai", { auth: dp.session("codex", { refresh: "write_back" }), fetch: scripted(fixture, seen) });
    const req = requestOf(fixture);
    const r = await p.chat(req.messages, { model: req.model });
    assert.equal(r.text, doneResponse(fixture).content.at(-1).text);
    assert.equal(seen[0].url, "https://auth.openai.com/oauth/token");
    assert.equal(seen[1].url, "https://chatgpt.com/backend-api/codex/responses");
    assert.equal(seen[1].init.headers["chatgpt-account-id"], "acct_1");
    const written = JSON.parse(readFileSync(join(dir, "auth.json"), "utf8"));
    assert.equal(written.tokens.refresh_token, "rt_d");
    assert.equal((await p.authStatus()).session, "codex");
    assert.throws(() => dp.session("codex", { refresh: "sometimes" }), TypeError);
  } finally {
    delete process.env.CODEX_HOME;
    rmSync(dir, { recursive: true, force: true });
  }
});

test("expired borrowed session names the owner", async () => {
  const fixture = load(join(fixtures, "wire", "openai.responses", "session_codex_expired_never.json"));
  const dir = mkdtempSync(join(tmpdir(), "ducktape-codex-"));
  sessionStore(fixture, dir, "/home/u/.codex");
  process.env.CODEX_HOME = dir;
  try {
    const p = new Provider("openai", { auth: dp.session("codex"), fetch: scripted(fixture) });
    await assert.rejects(p.chat([user("hi")], { model: "m" }), (e) => e instanceof dp.AuthError && e.code === "credential_expired" && /codex login/.test(e.message));
    const status = await p.authStatus();
    assert.equal(status.source, "session:codex");
    assert.equal(status.expires_at, "2025-10-28T23:59:55Z");
  } finally {
    delete process.env.CODEX_HOME;
    rmSync(dir, { recursive: true, force: true });
  }
});

test("codex keychain fallback goes through the keychain reader", async () => {
  const fixture = load(join(fixtures, "wire", "openai.responses", "session_codex_keychain_fallback.json"));
  const dir = mkdtempSync(join(tmpdir(), "ducktape-codex-"));
  process.env.CODEX_HOME = dir;
  try {
    const [[key, stored]] = Object.entries(fixture.keychain);
    const [service, account] = key.split("|", 2);
    const b64url = (o) => Buffer.from(JSON.stringify(o)).toString("base64url");
    const doc = JSON.parse(stored);
    doc.tokens.access_token = `${b64url({ alg: "RS256" })}.${b64url({ exp: Math.floor(Date.now() / 1000) + 3600 })}.sig`;
    const asked = [];
    const keychain = async (s, a) => {
      asked.push([s, a]);
      return s === service ? JSON.stringify(doc) : null;
    };
    const seen = [];
    const p = new Provider("openai", { auth: dp.session("codex"), keychain, fetch: scripted(fixture, seen) });
    const req = requestOf(fixture);
    const r = await p.chat(req.messages, { model: req.model });
    assert.equal(r.text, doneResponse(fixture).content.at(-1).text);
    assert.equal(asked[0][0], "Codex Auth");
    assert.ok(asked[0][1].startsWith("cli|") && asked[0][1] !== account);
    assert.equal(seen[0].init.headers["chatgpt-account-id"], "acct_1");
    assert.throws(() => new Provider("openai", { keychain: "nope" }), TypeError);
  } finally {
    delete process.env.CODEX_HOME;
    rmSync(dir, { recursive: true, force: true });
  }
});

test("the default keychain reader never throws", async () => {
  const { readKeychain } = await import("../src/keychain.js");
  const value = await readKeychain("ducktape-provider-test", "nobody");
  assert.ok(value === null || value instanceof Uint8Array);
});

test("a loaded core can be shared between providers", async () => {
  const core = await dp.loadCore();
  const { p, fixture, seen } = provider("chat_text_reasoning", { core });
  const req = requestOf(fixture);
  const r = await p.chat(req.messages, { model: req.model });
  assert.equal(r.text, doneResponse(fixture).content.at(-1).text);
  assert.equal(seen.length, 1);
});
