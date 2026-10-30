/** The Provider class and its streams. */
import { Core, defaultCore, loadCore } from "./core.js";
import { drive } from "./driver.js";
import { NotFoundError } from "./errors.js";
import { readKeychain } from "./keychain.js";
import { isBrowser } from "./platform.js";
import { blocks, withViews } from "./sugar.js";

const NAMESPACES = ["anthropic", "openai", "google", "ollama"];
const OPTION_KEYS = new Set([
  "max_output_tokens",
  "reasoning",
  "reasoning_summary",
  "tool_choice",
  "parallel_tools",
  "output",
  "stop_sequences",
  "cache",
  "retries",
  "user_id",
]);
const TIMEOUT_KEYS = new Set(["connect", "read", "total"]);
const CALLBACKS = ["loopback", "paste"];

function timeoutOf(value) {
  if (value === undefined || value === null) return undefined;
  if (typeof value === "number") return { total: value };
  if (typeof value !== "object") throw new TypeError("timeout must be seconds or {connect, read, total}");
  for (const key of Object.keys(value)) if (!TIMEOUT_KEYS.has(key)) throw new TypeError(`unknown timeout key ${key}`);
  return dropNulls(value);
}

function dropNulls(object) {
  if (!object) return object;
  const out = {};
  for (const [k, v] of Object.entries(object)) if (v !== undefined && v !== null) out[k] = v;
  return out;
}

function toolChoiceOf(value) {
  if (typeof value === "string") return { mode: value };
  if (value && typeof value === "object" && !("mode" in value) && "name" in value) return { mode: "named", name: value.name };
  return value;
}

function optionsOf(value) {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "object") throw new TypeError("options must be an object");
  for (const key of Object.keys(value)) if (!OPTION_KEYS.has(key)) throw new TypeError(`unknown option ${key}`);
  const out = { ...value };
  if ("tool_choice" in out && out.tool_choice !== null) out.tool_choice = toolChoiceOf(out.tool_choice);
  return out;
}

function systemOf(value) {
  if (value === undefined || value === null) return undefined;
  if (typeof value === "string") return [{ type: "text", text: value }];
  return blocks(Array.isArray(value) ? value : [value]);
}

function toolsOf(value) {
  if (value === undefined || value === null) return undefined;
  if (!Array.isArray(value)) throw new TypeError("tools must be an array");
  for (const t of value) if (!t || typeof t.type !== "string") throw new TypeError("a tool needs a type: function or provider");
  return value;
}

function messagesOf(value) {
  if (!Array.isArray(value) || value.length === 0) throw new TypeError("messages must be a non-empty array");
  for (const m of value) if (!m || typeof m.role !== "string" || !Array.isArray(m.content)) throw new TypeError("a message needs role and content");
  return value;
}

function vendorOf(value) {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== "object") throw new TypeError("vendor must be an object keyed by namespace");
  const out = {};
  for (const [ns, fields] of Object.entries(value)) out[ns] = fields && typeof fields === "object" ? dropNulls(fields) : fields;
  return out;
}

function without(defaults, call) {
  if (!defaults || !call) return defaults;
  const nulls = new Set(Object.entries(call).filter(([, v]) => v === null).map(([k]) => k));
  const out = {};
  for (const [k, v] of Object.entries(defaults)) if (!nulls.has(k)) out[k] = v;
  return Object.keys(out).length ? out : undefined;
}

function withoutVendor(defaults, call) {
  if (!defaults || !call) return defaults;
  const out = {};
  for (const [ns, fields] of Object.entries(defaults)) {
    if (ns in call && call[ns] === null) continue;
    out[ns] = fields && typeof fields === "object" && call[ns] && typeof call[ns] === "object" ? without(fields, call[ns]) ?? {} : fields;
  }
  return Object.keys(out).length ? out : undefined;
}

class Cancel {
  constructor() {
    this.flag = false;
    this.abort = null;
  }

  set() {
    this.flag = true;
    this.abort?.();
  }
}

/** A streamed chat reply: iterate the events, `cancel()` to stop. */
export class Stream {
  constructor(gen, cancel) {
    this.gen = gen;
    this.cancelState = cancel;
    this.response = null;
  }

  [Symbol.asyncIterator]() {
    return this;
  }

  async next() {
    const r = await this.gen.next();
    if (r.done) return { done: true, value: undefined };
    if (r.value.type === "finish") this.response = withViews(r.value.response);
    return r;
  }

  async return() {
    await this.gen.return();
    return { done: true, value: undefined };
  }

  cancel() {
    this.cancelState.set();
  }
}

/** One endpoint of one vendor. */
export class Provider {
  constructor(ns, options = {}) {
    if (!NAMESPACES.includes(ns)) throw new TypeError(`unknown namespace ${ns}; one of ${NAMESPACES.join(", ")}`);
    const {
      base_url,
      auth,
      headers,
      timeout,
      default_model,
      default_embedding_model,
      default_system,
      default_tools,
      default_options,
      default_vendor,
      overrides,
      strict,
      observer,
      dangerously_allow_browser,
      fetch: doFetch,
      core,
      keychain,
      ...rest
    } = options;
    const unknown = Object.keys(rest);
    if (unknown.length) throw new TypeError(`unknown provider option ${unknown[0]}`);
    this.ns = ns;
    this.secrets = new Map();
    this.resolvers = new Map();
    this.observer = observer ?? null;
    this.fetch = doFetch ?? globalThis.fetch.bind(globalThis);
    if (keychain !== undefined && typeof keychain !== "function") throw new TypeError("keychain must be a function taking (service, account)");
    this.keychain = keychain ?? readKeychain;
    this.corePromise = core ? (core instanceof Core ? Promise.resolve(core) : core.then ? core : loadCore(core)) : null;
    const config = { ns };
    if (base_url) config.base_url = base_url.replace(/\/+$/, "");
    if (auth !== undefined || dangerously_allow_browser !== undefined) {
      config.auth = { sources: this.sourcesOf(auth) };
      if (dangerously_allow_browser !== undefined) config.auth.dangerously_allow_browser = !!dangerously_allow_browser;
    }
    if (isBrowser && !dangerously_allow_browser && config.auth?.sources.some((s) => s.type === "inline" || s.type === "env")) {
      throw new TypeError("an API key in a browser needs dangerously_allow_browser: true; prefer resolver() against your backend");
    }
    if (headers) config.headers = { ...headers };
    if (timeout !== undefined) config.timeout = timeoutOf(timeout);
    if (default_model) config.default_model = default_model;
    if (default_embedding_model) config.default_embedding_model = default_embedding_model;
    if (default_system !== undefined) config.default_system = systemOf(default_system);
    if (default_tools !== undefined) config.default_tools = toolsOf(default_tools);
    if (default_options !== undefined) config.default_options = dropNulls(optionsOf(default_options));
    if (default_vendor !== undefined) config.default_vendor = vendorOf(default_vendor);
    if (overrides !== undefined) config.overrides = overrides;
    if (strict !== undefined) config.strict = !!strict;
    this.config = config;
    this.cache = { models: null, modelInfo: new Map(), capabilities: new Map() };
  }

  sourcesOf(auth) {
    const items = Array.isArray(auth) ? auth : [auth];
    const sources = [];
    items.forEach((item, i) => {
      if (item === undefined || item === null) return;
      if (typeof item === "string") {
        const ref = `inline:${i}`;
        this.secrets.set(ref, item);
        sources.push({ type: "inline", secret_ref: ref });
      } else if (item && item.type === "resolver" && typeof item.fn === "function") {
        const { fn, ...source } = item;
        source.resolver_id = `resolver:${i}`;
        this.resolvers.set(source.resolver_id, fn);
        sources.push(source);
      } else if (item && typeof item.type === "string") {
        sources.push({ ...item });
      } else {
        throw new TypeError("auth must be a string, env(), session(), resolver() or an array of those");
      }
    });
    return sources;
  }

  get host() {
    return { secrets: this.secrets, resolvers: this.resolvers, keychain: this.keychain };
  }

  core() {
    return this.corePromise ?? defaultCore();
  }

  async run(start, { signal, handleAction } = {}) {
    const gen = drive(await this.core(), start, this.host, {
      fetch: this.fetch,
      signal,
      observer: this.observer,
      handleAction,
    });
    for (;;) {
      const r = await gen.next();
      if (r.done) return r.value;
    }
  }

  chatStart(stream, messages, options) {
    const { model, system, tools, options: callOptions, vendor, timeout, headers, strict, include_raw, signal, ...rest } = options;
    const unknown = Object.keys(rest);
    if (unknown.length) throw new TypeError(`unknown chat option ${unknown[0]}`);
    const chosen = model ?? this.config.default_model;
    if (!chosen) throw new TypeError("no model: pass model or set default_model");
    const request = { model: chosen, messages: messagesOf(messages) };
    const opts = optionsOf(callOptions);
    const ven = vendorOf(vendor);
    if (system !== undefined) request.system = systemOf(system);
    if (tools !== undefined) request.tools = toolsOf(tools);
    if (opts && Object.keys(opts).length) request.options = dropNulls(opts);
    if (ven) request.vendor = ven;
    if (timeout !== undefined) request.timeout = timeoutOf(timeout);
    if (headers) request.headers = { ...headers };
    if (strict !== undefined) request.strict = !!strict;
    if (include_raw !== undefined) request.include_raw = !!include_raw;
    const config = { ...this.config };
    config.default_options = without(config.default_options, opts);
    config.default_vendor = withoutVendor(config.default_vendor, ven);
    return { provider: dropNulls(config), operation: { type: stream ? "stream" : "chat", request } };
  }

  /** Sends the conversation and returns the whole reply. */
  async chat(messages, options = {}) {
    const start = this.chatStart(false, messages, options);
    const result = await this.run(start, { signal: options.signal });
    return withViews(result.response);
  }

  /** Sends the conversation and streams events; the last is `finish`. */
  stream(messages, options = {}) {
    const start = this.chatStart(true, messages, options);
    const cancel = new Cancel();
    const self = this;
    const gen = (async function* () {
      yield* drive(await self.core(), start, self.host, {
        fetch: self.fetch,
        signal: options.signal,
        cancel,
        observer: self.observer,
      });
    })();
    return new Stream(gen, cancel);
  }

  /** Embeds one string or a batch. */
  async embed(input, options = {}) {
    const { model, options: embedOptions, vendor, timeout, include_raw, signal, ...rest } = options;
    const unknown = Object.keys(rest);
    if (unknown.length) throw new TypeError(`unknown embed option ${unknown[0]}`);
    const texts = typeof input === "string" ? [input] : input;
    if (!Array.isArray(texts) || texts.length === 0 || !texts.every((t) => typeof t === "string")) {
      throw new TypeError("embed input must be a non-empty string or array of strings");
    }
    const chosen = model ?? this.config.default_embedding_model;
    if (!chosen) throw new TypeError("no embedding model: pass model or set default_embedding_model");
    const request = { model: chosen, input: texts };
    if (embedOptions) {
      for (const key of Object.keys(embedOptions)) if (!["dimensions", "retries"].includes(key)) throw new TypeError(`unknown embed option ${key}`);
      request.options = dropNulls(embedOptions);
    }
    if (vendor !== undefined) request.vendor = vendorOf(vendor);
    if (timeout !== undefined) request.timeout = timeoutOf(timeout);
    if (include_raw !== undefined) request.include_raw = !!include_raw;
    const result = await this.run({ provider: this.config, operation: { type: "embed", request } }, { signal });
    return result.response;
  }

  /** Lists the models the endpoint offers; cached until `refresh`. */
  async models({ refresh = false, signal } = {}) {
    if (!this.cache.models || refresh) {
      const result = await this.run({ provider: this.config, operation: { type: "models", refresh } }, { signal });
      this.cache.models = result.models;
    }
    return [...this.cache.models];
  }

  /** Describes one model; unknown ids are `not_found`. */
  async modelInfo(id, { signal } = {}) {
    if (!this.cache.modelInfo.has(id)) {
      const result = await this.run({ provider: this.config, operation: { type: "model_info", model: id } }, { signal });
      if (!result.info) throw new NotFoundError(`unknown model ${id}`, { code: "model_not_found" });
      this.cache.modelInfo.set(id, result.info);
    }
    return this.cache.modelInfo.get(id);
  }

  /** What a model can do, from the shipped table, the vendor or overrides. */
  async capabilities(model, { signal } = {}) {
    if (!this.cache.capabilities.has(model)) {
      const result = await this.run({ provider: this.config, operation: { type: "capabilities", model } }, { signal });
      if (!result.capabilities) throw new NotFoundError(`unknown model ${model}`, { code: "model_not_found" });
      this.cache.capabilities.set(model, result.capabilities);
    }
    return this.cache.capabilities.get(model);
  }

  /** Logs into the provider's session source; `handleAction` shows what the user must do. */
  async login(handleAction, { callback = "loopback", timeout, signal } = {}) {
    if (!CALLBACKS.includes(callback)) throw new TypeError(`callback must be one of ${CALLBACKS.join(", ")}`);
    const operation = { type: "login", callback };
    if (timeout !== undefined) operation.timeout_ms = Math.round(timeout * 1000);
    const result = await this.run({ provider: this.config, operation }, { signal, handleAction });
    return result.status;
  }

  /** Forgets the provider's own session. */
  async logout({ signal } = {}) {
    await this.run({ provider: this.config, operation: { type: "logout" } }, { signal });
  }

  /** Which credential the next call would use. */
  async authStatus({ signal } = {}) {
    const result = await this.run({ provider: this.config, operation: { type: "auth_status" } }, { signal });
    return result.status;
  }
}
