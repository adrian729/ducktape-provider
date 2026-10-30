/** The effect loop: runs one operation of the core over fetch. */
import { errorFromData, UnsupportedError, AuthError, CancelledError, NetworkError, TimeoutError } from "./errors.js";
import { fromBase64, getEnv, randomBytes, etagOf, readFile, toBase64, writeFile } from "./platform.js";

/** The current time and fresh entropy. */
export function clock() {
  return { now_ms: Date.now(), entropy: toBase64(randomBytes(32)) };
}

function classify(error) {
  const message = error?.message || String(error);
  const code = error?.cause?.code ?? error?.code ?? "";
  const lowered = `${code} ${message} ${error?.cause?.message ?? ""}`.toLowerCase();
  if (error?.name === "TimeoutError" || code === "UND_ERR_CONNECT_TIMEOUT" || code === "ETIMEDOUT") return ["timeout", message];
  if (lowered.includes("enotfound") || lowered.includes("eai_again") || lowered.includes("getaddrinfo")) return ["dns", message];
  if (lowered.includes("cert") || lowered.includes("tls") || lowered.includes("ssl")) return ["tls", message];
  if (lowered.includes("econnreset") || lowered.includes("socket hang up") || lowered.includes("terminated")) return ["reset", message];
  if (lowered.includes("econnrefused") || lowered.includes("ehostunreach") || lowered.includes("fetch failed")) return ["connect", message];
  return ["other", message];
}

function secretOf(value) {
  if (value === undefined || value === null) return null;
  if (typeof value === "object") value = value.secret;
  if (value instanceof Uint8Array) return new TextDecoder().decode(value);
  if (typeof value !== "string") throw new TypeError("a resolver must return a string or {secret}");
  return value;
}

function sleep(ms, signal) {
  return new Promise((resolve) => {
    const timer = setTimeout(done, ms);
    function done() {
      signal?.removeEventListener("abort", done);
      clearTimeout(timer);
      resolve();
    }
    signal?.addEventListener("abort", done, { once: true });
  });
}

const CALLBACK_HEADER_LIMIT = 8192;
const CALLBACK_CONNECTION_MS = 5000;
const CALLBACK_PAGE = "<!doctype html><meta charset=utf-8><title>Signed in</title><p>The sign-in finished. You can close this window.</p>";

function callbackReply(status, reason, body) {
  const length = Buffer.byteLength(body);
  return `HTTP/1.1 ${status} ${reason}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: ${length}\r\nConnection: close\r\n\r\n${body}`;
}

/** Waits on a loopback listener for the sign-in redirect and returns its request target. */
async function awaitLoopbackCallback(effect, signal, showAction) {
  let net;
  try {
    net = await import("node:net");
  } catch {
    net = null;
  }
  if (!net?.createServer) {
    throw new UnsupportedError('the loopback login callback needs Node; headless apps can use callback "paste"');
  }
  const redirect = new URL(effect.redirect_uri);
  const host = redirect.hostname === "[::1]" ? "::1" : redirect.hostname;
  if (host !== "127.0.0.1" && host !== "::1") {
    throw new UnsupportedError(`the login callback host ${redirect.hostname} is not a loopback address`);
  }
  const port = Number(redirect.port);
  const timeoutMs = effect.timeout_ms ?? 300000;
  const server = net.createServer();
  const sockets = new Set();
  let finish;
  const outcome = new Promise((resolve) => {
    finish = resolve;
  });
  server.on("connection", (socket) => {
    sockets.add(socket);
    socket.on("close", () => sockets.delete(socket));
    socket.on("error", () => socket.destroy());
    socket.setTimeout(CALLBACK_CONNECTION_MS, () => socket.destroy());
    let received = Buffer.alloc(0);
    const onData = (chunk) => {
      received = Buffer.concat([received, chunk]);
      const end = received.indexOf("\r\n\r\n");
      if (end < 0 && received.length <= CALLBACK_HEADER_LIMIT) return;
      socket.off("data", onData);
      const line = received.toString("latin1", 0, end < 0 ? received.length : end).split("\r\n")[0];
      const [method, target] = line.split(" ");
      const path = target?.split("?")[0];
      if (method === "GET" && path === redirect.pathname) {
        socket.end(callbackReply(200, "OK", CALLBACK_PAGE));
        finish({ target });
      } else {
        socket.end(callbackReply(404, "Not Found", "not found"));
      }
    };
    socket.on("data", onData);
  });
  try {
    await new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen({ host, port }, resolve);
    });
  } catch (error) {
    throw new NetworkError(`${redirect.host} is in use; close the other login or use callback: 'paste'`, { code: "callback_port_busy", cause: error });
  }
  let timer;
  const onAbort = () => finish({ cancelled: true });
  try {
    if (signal?.aborted) return { cancelled: true };
    signal?.addEventListener("abort", onAbort, { once: true });
    timer = setTimeout(() => finish({ timedOut: true }), timeoutMs);
    await showAction();
    const result = await outcome;
    if (result.timedOut) {
      throw new TimeoutError(`no sign-in callback arrived within ${Math.round(timeoutMs / 1000)} seconds`, { code: "login_timeout" });
    }
    return result;
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
    for (const socket of sockets) socket.destroy();
    await new Promise((resolve) => server.close(resolve));
  }
}

class Timer {
  constructor(seconds, onFire) {
    this.ms = seconds ? seconds * 1000 : 0;
    this.onFire = onFire;
    this.handle = null;
  }

  arm() {
    this.clear();
    if (this.ms) this.handle = setTimeout(this.onFire, this.ms);
  }

  clear() {
    if (this.handle) clearTimeout(this.handle);
    this.handle = null;
  }
}

/** Runs one operation: yields stream events and returns the operation result. */
export async function* drive(core, start, host, { fetch: doFetch, signal, cancel, handleAction, observer }) {
  const [op, first] = core.start({ ...start, clock: clock() });
  if (!op) throw errorFromData(first.effect.error ?? { kind: "bad_request", message: "invalid start document" });
  let step = first;
  const notes = function* (next) {
    step = next;
    for (const note of step.notify ?? []) {
      if (note.type === "event") yield note.event;
      else if (note.type === "observer") observer?.[note.hook]?.(note.data);
    }
  };
  let reader = null;
  let controller = null;
  let lastValue;
  const showAction = async (action) => {
    if (!handleAction) throw new UnsupportedError("this call cannot ask the user for anything");
    try {
      lastValue = await handleAction(action);
    } catch (error) {
      throw new AuthError("the login handler failed", { code: "user_action_failed", cause: error });
    }
  };
  const closeBody = () => {
    if (reader) {
      reader.cancel().catch(() => {});
      reader = null;
    }
  };
  try {
    yield* notes(first);
    for (;;) {
      const effect = step.effect;
      if (effect.type === "done") return effect.result;
      if (effect.type === "failed") throw errorFromData(effect.error);
      if (cancel?.flag || signal?.aborted) {
        closeBody();
        yield* notes(core.resume(op, { clock: clock(), input: { type: "cancelled" } }));
        continue;
      }
      let input;
      switch (effect.type) {
        case "need_http": {
          closeBody();
          const req = effect.request;
          controller = new AbortController();
          let timedOut = false;
          const abort = () => controller.abort();
          signal?.addEventListener("abort", abort, { once: true });
          if (cancel) cancel.abort = abort;
          const headersWait = req.timeout?.connect ? req.timeout.connect + (req.timeout.read || 0) : 0;
          const connect = new Timer(headersWait, () => {
            timedOut = true;
            controller.abort();
          });
          const read = new Timer(req.timeout?.read, () => {
            timedOut = true;
            controller.abort();
          });
          connect.arm();
          try {
            const response = await doFetch(req.url, {
              method: req.method,
              headers: req.headers,
              body: req.body ?? undefined,
              signal: controller.signal,
              redirect: "manual",
            });
            connect.clear();
            const headers = {};
            response.headers.forEach((value, name) => {
              headers[name.toLowerCase()] = value;
            });
            reader = response.body ? response.body.getReader() : null;
            input = { type: "http_response", status: response.status, headers };
            yield* notes(core.resume(op, { clock: clock(), input }));
            while (step.effect.type === "receive") {
              if (cancel?.flag || signal?.aborted) break;
              read.arm();
              let chunk;
              try {
                chunk = reader ? await reader.read() : { done: true };
              } catch (error) {
                read.clear();
                closeBody();
                const kind = timedOut ? "timeout" : controller.signal.aborted ? "cancelled" : classify(error)[0];
                yield* notes(core.resume(op, { clock: clock(), input: { type: "http_error", error: { kind, message: error?.message ?? kind } } }));
                break;
              }
              read.clear();
              if (chunk.done) {
                reader = null;
                yield* notes(core.resume(op, { clock: clock(), input: { type: "http_end" } }));
              } else {
                yield* notes(core.feed(op, chunk.value, clock()));
              }
            }
            continue;
          } catch (error) {
            connect.clear();
            const kind = timedOut ? "timeout" : controller.signal.aborted ? "cancelled" : classify(error)[0];
            input = { type: "http_error", error: { kind, message: error?.message ?? kind } };
          } finally {
            signal?.removeEventListener("abort", abort);
            if (cancel) cancel.abort = null;
          }
          break;
        }
        case "receive":
          input = { type: "http_end" };
          break;
        case "need_sleep":
          closeBody();
          await sleep(effect.ms, signal);
          input = { type: "slept" };
          break;
        case "need_secret": {
          const ref = effect.secret_ref;
          const source = effect.source;
          let value = null;
          if (source.type === "inline") value = host.secrets.get(ref) ?? null;
          else if (source.type === "env") {
            for (const name of source.names ?? []) {
              const v = getEnv(name);
              if (v) {
                value = v;
                break;
              }
            }
          } else if (source.type === "resolver") {
            const fn = host.resolvers.get(source.resolver_id);
            if (fn) {
              try {
                value = secretOf(await fn());
              } catch (error) {
                throw new AuthError("the credential resolver failed", { code: "resolver_failed", cause: error });
              }
            }
          }
          if (value) core.setSecret(op, ref, new TextEncoder().encode(value));
          input = { type: "secret", secret_ref: ref, found: !!value };
          break;
        }
        case "need_read": {
          const locator = effect.locator;
          let bytes = null;
          if (locator.type === "file") bytes = await readFile(locator.path);
          else if (locator.type === "env") {
            const v = getEnv(locator.name);
            if (v !== undefined) bytes = new TextEncoder().encode(v);
          } else if (locator.type === "keychain" && host.keychain) {
            let v = null;
            try {
              v = await host.keychain(locator.service, locator.account);
            } catch {
              v = null;
            }
            if (typeof v === "string") v = new TextEncoder().encode(v);
            if (v && v.length) bytes = v;
          }
          input = bytes ? { type: "read", found: true, bytes: toBase64(bytes) } : { type: "read", found: false };
          if (bytes && locator.type === "file") input.etag = await etagOf(bytes);
          break;
        }
        case "need_write": {
          const locator = effect.locator;
          const bytes = fromBase64(effect.bytes);
          if (locator.type === "keychain") {
            let ok = false;
            if (host.keychainWrite) {
              try {
                ok = !!(await host.keychainWrite(locator.service, locator.account, bytes));
              } catch {
                ok = false;
              }
            }
            input = { type: "written", ok };
            break;
          }
          if (locator.type !== "file") {
            input = { type: "written", ok: false };
            break;
          }
          if (effect.expect) {
            const current = await readFile(locator.path);
            if (!current || (await etagOf(current)) !== effect.expect) {
              input = { type: "written", ok: false, conflict: true };
              break;
            }
          }
          input = { type: "written", ok: await writeFile(locator.path, bytes, effect.private ?? true) };
          break;
        }
        case "need_user_action":
          await showAction(effect.action);
          input = { type: "user_action", value: typeof lastValue === "string" ? lastValue : undefined };
          break;
        case "need_callback": {
          const result = await awaitLoopbackCallback(effect, signal, () => showAction(effect.action));
          if (result.cancelled) {
            input = { type: "cancelled" };
            break;
          }
          input = { type: "callback", url: new URL(effect.redirect_uri).origin + result.target };
          break;
        }
        default:
          throw new UnsupportedError(`the core asked for an unknown effect ${effect.type}`);
      }
      yield* notes(core.resume(op, { clock: clock(), input }));
    }
  } finally {
    closeBody();
    core.free(op);
  }
}

export { CancelledError };
