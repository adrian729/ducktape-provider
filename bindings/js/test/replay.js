import { etagOf, toBase64 } from "../src/platform.js";

const ENTROPY = toBase64(new Uint8Array(32).fill(128));

/** The scripted clock: fixed start, 100 ms per step. */
export function clock(tick) {
  return { now_ms: 1_761_696_000_000 + tick * 100, entropy: ENTROPY };
}

/** A step with its request body parsed, so JSON key order does not matter. */
export function normalize(step) {
  const copy = JSON.parse(JSON.stringify(step));
  const request = copy.effect?.request;
  if (request && typeof request.body === "string") {
    try {
      request.body = JSON.parse(request.body);
    } catch {}
  }
  return copy;
}

const encode = (text) => toBase64(new TextEncoder().encode(text));
const etag = (text) => etagOf(new TextEncoder().encode(text));

/** Replays a wire fixture's script through the core and returns the exchange it produced. */
export async function replay(core, fixture) {
  const [op, first] = core.start({ ...fixture.start, clock: clock(0) });
  let step = first;
  const http = fixture.http[Symbol.iterator]();
  const writes = (fixture.writes ?? [])[Symbol.iterator]();
  let queue = [];
  const out = [];
  let tick = 0;
  try {
    for (let i = 0; i < 10_000; i++) {
      const effect = step.effect;
      if (effect.type === "done" || effect.type === "failed") {
        out.push({ step });
        return out;
      }
      let input;
      switch (effect.type) {
        case "need_secret": {
          const value = fixture.secrets?.[effect.secret_ref];
          if (value !== undefined) core.setSecret(op, effect.secret_ref, new TextEncoder().encode(value));
          input = { type: "secret", secret_ref: effect.secret_ref, found: value !== undefined };
          break;
        }
        case "need_http": {
          const h = http.next().value;
          if (h.error) input = { type: "http_error", error: h.error };
          else {
            const chunks = h.chunks?.length ? h.chunks : h.body !== undefined ? [h.body] : [];
            queue = chunks.map((c) => ({ type: "http_chunk", bytes: encode(c) }));
            queue.push({ type: "http_end" });
            queue.reverse();
            input = { type: "http_response", status: h.status ?? 200, headers: h.headers ?? {} };
          }
          break;
        }
        case "receive":
          input = queue.pop() ?? { type: "http_end" };
          break;
        case "need_sleep":
          input = { type: "slept" };
          break;
        case "need_read": {
          const loc = effect.locator;
          const value =
            loc.type === "file"
              ? fixture.files?.[loc.path]
              : loc.type === "env"
                ? fixture.env?.[loc.name]
                : fixture.keychain?.[`${loc.service}|${loc.account}`];
          input = value === undefined ? { type: "read", found: false } : { type: "read", found: true, bytes: encode(value) };
          if (value !== undefined && loc.type === "file") input.etag = await etag(value);
          break;
        }
        case "need_write": {
          const w = writes.next().value ?? { ok: true };
          input = { type: "written", ok: w.ok, conflict: w.conflict ?? false };
          break;
        }
        default:
          throw new Error(`unsupported effect in replay: ${effect.type}`);
      }
      out.push({ step, input });
      tick += 1;
      step = core.resume(op, { clock: clock(tick), input });
    }
    throw new Error("replay did not finish");
  } finally {
    core.free(op);
  }
}

/** Compares a replay with the fixture's golden exchange; returns the first mismatch or null. */
export function mismatch(actual, expected) {
  if (actual.length !== expected.length) return `length ${actual.length} != ${expected.length}; last ${JSON.stringify(actual.at(-1))}`;
  for (let i = 0; i < actual.length; i++) {
    const a = JSON.stringify(sortKeys(normalize(actual[i].step)));
    const e = JSON.stringify(sortKeys(normalize(expected[i].step)));
    if (a !== e) return `step ${i}: ${a} != ${e}`;
    const ai = JSON.stringify(sortKeys(actual[i].input ?? null));
    const ei = JSON.stringify(sortKeys(expected[i].input ?? null));
    if (ai !== ei) return `input ${i}: ${ai} != ${ei}`;
  }
  return null;
}

function sortKeys(value) {
  if (Array.isArray(value)) return value.map(sortKeys);
  if (value && typeof value === "object") return Object.fromEntries(Object.keys(value).sort().map((k) => [k, sortKeys(value[k])]));
  return value;
}
