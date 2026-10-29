import assert from "node:assert/strict";
import { test } from "node:test";

import { loadCore } from "../src/core.js";
import { load, wireFixtures } from "./helpers.js";

const ENTROPY = Buffer.alloc(32, 128).toString("base64");

function clock(tick) {
  return { now_ms: 1_761_696_000_000 + tick * 100, entropy: ENTROPY };
}

function normalize(step) {
  const copy = JSON.parse(JSON.stringify(step));
  const request = copy.effect?.request;
  if (request && typeof request.body === "string") {
    try {
      request.body = JSON.parse(request.body);
    } catch {}
  }
  return copy;
}

function replay(core, fixture) {
  const [op, first] = core.start({ ...fixture.start, clock: clock(0) });
  let step = first;
  const http = fixture.http[Symbol.iterator]();
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
            queue = chunks.map((c) => ({ type: "http_chunk", bytes: Buffer.from(c).toString("base64") }));
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
          input = value === undefined ? { type: "read", found: false } : { type: "read", found: true, bytes: Buffer.from(value).toString("base64") };
          break;
        }
        case "need_write":
          input = { type: "written", ok: true, conflict: false };
          break;
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

const core = await loadCore();

for (const f of wireFixtures()) {
  test(`${f.wire}/${f.name}`, () => {
    const fixture = load(f.path);
    const actual = replay(core, fixture);
    const expected = fixture.exchange;
    assert.equal(actual.length, expected.length, JSON.stringify(actual.at(-1)));
    actual.forEach((a, i) => {
      assert.deepEqual(normalize(a.step), normalize(expected[i].step), `step ${i}`);
      assert.deepEqual(a.input, expected[i].input, `input ${i}`);
    });
  });
}
