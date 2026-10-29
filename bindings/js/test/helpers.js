import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

export const root = fileURLToPath(new URL("../../../", import.meta.url));
export const fixtures = join(root, "fixtures");

export function load(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

export function wireFixtures() {
  const out = [];
  const dir = join(fixtures, "wire");
  for (const wire of readdirSync(dir).sort()) {
    for (const name of readdirSync(join(dir, wire)).sort()) {
      if (name.endsWith(".json")) out.push({ wire, name: name.replace(/\.json$/, ""), path: join(dir, wire, name) });
    }
  }
  return out;
}

export function scripted(fixture, seen = []) {
  const script = fixture.http[Symbol.iterator]();
  return async (url, init) => {
    seen.push({ url, init, body: init.body ? JSON.parse(init.body) : null });
    const h = script.next().value;
    if (!h) throw new Error("script exhausted");
    if (h.error) throw new TypeError("fetch failed", { cause: { code: "ECONNREFUSED", message: h.error.message } });
    const chunks = h.chunks?.length ? h.chunks : h.body !== undefined ? [h.body] : [];
    const encoder = new TextEncoder();
    const body = new ReadableStream({
      pull(controller) {
        if (chunks.length) controller.enqueue(encoder.encode(chunks.shift()));
        else controller.close();
      },
    });
    return new Response(body, { status: h.status ?? 200, headers: h.headers ?? {} });
  };
}

export function doneResponse(fixture) {
  return fixture.exchange.at(-1).step.effect.result.response;
}

export function requestOf(fixture) {
  return fixture.start.operation.request;
}

export function withoutAuthSource(content) {
  return JSON.parse(JSON.stringify(content)).map((block) => {
    if (block.origin) delete block.origin.auth_source;
    return block;
  });
}
