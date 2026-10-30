import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { serve } from "./serve-browser.js";

const candidates = [process.env.CHROME_BIN, "google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"].filter(Boolean);

function launch(url, profile) {
  return new Promise((resolve, reject) => {
    const tryNext = (i) => {
      if (i >= candidates.length) return reject(new Error("no Chrome found; set CHROME_BIN"));
      const child = spawn(candidates[i], ["--headless=new", "--disable-gpu", "--no-first-run", `--user-data-dir=${profile}`, "--remote-debugging-port=0", url], { stdio: ["ignore", "ignore", "pipe"] });
      let text = "";
      child.on("error", () => tryNext(i + 1));
      child.stderr.on("data", (chunk) => {
        text += chunk;
        const m = text.match(/DevTools listening on (ws:\/\/[^\s]+)/);
        if (m) resolve({ child, endpoint: m[1] });
      });
      child.on("exit", (code) => reject(new Error(`Chrome exited with ${code}: ${text}`)));
    };
    tryNext(0);
  });
}

async function pageSocket(endpoint) {
  const base = endpoint.replace(/^ws:\/\/([^/]+).*/, "http://$1");
  for (let i = 0; i < 50; i++) {
    const targets = await (await fetch(`${base}/json`)).json();
    const page = targets.find((t) => t.type === "page" && t.url.startsWith("http://127.0.0.1"));
    if (page) return page.webSocketDebuggerUrl;
    await new Promise((r) => setTimeout(r, 200));
  }
  throw new Error("harness page not found");
}

function evaluate(ws, expression) {
  return new Promise((resolve, reject) => {
    const id = Math.floor(Math.random() * 1e9);
    const onMessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.id !== id) return;
      ws.removeEventListener("message", onMessage);
      if (msg.error) reject(new Error(msg.error.message));
      else resolve(msg.result.result.value);
    };
    ws.addEventListener("message", onMessage);
    ws.send(JSON.stringify({ id, method: "Runtime.evaluate", params: { expression, returnByValue: true } }));
  });
}

const profile = mkdtempSync(join(tmpdir(), "ducktape-chrome-"));
const { server, port } = await serve(0);
let child = null;
let failed = 1;
try {
  const url = `http://127.0.0.1:${port}/`;
  const launched = await launch(url, profile);
  child = launched.child;
  const ws = new WebSocket(await pageSocket(launched.endpoint));
  await new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve);
    ws.addEventListener("error", reject);
  });
  const deadline = Date.now() + 120_000;
  let results = null;
  while (Date.now() < deadline) {
    const text = await evaluate(ws, "JSON.stringify(window.__conformance ?? null)");
    results = text ? JSON.parse(text) : null;
    if (results?.done) break;
    await new Promise((r) => setTimeout(r, 300));
  }
  ws.close();
  if (!results?.done) throw new Error("the harness did not finish in time");
  if (results.error) throw new Error(results.error);
  for (const f of results.failures) console.log(`✖ ${f}`);
  console.log(`browser conformance: ${results.pass} passed, ${results.fail} failed; chat text ${JSON.stringify(results.provider.chat).slice(0, 40)}; stream ${results.provider.stream.length} events ending in ${results.provider.stream.at(-1)}; browser gate ${results.provider.browser_gate ? "enforced" : "MISSING"}`);
  failed = results.fail || results.provider.stream.at(-1) !== "finish" || !results.provider.browser_gate ? 1 : 0;
} finally {
  child?.kill();
  server.close();
  rmSync(profile, { recursive: true, force: true });
}
process.exit(failed);
