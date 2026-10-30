import { createServer } from "node:http";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".json": "application/json", ".wasm": "application/wasm" };

function fixtures() {
  const out = [];
  const dir = join(root, "fixtures", "wire");
  for (const wire of readdirSync(dir).sort()) {
    for (const name of readdirSync(join(dir, wire)).sort()) {
      if (name.endsWith(".json")) out.push({ wire, name: name.replace(/\.json$/, ""), path: `/fixtures/wire/${wire}/${name}` });
    }
  }
  return out;
}

/** Serves the repository with the browser harness at `/`; resolves to the bound port. */
export function serve(port = 0) {
  const server = createServer((req, res) => {
    const url = new URL(req.url, "http://localhost");
    if (url.pathname === "/fixtures.json") {
      res.writeHead(200, { "content-type": types[".json"] });
      res.end(JSON.stringify(fixtures()));
      return;
    }
    if (url.pathname === "/") {
      res.writeHead(302, { location: "/bindings/js/test/browser/index.html" });
      res.end();
      return;
    }
    const file = normalize(join(root, decodeURIComponent(url.pathname)));
    if (!file.startsWith(root)) {
      res.writeHead(403).end();
      return;
    }
    try {
      if (statSync(file).isDirectory()) throw new Error("directory");
      res.writeHead(200, { "content-type": types[extname(file)] ?? "application/octet-stream" });
      res.end(readFileSync(file));
    } catch {
      res.writeHead(404).end();
    }
  });
  return new Promise((resolve) => server.listen(port, "127.0.0.1", () => resolve({ server, port: server.address().port })));
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const { port } = await serve(Number(process.env.PORT ?? 8765));
  console.log(`http://127.0.0.1:${port}/`);
}
