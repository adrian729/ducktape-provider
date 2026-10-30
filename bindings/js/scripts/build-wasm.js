import { execFileSync } from "node:child_process";
import { copyFileSync, renameSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const out = fileURLToPath(new URL("../core.wasm", import.meta.url));
execFileSync("cargo", ["build", "--profile", "wasm", "--target", "wasm32-unknown-unknown", "-p", "ducktape-abi"], {
  cwd: root,
  stdio: "inherit",
});
copyFileSync(fileURLToPath(new URL("../../../target/wasm32-unknown-unknown/wasm/ducktape_abi.wasm", import.meta.url)), out);
const raw = statSync(out).size;
const wasmOpt = process.env.WASM_OPT ?? "wasm-opt";
const WASM_FEATURES = [
  "--enable-bulk-memory",
  "--enable-multivalue",
  "--enable-mutable-globals",
  "--enable-nontrapping-float-to-int",
  "--enable-reference-types",
  "--enable-sign-ext",
];
try {
  execFileSync(wasmOpt, ["-O3", ...WASM_FEATURES, out, "-o", `${out}.opt`], { stdio: "inherit" });
  renameSync(`${out}.opt`, out);
  console.log(`core.wasm: ${raw} bytes, ${statSync(out).size} after ${wasmOpt} -O3`);
} catch (e) {
  if (e.code !== "ENOENT") throw e;
  console.log(`core.wasm: ${raw} bytes (${wasmOpt} not found, skipped)`);
}
