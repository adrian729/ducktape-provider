import { execFileSync } from "node:child_process";
import { copyFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("../../../", import.meta.url));
execFileSync("cargo", ["build", "--profile", "wasm", "--target", "wasm32-unknown-unknown", "-p", "ducktape-abi"], {
  cwd: root,
  stdio: "inherit",
});
copyFileSync(
  new URL("../../../target/wasm32-unknown-unknown/wasm/ducktape_abi.wasm", import.meta.url),
  new URL("../core.wasm", import.meta.url),
);
