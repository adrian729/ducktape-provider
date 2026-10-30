/** What the host runtime offers: Node file system and environment, or nothing in a browser. */
export const isNode = typeof process !== "undefined" && !!process.versions?.node;

export const isBrowser = typeof window !== "undefined" && typeof document !== "undefined";

/** Reads an environment variable where the runtime has them. */
export function getEnv(name) {
  if (typeof process !== "undefined" && process.env) return process.env[name];
  if (typeof Deno !== "undefined") return Deno.env.get(name);
  return undefined;
}

/** Reads a file as bytes, or null when it does not exist or files are unavailable. */
export async function readFile(path) {
  if (!isNode) return null;
  try {
    const fs = await import("node:fs/promises");
    return new Uint8Array(await fs.readFile(path));
  } catch {
    return null;
  }
}

/** Writes a file atomically, private when asked; false when files are unavailable. */
export async function writeFile(path, bytes, isPrivate) {
  if (!isNode) return false;
  try {
    const fs = await import("node:fs/promises");
    const nodePath = await import("node:path");
    await fs.mkdir(nodePath.dirname(path), { recursive: true });
    const tmp = `${path}.tmp${process.pid}`;
    await fs.writeFile(tmp, bytes, { mode: isPrivate ? 0o600 : 0o644 });
    await fs.rename(tmp, path);
    return true;
  } catch {
    return false;
  }
}

/** Lower-case hex SHA-256 of bytes, the etag hosts report for file reads. */
export async function etagOf(bytes) {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

/** Random bytes from the runtime. */
export function randomBytes(n) {
  const out = new Uint8Array(n);
  globalThis.crypto.getRandomValues(out);
  return out;
}

/** Base64 of bytes. */
export function toBase64(bytes) {
  if (isNode) return Buffer.from(bytes).toString("base64");
  let s = "";
  for (const b of bytes) s += String.fromCharCode(b);
  return btoa(s);
}

/** Bytes from base64. */
export function fromBase64(text) {
  if (isNode) return new Uint8Array(Buffer.from(text, "base64"));
  const s = atob(text);
  const out = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i);
  return out;
}
