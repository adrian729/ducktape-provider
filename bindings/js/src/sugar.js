/** Shortcuts over the normalized types. */
import { toBase64 } from "./platform.js";

const responseViews = {
  get text() {
    return this.content.filter((b) => b.type === "text").map((b) => b.text).join("");
  },
  get reasoning() {
    return this.content.filter((b) => b.type === "reasoning").map((b) => b.text).join("");
  },
  get toolCalls() {
    return this.content.filter((b) => b.type === "tool_call");
  },
};

/** Gives a response its `text`, `reasoning` and `toolCalls` views. */
export function withViews(response) {
  return Object.setPrototypeOf(response, responseViews);
}

function isBytes(value) {
  return value instanceof Uint8Array || value instanceof ArrayBuffer;
}

function encodeBytes(value) {
  return toBase64(value instanceof ArrayBuffer ? new Uint8Array(value) : value);
}

/** A text block, with a cache mark when `cache` names a ttl. */
export function text(value, { cache } = {}) {
  return cache ? { type: "text", text: value, cache: { ttl: cache } } : { type: "text", text: value };
}

/** An image block from bytes, base64 text, a URL or a vendor file id. */
export function image(data, { url, file_id, media_type } = {}) {
  const block = { type: "image" };
  if (data !== undefined && data !== null) block.data = isBytes(data) ? encodeBytes(data) : data;
  if (url) block.url = url;
  if (file_id) block.file_id = file_id;
  if (media_type) block.media_type = media_type;
  return block;
}

/** A document block from bytes, base64 text, a URL, a vendor file id or plain text. */
export function document(data, { url, file_id, text: plain, media_type, name } = {}) {
  const block = { type: "document" };
  if (data !== undefined && data !== null) block.data = isBytes(data) ? encodeBytes(data) : data;
  if (url) block.url = url;
  if (file_id) block.file_id = file_id;
  if (plain !== undefined) block.text = plain;
  if (media_type) block.media_type = media_type;
  if (name) block.name = name;
  return block;
}

/** Turns strings, bytes and blocks into blocks. */
export function blocks(parts) {
  const out = [];
  for (const part of parts) {
    if (typeof part === "string") out.push({ type: "text", text: part });
    else if (isBytes(part)) out.push(image(part));
    else if (Array.isArray(part)) out.push(...blocks(part));
    else if (part && typeof part === "object" && typeof part.type === "string") out.push(part);
    else throw new TypeError(`cannot make a block from ${part === null ? "null" : typeof part}`);
  }
  return out;
}

/** The result of one tool call. */
export function toolResult(tool_call_id, content, { is_error } = {}) {
  const block = { type: "tool_result", tool_call_id, content: blocks(Array.isArray(content) ? content : [content]) };
  if (is_error !== undefined) block.is_error = is_error;
  return block;
}

/** A user turn. */
export function user(...parts) {
  return { role: "user", content: blocks(parts) };
}

/** An assistant turn, from parts or a whole response. */
export function assistant(...parts) {
  const content = [];
  for (const part of parts) {
    if (part && typeof part === "object" && Array.isArray(part.content) && part.stop) content.push(...part.content);
    else content.push(...blocks([part]));
  }
  return { role: "assistant", content };
}

/** A mid-conversation system turn. */
export function system(...parts) {
  return { role: "system", content: blocks(parts) };
}

/** A tool turn holding every result of one round. */
export function tool(...results) {
  return { role: "tool", content: blocks(results) };
}

/** Reads the credential from the first of these environment variables that is set. */
export function env(...names) {
  if (names.length === 0) throw new TypeError("env() needs at least one variable name");
  return { type: "env", names };
}

/** Uses a login session of the given preset. */
export function session(preset, { store, client_id } = {}) {
  const source = { type: "session", preset };
  if (store) source.store = store;
  if (client_id) source.client_id = client_id;
  return source;
}

/** Calls `fn` for the credential; it returns a string or `{secret}`, possibly as a promise. */
export function resolver(fn, { kind, placement } = {}) {
  const source = { type: "resolver", fn };
  if (kind) source.kind = kind;
  if (placement) source.placement = placement;
  return source;
}
