/** The wasm core: JSON documents in, JSON steps out, opaque operation handles. */
import { isNode } from "./platform.js";

const encoder = new TextEncoder();
const decoder = new TextDecoder();

/** Loads `core.wasm` from a URL, bytes or a compiled module. */
export async function loadCore(source) {
  let module;
  if (source instanceof WebAssembly.Module) {
    module = source;
  } else if (source instanceof Uint8Array || source instanceof ArrayBuffer) {
    module = await WebAssembly.compile(source);
  } else {
    const url = source ? new URL(source, import.meta.url) : new URL("../core.wasm", import.meta.url);
    if (url.protocol === "file:") {
      const fs = await import("node:fs/promises");
      module = await WebAssembly.compile(await fs.readFile(url));
    } else if (typeof WebAssembly.compileStreaming === "function") {
      module = await WebAssembly.compileStreaming(fetch(url));
    } else {
      module = await WebAssembly.compile(await (await fetch(url)).arrayBuffer());
    }
  }
  const instance = await WebAssembly.instantiate(module, {});
  return new Core(instance);
}

let shared = null;

/** The core every provider shares unless given its own. */
export function defaultCore() {
  shared ??= loadCore();
  return shared;
}

export class Core {
  constructor(instance) {
    this.ex = instance.exports;
    if (this.ex.ducktape_abi_version() !== 1) throw new Error("core.wasm has an unknown ABI version");
    this.scratch = this.ex.ducktape_alloc(8);
  }

  put(bytes) {
    if (bytes.length === 0) return 0;
    const ptr = this.ex.ducktape_alloc(bytes.length);
    new Uint8Array(this.ex.memory.buffer, ptr, bytes.length).set(bytes);
    return ptr;
  }

  take() {
    const view = new DataView(this.ex.memory.buffer);
    const ptr = view.getUint32(this.scratch, true);
    const len = view.getUint32(this.scratch + 4, true);
    const text = decoder.decode(new Uint8Array(this.ex.memory.buffer, ptr, len));
    this.ex.ducktape_free(ptr, len);
    return JSON.parse(text);
  }

  call(fn, bytes, ...args) {
    const ptr = this.put(bytes);
    try {
      return fn(ptr, bytes.length, ...args);
    } finally {
      this.ex.ducktape_free(ptr, bytes.length);
    }
  }

  /** Opens an operation and returns its handle with the first step. */
  start(doc) {
    const op = this.call(this.ex.ducktape_op_start, encoder.encode(JSON.stringify(doc)), this.scratch, this.scratch + 4);
    const step = this.take();
    return [op, step];
  }

  /** Continues an operation with the host's answer. */
  resume(op, doc) {
    this.call(this.ex.ducktape_op_resume.bind(null, op), encoder.encode(JSON.stringify(doc)), this.scratch, this.scratch + 4);
    return this.take();
  }

  /** Delivers raw body bytes of the open response. */
  feed(op, bytes, clock) {
    const clockBytes = encoder.encode(JSON.stringify(clock));
    const clockPtr = this.put(clockBytes);
    try {
      this.call(
        (ptr, len) => this.ex.ducktape_op_feed(op, ptr, len, clockPtr, clockBytes.length, this.scratch, this.scratch + 4),
        bytes,
      );
    } finally {
      this.ex.ducktape_free(clockPtr, clockBytes.length);
    }
    return this.take();
  }

  /** Stores a secret for an operation. */
  setSecret(op, ref, bytes) {
    const refBytes = encoder.encode(ref);
    const refPtr = this.put(refBytes);
    try {
      this.call((ptr, len) => this.ex.ducktape_op_set_secret(op, refPtr, refBytes.length, ptr, len), bytes);
    } finally {
      this.ex.ducktape_free(refPtr, refBytes.length);
    }
  }

  /** Frees an operation handle. */
  free(op) {
    if (op) this.ex.ducktape_op_free(op);
  }
}

export { isNode };
