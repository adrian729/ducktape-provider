# Delivery across languages

One implementation, several languages. As of 2026-10-28. Decided: option A' (Rust sans-I/O core), all laya languages in v1 (decisions 1 to 3 in [[Design draft]]). Toolchain detail and measurements superseded by [[Build and packaging]].

## The decision that matters: who owns the socket

- Core owns HTTP (TLS, async runtime, sockets): bridge Tokio to asyncio/libuv/Go (Temporal calls this the hardest part), cancellation gaps (napi AbortSignal only pre-start, UniFFI none built in), users' proxies/CAs/mocks/instrumentation go blind, no edge or browser runtimes, 4-6 MB wheels per platform, C toolchain for TLS (aws-lc needs NASM on Windows).
- Core is sans-I/O: pure, synchronous, deterministic functions. Host language does HTTP with its native client. Boundary is CPU-only, easy to bind, trivially cancellable (host closes its own connection), replayable from fixtures. Same shape as ducktape-laya's "CPU library behind a narrow ABI".

## Precedents

| Project | Core | Bindings | Core does HTTP |
|---|---|---|---|
| Temporal | Rust + tokio | PyO3 + pyo3-async-runtimes, Neon, C bridge for .NET, Magnus | yes |
| BAML (LLM runtime) | Rust | PyO3, napi-rs (recommended over Neon), cgo over cffi cdylib, wasm for VSCode | yes |
| 1Password SDKs | Rust | Go via Extism + wazero, `core.wasm` embedded, no cgo, mutex | host-mediated |
| Bitwarden SM SDK | Rust | C ABI with 3 functions: init / run_command(json) / free | yes |
| libdatadog | Rust, cbindgen | Ruby, .NET, PHP | yes |
| Matrix rust-sdk | Rust | UniFFI (Kotlin, Swift), wasm-bindgen; streams as `TaskHandle` with cancel | yes |
| Turso | Rust | Go via purego, no cgo | n/a |
| TensorZero | Rust gateway | Python embeds gateway in wheel (37-42 MB); others via HTTP | yes |
| gRPC | C-core | Node and .NET native cores deprecated for pure implementations to avoid native builds | yes |
| sentry-native | C | WinHTTP on Windows, curl elsewhere, now pluggable transport | yes |
| OpenAI/Anthropic Python SDKs | Python | jiter (Rust) only for partial JSON | - |

## Binding tools (verified versions)

- PyO3 0.30 + maturin; abi3 wheels keep the matrix small (pyreqwest ships 92 files without abi3). Free-threaded CPython needs separate wheels.
- napi-rs 3.15: async on managed Tokio, `ReadableStream` from Rust stream, async iterators; AbortSignal cancels only unstarted work; wasm32-wasip1 fallback.
- wasm-bindgen: one package for Node, Deno, Bun, browsers, Workers; `fetch` as transport.
- UniFFI 0.32: first-party Kotlin, Swift, Python, Ruby. Go/C#/Node generators third-party and lagging (livekit node generator abandoned). No built-in cancellation, no streams (listener callbacks). Fits mobile.
- Diplomat: C, C++, Dart, JS, .NET, Kotlin, Python; sync APIs only. Fits sans-I/O, fewer users.
- Go: cgo + cbindgen (breaks `CGO_ENABLED=0`, cross-compile pain), purego (dlopen, no cgo), wazero (pure-Go wasm).
- Java: FFM (JDK 22+) + jextract, or Chicory (pure-Java wasm). .NET: csbindgen P/Invoke, or wasm runtime.
- reqwest 0.13: rustls default, aws-lc-rs provider, platform verifier. Only relevant if core owns I/O.

## Options

| Option | Single logic | Streaming across boundary | Cancel | Async | TLS/proxies | Size | Packaging | Edge/browser |
|---|---|---|---|---|---|---|---|---|
| A Rust full-I/O + PyO3/napi/UniFFI | yes | Rust stream to ReadableStream / awaitable / callbacks | hard | bridge runtimes | own, diverges from host | 4-6 MB wheel, ~9 MB node | maturin matrix + prebuilds + TLS toolchain | no |
| A' Rust sans-I/O + host HTTP | yes | host pushes bytes, events return synchronously | trivial | host's own | host's | <1 MB native, few hundred KB wasm (est.) | same matrix minus TLS; one plain-export wasm for JS, Go and Java, native C ABI for C and .NET, PyO3 for Python | yes |
| B C++ full-I/O + C ABI + libcurl | yes | C callbacks from curl thread | progress-callback abort | own loop thread | OpenSSL per platform, dual-OpenSSL risk in CPython | several MB | hardest | no |
| B' C++ sans-I/O + C ABI | yes | host pushes bytes, events return synchronously | trivial | host's own | host's | <1 MB native, wasm via Emscripten (est.) | laya pipeline as-is: C ABI + seven bindings + CI; no PyO3/wasm-bindgen native types | via Emscripten, less mature than wasm-bindgen |
| C Spec-first codegen (Stainless, Fern, Speakeasy) | types only | native | native | native | host's | 0 | easiest | yes |
| D Sidecar gateway | yes | SSE over loopback | close socket | native client | sidecar's | 10s of MB binary | process lifecycle burden | no |
| E WASM component + wasi:http | yes | `stream<T>` in WASI 0.3 (ratified 2026-06) | spec'd, host-dependent | host loop | host's wasi:http impl, uneven | runtime 10s of MB | immature outside Rust/JS | partial |
| F Hand-written per language + shared fixtures | no | native | native | native | host's | 0 | easiest | yes |

Notes:

- C: translation logic (block mapping, stream re-indexing, error classification) is not expressible in OpenAPI. Use only to generate normalized types from JSON Schema. Stainless hosted generator winding down after Anthropic acquisition.
- D: fine as optional later product on the same core; not the library's integration path.
- E: too early for a client library in 2026. Plain core-wasm without wasi:http as a sans-I/O core is viable now (1Password).
- B': the fair alternative to A'. Same boundary, same fixtures; pays in hand-written JSON (no serde), hand-written Python/TS wrappers over the C ABI, and a C++ build matrix; gains the laya release pipeline unchanged and no Rust requirement.
- F: the status quo. Cheapest for one language, about half of A' for two; drift between hand-written stream state machines is caught by fixtures, not prevented. The fixture suite is required by A' and B' anyway.

## Performance reality

- Model latency dominates (TTFT 0.5-1 s, generation seconds). Per-event boundary crossing (≪10 µs) is noise against inter-token gaps (tens of ms).
- LiteLLM proxy overhead p50/p95/p99 = 2/8/13 ms on small instances (own benchmarks). Rust/Go gateways claim sub-ms (vendor claims, not reproduced).
- Python hurts at: httpx pool under thousands of concurrent requests (NVIDIA NeMo Gym: O(n²), switched to aiohttp), GC churn, pure-Python partial-JSON parsing (OpenAI/Anthropic SDKs ship jiter). Not at: SSE line splitting + json.loads for one stream.
- Implication: a shared core buys correctness and parity, not speed.

## Recommendation

1. Phase 0: language-neutral conformance suite, recorded fresh from live vendor traffic against the chosen surfaces (never derived from the old implementation). Per provider: normalized input to expected HTTP request; recorded responses and SSE/NDJSON byte streams (split-chunk boundaries, errors, 429 + retry-after) to expected normalized output; model-list payloads. JSON fixtures + JSON Schema of normalized types.
2. Phase 1: Rust sans-I/O core (serde_json, no tokio, no TLS) passing the suite. Python via PyO3 abi3 with `httpx` as default transport (decision 4). JS via plain `core.wasm` + fetch, AbortSignal, async iterators.
3. Phase 2: Go via wazero and Java via Chicory on the same `core.wasm`; .NET and C/C++ on the native C ABI; Rust crate with optional `reqwest` feature. See [[Build and packaging]].
4. Optional later: in-core fast transport (reqwest via PyO3/napi) for high-concurrency users; sidecar binary on the same core.

Boundary style: PyO3 gets native types; everything else goes through one narrow `extern "C"` surface (JSON in and out, opaque handles) compiled to wasm or native, with hand-written idiomatic wrappers. Rust (A') chosen over C++ (B') on 2026-10-28; toolchain chosen on its own merits in [[Build and packaging]], not copied from laya.

## Risks

- Policy logic leaking into host glue (retries, auth, timeouts, matching). Keep policy as pure core functions; glue = I/O only.
- Type churn across the boundary: generate per-language types from one schema; version the protocol.
- Wheel matrix without abi3; free-threaded CPython wheels.
- wazero/Chicory single-threaded per instance: pool or mutex; memory caps.
- Rust becomes a hard requirement for touching normalization logic.
- Ecosystem volatility: third-party generators orphaned, hosted codegen vendors disappearing, WASI 0.3 very new.
