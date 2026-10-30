# Design draft

Design with decisions (Settled section, dated). Clean-sheet design: the old implementation is not a base, not a source, not a reference. Builds on [[Provider API matrix]], [[Vendor APIs]], [[Prior art]], [[Agents]], [[Delivery across languages]], [[Build and packaging]], [[Auth model]].

## Principles

- Deep modules: `chat` and `stream` are the product; `embed`, discovery (`models`, `model_info`, `capabilities`) and auth (`login`, `logout`, `auth_status`) complete it; everything else inside. Surface in [[Interface spec]].
- Stateless. Caller owns history. History is append-only and carries opaque vendor state.
- Normalize intent, pass through detail. Tiny portable option set; vendor fields in a namespaced map; typed warnings on loss; never silent drops; never translate opaque state across vendors.
- Codec per wire protocol, not per vendor. Provider = endpoint (wire, URL, auth, dialect) + model profile.
- Core is sans-I/O. Host does HTTP. Fixtures are the contract.
- Defaults for the common case: stateless chat, retries on retryable errors, reasoning vendor default, no auto-match magic.
- "Codex" in the brief = OpenAI's models (incl. `gpt-5-codex`) through Responses in v1; the Codex agent (app-server) is a v2 agent backend.

## Architecture

```
host language glue (thin)           core (sans-I/O, one implementation)
  HTTP client, TLS, proxies           Operation.start(provider, input) -> Effect
  async / sync variants               Operation.resume(effect_result) -> Effect
  cancellation = close socket           Effect = NeedHttp{req} | NeedSleep{ms} | NeedSecret{ref} | NeedRead{locator} | NeedWrite{locator} | NeedUserAction | NeedCallback | Notify{observer event} | Done{result} | Failed{error}
  passes now + entropy on start/resume  (schema/effects.schema.json: Start, Resume, Step)
  secrets resolution (callbacks)      StreamDecoder(provider).feed(bytes) -> [Event]; finish() -> Response | Error
  effect loop (drive until Done)      pure helpers: build_request, decode_response, decode_models, classify_error, retry_decision
  idiomatic types / errors
```

- Core = effects state machine, host = effect loop. One loop per language covers everything that takes more than one request: retries with backoff, Ollama `tags` + `show`, `pause_turn` continuation (capped at 10, then `continue_required`), URL images fetched for byte-only vendors, token refresh. The host passes `now` and entropy bytes on every start and resume (PKCE, jitter, synthetic ids; fixtures pin them). Glue never carries policy.
- Core state is per operation. No sockets, no threads, no runtime.
- Boundary (decision 14, [[Build and packaging]]): one `extern "C"` surface (`op_start`, `op_resume`, `decoder_feed`, `set_secret`, `alloc`, `free`, `abi_version`; UTF-8 JSON in and out, no callbacks). Compiled twice: `core.wasm` (plain exports, zero imports) for JS, Go (wazero) and Java (Chicory); native cdylib + cbindgen header for C/C++ and .NET. Python gets PyO3 native objects; Rust gets the source crate. Java ships two artifacts with one API: Chicory on the wasm (Java 11+) and FFM on the native library (JDK 22+).
- Normalized types published as JSON Schema; per-language types generated from it.
- Namespace rule (decision 15): one short vendor namespace per wire, company names (`anthropic`, `openai`, `google`, `ollama`) used everywhere a vendor-specific name appears: `vendor` maps, `custom.kind`, `ProviderTool.id`, citations, warnings. The codec id (`anthropic.messages`) names the wire only and appears in user data solely as `origin.wire`. If a vendor ever runs two wires (API generation change), the namespace stays and `origin.wire` disambiguates meaning: opaque state is never replayed across wires, old-wire-only fields warn on the new wire, users migrate by changing the endpoint config.
- Conformance suite: fixtures (input to expected effects; bytes in to events out) run by every binding.
- Performance targets (to measure, not promise): boundary crossing per event under 10 µs; decoding a 1 MB SSE stream under 50 ms; Python wheel and wasm under 1 MB; import under 50 ms. Model latency dominates; the core must simply never show up in a profile.

## Wires and providers (first wave)

Rule: only surfaces a vendor recommends for new work. Chat Completions is never one of them, so third-party endpoints are reachable only through Responses, Messages or a native wire. Full comparison in [[Provider API matrix]].

| Wire codec | Covers | Wave |
|---|---|---|
| `anthropic.messages` | Claude; DeepSeek `/anthropic`; OpenRouter Messages | 1 |
| `openai.responses` | OpenAI; xAI; Groq and OpenRouter (beta) | 1 |
| `google.interactions` | Gemini API | 1 if the fixture spike is clean, else 2 |
| `ollama.native` | Local and cloud Ollama: chat, embed, model listing, `/api/show` capabilities, think levels, NDJSON | 1 |
| Bedrock, Vertex, Foundry | `anthropic.messages` and `openai.responses` under SigV4 / GCP / Azure auth | 2 |

Not a wire: OpenAI Chat Completions (anywhere), Gemini `generateContent` (labelled legacy), Bedrock Converse, Ollama `/v1/*` compat layers.

Provider config = `{endpoint, profile}`. `endpoint = {wire, base_url, auth, headers, api_version?, dialect}` (`auth` shape in [[Auth model]]): where to send and what the endpoint accepts (dialect = per-host deviations, e.g. a compat server that lacks `tool_choice`). `profile`: model-family quirks (schema subset, sampling params allowed, tool_choice allowed, reasoning levels, default max output). Adding Groq = config, not code. Profile source (decision 6c): a shipped table in core, refreshed from the vendor's model endpoint where one exists (Claude `/v1/models`, Ollama `/api/show`, Gemini models endpoint; OpenAI exposes nothing machine-readable), overridable by the user for custom endpoints. `capabilities()` and the loss policy read the same table, so UI and runtime never disagree.

## Surface sketch (language-neutral)

Superseded by [[Interface spec]] (frozen 2026-10-29); kept as the origin of the shape. Where the two differ, the spec wins.

```
Provider(config) | Provider.from_env(name) | Provider.from_session(name, preset)   # one provider = one endpoint + profile
  chat(request) -> Response
  stream(request) -> Iterator[Event]        # cancellable, ends with finish
  embed(request) -> EmbedResponse           # EmbedRequest {model, input[], options?: {dimensions?}, vendor?, timeout?}
  models() -> [ModelInfo]                   # id, kind: chat|embedding|unknown, context_window?, max_output?, capabilities?, raw
  model(id) -> ModelInfo                    # single lookup where the vendor has one, else from models()
  capabilities(model) -> Capabilities       # what this model accepts; same table the loss policy reads
  observer?: on_request(req) | on_response(res, raw) | on_event(ev) | on_retry(err, attempt)
  login(method: chatgpt_plan | anthropic_profile) -> effects (NeedUserAction, NeedCallback, NeedHttp, NeedWrite) ; logout(method) ; auth_status() -> {source, expires_at?}

Request
  model, messages[], system?: text | blocks[], tools?: (ToolDef | ProviderTool)[]
  options?: { max_output_tokens: int (default: vendor default; Claude: model max output from capabilities),
              stop_sequences: [str],
              reasoning: default|none|minimal|low|medium|high|xhigh|max (default: default),
              tool_choice: auto|none|required|{name} (default: auto),
              parallel_tools: bool (default: vendor),
              output: {json_schema, name?} ,
              cache: none|auto|explicit (default: auto) }
  vendor?: { anthropic: {...}, openai: {...} }   # vendor namespace; merged into body; conflicts with normalized fields = error
  timeout?: {connect?, read?, total?}, headers?
  strict?: bool   # loss policy switch, decision 6a

Message { role: user|assistant, content: Block[], origin?: {wire, provider, model} }   # origin set on assistant messages the library produced
Block (all: id?, cache?: {ttl: short|long}, vendor?: {ns: {...}}, provider_executed?: bool)
  text {text}                                        # citations under vendor.<ns>.citations
  | image {bytes|url, media_type} | document {bytes|url|text, media_type, name?}
  | tool_call {id, name, arguments, truncated?, status?}  # status reserved for agents (v2)
  | tool_result {call_id, content: text | Block[], is_error?}
  | reasoning {text, vendor_state: opaque}  | compaction {vendor_state: opaque, text?}
  | custom {kind: "<ns>.<type>", data}               # hosted-tool calls/results, unknown kinds, anything else

ToolDef { name, description, input_schema, strict?, cache?, vendor? }
ProviderTool { type: "provider", provider_tool: "anthropic.web_search", config }   # hosted tools

Response
  content: Block[], stop: {reason, raw}, usage: {input: {total, cache_read?, cache_write?}, output: {total, reasoning?}, raw} | null
  warnings: [{kind: unsupported|compatibility|deprecated, feature, detail}]
  origin: {wire, provider, model}, latency_ms, ttft_ms?, raw?
EmbedResponse { embeddings: [[float]], usage?: {input}, warnings, origin, latency_ms, raw? }

Capabilities (every value yes | no | unknown, never guessed)
  reasoning: {levels: [none|minimal|low|medium|high|xhigh|max], can_disable}
  tool_choice: [auto, none, required, named], parallel_tools: {supported, toggle}
  output: {json_schema}, cache: {breakpoints, ttls: [short, long]}
  inputs: [text, image, document, audio, video], tools: {functions, hosted: ["anthropic.web_search", ...]}
  limits: {context_window?, max_output?}, embeddings, source: shipped|vendor|user

Event (all block-scoped events carry index; parent_id? reserved for agents)
  block_start {index, kind, id?, name?} | text_delta | reasoning_delta | tool_call_delta {arguments_json}
  | block_stop {index, block}  | finish {response}   # warnings travel in response
  | raw {vendor event}   # opt-in

Error (all: status?, code?, retryable, retry_after?, body, request_id?)
  Auth {code: no_credential | credential_expired | resolver_failed | rejected} | RateLimit | Quota | Server | Timeout | Cancelled | ContextOverflow | RequestTooLarge | Malformed | Request (400) | Unsupported | Api (base)
```

- Stop reasons: complete, tool_call (derived from blocks), max_output_tokens, stop_sequence, context_window_exceeded, refusal, content_filtered, continue_required (Claude `pause_turn`, OpenAI steered), compaction, malformed_output, other.
- `reasoning.vendor_state` and `compaction.vendor_state` are opaque, replayed verbatim when the block's `origin.wire` and `origin.base_url` match the call (decision 17); otherwise dropped with a warning.
- Loss policy (decision 6a): intent options the provider cannot honor (reasoning level, cache, parallel toggle) degrade with a `warning`; options that change semantics (`tool_choice: required` or `{name}` unsupported, `output.json_schema` without native or tool fallback) raise `Unsupported`; `strict: true` turns every warning into an error.
- Forward compatibility (decision 8): unknown block kinds decode as `custom`; unknown stop reasons map to `other` with raw kept; unknown event types go to `raw`; unknown fields preserved and ignored; generated enums are open; `parent_id`, `tool_call.status`, `provider_executed` are reserved now so v2 adds without changing. Schema carries one semver; bindings check the major at load.
- Cache: `auto` = one breakpoint at the end of the system prompt and tools where the vendor supports breakpoints, implicit elsewhere; `explicit` = only the caller's block-level `cache` marks; `ttl: short|long` maps to 5m/1h on Claude and 30m/12h on OpenAI.
- Model id and provider are separate. No prefix inference. Optional `Provider.default_model`.
- Structured output: one `output.json_schema` intent, mapped per vendor (native format, or forced tool fallback with warning). Own design pass.
- Compaction: Claude/OpenAI only; expose as `vendor` passthrough + opaque block in v1; promote to `compact()` only if two consumers need it.
- Timeouts: `connect`, `read` (between bytes on a stream), `total`; library defaults, overridable per endpoint and per call. Host-cancelled calls raise `Cancelled`.
- Not in v1: batch, token counting, server-side state, hosted-tool normalization, audio.

## Agents (v2, same types)

- Separate `Agent` / `Session` interface, see [[Agents]]. Shares Block, Usage, Error, core Events. Adds `tool_call_update`, `permission_request`, `plan`, `file_change`, `terminal_output`, `turn_stop`.
- First backend: ACP client (subprocess, JSON-RPC). Native Codex app-server, Claude stream-json, Managed Agents, OpenAI Agents API later.
- Design the v1 Block and Event types so v2 adds, never changes (reserved fields above).

## Language rollout

| Phase | Deliverable |
|---|---|
| 0 | Interface spec frozen ([[Interface spec]], 2026-10-29). JSON Schema of the normalized types (`schema/provider.schema.json`, done). Fixtures recorded on demand from live vendor traffic on the chosen surfaces, scrubbed (decision 11): pending. |
| 1 | Effects protocol schema (`schema/effects.schema.json`, done 2026-10-29). Rust workspace: `ducktape-core` (types, effects, operation driver, retry, auth), `ducktape-abi`, conformance runner over `fixtures/types/` and `fixtures/wire/` (done). All four codecs done with synthetic wire fixtures: `anthropic.messages`, `openai.responses`, `ollama.native`, `google.interactions` (2026-10-29). Python done (2026-10-30): `crates/py` PyO3 0.30 module over native objects, `bindings/python` host package with httpx (blocking and asyncio twins, streams with `cancel()`, errors by kind, sugar), conformance replays every wire fixture through the native module plus end-to-end tests over httpx mock transports; native profiles unwind with `catch_unwind` in the ABI, wasm built with `--profile wasm` (abort). JS done (2026-10-30): `bindings/js` ESM package over the plain `core.wasm` (zero imports) with `fetch`, `AbortSignal` on every call, async-iterator streams with `cancel()`, connect and read timeouts as abort timers, response views as prototype getters, hand-written `index.d.ts`; conformance replays every wire fixture through the wasm ABI plus end-to-end tests over a scripted `fetch`. Browser gate (`dangerously_allow_browser`) enforced in the binding. Not yet: headless-browser run, `wasm-opt`, `.d.ts` generated from the schema. `core.wasm` is 1.06 MB stripped with four codecs, above the 500 KB estimate. Session auth done (2026-10-30): presets `anthropic_profile`, `claude_code`, `claude_code_token_env`, `codex` as a core state machine over `NeedRead` / `NeedHttp` / `NeedWrite` with refresh modes, 401 forced refresh and `auth_status` reporting expiry; 40 synthetic session fixtures replayed by all three runners; decisions in [[Auth model]]. `login`, `logout` and `chatgpt_plan` stay unsupported until a recorded fixture. Packaging in CI done (2026-10-30): `.github/workflows/package.yml` builds abi3 wheels for linux x86_64 and aarch64, windows x64, macOS arm64 and x86_64 with `maturin-action`, runs the Python tests against the installed wheel on native runners, builds the sdist, and builds `core.wasm` with `wasm-opt -O3` (binaryen version_133, pinned by SHA-256), runs the JS tests on the optimized file and packs the npm tarball; all uploaded as artifacts, nothing published. Local wheel 916 KB, sdist 86 KB and installs from source, npm tarball 383 KB packed. Not yet: free-threaded wheels (`cp314t` / `abi3t`), publishing, release tags. Keychain reads in both bindings (2026-10-30), see [[Auth model]]. Browser run done (2026-10-30): `bindings/js/test/browser/index.html` replays every wire fixture through `core.wasm` over `fetch` plus a chat, a stream and the browser gate; `npm run test:browser` drives headless Chrome over the DevTools protocol (no browser download, uses the installed Chrome) and runs in CI; it caught one binding bug (a loaded `Core` passed as `core` was treated as a URL). `.d.ts` from the schema done (2026-10-30): `scripts/gen_dts.py` writes `bindings/js/schema.d.ts` (42 definitions; open enums as `literal | (string & {})`, discriminated unions, nested objects inline) and CI checks it is current; `index.d.ts` keeps only the API and augments `Response` with the views; `test/types.test-d.ts` is type-checked with `tsc` (positive and `@ts-expect-error` cases). Next: login flow; free-threaded wheels |
| 2 | Go (wazero) and Java (Chicory) on the same `core.wasm`; .NET and C/C++ on the native C ABI; Rust crate (+ optional reqwest). All part of the v1 release (decision 1); phases are build order only. |
| later | fast in-core transport; sidecar binary; agents |

## Settled (unless reopened)

- Decision 1 (2026-10-28): v1 ships every laya language: C/C++, Rust, Node, Python, Go, Java/Kotlin, .NET, plus the browser build. Platforms as laya: Linux x86-64, Linux arm64, Windows x64, macOS arm64 and x86-64. A shared core is therefore mandatory; option F is out.
- Decision 2 (2026-10-28): core in Rust. Build, packaging and binding tooling are to be chosen on their own merits, not copied from laya; investigation in [[Build and packaging]].
- Decision 3 (2026-10-28): core is an effects state machine (`NeedHttp`, `NeedSleep`, `NeedSecret`, `Done`, `Failed`; decision 10 adds `NeedRead`, `NeedWrite`, `NeedUserAction`, `NeedCallback`); each binding is a small effect loop with no policy. Retries, multi-request operations, continuation and token refresh live in core.
- Decision 4 (2026-10-28): every binding is batteries-included: effect loop + default transport from the host platform (Node and browser `fetch`, Go `net/http`, Java `java.net.http`, .NET `HttpClient`), transport swappable by the user. Python default is `httpx` (sync and async in one API, pooling, per-phase timeouts, cancel by closing the response); Rust gets an optional `reqwest` feature; C/C++ ships no default transport.
- Decision 5 (2026-10-28): no OpenAI Chat Completions codec, for any vendor. Third parties that serve only that format are out until they serve Responses or Messages.
- Decision 6a (2026-10-28): loss policy split by effect. Degradable intent (reasoning level, cache hints, parallel toggle) proceeds with a typed warning; impossible semantics (forced tool choice unsupported, JSON schema output with no fallback) fails before sending with `Unsupported`; `strict: true` per call or per provider turns every warning into an error. Never a silent drop.
- Decision 6b (2026-10-28): `max_output_tokens` defaults to the vendor default where one exists; for Claude, which requires it, the model's max output from the capability table. Option enums as sketched: `reasoning` rounded to the nearest vendor level with a warning, `tool_choice` `auto|none|required|{name}`, `cache` `none|auto|explicit` default `auto`.
- Decision 6c (2026-10-28): `capabilities(model)` added to the surface. Data layered: shipped table, then vendor model endpoint, then user override; values yes/no/unknown; same table drives warnings and errors.
- Decision 7 (2026-10-28): Gemini Interactions gets a fixture-recording spike (plain chat, streamed tool call, thinking round-trip via `steps`, blocked response, overflow). Clean recordings = wave 1; otherwise wave 2. v1 does not wait for it.
- Decision 8 (2026-10-28): open-world decoding. Unknown block kinds decode as `custom`, unknown stop reasons as `other` with raw kept, unknown events go to `raw`, unknown fields are preserved and ignored, every generated enum is open (known set + string fallback). Reserved for v2: `parent_id` on events, `tool_call.status`, `provider_executed`. One semver on the JSON Schema (`$id`); minor = additive, major = breaking; bindings check the major against the core at load.
- Decision 9 (2026-10-28): each codec pins the API version its fixtures were recorded against (`anthropic-version: 2026-10-21`, Gemini `v1beta`); endpoint config may override with `api_version`, which adds a `compatibility` warning.
- Decision 10 (2026-10-28): auth model as in [[Auth model]]: kind + source, explicit wins, session discovery only by explicit ordered list, all logic in core, stores via `NeedRead`/`NeedWrite`, borrowed CLI stores read-only by default. Gemini CLI preset out (legacy shape; exception possible later). Login is part of the interface, UI-free: `login(method)` runs as a core state machine emitting `NeedUserAction {url, code?, instructions}` and `NeedCallback {redirect_uri}` effects the host renders however it wants (print, open browser, QR); tokens we obtain go to our own store, never into another tool's. `logout`, `auth_status`. Keychain reads in v1 through `NeedRead {store: keychain}` with a default implementation per binding; our own store uses the keychain where available, else a 0600 file.
- Decision 11 (2026-10-28): no paid vendor traffic in CI. Weekly GitHub Actions cron (free) diffs vendor OpenAPI specs, Gemini discovery doc, Ollama `openapi.yaml` and official SDK releases, opens an issue on change. Fixtures recorded on demand by a human with their own key (`record` command), locally for Ollama, or by opt-in capture mode in consumer projects (scrubbed). Open-world decoding absorbs the rest.
- Decision 12 (2026-10-28): observer hooks (`on_request`, `on_response`, `on_event`, `on_retry`) in v1. Middleware later only if it can live in the core (one implementation, e.g. as effects), never as eight per-binding chains.
- Decision 13 (2026-10-28): one monorepo: core, schema, fixtures, every binding incl. Go. One version from the workspace, one release workflow, tags `vX.Y.Z` plus `bindings/go/vX.Y.Z`; `core.wasm` committed under the Go module at release commits.
- Decision 14 (2026-10-28): boundary and toolchain as recommended in [[Build and packaging]]: one `core.wasm` for JS, Go (wazero) and Java (Chicory); native C ABI for C/C++ and .NET; PyO3 for Python; Rust source crate; Java additionally gets an FFM artifact over the native library in v1; 5 CI jobs; trusted publishing where available.
- Decision 16 (2026-10-29): `chat` runs over the vendor's streaming wire and aggregates in the core. One decoder per wire, `read` timeout meaningful, cancellable, `first_token_ms` always known; non-streamed long generations no longer hit the read timeout.
- Decision 17 (2026-10-29): opaque state (`reasoning`, `compaction`, `custom`) carries its own `origin` on the block; replayed verbatim when `wire` and `base_url` match the call, whatever the model (Anthropic drops unreadable blocks unbilled; stripping client-side counts as editing history); dropped with a warning on another endpoint.
- Decision 18 (2026-10-29): the vendor's env credentials are used only for the vendor's own `base_url`; a custom host needs explicit `auth`. Auth sources are tried in literal list order; kind and placement live on each source.
- Decision 19 (2026-10-29): `tool` role for tool results (vendor-neutral); `reasoning: auto` = enabled at the vendor's default depth wherever the model can reason; `store: false` on OpenAI and Google by default; `pause_turn` continued by the core up to 10 times.
- Decision 20 (2026-10-29): secrets never enter config JSON (`inline` is a handle filled through `set_secret`); every schema union carries a discriminator; canonical object forms at the boundary, string sugar in bindings.
- Decision 15 (2026-10-28): short vendor namespaces, company names (`anthropic`, `openai`, `google`, `ollama`), for all vendor-specific names; `origin.wire` carries the codec id. Public names are vendor-neutral: `reasoning`, `tool_call`, `finish`, `complete`, `max_output_tokens`, `arguments`, `input_schema`, `vendor_state`.
- Sans-I/O core, host-native HTTP, effects state machine, fixtures as contract.
- Four wires on vendor-recommended surfaces; no Chat Completions, no `generateContent`, no Converse, no Ollama compat layers.
- Retries default on: 2 attempts with backoff on retryable errors, honoring `retry_after`, never after the first streamed byte. Policy lives in core as a pure function.
- `compact()` and `count_tokens` out of v1; compaction reachable through `vendor` and the opaque block.
- Explicit provider per call. No auto-match. Optional `default_model`.
- Agents: types reserved now, backend in v2.

## Decision log

Decisions 1 to 20 settled (2026-10-28 and 29), see Settled. Phase 1 starts with the effects-protocol schema, before any codec. Still open:

- Gemini Interactions fixture spike (decision 7) and its wave.
- Fixture-time checks: OpenAI `encrypted_content` needs `include`; OpenAI summary gating on org verification; OpenAI and Google embedding overflow behavior; Codex backend request constraints; the `anthropic_profile` client id.
- Structured output: own design pass (native format vs forced-tool fallback per vendor).
- Claude Code session preset shipped read-only (2026-10-30) on the unverified file layout; refresh stays off until the endpoint and client id are verified ([[Auth model]]).
- Agents v2 event vocabulary, after v1 freezes.
