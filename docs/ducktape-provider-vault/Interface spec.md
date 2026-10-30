# Interface spec

Phase 0. The normalized surface, language-neutral, frozen section by section (date per section). Rules: the common case needs no options; everything the vendors offer stays reachable through options or namespaced passthrough; names are vendor-neutral (no vendor's wire term where another vendor uses a different one); see [[Design draft]] for the decisions behind it. Signatures are written Python-style; section 11 maps them per language.

## 1. Provider constructor (2026-10-29)

```
Provider(vendor, *, base_url?, auth?, headers?, timeout?,
         default_model?, default_embedding_model?, default_system?, default_tools?, default_options?, default_vendor?,
         overrides?, strict?, observer?)
```

| Param | Type | Default | What it does |
|---|---|---|---|
| `vendor` | `anthropic` / `openai` / `google` / `ollama` (company names) | required | Picks the wire, the vendor URL, env var names, header placement, capability table |
| `base_url` | URL | vendor's own | Same wire on another host: Groq, xAI, OpenRouter, DeepSeek, a proxy, remote Ollama, later Bedrock / Vertex / Foundry |
| `auth` | string, `env(names)`, `session(preset, store?, client_id?)`, `resolver(fn)`, or a list of those | env vars for `vendor`, only when `base_url` is the vendor's own | String = that key. List = tried in literal order. With a custom `base_url`, `auth` is required: the vendor's env key is never sent to another host. See [[Auth model]] |
| `headers` | map | none | Sent on every request |
| `timeout` | seconds or `{connect, read, total}` | connect 10 s, read 60 s, total none | Default for every call |
| `default_model` | string | none | Model used when a call gives none |
| `default_embedding_model` | string | none | Model for `embed` when a call gives none |
| `default_system` | text or blocks | none | System prompt for every call; a call's `system` replaces it |
| `default_tools` | tool list | none | Tools for every call; a call's `tools` replaces it. Typical: set once, never per call |
| `default_options` | map | none | Option defaults merged under every call, call wins |
| `default_vendor` | map | none | Vendor passthrough merged under every call, call wins |
| `overrides` | `{api_version?, capabilities?, dialect?}` | none | Advanced: pin another API version, declare what a custom host can do (`Capabilities.source: user`), name endpoint deviations |
| `strict` | bool | false | Every warning becomes an error; a call can override |
| `observer` | callbacks | none | `on_request`, `on_response`, `on_event`, `on_retry`; redacted data |

- No network at construction; the first call resolves credentials. Missing credential = `auth` error, code `no_credential`, naming what was tried.
- One `Provider` = one endpoint. No registry; the caller keeps a map.
- Wire id, API version, model profile and dialect come from the shipped table keyed by `vendor` and host; `overrides` is the only way to change them.
- A session preset pins its own `base_url`, headers and dialect (Codex sessions talk to the Codex backend, not `api.openai.com`); `base_url` is ignored with a warning when a session is the active source.
- Browser build: env, session and file sources do not exist; an API key requires `dangerously_allow_browser=True` (same gate as the vendor SDKs); the recommended pattern is a `resolver` fetching short-lived tokens from the app's backend.
- Unknown `vendor` or option key is an argument error before any request.

```python
Provider("anthropic")
Provider("anthropic", auth="sk-ant-...")
Provider("openai", auth=session("codex"))                                  # Codex login; backend constraints unverified, see Auth model
Provider("openai", base_url="https://api.groq.com/openai", auth=env("GROQ_API_KEY"))
Provider("ollama", base_url="http://gpu-box:11434")
```

## 2. Methods (2026-10-29)

```
chat(messages, *, model?, system?, tools?, options?, vendor?, timeout?, headers?, strict?, include_raw?) -> Response
stream(same args) -> Stream[Event]              # has cancel(); last event is finish
embed(input, *, model?, options?, vendor?, timeout?, include_raw?) -> EmbedResponse

models(refresh?) -> [ModelInfo]
model_info(id) -> ModelInfo
capabilities(model) -> Capabilities

login(handle_action, *, callback?, timeout?, app_name?) -> AuthStatus     # logs into the provider's own session source; section 10
logout()
auth_status() -> AuthStatus                                    # may read stores and call resolvers; has an async twin
```

| Group | Methods | Common case |
|---|---|---|
| Core | `chat`, `stream` | `chat(messages)` with `default_model` |
| Embeddings | `embed` | `embed(texts)` with `default_embedding_model`; never falls back to the chat model |
| Discovery | `models`, `model_info`, `capabilities` | pickers, hiding unsupported options |
| Auth | `login`, `logout`, `auth_status` | apps that sign users in; no UI in the library |

- `chat` and `stream` are separate methods (honest return types). `chat` runs over the vendor's streaming wire and aggregates in the core (decision 16): one decoder per wire, `read` timeout meaningful, cancellable, `first_token_ms` always known.
- `model` falls back to `default_model`; neither set = error before any request.
- Every method has an async twin wherever the language has async.
- Nothing else is public.

## 3. `chat` and `stream` arguments (2026-10-29)

| Param | Default | Meaning |
|---|---|---|
| `messages` | required | The conversation, section 4 |
| `model` | `default_model` | Vendor model id, no prefix tricks |
| `system` | `default_system` | Text; or a list of text blocks when a cache mark is wanted on it |
| `tools` | `default_tools` | Function tools and hosted tools, section 5 |
| `options` | `default_options` | Portable set below |
| `vendor` | `default_vendor` | `{anthropic: {...}}` raw body fields |
| `timeout` | provider's | Seconds (= total), or `{connect, read, total}`; `read` = gap between bytes; `total` spans retries and sleeps, so each attempt is sent with the time still left; hosts that cannot time the TCP connect alone bound the wait for response headers by connect + read |
| `headers` | none | Extra headers for this call |
| `strict` | provider's | Warnings become errors |
| `include_raw` | false | Attach the vendor's reply body as `response.raw`; on `stream`, also deliver `raw` events |

Options, all optional:

| Option | Default | Values | Mapping |
|---|---|---|---|
| `max_output_tokens` | vendor default; Claude: model max | int | all |
| `reasoning` | `auto` | `auto` `none` `minimal` `low` `medium` `high` `xhigh` `max` | `auto` = reasoning enabled at the vendor's default depth where the model supports it (Claude `thinking: adaptive`, OpenAI `reasoning.effort` default, Google thinking on, Ollama `think: true`); levels rounded to the nearest vendor level with a warning; `none` where the model allows it |
| `reasoning_summary` | `auto` | `none` `auto` | Claude `thinking.display: summarized`, OpenAI `reasoning.summary: auto`, Google `thinking_summaries: auto`; Ollama raw thinking, `none` warns. Verify at fixture time whether OpenAI gates summaries on org verification |
| `tool_choice` | `auto` | `auto` `none` `required` or `{name: "add"}` | `required` / named unsupported = `unsupported` error |
| `parallel_tools` | vendor default | bool | Claude, OpenAI; others warn |
| `output` | none | `{json_schema, name?}` | native format or forced-tool fallback with a warning |
| `stop_sequences` | none | list of strings | Claude, Google, Ollama; OpenAI warns |
| `cache` | `auto` | `none` `auto` `explicit` | `auto` = one mark after system and tools where supported |
| `retries` | 2 | int, 0 disables | retries after the first attempt (Stainless parity: 3 attempts); see section 8 |
| `user_id` | none | string | Claude `metadata.user_id`, OpenAI `safety_identifier`, Google `user_metadata` (unverified); Ollama warns |

- Reasoning is on and visible by default: `reasoning` blocks and `reasoning_delta` events arrive wherever the model can reason, with no configuration.
- Server-side storage is off by default: `openai.responses` sends `store: false` and `include: ["reasoning.encrypted_content"]`, `google.interactions` sends `store: false`; `vendor` can override. Verify both at fixture time.
- `temperature`, `top_p`, `seed`, `service_tier`, `metadata` maps, `truncation`, `max_tool_calls`, `store` are one- or two-vendor fields and go through `vendor`.
- Merge rules: `default_options` and `options` merge key by key, call wins; `default_vendor` and `vendor` merge key by key inside each namespace, call wins; `headers` merge key by key, call wins; `system` and `tools` replace. `null` unsets a default.
- Clash rule: a `vendor` field that targets a wire field an option the caller set (per call or in `default_options`) also writes is an argument error. Library defaults (`reasoning: auto`, `reasoning_summary`, `cache: auto`, Claude `max_tokens`, `store`) yield to `vendor`: `vendor.anthropic.thinking = {type: between_tools}` or `vendor.openai.reasoning = {summary: detailed}` simply win.
- Unknown capability (custom host, no shipped table, no `overrides.capabilities`): send as asked, no warning; the vendor's own error surfaces.

## 4. Messages and blocks (2026-10-29)

```
Message { role: user | assistant | system | tool, content: Block[] }
```

| Role | Holds | Wire mapping |
|---|---|---|
| `user` | text, image, document | all |
| `assistant` | the response's content, returned untouched | all |
| `system` | text, mid-conversation | Claude `system` role on models that allow it (profile), OpenAI `developer` item, Ollama `system` message, Google folded into the system instruction with a warning |
| `tool` | `tool_result` blocks, all parallel results of one turn | Claude user turn with `tool_result` blocks, OpenAI `function_call_output` items, Google `function_result`, Ollama `tool` messages |

| Block | Fields | Notes |
|---|---|---|
| `text` | `text` | citations under `vendor.<vendor>.citations` |
| `image` | one of `data` / `url` / `file_id`; `media_type?` (sniffed from `data` for PNG, JPEG, GIF, WebP) | `url` fetched by the core for Ollama (not in the browser build: `unsupported`); `detail` and similar via `vendor` |
| `document` | one of `data` / `url` / `file_id` / `text`; `media_type`, `name?` | user turns only on Claude |
| `tool_call` | `id`, `name`, `arguments`, `truncated?`, `arguments_json?` | `arguments` parsed object; when the JSON is cut or invalid: `truncated: true`, `arguments: {}`, raw text in `arguments_json` |
| `tool_result` | `tool_call_id`, `content: Block[]`, `is_error?` | images inside on Claude, OpenAI, Google |
| `reasoning` | `text`, `vendor_state?`, `origin?` | `origin` set by the library; redacted reasoning = empty text |
| `compaction` | `vendor_state`, `text?`, `origin` | |
| `custom` | `kind: "<vendor>.<type>"`, `data`, `origin?` | hosted-tool blocks, Google image / audio / video output, unknown kinds; never fails to decode |

Every block may carry `id?` (the vendor's id when it has one), `cache?: {ttl: short / long}`, `vendor?: {<vendor>: {...}}`.

- Opaque state carries its own `origin {wire, vendor, base_url, model, auth_source?}` on the block, so a bare `response.content` array describes itself; no message-level origin.
- Replay rule (decision 17): opaque blocks are sent verbatim when the block's `wire` and `base_url` match the call, whatever the model (Anthropic's guidance: pass thinking blocks back unchanged across models, the API drops what the target cannot read, unbilled; stripping client-side counts as editing history). On a different endpoint they are dropped with a warning; a dropped `compaction` warns that summarized history is lost. Hand-written blocks have no `origin` and are never treated as opaque state.
- History is append-only: append `assistant(response)` and go on.
- Ollama sends no tool-call ids: the core generates `call_<n>` ids and maps `tool_result` back by name and order.
- System prompt placement: Claude top-level `system` (blocks keep cache marks); OpenAI `instructions` for plain text, a `developer` item when blocks carry cache marks; Google `system_instruction` string, blocks joined with blank lines, cache marks warn; Ollama first `system` message. The `chatgpt_plan` dialect allows no system items: folded into the first user turn with a warning.
- Shortcuts over the types: `user(...)`, `assistant(...)`, `system(...)`, `tool(...)` take strings and blocks and return a message; `assistant(response)` takes a response. `text()`, `image()`, `document()`, `tool_result()` return blocks. Clipboard image: `user("what is this?", image(png_bytes))`.
- Naming is vendor-neutral by rule: `reasoning` (not thinking / thought), `tool_call` (not tool_use / function_call), `tool` role (not a user turn), `arguments` and `input_schema` (MCP), stream terminal event `finish` (not message_stop), stop reasons `complete` / `max_output_tokens` (not end_turn / max_tokens).

## 5. Tools (2026-10-29)

```
ToolDef      { type: "function", name, description, input_schema, strict?, cache?, vendor? }
ProviderTool { type: "provider", provider_tool: "<vendor>.<name>", config? }       # tools the vendor runs itself
tools: Tool[]                                                                 # usually set once as default_tools
```

| Field | Notes |
|---|---|
| `type` | discriminator; `ToolDef(...)` fills it in bindings with constructors |
| `input_schema` | JSON Schema; vendor subset limits surface as warnings, never silent edits |
| `strict` | schema enforced by Claude and OpenAI; others warn |
| `cache` | mark the tool list for caching; same `ttl` as blocks |
| `vendor` | per-vendor extras: Claude `eager_input_streaming`, `defer_loading`; OpenAI `custom` freeform tools, `namespace` |
| `provider_tool` | `openai.web_search`, `anthropic.web_fetch`, `google.google_search`, `anthropic.code_execution`, ...; `config` is the vendor's own object, untranslated in v1 |

```python
p = Provider("openai", default_model="gpt-5.5", default_tools=[ToolDef("add", "adds two numbers", schema)])
r = p.chat(msgs)
msgs += [assistant(r), tool(*[tool_result(c.id, run(c)) for c in r.tool_calls])]
```

- The library never runs tools and never loops over them. `r.tool_calls` is the filtered list of `tool_call` blocks.
- Hosted-tool calls and results arrive as `custom` blocks and go back untouched.
- On the wire every request carries the tool list (all vendors are stateless per request); `cache: auto` marks it so repeats are cheap.

## 6. Response (2026-10-29)

```
Response
  content: Block[]               # text, reasoning, tool_call, compaction, custom, in order
  stop: {reason, raw}            # reason: complete | tool_call | max_output_tokens | stop_sequence | context_window_exceeded
                                 #   | refusal | content_filtered | continue_required | compaction | malformed_output | other
  usage: {input: {total, cache_read?, cache_write?}, output: {total, reasoning?}, raw} | null
  warnings: [{kind: unsupported | compatibility | deprecated, feature, detail}]
  origin: {wire, vendor, base_url, model, auth_source?}
  latency_ms, first_token_ms     # whole call incl. retries; first_token_ms = request start to the first delta
  raw?                           # vendor reply body, only with include_raw=True
```

| View over `content` | Returns |
|---|---|
| `text` | all `text` blocks concatenated, no separator |
| `reasoning` | all `reasoning` blocks' text concatenated |
| `tool_calls` | the `tool_call` blocks |

- `usage.input.total` includes cached tokens on every vendor. `usage` is `null` when the vendor sent nothing, never absent.
- `stop.reason` is `tool_call` whenever a complete `tool_call` block is present, whatever the vendor said; if any tool call is truncated it is `max_output_tokens`. `stop.raw` keeps the vendor value.
- Server-tool long turns (Claude `pause_turn`) are continued by the core automatically, up to 10 continuations per call; `continue_required` is reported only when that cap is hit or a vendor asks for a continuation the core cannot perform: then send `assistant(response)` back and call again.
- `warnings` is always a list, empty when nothing degraded.
- `raw` is opt-in: the common case never needs it; it is for debugging and unmodeled vendor fields.

## 7. Stream events (2026-10-29)

```
stream(...) -> Stream[Event]     # iterator with cancel()

Event (block-scoped events carry index = position in the final content)
  block_start      {index, kind, id?, name?}     # kind: text | reasoning | tool_call | compaction | custom
  text_delta       {index, text}
  reasoning_delta  {index, text}
  tool_call_delta  {index, arguments_json}       # fragment of the arguments JSON
  block_stop       {index, block}                # the finished block, with vendor_state when any
  finish           {response}                    # the same Response chat returns; always last
  raw              {event}                       # vendor event, only with include_raw=True
```

```python
for ev in p.stream(msgs):
    if ev.type == "text_delta":
        print(ev.text, end="")
    elif ev.type == "finish":
        r = ev.response
```

- Errors mid-stream raise the same errors as `chat` (section 8). A stream dropped before `finish` is an error, never a silent short response.
- `cancel()` closes the host connection and flags the operation; the core is never touched from another thread. The consumer's next iteration raises `cancelled`. In async languages, cancelling the task does the same; `chat` is cancellable the same way (section 11).
- Same event sequence on every vendor: a tool call that arrives whole (Ollama, some Google cases) still yields `block_start`, one `tool_call_delta` with the full JSON, `block_stop`. One consumer code path.
- Vendor-only stream events (hosted-tool progress, citations, keepalives) are not normalized; they appear as `raw` when asked.

## 8. Errors (2026-10-29)

Defined as data so every language maps it idiomatically (classes, a struct with `kind`, a C code):

```
ProviderError { kind, message, status?, code?, retryable, retry_after?, attempts, body?, request_id?, origin }
```

| `kind` | When | Retryable |
|---|---|---|
| `auth` | no credential, expired, resolver failed, rejected (401 / 403); `code` says which | no |
| `rate_limit` | 429 with a retry window | yes |
| `quota` | 429 or 402 billing or spend limit (vendor code) | no |
| `server` | 5xx, overloaded | yes |
| `connection` | DNS, refused, TLS, dropped | before the first response byte only |
| `timeout` | connect, read or total exceeded | before the first response byte only |
| `cancelled` | `cancel()` or host cancellation | no |
| `not_found` | unknown model or endpoint (404) | no |
| `context_overflow` | input too long for the model | no |
| `request_too_large` | body over the vendor's byte limit | no |
| `bad_request` | any other 4xx | no |
| `malformed_response` | vendor reply could not be parsed | no |
| `unsupported` | the provider cannot do what was asked (loss policy, decision 6a) | no |

- `retryable` is computed per vendor from status and code, never from status alone.
- Retries happen in the core before an error reaches the caller: `options.retries` retries after the first attempt (default 2) with backoff on `rate_limit`, `server`, `connection`, `timeout`, honoring `retry_after` up to 60 s (longer = no retry), never after the first response byte, always inside `timeout.total`. `attempts` reports how many were made; `on_retry` fires per retry.
- Blocked or malformed model output is a `Response`, not an error, on every vendor. Google: a blocked reply (`safety` / `recitation` / `prohibited_content` / `spii` / `blocklist` / `content_blocked`, whether as an HTTP error before output or as an error event mid-stream) becomes `stop.reason = content_filtered` with the partial content kept; `malformed_function_call` / `malformed_tool_call` / `unexpected_tool_call` / `too_many_tool_calls` become `malformed_output`; the code is in `stop.raw`.
- Ollama chat truncates an over-long prompt silently; the core warns (`compatibility`) when `prompt_eval_count` reaches the model's context length. Best effort.
- `body` is the vendor's error body with secrets redacted; `message` is ours, short, vendor text appended.
- Argument mistakes raise the language's own errors before any request.
- One base, so catching `ProviderError` catches everything the library raises.

## 9. Embeddings and discovery (2026-10-29)

```
embed(input, *, model?, options?, vendor?, timeout?, include_raw?) -> EmbedResponse
  input: one string or a list of strings; one string = a one-element batch
  options: {dimensions?, retries?}
EmbedResponse { embeddings: [[float]], usage: {input: {total}} | null, warnings, origin, latency_ms, raw? }
  view: dimensions = length of the first vector
```

- `model` falls back to `default_embedding_model`, never to `default_model`.
- `embeddings[i]` is the vector for `input[i]`; vectors are always floats.
- `dimensions` maps to OpenAI `dimensions`, Google `output_dimensionality`, Ollama `dimensions`. Google `task_type` and other single-vendor knobs go through `vendor`.
- Too-long input: Ollama is sent `truncate: false` so it errors like the others; `vendor.ollama.truncate` opts back in. That OpenAI and Google report overflow as an error is unverified; mapped to `context_overflow` where they do.
- Batch and token limits are the vendor's and come back as `bad_request`. Empty input is a language error before any request. Claude: `unsupported`.

```
models(refresh?) -> [ModelInfo]
model_info(id) -> ModelInfo                    # unknown id: not_found
ModelInfo { id, kind: chat | embedding | unknown, context_window?, max_output_tokens?, capabilities?, raw }

capabilities(model) -> Capabilities            # every value yes | no | unknown
  reasoning: {levels: [...], can_disable}
  tool_choice: [auto, none, required, named], parallel_tools: {supported, toggle}
  output: {json_schema}, cache: {breakpoints, ttls: [short, long]}
  inputs: [text, image, document, audio, video], tools: {functions, hosted: ["openai.web_search", ...]}
  limits: {context_window?, max_output_tokens?}, embeddings, source: shipped | vendor | user
```

- Discovery results are cached per provider for the process lifetime; `models(refresh=True)` re-fetches.
- `kind` is `unknown` where the listing does not separate chat from embedding models (Ollama). OpenAI lists ids only; `capabilities()` fills limits from the shipped table. A custom host with no table answers `unknown` unless `overrides.capabilities` says otherwise.

## 10. Auth and observer (2026-10-29)

```
login(handle_action, *, callback?, timeout?, app_name?) -> AuthStatus
  logs into the provider's session source (its preset, store and client_id); library-owned presets only: chatgpt_plan (anthropic_profile still unsupported)
  handle_action(action) -> string?               # the app shows it; returns the pasted value for paste
    action: open_url {url, instructions} | show_code {url, code, instructions} | paste {prompt}
  callback: loopback (default) | paste            # paste for headless apps; the browser build has no login
  timeout: default 5 minutes; expiry fails with timeout{login_timeout}
  app_name: the name the vendor shows the user at first registration; default ducktape-provider
logout()                                          # library-owned sessions only; codex / claude_code: unsupported, message names the CLI command
auth_status() -> AuthStatus { source, expires_at?, session? }

observer: { on_request(req), on_response(res), on_event(ev), on_retry(err, attempt) }   # all optional; redacted data only
```

- The flow (PKCE, state, nonce, loopback listener on `127.0.0.1:1455`, token exchange, id token checks, store) runs in the core and the binding's effect loop; the app never sees a token. A busy port fails with `connection{callback_port_busy}`; the handler's return value is ignored with the loopback callback and is the pasted redirect URL with `paste`. ChatGPT plan calls drop the request fields the plan endpoint rejects (`max_output_tokens`, `temperature`, `top_p`, `metadata`, `safety_identifier`, `user`, `service_tier`, `previous_response_id`, `truncation`) with an `unsupported` warning each, and `models()` lists the plan's models.
- `chatgpt_plan` tokens go to the library's own store, shared on the machine (`$XDG_CONFIG_HOME/ducktape/chatgpt_plan.json`, `~/.config/ducktape/chatgpt_plan.json` or `%APPDATA%\ducktape\chatgpt_plan.json`, a 0600 file; the keychain variant is not built), so one login serves every app using the library; `session(preset, store=path)` isolates an app. `anthropic_profile` writes a profile in Anthropic's documented shared location (`~/.config/anthropic`), readable by the `ant` CLI; how a third party obtains the OAuth client id is unverified, so this preset ships only once a fixture proves the flow.
- `session(preset, client_id=...)` for vendors that issue ids per app.
- Not logged in: the first call raises `auth{no_credential}` with the hint to call `login`. Apps check `auth_status()` first or catch that.
- `auth_status().source` is the same label every response carries in `origin.auth_source`: `env:OPENAI_API_KEY`, `session:codex`, `resolver`, `inline`.
- `login` and `auth_status` have async twins like every method.

```python
p = Provider("openai", auth=session("chatgpt_plan"))
if p.auth_status().source is None:
    p.login(lambda a: print(a.instructions, a.url))
r = p.chat([user("hi")])
```

## 11. Per-language mapping (2026-10-29)

| Language | Keyword arguments | Cancellation | Sugar unions (string or object) |
|---|---|---|---|
| Python | native | `cancel()`; `asyncio` task cancel | native |
| JS / TS | options object | `AbortSignal` on every call | native |
| Go | options struct, `ctx context.Context` first parameter | `ctx` | canonical object forms only |
| Java / Kotlin | builder | `cancel()` on the stream, `CompletableFuture.cancel` | canonical object forms only |
| .NET | optional parameters / options record | `CancellationToken` on every call | canonical object forms only |
| Rust | builder | drop the future / `cancel()` | enums |
| C | JSON documents through the C ABI, the host runs the effect loop | resume with `cancelled` | JSON only |

- Names: one name per concept in every language, in that language's casing (`model_info`, `modelInfo`, `ModelInfo`); a language's own rules win where they ask for more, so .NET methods take a verb (`GetModelsAsync`, `GetResponse`) and C# keeps `SystemMessage` and `ToolMessage` where the bare word would clash.
- Canonical forms (what the schema and the core accept): `system` = `TextBlock[]`, `tool_result.content` = `Block[]`, `embed.input` = `string[]`, `timeout` = `{connect, read, total}`, `tool_choice` = `{mode, name?}`, tools tagged with `type`. Strings and scalars are binding sugar converted before the boundary.
- `cancel` on any call, streamed or not, closes the host connection; the core sees it on the next resume.
- Conformance runs against both boundaries (PyO3 native objects and the JSON C ABI / wasm) so they cannot drift.

## JSON Schema (2026-10-29)

`schema/provider.schema.json` in the repo: JSON Schema 2020-12, version in `$id` (`1.0.0`). Open world per decision 8: open enums (`anyOf` known set / string), unknown fields allowed, unknown blocks and events as `custom` / `raw`; the argument errors of sections 1 and 3 are binding-side validation, the schema stays open for forward compatibility. Every union carries a discriminator (`type`, `mode`, source key) so Go, Java, .NET and C types generate cleanly. Covers Message and every Block, Tool, ChatRequest and Options, Response, Usage, Warning, Event, ProviderError, EmbedRequest and EmbedResponse, ModelInfo, Capabilities, AuthStatus, LoginAction, Auth and ProviderConfig. Secrets never appear in config JSON: `inline` holds a secret handle filled through `set_secret`. The core-to-binding effects protocol is `schema/effects.schema.json` (`Start` / `Resume` in, `Step { notify[], effect }` out; `Clock { now_ms, entropy }` on every call; body bytes may cross raw through `op_feed`). Type samples in `fixtures/types/` are validated against both schemas and round-tripped through the Rust types by the conformance crate.
