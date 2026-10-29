# Provider API matrix

What each provider's current API offers and how. As of 2026-10-28, from official docs (Claude: the bundled `claude-api` reference, model table cached 2026-10-06, API-version migration guide for `2026-10-21`; OpenAI, Gemini, Ollama, Bedrock: downloaded reference pages). Rule applied: for a vendor's own models, target the surface the vendor recommends for new work and skip surfaces it steers away from. Chat Completions is excluded everywhere (decided, see [[Design draft]]).

## Surface choice per vendor

| Vendor | Target | Avoid | Why |
|---|---|---|---|
| Anthropic | Messages API `POST /v1/messages`, `anthropic-version: 2026-10-21` | Text Completions (removed from SDK 1.x); `output_format` (replaced by `output_config.format`); `budget_tokens` thinking; `computer_20251124` on newest models | Single endpoint for everything. Version 2026-10-21 removes sampling params, rejects `document` blocks in assistant turns and `citations` text blocks in user turns |
| OpenAI | Responses API `POST /v1/responses` | Chat Completions: "remains supported, Responses is recommended for all new projects"; no encrypted reasoning continuity; since GPT-5.4 no tool calling with `reasoning_effort` other than `none`. Assistants (sunset 2026-08-26). Completions | Responses is the forward path and the only shape with full reasoning state and hosted tools |
| Google | Interactions API `POST /v1beta/interactions` (reference viewed is v1beta; a stable v1 exists; GA 2026-06) | `generateContent` (docs index labels it "legacy Gemini Generate Content API"); explicit `cachedContents` (not on Gemini 4 or Interactions); `thinking_budget` | New features land on Interactions (steps, background, agents); `generateContent` stays served but frozen. Embeddings, token counting and batch keep their own endpoints |
| Ollama | Native `/api/chat`, `/api/embed`, `/api/tags`, `/api/show` | `/api/embeddings` (gone from the OpenAPI spec; `/api/embed` replaces it); `/api/generate` for chat; compat layers `/v1/chat/completions`, `/v1/responses` (stateless), `/v1/messages` (drop `tool_choice`, limited stop reasons) | Native carries model metadata, think levels, capabilities; compat layers are lossy |
| AWS Bedrock | For Claude: Messages API on the Mantle endpoint (`bedrock-mantle`, Messages shape; the Claude SDKs' preferred client). For OpenAI models: Responses on `bedrock-runtime` (synchronous only, no server-side tools) | Converse (AWS still recommends it "for a consistent interface across all models", but it is a third normalized shape with its own binary framing); InvokeModel path (`bedrock-runtime` Messages goes through InvokeModel; the Claude SDK calls it legacy) | AWS "new to Bedrock" advice: `bedrock-runtime` with Messages, Chat Completions or Responses. Version `2026-10-21` on Bedrock only for Claude 4.7+. Wire-compatible, SigV4 auth, not wave 1 |
| Google Vertex | Messages shape for Claude; Gemini via Vertex's own surface (Interactions availability on Vertex unverified) | OpenAI-compat chat endpoint | Same wires, GCP auth. Not wave 1 |
| Azure / Foundry | Responses on `/openai/v1/`; Messages for Claude | Chat Completions | Same wires, Azure auth. Not wave 1 |
| xAI, Groq, OpenRouter, DeepSeek, Together, Mistral | Where they serve Responses (xAI recommended; Groq and OpenRouter beta) or Messages (DeepSeek `/anthropic`, OpenRouter) | Their Chat Completions endpoints | Vendors reachable only through Chat Completions (Together, Mistral native, llama.cpp server, LM Studio, SGLang, Fireworks, Gemini's OpenAI-compat) are out until they ship Responses or Messages. vLLM reportedly serves `/v1/responses` (unverified). Ollama stays reachable natively |

## Columns

C = Claude Messages. O = OpenAI Responses. G = Gemini Interactions. L = Ollama native. Bedrock, Vertex, Azure reuse C or O. In-cell alternatives are written with `/`.

## 1. Request shape

| | C | O | G | L |
|---|---|---|---|---|
| Endpoint | `POST /v1/messages` | `POST /v1/responses` | `POST /v1beta/interactions` | `POST /api/chat` |
| Auth | `x-api-key` or `Authorization: Bearer` (both accepted; the 1.x SDKs send Bearer), `anthropic-version` header | `Authorization: Bearer` | `x-goog-api-key` or Bearer | none local; `Authorization: Bearer` on ollama.com |
| Model | `model` | `model` | `model` or `agent` (one required) | `model` |
| History | `messages[{role: user / assistant, content: blocks}]`, stateless, consecutive same roles merged | `input`: string or item array (`message` with roles user / assistant / system / developer, `function_call`, `function_call_output`, `reasoning`, `compaction`); or `previous_response_id` / `conversation` | `input`: string, Content, Content[] or Step[]; stateless replay = send the previous response's `steps` back as `input` plus the new user content; or `previous_interaction_id` (server keeps history; `store=true` default, `store=false` opts out and disables chaining and background) | `messages[{role: system / user / assistant / tool, content: string}]`, stateless |
| System prompt | top-level `system` (string or text blocks); mid-conversation `role: system` on newest models | `instructions` (not inherited across `previous_response_id`, cannot hold a cache breakpoint) or system / developer items | `system_instruction` (string, interaction-scoped, not inherited across `previous_interaction_id`) | `system` message |
| Assistant prefill | removed (400 on 4.6+) | n/a | n/a | last assistant message continues (model-dependent) |
| Max output | `max_tokens` (required) | `max_output_tokens` | `generation_config.max_output_tokens` | `options.num_predict` |
| Stop sequences | `stop_sequences` | none in Responses | `generation_config.stop_sequences` | `options.stop` |
| Sampling | removed on 2026-10-21 (`temperature` / `top_p` / `top_k` = 400) | `temperature`, `top_p` (effect on reasoning models unverified) | `temperature` / `top_p` / `top_k` deprecated (changelog 2026-07-21, scope unconfirmed); absent from Interactions `GenerationConfig`; `seed` | `options.temperature`, `top_p`, `top_k`, `seed`, ... |
| Streaming flag | `stream: true` | `stream: true` | `stream: true` | `stream` default true |
| Metadata | `metadata.user_id` | `metadata`, `safety_identifier`, `prompt_cache_key` | `labels`, `user_metadata` | none |
| Passthrough room | `output_config`, betas via `anthropic-beta` header | `include[]`, `truncation`, `service_tier`, `store` | `generation_config.*`, `safety_settings`, `service_tier`, `environment_id`, `agent_config`, `webhook_config` | `options.*`, `keep_alive` |

## 2. Content blocks (input)

| | C | O | G | L |
|---|---|---|---|---|
| Text | `text` | `input_text` | `text` | string |
| Image | `image` source `base64` / `url` / `file` (Files API) | `input_image` from URL, data URL or `file_id`; `detail` | `image` source `inline` (base64) / `remote` (URL) / `gcs` / Files API | `images[]` base64 only |
| Document / PDF | `document` from base64 / url / file / plain text / custom content; `citations: {enabled}`; user turns only (2026-10-21) | `input_file` from `file_id`, `file_data` (base64) or URL | `document` inline / remote / gcs / Files API | none |
| Audio | none | not in Responses | `audio` inline / remote | none for chat |
| Video | none | none | `video` inline / remote | none |
| Tool result | `tool_result {tool_use_id, content: string or [text, image, document, search_result], is_error}` in a user message; all parallel results in one message | `function_call_output {call_id, output: string or [input_text, input_image, input_file]}` item | `function_result {call_id, name, result: Content[], is_error}` | `role: tool` message `{tool_name, content}`; no call id |
| Thinking round-trip | `thinking {thinking, signature}` blocks verbatim; bound to model + prefix (preserved thinking); unreadable blocks dropped unbilled | `reasoning` item with `encrypted_content` (returned by default); resend all output items; assistant `phase` must be resent | `thought` step `{signature, summary[]}` replayed inside `steps`; signature required for function calling on Gemini 3/4 (error `missing_thought_signature`) | `thinking` string, no state |
| Cache breakpoint | `cache_control {type: ephemeral, ttl: 5m / 1h}` on text / image / document / tool_result / tool / system blocks, max 4; or top-level `cache_control` auto | `prompt_cache_breakpoint` on content (GPT-5.6+, max 4 writes, min 1024 tokens); `prompt_cache_options {mode: implicit / explicit, ttl: 30m / 12h}` | none (implicit only) | none (KV reuse, `keep_alive`) |
| Opaque vendor blocks | `compaction`, `fallback`, server-tool results, `search_result`, `redacted_thinking` | `compaction`, hosted tool call / output items, `item_reference` | `code_execution_*`, `google_search_*`, `google_maps_*`, `file_search_*`, `mcp_server_tool_*`, `url_context_*` steps | none |

## 3. Tools

| | C | O | G | L |
|---|---|---|---|---|
| Function tool | `{name, description, input_schema, strict?, cache_control?, eager_input_streaming?, defer_loading?, allowed_callers?}` | `{type: function, name, description, parameters, strict}`; also `custom` (freeform grammar), `namespace`, `tool_search` | `{type: function, name, description, parameters}` (JSON Schema) | `{type: function, function: {name, description, parameters}}` |
| Tool choice | `auto` / `any` / `tool {name}` / `none`; `disable_parallel_tool_use`; `any` and `tool` = 400 on Opus 5.5, Sonnet 5.5, Fable 5.1 | `none` / `auto` / `required` / `{type: function, name}` / `allowed_tools` / hosted tool; `parallel_tool_calls`; `max_tool_calls` | `tool_choice`: `auto` / `any` / `none` / `validated`, or `{allowed_tools: {mode, tools[]}}` | not supported |
| Parallel calls | default on; results back in one user message; toggle `disable_parallel_tool_use` | `parallel_tool_calls` (default on) | yes; no toggle; signature only on first call | yes (`index`); no toggle |
| Tool call in output | `tool_use {id, name, input}` block, `input` parsed object | `function_call {call_id, name, arguments}` item, `arguments` JSON string | `function_call {id, name, arguments}` step | `message.tool_calls[{function: {name, arguments}}]`, args parsed object, no id |
| Structured output | `output_config.format {type: json_schema, schema}` (+ `strict` tools); incompatible with citations; `messages.parse()` in SDK | `text.format {type: json_schema, name, schema, strict}` or `json_object` | `response_format` (JSON Schema; one format or an array) | `format: json` or JSON Schema object |
| Hosted tools | `web_search_20260209`, `web_fetch_20260209`, `code_execution_20260521`, `tool_search_*`, `computer_toolset_20260801`, memory, bash, text_editor, advisor, MCP connector (`mcp_servers` + `mcp_toolset`, beta) | `web_search`, `file_search`, `code_interpreter`, `computer`, `image_generation`, `mcp` (with approvals), `shell`, `apply_patch` | `google_search`, `google_maps`, `code_execution`, `url_context`, `file_search`, `computer_use`, `mcp_server`; agents (deep research, antigravity, dynamic) with `environment_id` | web search via cloud capability |
| Hosted tool results | content blocks (`web_search_tool_result`, `bash_code_execution_tool_result`, ...) in the same response; `pause_turn` for long turns | output items (`web_search_call`, `mcp_call`, `mcp_approval_request`, ...) | `*_call` / `*_result` steps | n/a |
| Tool errors | `tool_result.is_error` | no flag, text only | `function_result.is_error` | no flag |

## 4. Reasoning

| | C | O | G | L |
|---|---|---|---|---|
| Config | `thinking {type: adaptive, display: summarized / omitted / updates}`; `output_config.effort: low / medium / high / xhigh / max`; cannot disable on Opus 5.5 and Fable; Sonnet 5.5 off only via `{type: between_tools}`; `budget_tokens` removed on 4.7+ | `reasoning {effort: none..max, summary: auto / concise / detailed, mode: standard / pro, context: auto / current_turn / all_turns}` | `generation_config.thinking_level: minimal / low / medium / high`; `thinking_summaries: auto / none` | `think: bool` or model level string from `/api/show thinking.values` |
| Output | `thinking` blocks (summary or empty), `signature`; raw chain of thought never shown; `redacted_thinking` on older models | `reasoning` item: `summary[]`, optional `content[]`, `encrypted_content` | `thought` step `{signature, summary: [text]}` | `message.thinking` |
| Round-trip | required verbatim, same model, append-only history | resend items or use `previous_response_id`; `reasoning.context: all_turns` | replay `steps` verbatim; signature required for function calling in the current turn | not needed |
| Billing | thinking in `output_tokens` | `output_tokens_details.reasoning_tokens` | `total_thought_tokens` separate from output | `eval_count` includes thinking |

## 5. Streaming

| | C | O | G | L |
|---|---|---|---|---|
| Transport | SSE | SSE (WebSocket mode optional) | SSE | NDJSON |
| Lifecycle | `message_start` / `message_delta {stop_reason, usage}` / `message_stop` / `ping` / `error` | `response.created` / `in_progress` / `queued` / `completed` / `incomplete` / `failed`, `error`; `sequence_number` on every event; resumable with `starting_after` (background) | `interaction.created`, `interaction.status_update`, `interaction.completed {status, usage}`, `error`, `done` (`[DONE]`) | lines until `done: true` (`done_reason`, metrics); error line `{"error"}` with HTTP 200 |
| Block boundaries | `content_block_start {index, block}` / `content_block_stop` | `output_item.added` / `done {output_index}`; `content_part.added` / `done {content_index}` | `step.start {index, step}` / `step.stop` | none; infer on field change |
| Text | `content_block_delta text_delta` | `response.output_text.delta` / `done`; `output_text.annotation.added` | `step.delta {delta: text}` | `message.content` chunk |
| Thinking | `thinking_delta`, `signature_delta` | `reasoning_summary_text.delta` / `done`, `reasoning_text.delta` / `done`, `reasoning_summary_part.added` / `done` | `step.delta {delta: thought_summary}`, `step.delta {delta: thought_signature}` | `message.thinking` chunk |
| Tool args | `input_json_delta` (eager with `eager_input_streaming`) | `function_call_arguments.delta` / `done`; `custom_tool_call_input.delta` / `done` | `step.delta {delta: arguments_delta}` | whole `tool_calls` in one line |
| Refusal / blocked | `stop_reason: refusal` + `stop_details` | `response.refusal.delta` / `done` | `error` event with codes `safety`, `recitation`, `prohibited_content`, `spii`, `blocklist`, `content_blocked`, ... (errors, not statuses) | n/a |
| Hosted tool progress | whole result blocks; `compaction_delta`; `citations_delta` | `response.<tool>_call.*` (web_search, file_search, code_interpreter `_code.delta`, image_generation `partial_image`, mcp_call, shell), `compaction.compacting` | `*_call` / `*_result` steps | n/a |
| Usage | cumulative in `message_delta` | terminal `response.completed` only | `interaction.completed.interaction.usage` | final line |

## 6. Response

| | C | O | G | L |
|---|---|---|---|---|
| Envelope | `{id, model, role, content[], stop_reason, stop_details?, usage}` | `{id, status, output[], incomplete_details?, usage, ...}` | `{id, status, steps[], usage, model?, agent?}` | `{model, message, done, done_reason, metrics}` |
| Output blocks | `text`, `thinking`, `tool_use`, server tool blocks, `compaction`, `fallback` | items: `message` (parts `output_text`, `refusal`), `reasoning`, `function_call`, hosted tool items, `compaction` | steps: `text`, `thought`, `function_call`, hosted `*_call` / `*_result`, `image`, `audio`, `video` | `message.content`, `message.thinking`, `message.tool_calls` |
| Stop / status | `stop_reason`: `end_turn`, `tool_use`, `max_tokens`, `stop_sequence`, `pause_turn`, `refusal`, `compaction`, `model_context_window_exceeded`; `stop_details {category, explanation}` on refusal | `status`: `completed`, `incomplete` (`incomplete_details.reason: max_output_tokens / max_messages / content_filter / steered`), `failed`, `cancelled`, `queued`, `in_progress`; tool use inferred from items | `status`: `in_progress`, `requires_action`, `completed`, `failed`, `cancelled`, `incomplete`; blocked and malformed output arrive as errors (`safety`, `malformed_function_call`, `unexpected_tool_call`, `too_many_tool_calls`, ...) | `done_reason`: `stop`, `length`, (`load`, `unload` unverified) |
| Usage | `input_tokens` (excludes cache), `cache_creation_input_tokens`, `cache_read_input_tokens`, `output_tokens` (incl. thinking), `server_tool_use`, `iterations[]`, `speed`, `inference_geo` | `input_tokens` (incl. cached), `input_tokens_details {cached_tokens, cache_write_tokens}`, `output_tokens`, `output_tokens_details {reasoning_tokens}`, `total_tokens` | `total_input_tokens`, `total_cached_tokens`, `total_output_tokens`, `total_thought_tokens`, per-modality breakdowns | `prompt_eval_count`, `prompt_eval_cached_count`, `eval_count`, `*_duration` ns |
| Latency reported | no (`request-id` header) | no | no | `total_duration`, `load_duration`, `eval_duration` |
| Citations | `citations[]` on text blocks | annotations on `output_text` | grounding on search steps (shape unverified) | none |

## 7. Caching and server state

| | C | O | G | L |
|---|---|---|---|---|
| Prompt cache | explicit breakpoints, 5m / 1h, model-dependent minimum 512-4096 tokens, cache reads reported | implicit by default; explicit breakpoints GPT-5.6+, 30m / 12h; `prompt_cache_key` | implicit only (min 2048 / 4096 tokens); explicit caching gone on Gemini 4 and Interactions | KV reuse, `keep_alive` |
| Server conversation state | none (Managed Agents separate) | `store` (default true), `previous_response_id`, Conversations API | `store` (default true), `previous_interaction_id`; `store=false` blocks chaining and background | none |
| Background / async | Batches only | `background: true`, poll `GET /responses/{id}`, `POST .../cancel`, stream resume | `background: true`, `GET /interactions/{id}`, `POST .../cancel`; webhooks | none |
| Compaction | threshold `context_management.edits [{type: compact_20260112}]` (beta `compact-2026-01-12`); on-demand `compaction {type: summarize}` (beta `compact-2026-09-04`); readable signed block | `context_management [{type: compaction, compact_threshold}]`, `POST /responses/compact`, `compaction_trigger` item; encrypted | none | none |
| Context editing | `clear_tool_uses_20250919`, `clear_thinking_20251015` (beta) | `truncation: auto` | none | none |
| Cancel | close connection | close connection; `/cancel` for background | close connection (499 `cancelled`); `/cancel` for background | close connection |

## 8. Other endpoints

| | C | O | G | L |
|---|---|---|---|---|
| Embeddings | none | `POST /v1/embeddings` (`dimensions`, `encoding_format`) | `models/{m}:embedContent`, `batchEmbedContents` (`gemini-embedding-2`, `gemini-embedding-001`; `task_type`, `output_dimensionality`) | `POST /api/embed` (`input` string or list, `truncate`, `dimensions`) |
| Model listing | `GET /v1/models`, `GET /v1/models/{id}`: `max_input_tokens`, `max_tokens`, `capabilities` | `GET /v1/models`: id, created, owned_by only | `GET /v1beta/models`: `inputTokenLimit`, `outputTokenLimit`, `supportedGenerationMethods`, `thinking` | `GET /api/tags`; `POST /api/show`: `capabilities[]`, `model_info.*.context_length`, `thinking.values`; `GET /api/ps` |
| Token counting | `POST /v1/messages/count_tokens` | `POST /v1/responses/input_tokens` | `models/{m}:countTokens` (generateContent shape) | none |
| Batch | `POST /v1/messages/batches` (any order, 256 MB) | `POST /v1/batches` (JSONL) | `batchGenerateContent` (generateContent shape; Interactions batch unverified) | none |
| Files | Files API GA (`client.files.*`), `file_id` in blocks | Files API, `file_id` | Files API (48 h retention, 2 GB per file) | none |
| Agents | Managed Agents (separate REST + SSE, beta) | Agents API (hosted; `/v1/agents/sessions`, `/v1/agents/environments`; not yet evaluated, see [[Agents]]) | Interactions `agent=` + `agent_config`, `environment_id` | none |
| Usage / billing | Admin API usage and cost reports | Admin usage API | none in API | `/api/usage`, `/api/balance` (cloud) |

## 9. Errors

| | C | O | G | L |
|---|---|---|---|---|
| Body | `{type: error, error: {type, message}, request_id}` | `{error: {message, type, param, code}}` | `{error: {code: snake_case, message}}` | `{error: string}` |
| 400 | `invalid_request_error` | `invalid_request_error` | `invalid_request`, `failed_precondition`, `parameter_unknown`; blocked content (`safety`, `recitation`, `prohibited_content`, `spii`, `blocklist`, `content_blocked`); malformed output (`malformed_function_call`, `malformed_tool_call`, `unexpected_tool_call`, `too_many_tool_calls`, `missing_thought_signature`) | bad request |
| 401 / 403 | `authentication_error` / `permission_error` | invalid key / region or permission | `authentication` / `permission_denied` | 401 on cloud |
| 402 | `billing_error` | n/a | `payment_required` (do not retry) | n/a |
| 404 | `not_found_error` | not found | `not_found`, `model_not_found` | model not found |
| 409 | n/a | conflict | `already_exists`, `aborted` | n/a |
| 413 | `request_too_large` (32 MB request; Anthropic's error page also lists too many input tokens as a 413 cause) | n/a | n/a | n/a |
| 416 | n/a | n/a | `out_of_range` | n/a |
| 429 | `rate_limit_error` + `retry-after`; spend-cap 429 has no `retry-after` | type `rate_limit_error` (code `slow_down`) + `Retry-After`; billing codes `credit_balance_exhausted`, `organization_spend_limit_exceeded`, `project_spend_limit_exceeded`, `organization_usage_limit_exceeded` (type may be `insufficient_quota`) are not retryable | `rate_limit_exceeded`, `too_many_requests` (retry), `quota_exceeded` (daily, wait) | cloud rate limit |
| 499 | n/a | n/a | `cancelled` | n/a |
| 5xx | 500 `api_error`, 529 `overloaded_error` (also mid-stream) | 500, 503 `service_unavailable_error` / `server_is_overloaded` + `Retry-After` | 500 `api_error`, 501 `unimplemented`, 503 `service_unavailable` | 500; 502 cloud model unreachable |
| Context overflow | 400 "prompt is too long", or 413; input + `max_tokens` over window stops with `model_context_window_exceeded` | `context_length_exceeded` (400) | undocumented on Interactions | silent truncation (`num_ctx`); `truncate: false` on embed only |
| Mid-stream | `error` event | `error` event, `response.failed` | `error` event | NDJSON error line |

## Unverified cells

Interactions: citations / grounding shape, batch support, Vertex availability, context-overflow reporting. OpenAI: effect of sampling params on reasoning models. Gemini sampling deprecation scope. Ollama `done_reason` values beyond `stop` / `length`. Bedrock Messages feature parity with the Claude API (web fetch, fallbacks, fast mode are Claude API only per the reference). vLLM Responses support.
