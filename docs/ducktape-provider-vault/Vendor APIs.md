# Vendor APIs

Normalization verdicts, streaming, stop reasons, errors. Feature-by-feature matrix on the chosen surfaces is in [[Provider API matrix]]. As of 2026-10-28, from official docs. Columns: Claude Messages (2026-10-21), OpenAI Responses, Gemini Interactions (IX), Ollama native, Bedrock (Messages / Responses on `bedrock-runtime`; Converse only as a reference design).

## State of the vendors

- OpenAI and Google push stateful APIs: Responses (`previous_response_id`, Assistants sunset 2026-08), Interactions (`previous_interaction_id`, GA 2026-06, `generateContent` labelled legacy). Anthropic Messages stays stateless (Managed Agents is a separate product). Bedrock serves Messages, Chat Completions and Responses besides Converse. xAI deprecates Chat Completions.
- Opaque round-trip state is mandatory: Claude thinking `signature` + compaction signature; OpenAI reasoning `encrypted_content` (default) + `phase`; Gemini `thought` step `signature` (missing one = `missing_thought_signature` error on Gemini 3/4); Bedrock `reasoningContent.signature`.
- Claude "preserved thinking": thinking bound to model + history prefix; editing earlier history = 400 for new accounts. History must be append-only.
- Claude API `2026-10-21`: `temperature`/`top_p`/`top_k` removed (400); `document` blocks rejected in assistant turns; text blocks with `citations` rejected in user turns. Bearer auth accepted alongside `x-api-key` (the 1.x SDKs switched to Bearer). Newest models: forced `tool_choice` (`any`, `tool`) = 400 on Opus 5.5, Sonnet 5.5, Fable 5.1; thinking cannot be disabled on Opus 5.5 and Fable 5.1, Sonnet 5.5 only via `between_tools`.
- Caching converging on explicit breakpoints: Claude `cache_control` (5m/1h), OpenAI `prompt_cache_breakpoint` on GPT-5.6+ (30m/12h, `cache_write_tokens`), Bedrock `cachePoint` (5m/1h). Gemini 4 / IX: implicit only.
- Compatibility shapes: OpenAI Chat Completions = widest (Ollama, Groq, Together, OpenRouter, DeepSeek, xAI, Mistral-ish, Gemini compat, Azure). Responses = forward target (xAI, Groq beta, OpenRouter beta, Azure GA, Bedrock, Ollama `/v1/responses` stateless). Anthropic Messages = second de-facto shape (DeepSeek, Ollama, Bedrock, Vertex, Foundry). Compat layers routinely drop `tool_choice`, logprobs, reasoning round-trip, caching.

## Normalization verdicts

| Feature | Verdict | Notes |
|---|---|---|
| Roles user/assistant, tool results | clean | OpenAI developer vs system, Claude mid-conversation system, OpenAI `phase`: passthrough. Ollama tool msgs have no call id: library keeps ids, matches by name+order. |
| System prompt text | clean | As blocks with cache: no-op on OpenAI `instructions` (use developer message), Ollama. Gemini IX `system_instruction`: re-send every call. |
| Text | clean | Claude: citation-bearing text only in assistant turns (2026-10-21). |
| Image base64 | clean | |
| Image URL | loss | Ollama, Bedrock: bytes only (download or raise). Claude `url`, OpenAI URL, Gemini IX `remote` accept it. |
| Document / PDF | loss | Ollama none. Citations vendor-specific. Bedrock needs `name`. Claude 2026-10-21: user turns only. |
| Audio in | vendor-only | Gemini, Bedrock per model. |
| Tool defs (name, description, JSON Schema) | clean | `strict`, eager streaming, defer_loading, namespaces, custom/freeform: passthrough. Gemini IX takes JSON Schema in `parameters`. |
| Tool choice | loss | `auto`/`none` universal. `required`/named: 400 on newest Claude, absent on Ollama, some Bedrock models. Gemini adds `validated`. Capability-gated intent + warning. |
| Parallel tool calls | clean read, loss toggle | Toggle only Claude (`disable_parallel_tool_use`) and OpenAI (`parallel_tool_calls`). Gemini IX and Ollama: no toggle. |
| Tool result text | clean | |
| Tool result multimodal | loss | Images ok on Claude/OpenAI/Gemini IX/Bedrock; Ollama text only. `is_error`: Claude, Gemini IX `function_result.is_error`, Bedrock `status`; OpenAI and Ollama fold into text. |
| Structured output (JSON Schema) | clean intent | Schema subsets differ per vendor. Claude `output_config.format`, OpenAI `text.format`, Gemini IX `response_format`, Ollama `format`, Bedrock `outputConfig.textFormat`. |
| Reasoning effort | loss | One enum none..max maps to Claude `output_config.effort` (low..max), OpenAI `reasoning.effort` (none..max), Gemini IX `thinking_level` (minimal..high), Ollama `think` levels (model-defined, from `/api/show`). Bedrock passthrough. Off fails on several Claude models and Gemini Pro. |
| Reasoning output | loss | Summary vs raw vs omitted vs encrypted. Normalize: thinking text (may be empty) + opaque state. |
| Reasoning round-trip | must carry opaque | Never translate across vendors or models. Gemini IX: replay whole `steps`. |
| Streaming | clean core | See below. Hosted-tool progress, annotations: raw only. |
| Stop reason | loss | Keep raw. See below. |
| Usage | loss | Claude `input_tokens` excludes cache; OpenAI includes; Gemini `total_thought_tokens` separate from output. Define input = total incl. cache; compute. |
| Cache tokens | read clean, write loss | Gemini, Ollama no write count; OpenAI only new models. |
| Cache breakpoint control | loss | "breakpoint after this block + ttl" maps to Claude, Bedrock, OpenAI GPT-5.6+. No-op Gemini 4/IX, Ollama. TTL sets differ (5m/1h vs 30m/12h). |
| Hosted server tools | passthrough | Claude web_search/web_fetch/code_execution/computer/MCP connector; OpenAI web/file/code/computer/image/mcp/shell/apply_patch; Gemini IX google_search/google_maps/code_execution/url_context/file_search/computer_use/mcp_server. Names, results, billing all differ. |
| Server-side state | vendor-only | OpenAI `previous_response_id`/Conversations/background; Gemini IX `previous_interaction_id`/background (`store=true` default). Keep library stateless; opt-in extension later. |
| Compaction | loss | Claude readable signed block (threshold or on-demand); OpenAI encrypted item + `/responses/compact`. Others none. |
| Embeddings | clean | Task type (Gemini), truncate (Ollama), dimensions: passthrough. Claude none. |
| Model listing | loss | Context window: Claude, Gemini, Ollama only. OpenAI `/v1/models` has no window. Capabilities machine-readable: Claude, Ollama, partly Gemini. |
| Batch | vendor-only | All differ; Ollama none; Gemini IX batch unverified. |
| Token counting | loss | Claude, OpenAI, Gemini (generateContent shape), Bedrock (unverified); Ollama none. |
| Cancellation | clean | Close connection (Gemini reports 499 `cancelled`). Server-side cancel only for background jobs. |

## Streaming event models

- Claude SSE: `message_start`, per block `content_block_start/delta/stop` (deltas: text, input_json, thinking, signature, citations, compaction), `message_delta` (stop reason, cumulative usage), `message_stop`, `ping`, `error`. `index` = position in final content.
- OpenAI Responses SSE: `sequence_number` on all. `response.created/in_progress/completed/incomplete/failed`; `output_item.added/done`; `content_part.added/done`; `output_text.delta`; `function_call_arguments.delta`; `reasoning_summary_text.delta`, `reasoning_text.delta`; hosted tool progress `response.<tool>_call.*`; `error`. Encrypted reasoning only complete in `output_item.done`. Usage only on terminal. Block id = (output_index, content_index): flatten.
- Gemini IX SSE: `interaction.created`, `step.start {index, step}`, `step.delta {delta: text | thought_summary | thought_signature | arguments_delta}`, `step.stop`, `interaction.status_update`, `interaction.completed {status, usage}`, `error`, `done`. Block boundaries native; function calls stream as `arguments_delta`.
- Ollama NDJSON: `{message:{content?, thinking?, tool_calls?}, done:false}`; tool calls whole; final `done:true, done_reason` + metrics; mid-stream error = `{"error":...}` line with HTTP 200.
- Bedrock: AWS binary event stream for Converse; Messages and Responses on `bedrock-runtime` reuse the vendor SSE shapes (unverified in detail).

Minimal common vocabulary, synthesizable from all five:

| Event | Fields | Synthesis |
|---|---|---|
| `block_start` | index, kind, id?, name? | Claude/Gemini IX/Bedrock native; OpenAI from item/part added; Ollama on field change or whole call |
| `text_delta` | index, text | all |
| `thinking_delta` | index, text | summary or raw, may be empty |
| `tool_call_start` | index, id, name | Ollama: id generated when missing |
| `tool_call_delta` | index, partial_json | Ollama: one delta with full JSON |
| `block_stop` | index, final block incl. opaque state | OpenAI: attach `output_item.done`; Claude: signature gathered before stop; Gemini: `thought_signature` delta before stop |
| `message_stop` | response (stop, usage, content) | wait for usage before emitting |
| error | typed exception, same as non-stream | all have in-stream form |
| `raw` (opt-in) | vendor event | hosted-tool progress, annotations, keepalive |

Do not normalize: mid-stream usage, citation deltas, hosted-tool lifecycle, OpenAI index pairs.

## Stop reasons

| Normalized | Claude | OpenAI Responses | Gemini IX | Ollama | Bedrock Converse |
|---|---|---|---|---|---|
| end_turn | end_turn | completed, no tool items | completed, no function_call steps | stop | end_turn |
| tool_use | tool_use | completed + function_call items | requires_action / function_call steps | stop + tool_calls | tool_use |
| max_tokens | max_tokens | incomplete/max_output_tokens | incomplete (reason unverified) | length | max_tokens |
| stop_sequence | stop_sequence | n/a | = completed | = stop | stop_sequence |
| context_window_exceeded | model_context_window_exceeded | error | error (undocumented) | truncates silently | model_context_window_exceeded |
| refusal | refusal + stop_details | refusal part | - | - | - |
| content_filtered | - | incomplete/content_filter, failed codes | error codes safety, recitation, prohibited_content, spii, blocklist, content_blocked | - | guardrail_intervened, content_filtered |
| continue_required | pause_turn | steered, background | in_progress (background) | - | - |
| compaction | compaction | compaction item | - | - | - |
| malformed_output | - | - | error codes malformed_function_call, malformed_tool_call, unexpected_tool_call, too_many_tool_calls | - | malformed_model_output, malformed_tool_use |
| other | unknown | cancelled, max_messages | cancelled, failed | load/unload | - |

Rules: derive `tool_use` from presence of tool-call blocks, never trust vendor field. Always keep raw. On-demand compaction = own result type, not a stop reason. Gemini blocked or malformed output is an error on the wire; the codec turns it into a Response with `content_filtered` or `malformed_output` and the code in `stop.raw`, so consumers have one code path across vendors ([[Interface spec]] section 8).

## Errors

| | Claude | OpenAI | Gemini IX | Ollama | Bedrock |
|---|---|---|---|---|---|
| Body | `{type:"error", error:{type,message}, request_id}` | `{error:{message,type,param,code}}` | `{error:{code: snake_case, message}}` | `{error:"string"}` | exception name + message |
| 401/403 | authentication_error / permission_error | bad key / region | authentication / permission_denied | 401 cloud | AccessDeniedException |
| 402 | billing_error | - | payment_required | - | - |
| 413 | request_too_large (32 MB; Anthropic's error page also lists too many input tokens) | - | - | - | - |
| 429 | rate_limit_error + retry-after; spend-cap 429 has no retry-after, never clears | type rate_limit_error, code slow_down + Retry-After; billing codes credit_balance_exhausted, organization_spend_limit_exceeded, project_spend_limit_exceeded, organization_usage_limit_exceeded are 429 but not retryable | rate_limit_exceeded, too_many_requests (retry), quota_exceeded (wait for reset); no Retry-After documented | cloud rate limit | ThrottlingException, ModelNotReadyException |
| Overload | 529 overloaded_error (also mid-stream) | 503 service_unavailable_error / server_is_overloaded + Retry-After | 503 service_unavailable | 502 cloud | 503, 424 ModelErrorException |
| Overflow | 400 "prompt is too long", or 413 | code context_length_exceeded | undocumented | silent truncation | ValidationException text |
| Mid-stream | `error` event | `error` / response.failed | `error` event | NDJSON error line | exception union |

Rules: classify 429 by code into retryable vs quota, not by status. 529/503/424 retryable. Honor retry-after where present. Overflow needs per-vendor matcher; Ollama needs explicit check. 402 and quota codes never retry.

## Unverified

Gemini IX overflow and `incomplete` reasons; Gemini RetryInfo; OpenAI x-ratelimit headers; Ollama done_reason values and overflow behavior; Bedrock CountTokens, URL images, Messages/Responses stream shapes on `bedrock-runtime`; Together/Mistral/OpenRouter/Groq details from snippets.
