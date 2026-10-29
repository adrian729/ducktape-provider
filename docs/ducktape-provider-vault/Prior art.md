# Prior art

Multi-provider libraries and gateways, as of 2026-10-28. Verified in source or docs unless marked (inferred).

## Projects, one line each

- Vercel AI SDK 7, provider spec `LanguageModelV4`: `doGenerate`/`doStream`; prompt = roles + typed parts (text, file, reasoning, tool-call, tool-result, custom); `providerOptions` on every object; result = ordered content, `finishReason {unified, raw}`, usage with nullable breakdown + raw, typed `warnings[]`; stream = start/delta/end per block id + finish + raw. Most mature. Churn criticized (v4 to v7 in two years).
- LiteLLM: OpenAI Chat shape + bolt-ons (`reasoning_content`, `thinking_blocks`, `provider_specific_fields`), `drop_params`. 204 MB install (core 138 MB), 1.7k open issues, network fetch at import, proxy code in SDK paths, backdoored releases 2026-03 (~47k downloads).
- aisuite: OpenAI shape, `"provider:model"`, agent loop inside the chat call (`max_turns`). Weak maintenance.
- any-llm (Mozilla.ai): wraps vendor SDKs; three input surfaces (chat, responses, messages) because one was not enough; ~15 error classes; Go port exists.
- simonw llm 0.32+: refactored from strings to messages + typed Parts; `StreamEvent {text, reasoning, tool_call_name, tool_call_args, tool_result}` grouped by family and `tool_call_id`; opaque tokens in `provider_metadata` per provider.
- pydantic-ai: Provider (auth/endpoint) / Model (wire) / Profile (model-family quirks) split; `PartStart/Delta/End` events; messages module ~6.8k lines, mixes runtime concerns in.
- LangChain v1: retrofitted typed `content_blocks` after loose `str | list[dict]` content.
- OpenAI Agents SDK: internal lingua franca = Responses items; other vendors squeezed through it.
- rig (Rust): typed content enums + `Native` vendor item kept next to canonical block + `Origin {api, provider, model}` + fingerprint; replays native only on exact match, else rebuilds. Solves provider switch mid-conversation.
- genai (Rust): adapter inferred from model-name prefix, effort suffix in model name. Both ambiguous.
- langchaingo: `[]byte` streaming callback, untyped metadata maps. Avoid.
- Bifrost (Go gateway): OpenAI + bolt-ons, separate `Provider`/`Model` fields, `RawRequestBody`. Perf claims inconsistent (11 µs vs 200 µs).
- OpenRouter: OpenAI Chat, unknown params ignored silently, `native_finish_reason`, `reasoning_details[]` typed with per-vendor `format` tag, must be echoed in order.
- Kong AI proxy: OpenAI normalization by default, added `llm_format` native passthrough (admission of loss).
- Bedrock Converse: vendor-made normalized API. Good: typed block unions, start/delta/stop, rich `stopReason`, `cachePoint` block, `reasoningContent` with signature, `additionalModelRequestFields` + `additionalModelResponseFieldPaths`, `metrics.latencyMs`. Weak: reasoning config passthrough, per-model support matrix, no URL images, no hosted tools.
- Open Responses (LF spec): OpenAI Responses-derived wire standard; `slug:` prefixed extensions; backed by OpenRouter, Vercel, Portkey; Anthropic and Google reportedly absent. Target codec, not core.
- Microsoft.Extensions.AI `IChatClient`: three methods (`GetResponseAsync`, `GetStreamingResponseAsync`, `GetService`); typed `AIContent` list incl. approvals and hosted tools; `RawRepresentationFactory` passthrough; middleware for retries/tool loop/telemetry. Cleanest minimal interface.
- LangChain4j: typed stream handler + `onUnmappedRawEvent`. Spring AI: `Flux<ChatResponse>`, too coarse.
- Stainless SDKs (OpenAI, Anthropic): user baseline. 2 retries default with backoff on 408/409/429/5xx + retry-after, streams not retried, `extra_headers/body`, `with_raw_response`. Stainless acquired by Anthropic 2026-05, hosted generator winding down.
- MCP tools: `{name, description, inputSchema, outputSchema?}`; results = content list + `isError`. Vercel and others adopted `inputSchema`.

## Convergence

Independent designs that landed on the same core: Vercel V4, MEAI, Bedrock Converse, pydantic-ai, rig, LangChain v1 blocks, llm 0.32.

- Role + ordered typed parts, both directions.
- Reasoning as first-class part with opaque vendor state.
- Tool call / tool result parts with ids.
- Block-lifecycle streaming keyed by id.
- Namespaced passthrough in and out.
- No `choices[]`.

## Why not OpenAI Chat shape as the core

Evidence, not a published verdict:

- OpenAI recommends Responses; reasoning continuity (`encrypted_content`) exists only there; since GPT-5.4, Chat Completions does not support tool calling with `reasoning_effort` other than `none`.
- Anthropic's OpenAI-compat page: "not production-ready"; drops thinking, cache breakpoints, `strict`, `response_format`; hoists system messages.
- Every Chat-shaped normalizer grew side fields (LiteLLM, OpenRouter, Bifrost) or added native passthrough / extra surfaces (Kong, Portkey, any-llm).
- Block-model designs converged elsewhere (above).
- Secondary: tianpan.co 2026-07 critique (stripped cache markers, 5-8x input spend).

Decided: no Chat Completions codec at all; see [[Design draft]].

## Ideas to steal

1. Role + typed blocks; stable `id` on every streamed block.
2. Start/delta/stop per block + final finish with usage. llm's grouping rule as minimal reference.
3. Namespaced opaque metadata on every block, in and out: `{"anthropic": {"signature": ...}}`.
4. rig: keep native vendor item + origin tag; replay verbatim only on exact (wire, provider, model) match.
5. Finish reason `{unified, raw}`.
6. Usage with nullable breakdown (input total/no_cache/cache_read/cache_write, output total/text/reasoning) + raw.
7. Typed `warnings[]` (unsupported, compatibility, deprecated) instead of silent drops.
8. Typed, namespaced escape hatches: custom block `provider.kind`, raw stream events opt-in, raw request merge, response field extraction paths.
9. Function tools separate from provider (hosted) tools; `provider_executed` flag on resulting blocks.
10. MCP-aligned tool shape and result content list + error variant.
11. Computed `retryable` on the error base + `retry_after`; retry policy outside the wire codec.
12. Provider / wire / model-profile split. OpenAI-compatible endpoints become config, not code.
13. Spec version field on the adapter interface.
14. One reasoning effort enum mapped per vendor; vendor fields win via passthrough.
15. Stateless by default; optional continuation token.
16. Serialized JSON of the normalized types is the cross-language contract; publish JSON Schema.
17. Tiny core + middleware (MEAI).

## Mistakes to avoid

1. Scope creep in one package (LiteLLM).
2. Base shape + bolt-ons.
3. Silent drops of unsupported features.
4. Assuming "common params" are portable (temperature, tool_choice no longer are).
5. Provider inferred from model-name prefix; options encoded in the name.
6. Untyped or token-only streams; consumers reassembling `delta.tool_calls[index]`.
7. Loose content typing needing a later retrofit.
8. Agent loop inside the chat call.
9. Dependency extremes: wrapping every vendor SDK vs reimplementing without a spec. Middle: raw HTTP + small typed codecs + recorded fixtures.
10. User-facing breaking churn.
