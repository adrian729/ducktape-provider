# Lessons from first attempt

Scope of this note: mistakes not to repeat and consumer needs that surfaced. The discarded implementation is not a base, source or reference for the new design; nothing is extracted or ported from it.

Discarded attempt: Python, stdlib only, 5,506 source lines (3 adapters: Claude Messages, OpenAI Responses, Ollama native), ~15k test lines. One real consumer: ducktape-harness (a REPL agent, Python, sync streaming only).

## Choices that matched what the 2026 research found

- Block-based message model (Claude's shape as superset). Matches 2026 convergence.
- Stream vocabulary `text_delta`, `thinking_delta`, `tool_use_start/delta`, `block_stop`, `message_stop`. Matches the minimal common vocabulary derived from five vendors. Missing: block_start, opaque per-block state on stop, raw escape hatch.
- Error taxonomy: Auth, RateLimit(retry_after), Server(retry_after), Timeout, ContextOverflow, Malformed, Unsupported*, APIError base with status/body. Consumer used every class.
- Usage normalization: input = total incl. cache; cache_read/cache_write.
- `truncated` flag on tool_use when args JSON cannot be parsed (max_tokens cutoff). Consumer needed it.
- latency_ms, ttft_ms self-measured.
- Explicit stateless design: no routing, no budgets, no config files, no conversation storage.
- Two methods `chat` / `stream_chat` rather than a `stream` flag (return types).
- Conformance tests across backends, partially done.

## What cost the most (and why)

- Secret hygiene for stdlib urllib: `_Secret` wrapper, traceback clearing, redirect protection, header validation, transport allowlist. Hundreds of lines; still leaked keys into frame locals on every error path (both reviews, H1/M1). Cause: library owned the transport. Sans-I/O core removes this class.
- Sync-to-async bridging via threads + bounded queue (`_stream_off_loop`). Subtle, load-bearing. Cause: same.
- Plugin discovery with name-collision matrix (`_register_plugins`, ~110 lines). Low value; `autodiscover=True` is arbitrary-code execution from installed packages.
- Auto-match provider from model id with cache + eviction rules on 404/Auth. Consumer (harness) ended up pinning providers and warning that `:` in ids breaks prefix schemes. Drop or reduce to a dumb lookup.
- Config merging layers (`Provider(config)`, per-call, `providers.<name>`, headers as callables, compaction tri-state). Many validation paths. Simplify: one request object, one namespaced passthrough.
- TypedDict everywhere: passthrough blocks not statically visible; `Config` could not be closed. New design: typed core + explicit `vendor` map.

## Gaps found by the consumer (now requirements)

- Model metadata (context window, max output) for context-usage display.
- Multimodal tool results (images back to the model).
- Abort an in-flight request immediately (`cancel()`), on sync and async paths.
- Usage must be present or explicitly null, never absent.
- Truncated tool args must be distinguishable from empty args.
- Claude 413 `request_too_large` is a 32 MB request-size limit, not context overflow. The attempt mapped it to overflow; that was wrong. Needs its own error.
- Headers per provider (Ollama Cloud auth, org/workspace headers).
- URL images, documents, cache usage fields, pause_turn, hosted tools composing with user tools (passthrough `tools` replaced the list).

## Ideas from the old notes still valid

- Divergences matrix generated from the conformance suite as the honest README table.
- "The abstraction to interact with each API/model, no more, no less."
- Pricing/cost estimate dropped: no vendor reports price; caller computes from usage.
- Structured output needs its own design pass (three different vendor mechanisms).

## Numbers

| | v0.1 |
|---|---|
| src LOC | 5,506 |
| test LOC | ~15,000 |
| adapters | 3 |
| runtime deps | 0 |
| share of src that is transport/plumbing | not measured; the two largest modules were transport |
