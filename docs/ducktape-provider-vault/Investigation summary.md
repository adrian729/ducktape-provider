# Investigation summary

Date: 2026-10-28. Status: research only, no code. Companion notes: [[Provider API matrix]], [[Vendor APIs]], [[Prior art]], [[Agents]], [[Delivery across languages]], [[Build and packaging]], [[Auth model]], [[Lessons from first attempt]], [[Design draft]].

## Goal (from [[Ducktape Provider]])

Normalized provider for models and agents. Several languages, one implementation. Small surface, easy integration, good defaults, fast. Deep modules.

## Headline findings

- Seven independent designs (Vercel V4, MEAI, Bedrock Converse, pydantic-ai, rig, LangChain v1, llm 0.32) converged on the same core: role + ordered typed blocks, block-lifecycle streaming with ids, namespaced opaque vendor state on each block. Claude's block model generalized. See [[Prior art]].
- OpenAI Chat Completions shape as the base loses data (reasoning state, cache breakpoints, strict tools). Evidence: OpenAI pushes Responses, Anthropic's compat page lists the drops, every Chat-shaped normalizer grew bolt-on fields. Conclusion ours. Not the core, and not a codec either: Chat Completions is out entirely.
- 2026 vendor reality: history is append-only and carries mandatory opaque state (Claude thinking signatures, OpenAI encrypted reasoning, Gemini thought signatures). Dropping it is a 400 or quality loss. See [[Vendor APIs]].
- Even `temperature` is no longer portable (Claude API 2026-10-21 rejects it; newest Claude models reject forced `tool_choice`). Normalized option set must be tiny; everything else is namespaced passthrough with warnings, never silent drops.
- Agents are not models. Agents own history and run their own tools; they need sessions, permissions, cancel. Separate interface, shared block and event types. ACP (Agent Client Protocol, SDK 1.0, ~48 agents) is the portable backend, with gaps (no tool-arg deltas, single-turn prompt, editor-client assumptions). See [[Agents]].
- Multi-language: the real decision is who owns the socket. Sans-I/O core (pure functions: build request, decode response, feed stream bytes) + host-native HTTP per language avoids async bridging, TLS packaging, cancellation and mocking pain. Same packaging shape as ducktape-laya. See [[Delivery across languages]].
- Language choice buys parity, not speed. Model latency dominates; Python only hurts at thousands of concurrent streams.
- The discarded first attempt is not a source for this design. Its one lesson worth keeping: owning the HTTP transport (secret hygiene, sync-to-async bridging) cost more than the normalization itself. See [[Lessons from first attempt]] for the mistakes not to repeat.

## Direction (decided 2026-10-28)

- Rust sans-I/O core, JSON-schema'd normalized types, language-neutral conformance fixtures as the contract.
- All laya languages in v1. Thin host glue per language, batteries-included: Python (PyO3, httpx), one plain `core.wasm` for JS, Go (wazero) and Java (Chicory), native C ABI for C/C++ and .NET, Rust crate. Detail in [[Build and packaging]].
- Codecs per wire protocol, not per vendor, current surfaces only: `anthropic.messages`, `openai.responses`, `google.interactions`, `ollama.native`. No `generateContent`, no Converse, no Ollama `/v1/*` (vendors steer away from them). No Chat Completions anywhere (decided). Provider = endpoint (wire, URL, auth, dialect) + model-family profile. Per-feature comparison in [[Provider API matrix]].
- Surface v1: `chat`, `stream`, `embed`, `models`, `model_info`, `capabilities`, plus `login`/`logout`/`auth_status` ([[Auth model]]). Agents v2 behind the same block/event types.

## Decisions

All 20 settled; log and remaining items in [[Design draft]] § Decision log. Surface frozen in [[Interface spec]].

## Raw research

Subagent downloads (vendor docs, Vercel V4 spec, OpenCode OpenAPI, Claude SDK sources) are in the session scratchpad, not in the repo: `/tmp/claude-1000/-home-ranx729-projects-ducktape-provider/d6d5cee2-d601-4fe8-8b32-8694c30187a4/scratchpad/`. Temporary.
