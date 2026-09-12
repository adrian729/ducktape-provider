# ducktape-provider vs. vendor APIs — coverage check

Compared the three adapters (`src/ducktape_provider/adapters/*.py`) against the
vendor research in `little-agent/research/*.md` (Claude Messages API, OpenAI
Chat Completions/Responses, Ollama). Report only — no code changes.

## Real gaps (not reachable even via the existing `config` passthrough)

**No way to add custom HTTP headers, on any adapter.** All three
`_build_request` methods hardcode their `headers={...}` dict; `config` only
merges into the JSON _body_ (`payload.update(config or {})`), never into
headers. Concretely this blocks:

- **Ollama Cloud** (`ollama-api.md`): needs
  `Authorization: Bearer $OLLAMA_API_KEY` when pointed at `ollama.com` instead
  of localhost. `OllamaLocalAdapter` reads `OLLAMA_HOST` for the URL but has no
  auth-header path at all — so despite the research's own conclusion that Cloud
  "can be treated as Ollama with a different base_url + API key," the adapter
  currently only does the base_url half.
- Anthropic's optional `anthropic-workspace-id` header, OpenAI's optional
  `OpenAI-Organization`/`OpenAI-Project` headers — same story, no hook to set
  them.

**Image blocks are base64-only.** `ImageBlock` (`types.py`) has no way to
express a remote-URL image. Both Claude (`source:{type:"url",...}`) and OpenAI
(`image_url` accepts a plain `https://...` URL, not just a data-URL) support
this per the research docs; the adapter's type/serialize code only ever handles
`data`+`media_type`.

**No `document` content-block type** (Claude PDFs) and no `redacted_thinking`
block — both real Claude content-block kinds the research notes exist, neither
modeled in `Block`.

**Claude `pause_turn` stop reason isn't mapped** — `ClaudeAdapter._deserialize`
handles `end_turn`/`tool_use`/`max_tokens`/`stop_sequence`/`refusal` explicitly
and buckets everything else (including `pause_turn`, used for long-running
server-tool turns) into `"other"`, losing the distinction.

**Claude prompt-caching usage fields are dropped.** `Usage` only has
`input_tokens`/`output_tokens`; Claude's response also carries
`cache_creation_input_tokens`/`cache_read_input_tokens`, silently discarded in
`_deserialize`. (The `raw` field preserves them for anyone who reaches in, but
the normalized `usage` doesn't.)

**No retry/backoff anywhere.** Every adapter's `chat`/`stream_chat` converts any
`HTTPError` straight into a `RuntimeError`. The Ollama Cloud research explicitly
flags 429/queuing behavior under concurrency limits as something an adapter
"should handle" — currently a transient 429 is indistinguishable from a hard
failure.

## Deliberate-looking gaps that are still worth naming

**OpenAI adapter targets the Responses API, not Chat Completions** — this is the
opposite of what `research/README.md` and `openai-api.md` actually recommend
("keep Chat Completions as the default OpenAI backend... special- case Responses
only when a caller needs a hosted tool or reasoning-quality parity"). Not
necessarily wrong (Responses is OpenAI's forward path and the research
acknowledges that), but it's a real divergence from the project's own documented
decision, undocumented anywhere in the adapter itself. -> This was already
discussed, that is the old API and it makes no sense that we implement targeting
what is or will be deprecated when we know it already.

**No hosted/server-side tools on either Claude or OpenAI** — `web_search`,
`code_interpreter`, `file_search`, `computer_use`, MCP-as-a-tool (OpenAI), or
Claude's equivalents. Given the current design, even passing these via `config`
wouldn't compose cleanly: `_build_request` does
`if tools: payload["tools"] = ...` from caller-supplied tool defs, then
`payload.update(config or {})` — so a `config={"tools": [...]}` passthrough
would _replace_ the user's function-tool list rather than merge a hosted tool
alongside it.

**No model-capability probing.** Ollama's `/api/show` → `capabilities`
(tools/vision support is model-dependent, per research) is never consulted — the
adapter will happily send `tool_calls`/`images` to a model that doesn't support
them and let it fail downstream. OpenAI has the analogous problem solved cheaply
(regex over `/v1/models`), Claude doesn't need it (uniform tool/vision support).

## Things that look missing but actually aren't

- Per-request tunables (`temperature`, `top_p`, `thinking`, `reasoning`,
  Ollama's `options`/`format`/`think`, OpenAI's
  `text.format`/`tool_choice`/`parallel_tool_calls`) — all reachable today via
  `config` passthrough into the JSON body, since every adapter does
  `payload.update(config or {})` before sending. Not wired up as first-class
  fields, but not blocked either.
- Ollama's missing `tool_call_id` correlation — research flagged this as
  something to fix; the adapter already does it (`_deserialize` synthesizes
  `call_{i}` ids, matching the research's own suggested workaround).
- Ollama Cloud's `keep_alive` no-op concern — moot since `keep_alive` is just a
  body field, already overridable via `config`.

## Bottom line

The single highest-leverage fix, if you want one: plumb an optional
headers-merge point into each `_build_request` (mirrors the existing
`config`-into-body pattern) — that alone unblocks Ollama Cloud auth and the two
vendors' optional org/workspace headers without any other design change.
Everything else above is either a deliberate scope boundary (no hosted tools, no
document blocks, chat-only) or a small, isolated addition (URL-based images,
`pause_turn` mapping, cache-token usage fields) — none of it forces a structural
change to the adapter shape you already have. -> dont take this idea as the
ground truth, take it into account, but if there is a better solution we should
go for that, not just for the lazy/easy one.
