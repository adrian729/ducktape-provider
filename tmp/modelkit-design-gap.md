# ducktape-provider vs. the original "modelkit" design (v0)

Compared the current implementation against
`sanchez.albanell.adrian/review/project-modelkit.md`, the earlier portfolio
design doc for the same idea (there called "modelkit"). Report only — no code
changes. Where a research-doc gap from `api-coverage-gaps.md` overlaps, it's
named but not re-argued.

## Scope: one verb shipped, one verb never started

modelkit's core definition was **two** verbs: `chat(...)` and `embed(texts,
provider=...)`. ducktape-provider only ever built chat — there is no `embed`
anywhere in the package, no embedding types, no vector/usage shape for it.
Everything downstream in the design doc that depends on embed (open question
4: local embedding model default / dimension handling; "mixing dimensions in
one index is a hard error") is therefore moot, not resolved — it was never
reached rather than decided against.

## The "real product" — conformance suite / divergences matrix — doesn't exist

modelkit's design doc calls this out explicitly: *"the same scenarios run
against every backend, and its output (a divergences matrix) is the honest
README table and the interview artifact."* ducktape-provider's tests are
per-adapter unit/HTTP-mock tests (`test_claude_adapter.py`,
`test_openai_adapter.py`, `test_ollama_adapter.py`) — each exercises its own
adapter in isolation with hand-picked fixtures. There is no shared scenario
suite run identically across all three backends, and no generated table of
where they diverge. This was named as the centerpiece of the portfolio
story; it's the single biggest gap between the plan and what got built.

## Result shape is thinner than planned

modelkit's `chat` result was meant to normalize to: content, **parsed
structured output**, tool calls, usage (tokens in/out, **cost estimate**,
**latency**), finish reason. The actual `Response` TypedDict (`types.py`) has:
`content`, `stop_reason`, `raw_stop_reason`, `usage` (`input_tokens`/
`output_tokens` only), `raw`. Missing relative to the plan:
- **No parsed structured-output field.** No unified "give me back a schema-
  validated object" mechanism (OpenAI's `text.format`, Claude's forced-tool
  trick, Ollama's `format` are each reachable only via raw `config`
  passthrough into that one vendor's body, with no shared shape or
  validation).
- **No cost estimate.** No pricing table anywhere in the package (the
  architecture sketch specifically calls for one: `core` = "types, protocols,
  registry, **pricing data file**"). Ties directly into open question 2
  ("token counting: trust provider-reported usage only, or bundle a local
  estimator") — that question was answered by default (trust provider usage
  only) but the cost-estimate half of the original ambition was dropped along
  with it, not consciously descoped.
- **No latency measurement.** Nothing times a call or exposes duration on the
  result.

## No normalized error taxonomy

modelkit planned "a normalized taxonomy (rate-limited, timeout,
context-overflow, refused, schema-invalid)" with retry/backoff configured
once, off by default. Every adapter today collapses every failure — 429, 500,
timeout, whatever — into a single generic `RuntimeError` with a
provider-prefixed string message (`f"claude chat failed: {e.code} {body}"`).
There's no programmatic way to distinguish "you got rate-limited, back off"
from "your API key is wrong" from "the model refused." (This overlaps with
`api-coverage-gaps.md`'s "no retry/backoff anywhere" finding, but the missing
taxonomy is the deeper piece — retry logic can't be built on top of an
undifferentiated `RuntimeError` in the first place.)

## No config file / no budgets

modelkit: "Keys from env; `modelkit.toml` holds defaults and per-project
budgets." ducktape-provider does the env-var half (matches — see README's
Configuration table) but has no project-level config file at all; `Config` is
a purely in-process `TypedDict` the caller builds and passes per-call. There's
also nothing resembling a budget: no cost data (see above) means there's
nothing to enforce a budget against even if the config-file mechanism
existed.

## Registry: simpler than planned, not necessarily worse

modelkit wanted "entry-points style" provider registration — plugin discovery
via Python package metadata, so a third party could `pip install` a new
adapter and have it register itself. `Provider.__init__(adapters: dict[str,
Adapter] | None = None)` is a plain constructor-injected dict instead — a
caller wires in custom adapters by passing the dict, no discovery mechanism.
This is a real simplification vs. the plan, but it's arguably the right call
for a "tiny stdlib-only library" (entry-points registration pulls in
`importlib.metadata` plumbing for a benefit — third-party adapter packages —
that doesn't exist yet). Flagging it as a deliberate-looking scope reduction,
not a defect.

## Model identity: two strings instead of one

modelkit: `"provider:model"` as a single string, no enums. Actual API is
`Provider.chat(provider, model, ...)` — two separate positional strings. Pure
shape difference, no capability lost.

## `chat`/`stream_chat` as two methods, not one verb with a flag

modelkit's one-verb sketch was `chat(request, provider=..., stream=...)`
yielding events *and* a final result from the same call. The project
considered exactly this during this session and deliberately kept them
separate (`Provider.chat` returns a buffered `Response`; `Provider.stream_chat`
returns an `Iterator[StreamEvent]`) specifically because a `stream: bool` in a
runtime dict can't be reflected in a static return type. Noting it here only
because it's a named divergence from the original design doc, not because
it's wrong — the reasoning for the current shape is already on record from
earlier in this session.

## Sync-only (open question 3 answered by default, not decided)

modelkit's open question 3 asked "sync vs async-first," noting both original
consumer apps were FastAPI. ducktape-provider is sync-only end to end
(`urllib.request`, blocking `with urlopen(...)`) — there is no async story at
all, not even as a documented non-goal. Given the "tiny stdlib-only library"
framing this is a defensible choice, but it's a live open question from the
plan that was resolved by omission rather than by a stated decision.

## What did get resolved cleanly

- **Open question 1 (tools at v0.1 vs v0.2)**: resolved in the more ambitious
  direction — full tool-definition support shipped on all three backends from
  the start, not deferred.
- **Open question 5 (name collision)**: resolved — renamed from "modelkit" to
  "ducktape-provider" before any publishing happened, exactly as the doc
  recommended.
- **"What it deliberately is not"** (routing, caching, load balancing, proxy,
  auth, eval framework, conversation/memory abstractions): the implementation
  stays out of all of these. Full alignment with the plan here.
- **Streaming event protocol**: modelkit wanted one shared shape (token delta,
  tool-call delta, usage, done) that adapters translate into. The actual
  `StreamEvent` union (`text_delta`, `thinking_delta`, `tool_use_start`,
  `tool_use_delta`, `block_stop`, `message_stop`) is a faithful, slightly
  richer realization of that same idea — usage arrives folded into the final
  `message_stop`'s `response` rather than as its own event type, a reasonable
  simplification.

## Portfolio-narrative gaps

Two things the design doc treats as the actual point of publishing this
library are currently absent from the finished package:
- The README never states the "~200 lines instead of borrowing LiteLLM's
  opacity" tradeoff out loud — the doc calls that sentence "the senior
  signal," and it's simply not there (checked `README.md` directly).
- No divergences matrix exists to be "talking material," per the conformance-
  suite gap above.
- Line count has also grown well past the "~200 lines" figure the pitch was
  built around (now several hundred lines across `types.py` + `adapter.py` +
  `provider.py` + three adapter modules, plus test files) — not a defect, but
  worth knowing before reusing that specific pitch language.
