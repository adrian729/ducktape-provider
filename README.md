<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/ducktape-lockup-dark.svg">
    <img src="assets/ducktape-lockup-light.svg" alt="ducktape provider" width="440">
  </picture>
</p>

One normalized interface over model vendors, one implementation, several languages.

Status: in development, not released. The interface is frozen in [`docs/ducktape-provider-vault/Interface spec.md`](docs/ducktape-provider-vault/Interface%20spec.md); the JSON contract is in [`schema/`](schema).

## Layout

| Path | |
|---|---|
| `schema/provider.schema.json` | Normalized types: messages, blocks, tools, options, responses, events, errors, config |
| `schema/effects.schema.json` | Core-to-host protocol: `Start`, `Resume`, `Step` |
| `crates/core` | Sans-I/O core: types, codecs, effects state machine |
| `crates/abi` | `extern "C"` surface, built natively and as `core.wasm` |
| `crates/conformance` | Fixture runner |
| `crates/py` | PyO3 module behind the Python package |
| `bindings/python` | Python package: `Provider`, `AsyncProvider`, messages, errors |
| `fixtures/` | Samples every binding must pass |

## Python

Not on PyPI yet. Build it in place with [maturin](https://www.maturin.rs):

```
cd bindings/python && uv venv && uv pip install httpx pytest && uvx maturin develop --uv
```

```python
from ducktape_provider import Provider, user, assistant, tool, tool_result, ToolDef

p = Provider("anthropic", default_model="claude-opus-5-5")        # key from ANTHROPIC_API_KEY
r = p.chat([user("hello")])
print(r.text, r.usage, r.stop.reason)

for ev in p.stream([user("tell me a story")]):
    if ev.type == "text_delta":
        print(ev.text, end="")

add = ToolDef("add", "Adds two numbers", {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}})
history = [user("what is 2+2?")]
r = p.chat(history, tools=[add])
if r.stop.reason == "tool_call":
    call = r.tool_calls[0]
    history += [assistant(r), tool(tool_result(call.id, "4"))]
    r = p.chat(history, tools=[add])
```

| Need | Write |
|---|---|
| Another host, same wire | `Provider("openai", base_url="https://api.groq.com/openai", auth=env("GROQ_API_KEY"))` |
| A key in code | `Provider("anthropic", auth="sk-ant-...")` |
| A token fetched at request time | `Provider("openai", auth=resolver(get_token))` |
| Local Ollama | `Provider("ollama", default_model="llama3")` |
| Options | `p.chat(msgs, options={"max_output_tokens": 500, "reasoning": "high", "tool_choice": {"name": "add"}})` |
| Vendor fields | `p.chat(msgs, vendor={"anthropic": {"temperature": 0.2}})` |
| Images | `user("what is this?", image(png_bytes))` or `image(url="https://...")` |
| Embeddings | `Provider("openai", default_embedding_model="text-embedding-3-small").embed(["a", "b"]).embeddings` |
| Discovery | `p.models()`, `p.model_info("gpt-5")`, `p.capabilities("gpt-5")` |
| asyncio | `AsyncProvider` has the same methods as coroutines; `stream()` is an async iterator |
| Your own `httpx` client | `Provider("openai", client=httpx.Client(proxy=...))` |

Every error is a `ProviderError` with `kind`, `status`, `code`, `retryable`, `retry_after`, `attempts`, `body`, `request_id` and `origin`. Subclasses by kind: `AuthError`, `RateLimitError`, `QuotaError`, `ServerError`, `NetworkError`, `TimeoutError`, `CancelledError`, `NotFoundError`, `ContextOverflowError`, `RequestTooLargeError`, `BadRequestError`, `MalformedResponseError`, `UnsupportedError`. Argument mistakes raise `ValueError` or `TypeError` before any request.

Types are dataclasses in [`bindings/python/src/ducktape_provider/_types.py`](bindings/python/src/ducktape_provider/_types.py); the JSON contract behind them is [`schema/provider.schema.json`](schema/provider.schema.json).

## Development

```
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --profile wasm --target wasm32-unknown-unknown -p ducktape-abi
pip install jsonschema referencing && python scripts/check_fixtures.py .
cd bindings/python && uvx maturin develop --uv && .venv/bin/python -m pytest
```

## License

[MIT](LICENSE)
