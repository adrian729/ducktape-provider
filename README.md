# ducktape-provider

One normalized chat/tool-use interface over Claude, OpenAI, and local Ollama. Stdlib only, no dependencies.

## Install

Not on PyPI yet — install straight from GitHub:

```
pip install git+https://github.com/adrian729/ducktape-provider
```

or with `uv`:

```
uv add git+https://github.com/adrian729/ducktape-provider
```

## Usage

```python
from ducktape_provider import Provider

provider = Provider()
response = provider.chat(
    "claude",
    "claude-opus-5",
    [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ],
)
```

`provider` is one of `"claude"`, `"openai"`, or `"ollama-local"` — same `Message`/`Response` shape regardless of which one you call.

`Provider.stream_chat(...)` gives the same call an incremental `StreamEvent` iterator instead of one buffered `Response`.

`Provider.async_chat(...)`/`Provider.async_stream_chat(...)` are async equivalents backed by `asyncio.to_thread`, not a native async HTTP client.

`Provider().providers()` reports which vendors are configured and reachable right now; `Provider().models()` lists the model ids each one can actually serve.

## Configuration

Vendor availability and model listing come from environment variables — no config file:

| Provider | Env var(s) |
|---|---|
| `claude` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `ollama-local` | `OLLAMA_HOST` (default `http://127.0.0.1:11434`) |

`config` is merged straight into the vendor's request body, so any field that vendor's API accepts works — `temperature`, `top_p`, Claude's `thinking`, OpenAI's `text.format`/`tool_choice`, Ollama's `options`/`format`, whatever's in that vendor's own docs.

```python
provider.chat("claude", "claude-opus-5", messages, config={"temperature": 0.2})
```

`Provider(timeout=...)` sets a default across all providers; `config={"providers": {"claude": {...}}}` overrides it for one provider only on a single call.

## Third-party adapters

`Provider(autodiscover=True)` picks up adapters registered by other installed packages under the `ducktape_provider.adapters` entry-point group, in addition to the 3 built-in ones — off by default, so nothing changes unless you opt in.

```toml
[project.entry-points."ducktape_provider.adapters"]
myvendor = "my_package.adapter:MyVendorAdapter"
```

The entry point must point to an `Adapter` subclass, not an instance — `Provider` instantiates it with no constructor args. A discovered name never overrides an explicit `adapters=` entry or a built-in default (`claude`, `openai`, `ollama-local`); it only fills in names not already present.

## License

[MIT](LICENSE)
