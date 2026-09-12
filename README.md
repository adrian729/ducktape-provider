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
response = provider.chat("claude", "claude-opus-5", [
    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
])
```

`provider` is one of `"claude"`, `"openai"`, or `"ollama-local"` — same `Message`/`Response` shape regardless of which one you call.

`Provider.stream_chat(...)` gives the same call an incremental `StreamEvent` iterator instead of one buffered `Response`.

`Provider().providers()` reports which vendors are configured and reachable right now; `Provider().models()` lists the model ids each one can actually serve.

## Configuration

Vendor availability and model listing come from environment variables — no config file:

| Provider | Env var(s) |
|---|---|
| `claude` | `ANTHROPIC_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `ollama-local` | `OLLAMA_HOST` (default `http://127.0.0.1:11434`) |

## License

[MIT](LICENSE)
