# ducktape-provider

One normalized chat/tool-use interface over Claude, OpenAI, and local Ollama. Stdlib only, no dependencies.

```python
from ducktape_provider import Provider

provider = Provider()
response = provider.chat("claude", "claude-opus-5", [
    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
])
```

`Provider.stream_chat(...)` gives the same call an incremental `StreamEvent` iterator instead of one buffered `Response`.

Vendor availability comes from environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and a reachable `OLLAMA_HOST` (default `http://127.0.0.1:11434`).
