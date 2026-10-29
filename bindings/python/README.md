# ducktape-provider

One normalized interface over model vendors: Anthropic, OpenAI, Google and Ollama behind one `Provider`.

```python
from ducktape_provider import Provider, user

p = Provider("anthropic", default_model="claude-opus-5-5")
print(p.chat([user("hello")]).text)
```

Documentation: https://github.com/adrian729/ducktape-provider
