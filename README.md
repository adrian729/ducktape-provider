# ducktape-provider

Simple AI chat and embeddings provider to normalize usage of different APIs.

Claude, OpenAI and self-hosted Ollama adapters included. Extensible by subclassing `Adapter`.

Stdlib only, no dependencies.

## Install

Requires Python 3.12+.

Not on PyPI yet, install straight from GitHub:

```
pip install git+https://github.com/adrian729/ducktape-provider
```

or with `uv`:

```
uv add git+https://github.com/adrian729/ducktape-provider
```

## Provider setup

Vendor availability and model listing come from environment variables — no config file:

| Provider       | Env var(s)                                       |
| -------------- | ------------------------------------------------ |
| `claude`       | `ANTHROPIC_API_KEY`                              |
| `openai`       | `OPENAI_API_KEY`                                 |
| `ollama-local` | `OLLAMA_HOST` (default `http://127.0.0.1:11434`) |

`ollama-local` works with any self-hosted Ollama: set `OLLAMA_HOST` to its `http(s)://` URL.

Or pass API keys in code with `api_keys`:

| Form | Example |
|---|---|
| Keys by provider name | `Provider(api_keys={"claude": "sk-ant-...", "openai": get_openai_key})` |
| One function for all | `Provider(api_keys=lambda name: vault.read(name))` |

- A value is a key or a function returning one; functions run on every request, possibly from several threads.
- A provider given a key never reads its env var.
- Also applies to Claude and OpenAI adapters passed in `adapters=`.

## Usage

All types the interface uses (`Message`, `Response`, `EmbedResponse`, `StreamEvent`, `Config`, errors, `Adapter`) are in [`types.py`](src/ducktape_provider/types.py).

```python
from ducktape_provider import Provider

provider = Provider()
response = provider.chat(
    "claude-opus-5",
    [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ],
    provider="claude",
)
```

### `Provider(...)`

| Param | Type | Default | |
|---|---|---|---|
| `adapters` | `Mapping[str, Adapter] \| None` | built-ins | Adapters by name |
| `timeout` | `float \| None` | adapter default | Seconds for every call; `None` disables |
| `autodiscover` | `bool \| Collection[str]` | `False` | Load [third-party adapters](#third-party-adapters) |
| `executor` | `Executor \| None` | asyncio default | Thread pool for async calls, except streams |
| `api_keys` | `Mapping[str, str \| Callable] \| Callable \| None` | env vars | API keys, see [Provider setup](#provider-setup) |
| `config` | `Config \| Mapping[str, Any] \| None` | none | Defaults for every call, see [Configuration](#configuration) |

### Chat methods

| Method | Description | Returns |
|---|---|---|
| `chat(model, messages, system, tools, config, provider)` | Send a chat, wait for the full reply | `Response` |
| `stream_chat(model, messages, system, tools, config, provider)` | Send a chat, get the reply as events | `Iterator[StreamEvent]` |
| `async_chat(model, messages, system, tools, config, provider)` | `await`able `chat` | `Response` |
| `async_stream_chat(model, messages, system, tools, config, provider)` | `async for` version of `stream_chat` | `AsyncGenerator[StreamEvent, None]` |

- `model`: vendor model id, e.g. `"claude-opus-5"`. If `provider` is not given, the provider is matched automatically from the model.
- `messages`: the conversation so far, as a `list[Message]`.
- `system`: optional system prompt.
- `tools`: optional `list[ToolDef]` the model may call.
- `config`: optional per-call settings and vendor request fields, see [Configuration](#configuration).
- `provider`: optional, keyword-only adapter name, e.g. `"claude"`, `"openai"`, `"ollama-local"`. If omitted, the first available provider that serves `model` is used and a warning is logged; the match is cached per `Provider` and dropped if a call through it fails with a 404, `AuthError`, or if the provider can't be reached.

### Other methods

| Method | Description | Returns |
|---|---|---|
| `providers()` | Configured providers and whether each is available (configured; for Ollama, running) | `dict[str, bool]` |
| `models(*, embeddings=False)` | Model ids of each available provider; embedding models when `embeddings=True` | `dict[str, list[str]]` |
| `model_info(model, *, provider=None)` | Context window / max output for a model, when known | `ModelInfo \| None` |
| `embed(model, input, config, provider)` | Embed one text or a batch of texts | `EmbedResponse` |
| `async_providers()` | `await`able `providers` | `dict[str, bool]` |
| `async_models(*, embeddings=False)` | `await`able `models` | `dict[str, list[str]]` |
| `async_model_info(model, *, provider=None)` | `await`able `model_info` | `ModelInfo \| None` |
| `async_embed(model, input, config, provider)` | `await`able `embed` | `EmbedResponse` |

`model_info` is best-effort: `None` means the provider doesn't expose it for that model, not that the model doesn't exist. Claude and self-hosted Ollama read it live from the vendor; OpenAI's API doesn't expose it, so it's always `None` there. It answers for chat models only. `ModelInfo`'s fields are in [`types.py`](src/ducktape_provider/types.py).

## Configuration

`config` sets options for a single call. It's a dict (`Config | Mapping[str, Any]`): `timeout` (seconds, `None` disables) and `headers` (extra HTTP headers) are handled by us, and every other key is sent as-is in the vendor's request body (`temperature`, `max_tokens`, …).

```python
provider.chat(
    "claude-opus-5",
    messages,
    config={
        "temperature": 0.2,
        "timeout": 30,
        "providers": {"claude": {"temperature": 0.7}},  # wins for claude
    },
)
```

`providers` holds per-provider config that overrides the rest for that provider; its fields are the same as that vendor's API.

`headers` merge key by key: a per-provider header adds to the call's headers rather than replacing them. Keys the call already sets (`model`, `messages`/`input`, `stream`) raise `ValueError`.

- `Provider(config=...)` sets the same options for every call; a call's `config` overrides it.
- Headers go per provider (`providers.<name>.headers`); values can be functions called per request.
- They are also sent when checking availability and listing models.
- They are only sent over https, or http to a loopback IP with no proxy; otherwise the provider reports unavailable and calls raise `ValueError`.

## Embedding

Turn text into dense vectors:

```python
single = provider.embed("text-embedding-3-small", "hello")
vectors = provider.embed("nomic-embed-text:latest", ["a", "b"])
```

- `input` is one `str` or a `Sequence[str]` (tuples work); a single string is sent as a one-element batch.
- `embeddings[i]` is the vector for input `i`, always in input order.
- `EmbedResponse` has `embeddings` (`list[list[float]]`), `usage` (`{"input_tokens": int}` or `None`), `dimensions` (`len(embeddings[0])`), `latency_ms`, and `raw` for vendor extras. Types are in [`types.py`](src/ducktape_provider/types.py).
- `provider` is optional and keyword-only, with the same auto-match as `chat`: `embed` matches each provider's embedding models. Use the model id exactly as the provider lists it, tag included (`nomic-embed-text:latest`, not `nomic-embed-text`); otherwise pass `provider=` explicitly.
- `config` is per-call and has the same shape as `chat`'s. Vendor body options like `dimensions` or `truncate` go there:

```python
provider.embed("text-embedding-3-small", "hello", config={"dimensions": 512})
provider.embed("nomic-embed-text:latest", "hello", config={"truncate": False})
```

- `Provider(config=...)` contributes only `timeout` to an embed call; chat defaults like `temperature` are ignored there. Put embed body options in the per-call `config`.
- Batch-size and token limits are the caller's responsibility and come back as `APIError` from the vendor. OpenAI caps a request at 2048 inputs, 8192 tokens per input and 300k tokens per request.
- Invalid input is rejected before the request is sent: `[]` and `""` raise `ValueError`, a non-string element raises `TypeError`, and a `dict` or `bytes` raises `TypeError`.
- Vectors are always floats; `encoding_format: "base64"` raises `ValueError`.
- Ollama truncates long input silently by default; pass `config={"truncate": False}` to error instead.
- A provider without an embeddings endpoint raises `UnsupportedOperationError`.

Discover embedding models:

```python
provider.models(embeddings=True)  # embedding ids per provider
provider.models()                 # chat ids per provider
```

The two listings never mix, and `embeddings` is keyword-only (`models(embeddings=True)`, not `models(True)`). A provider whose listing API doesn't separate model kinds (self-hosted Ollama's does not) can also show embedding models in the chat listing, so an id from `models()` is not a guarantee the chat methods accept it.

An embedding id passed to `chat`/`stream_chat`/`model_info` raises `KeyError`, and a chat id passed to `embed` raises `KeyError`. With an explicit `provider=` you get that provider's own error instead (usually `APIError` with status 404), because the library doesn't look the id up first. `model_info()` answers for chat models only.

## Streaming

`stream_chat` yields the reply as events; the last one is always `message_stop`, carrying the same `Response` that `chat` returns.

```python
for event in provider.stream_chat("claude-opus-5", messages):
    if event["type"] == "text_delta":
        print(event["text"], end="")
    elif event["type"] == "message_stop":
        response = event["response"]
```

| `type` | Fields | |
|---|---|---|
| `text_delta` | `index`, `text` | Chunk of text |
| `thinking_delta` | `index`, `thinking` | Chunk of reasoning |
| `tool_use_start` | `index`, `id`, `name` | Model starts a tool call |
| `tool_use_delta` | `index`, `partial_json` | Chunk of the tool call's JSON arguments |
| `block_stop` | `index` | Block finished |
| `message_stop` | `response` | Full `Response` |

`index` is the block's position in `response["content"]`. Errors mid-stream raise the same exceptions as `chat` (see [Errors](#errors)).

- The iterator `stream_chat` returns has a `cancel()` method for the built-in adapters: call it from another thread to stop the connection immediately, instead of waiting for the current read to finish. Safe at any point, including after the stream is already done. A third-party adapter's stream may not have it — check with `getattr(stream, "cancel", None)` if you need to support arbitrary ones uniformly.

## Latency

`latency_ms` is the time from sending the request until the reply is complete. Streamed responses also have `ttft_ms`: time until the first content event. Both are optional on `Response`; the built-in adapters always set `latency_ms`.

```python
response = provider.chat("claude-opus-5", messages)
print(response.get("latency_ms"), response.get("ttft_ms"))
```

## Errors

Import them from `ducktape_provider`. Streams raise the same errors as `chat`.

| Error | Raised when | Retry? |
|---|---|---|
| `AuthError` | API key missing or rejected (401/403) | No |
| `RateLimitError` | Rate limited (429) | Yes, after `retry_after` |
| `ServerError` | Vendor server error (5xx) | Yes, after `retry_after` |
| `RequestTimeoutError` | Request exceeded `timeout` | Yes |
| `ContextOverflowError` | Input too long for the model | No |
| `MalformedResponseError` | Vendor reply couldn't be parsed | No |
| `APIError` | Any other request failure, e.g. dropped connection | Depends on `status` |
| `UnsupportedBlockError` | Content the vendor doesn't support, e.g. URL image for Ollama | No |
| `UnsupportedOperationError` | The resolved provider doesn't support the operation, e.g. `embed()` on a provider with no embeddings endpoint | No |

- `APIError` is the base of the six above it; it has `status` (HTTP status or `None`) and `body`.
- `RateLimitError` and `ServerError` have `retry_after` (seconds or `None`).
- `DucktapeError` is the base of all of them; `UnsupportedBlockError` and `UnsupportedOperationError` are `DucktapeError`s but not `APIError`s.
- Invalid arguments raise `ValueError`/`TypeError` before any request is sent; an unknown `provider`, or no provider serving `model`, raises `KeyError`.

## Third-party adapters

Subclass `Adapter` and implement its four required methods:

| Method | Returns |
|---|---|
| `is_available()` | `bool`: whether the vendor is configured and usable |
| `models()` | `set[str]`: model ids it can serve; empty when unreachable |
| `chat(model, messages, system, tools, config)` | `Response`: only `content`, `stop_reason` and `usage` are required; `usage` is `None` when not known |
| `stream_chat(model, messages, system, tools, config)` | `Iterator[StreamEvent]`, ending with `message_stop` |
| `embed(model, input, config)` (optional) | `EmbedResponse`: dense vectors; the default raises `UnsupportedOperationError` |
| `embed_models()` (optional) | `set[str]`: model ids it can embed with; empty by default |

`config` arrives already merged (per-provider overrides applied); apply its `timeout` and `headers` to your HTTP request and send the rest to your vendor. Raise the errors from [Errors](#errors) so callers can handle every provider the same way.

Implementing `embed` without also overriding `embed_models()` leaves your adapter reachable only through an explicit `provider=`: auto-match uses `embed_models()` to find the provider, and an empty set never matches.

Use it directly:

```python
from ducktape_provider import ClaudeAdapter, Provider

provider = Provider(adapters={"claude": ClaudeAdapter(), "myvendor": MyVendorAdapter()})
```

Passing `adapters` replaces the built-ins, so include any you still want.

Or ship it as a plugin, registered in your package's `pyproject.toml` (the class must take no constructor arguments):

```toml
[project.entry-points."ducktape_provider.adapters"]
myvendor = "my_package.adapter:MyVendorAdapter"
```

Users load plugins with `Provider(autodiscover=True)`, or only some with `Provider(autodiscover={"myvendor"})` — the allowlist accepts either the plain name or a qualified `"my-package:myvendor"`. A plugin whose name collides with a built-in, a passed adapter, or another plugin registers as `<package>:<name>` instead; if that qualified name is also taken, it's skipped with a warning.

## Async

`async_chat`, `async_stream_chat`, `async_embed`, `async_providers`, `async_models` and `async_model_info` work like their sync versions without blocking the event loop.

```python
response = await provider.async_chat("claude-opus-5", messages)

async with contextlib.aclosing(
    provider.async_stream_chat("claude-opus-5", messages)
) as events:
    async for event in events:
        ...
```

- Use `contextlib.aclosing` to close a stream right away if you stop reading early.
- Cancelling an `async_stream_chat` call stops its HTTP request immediately, for the built-in adapters. A `chat`/`async_chat` call, or a stream from a third-party adapter that doesn't support this, still runs until done or `timeout`.
- `Provider(executor=...)` sets the thread pool for `async_chat`, `async_embed`, `async_providers`, `async_models` and `async_model_info`; it must be thread-based.

## Development

These are the commands CI runs. ty isn't a dev dependency: it's installed into the project venv from a hash-pinned requirements file, so rerun that install after `uv sync`, which removes it.

```
uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv pip install --require-hashes -r .github/ty-requirements.txt
uv run python -m ty check src tests
uv run python -W default -W error::ResourceWarning -m unittest discover -s tests
```

## License

[MIT](LICENSE)
