# ducktape-provider

Simple AI chat provider to normalize usage of different APIs.

Claude, OpenAI and local Ollama adapters included. Extensible by subclassing `Adapter`.

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

Or pass API keys in code with `api_keys`:

| Form | Example |
|---|---|
| Keys by provider name | `Provider(api_keys={"claude": "sk-ant-...", "openai": get_openai_key})` |
| One function for all | `Provider(api_keys=lambda name: vault.read(name))` |

- A value is a key or a function returning one; functions run on every request, possibly from several threads.
- A provider given a key never reads its env var.
- Also applies to Claude and OpenAI adapters passed in `adapters=`.

## Usage

All types the interface uses (`Message`, `Response`, `StreamEvent`, `Config`, errors, `Adapter`) are in [`types.py`](src/ducktape_provider/types.py).

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
| `models()` | Model ids of each available provider | `dict[str, list[str]]` |
| `async_providers()` | `await`able `providers` | `dict[str, bool]` |
| `async_models()` | `await`able `models` | `dict[str, list[str]]` |

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

- `APIError` is the base of the six above it; it has `status` (HTTP status or `None`) and `body`.
- `RateLimitError` and `ServerError` have `retry_after` (seconds or `None`).
- `DucktapeError` is the base of all of them.
- Invalid arguments raise `ValueError`/`TypeError` before any request is sent; an unknown `provider`, or no provider serving `model`, raises `KeyError`.

## Third-party adapters

Subclass `Adapter` and implement its four methods:

| Method | Returns |
|---|---|
| `is_available()` | `bool`: whether the vendor is configured and usable |
| `models()` | `set[str]`: model ids it can serve; empty when unreachable |
| `chat(model, messages, system, tools, config)` | `Response`: only `content` and `stop_reason` are required |
| `stream_chat(model, messages, system, tools, config)` | `Iterator[StreamEvent]`, ending with `message_stop` |

`config` arrives already merged (per-provider overrides applied); apply its `timeout` and `headers` to your HTTP request and send the rest to your vendor. Raise the errors from [Errors](#errors) so callers can handle every provider the same way.

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

`async_chat`, `async_stream_chat`, `async_providers` and `async_models` work like their sync versions without blocking the event loop.

```python
response = await provider.async_chat("claude-opus-5", messages)

async with contextlib.aclosing(
    provider.async_stream_chat("claude-opus-5", messages)
) as events:
    async for event in events:
        ...
```

- Use `contextlib.aclosing` to close a stream right away if you stop reading early.
- Cancelling a call doesn't stop its HTTP request; it runs until done or `timeout`.
- `Provider(executor=...)` sets the thread pool for `async_chat`, `async_providers` and `async_models`; it must be thread-based.

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
