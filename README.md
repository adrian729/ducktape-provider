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
| `crates/provider` | Rust crate: `Provider`, streams, messages, errors, optional `reqwest` transport |
| `bindings/python` | Python package: `Provider`, `AsyncProvider`, messages, errors |
| `bindings/js` | npm package on `core.wasm`: `Provider`, streams, messages, errors, `index.d.ts` |
| `bindings/go` | Go module on `core.wasm` through wazero: `Provider`, streams, messages, errors |
| `bindings/java` | Maven module on `core.wasm` compiled to bytecode by Chicory: `Provider`, streams, messages, errors |
| `bindings/dotnet` | NuGet package on the native library through P/Invoke: `Provider`, streams, messages, errors |
| `bindings/c` | Header, CMake and pkg-config files and a smoke test for the native library |
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
| An existing login | `Provider("openai", auth=session("codex"))`, `Provider("anthropic", auth=session("claude_code"))`, `session("anthropic_profile", profile="work")` |
| Refreshing another tool's login | `session("codex", refresh="memory")` keeps the new token in process; `refresh="write_back"` updates the tool's store (file or keychain, only if it did not change meanwhile; the call still uses the new token when the write is refused) |
| Login state | `p.auth_status()` gives `source`, `expires_at` and `session`; `source` is `None` when nothing is configured |
| Several sources in order | `Provider("anthropic", auth=[session("claude_code"), env("ANTHROPIC_API_KEY")])` |
| Your own keychain | `Provider("openai", auth=session("codex"), keychain=lambda service, account: keyring.get_password(service, account), keychain_write=lambda service, account, value: keyring.set_password(service, account, value.decode()))`; the defaults read and write macOS Keychain, Linux Secret Service and Windows Credential Manager; a reader without a writer makes `refresh="write_back"` keep the refreshed token in memory only |
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

## JavaScript

Not on npm yet. Build the wasm core into the package, then import it from `bindings/js`:

```
cd bindings/js && npm run build:wasm && npm test
```

```js
import { Provider, user, assistant, tool, toolResult } from "ducktape-provider";

const p = new Provider("openai", { default_model: "gpt-5" });            // key from OPENAI_API_KEY
const r = await p.chat([user("hello")]);
console.log(r.text, r.usage, r.stop.reason);

for await (const ev of p.stream([user("tell me a story")])) {
  if (ev.type === "text_delta") process.stdout.write(ev.text);
}

const add = { type: "function", name: "add", description: "Adds two numbers", input_schema: { type: "object", properties: { a: { type: "number" }, b: { type: "number" } } } };
let history = [user("what is 2+2?")];
let reply = await p.chat(history, { tools: [add] });
if (reply.stop.reason === "tool_call") {
  const call = reply.toolCalls[0];
  history = [...history, assistant(reply), tool(toolResult(call.id, "4"))];
  reply = await p.chat(history, { tools: [add] });
}
```

| Need | Write |
|---|---|
| Another host, same wire | `new Provider("openai", { base_url: "https://api.groq.com/openai", auth: env("GROQ_API_KEY") })` |
| A key in code | `new Provider("anthropic", { auth: "sk-ant-..." })` |
| A token fetched at request time | `new Provider("openai", { auth: resolver(getToken) })` |
| An existing login | `new Provider("openai", { auth: session("codex") })`, `session("anthropic_profile", { profile: "work" })` |
| Refreshing another tool's login | `session("codex", { refresh: "memory" })` or `{ refresh: "write_back" }`, same meaning as in Python |
| Login state | `await p.authStatus()` gives `source`, `expires_at` and `session` |
| Your own keychain | `new Provider("openai", { auth: session("codex"), keychain: async (service, account) => ..., keychain_write: async (service, account, bytes) => true })`; the defaults read and write macOS Keychain, Linux Secret Service and Windows Credential Manager |
| In a browser | `resolver()` against your backend; an API key needs `dangerously_allow_browser: true` |
| Cancel | `signal: controller.signal` on any call, or `stream.cancel()` |
| Options and vendor fields | same keys as Python: `{ options: { reasoning: "high" }, vendor: { openai: { temperature: 0.2 } } }` |
| Images | `user("what is this?", image(pngBytes))` or `image(null, { url: "https://..." })` |
| Your own transport | `new Provider("openai", { fetch: myFetch })` |
| Your own `core.wasm` location | `new Provider("openai", { core: new URL("/static/core.wasm", location.href) })` |

Data fields use the schema's names (`default_model`, `tool_call_id`); methods and views are camelCase (`modelInfo`, `toolCalls`). Errors are the same classes as in Python, with `kind` as the stable discriminator. Types: [`bindings/js/index.d.ts`](bindings/js/index.d.ts) for the API and [`bindings/js/schema.d.ts`](bindings/js/schema.d.ts), generated from the JSON schema, for the data.

## Go

Not tagged yet. Build the wasm core into the module, then use it from `bindings/go` (a `replace` directive or a workspace):

```
cargo build --profile wasm --target wasm32-unknown-unknown -p ducktape-abi && cp target/wasm32-unknown-unknown/wasm/ducktape_abi.wasm bindings/go/core.wasm
```

```go
import ducktape "github.com/adrian729/ducktape-provider/bindings/go"

p, err := ducktape.New("anthropic", ducktape.Config{DefaultModel: "claude-opus-5-5"})   // key from ANTHROPIC_API_KEY
r, err := p.Chat(ctx, []ducktape.Message{ducktape.UserText("hello")}, nil)
fmt.Println(r.Text(), r.Usage, r.Stop.Reason)

s, err := p.Stream(ctx, []ducktape.Message{ducktape.UserText("tell me a story")}, nil)
for s.Next() {
	if ev := s.Event(); ev.Type == "text_delta" {
		fmt.Print(ev.Text)
	}
}
err = s.Err()

add := ducktape.FunctionTool("add", "Adds two numbers", map[string]any{"type": "object", "properties": map[string]any{"a": map[string]any{"type": "number"}, "b": map[string]any{"type": "number"}}})
history := []ducktape.Message{ducktape.UserText("what is 2+2?")}
r, err = p.Chat(ctx, history, &ducktape.ChatOptions{Tools: []ducktape.Tool{add}})
if r.Stop.Reason == ducktape.StopReasonToolCall {
	call := r.ToolCalls()[0]
	history = append(history, ducktape.AssistantFrom(r), ducktape.ToolMessage(ducktape.ToolResultText(call.ID, "4")))
	r, err = p.Chat(ctx, history, &ducktape.ChatOptions{Tools: []ducktape.Tool{add}})
}
```

| Need | Write |
|---|---|
| Another host, same wire | `ducktape.New("openai", ducktape.Config{BaseURL: "https://api.groq.com/openai", Auth: []ducktape.Credential{ducktape.Env("GROQ_API_KEY")}})` |
| A key in code | `Auth: []ducktape.Credential{ducktape.Key("sk-ant-...")}` |
| A token fetched at request time | `Auth: []ducktape.Credential{ducktape.Resolver(func(ctx context.Context) (string, error) { ... })}` |
| An existing login | `ducktape.Session("codex")`, `ducktape.Session("anthropic_profile", ducktape.SessionOptions{Profile: "work"})` |
| Refreshing another tool's login | `ducktape.Session("codex", ducktape.SessionOptions{Refresh: ducktape.AuthSourceRefreshMemory})` or `AuthSourceRefreshWriteBack`, same meaning as in Python |
| Login state | `p.AuthStatus(ctx)` gives `Source`, `ExpiresAt` and `Session` |
| Several sources in order | `Auth: []ducktape.Credential{ducktape.Session("claude_code"), ducktape.Env("ANTHROPIC_API_KEY")}` |
| Your own keychain | `Keychain: func(ctx context.Context, service, account string) ([]byte, error) { ... }` and `KeychainWrite: func(ctx context.Context, service, account string, value []byte) (bool, error) { ... }`; the defaults read and write macOS Keychain, Linux Secret Service and Windows Credential Manager |
| Cancel | cancel the `ctx` of any call, or `s.Close()` on a stream |
| Options and vendor fields | `&ducktape.ChatOptions{Options: &ducktape.Options{Reasoning: ducktape.ReasoningLevelHigh}, Vendor: ducktape.VendorMap{"openai": {"temperature": 0.2}}}` |
| Images | `ducktape.User(ducktape.Text("what is this?"), ducktape.Image(pngBytes, "image/png"))` or `ducktape.ImageURL("https://...")` |
| Embeddings | `ducktape.New("openai", ducktape.Config{DefaultEmbeddingModel: "text-embedding-3-small"})` then `p.Embed(ctx, []string{"a", "b"}, nil)` |
| Discovery | `p.Models(ctx, false)`, `p.ModelInfo(ctx, "gpt-5")`, `p.Capabilities(ctx, "gpt-5")` |
| Your own transport | `Client: &http.Client{Transport: ...}` |
| One compiled core for many providers | `core, _ := ducktape.NewCore(ctx, wasmBytes)` then `Core: core` |

Every error is a `*ducktape.ProviderError` with `Kind`, `Status`, `Code`, `Retryable`, `RetryAfter`, `Attempts`, `Body`, `RequestID` and `Origin`; `errors.Is(err, ducktape.ErrRateLimit)` matches by kind (one sentinel per kind). Argument mistakes return a plain error before any request. Types are generated from the JSON schema into [`bindings/go/types_gen.go`](bindings/go/types_gen.go); unions are one struct with a `Type` field that says which fields apply.

## Java

Not on Maven Central yet. Build the wasm core into the module, then `mvn install` it (Java 17+):

```
cargo build --profile wasm --target wasm32-unknown-unknown -p ducktape-abi && cp target/wasm32-unknown-unknown/wasm/ducktape_abi.wasm bindings/java/core.wasm && cd bindings/java && mvn -B install
```

```java
import io.github.adrian729.ducktape.*;
import io.github.adrian729.ducktape.types.*;
import static io.github.adrian729.ducktape.Ducktape.*;

Provider p = Provider.builder("anthropic").defaultModel("claude-opus-5-5").build();   // key from ANTHROPIC_API_KEY
Response r = p.chat(List.of(user("hello")));
System.out.println(textOf(r) + " " + r.usage() + " " + r.stop().reason());

try (Stream s = p.stream(List.of(user("tell me a story")))) {
    for (Event ev : s) {
        if (ev.type().equals("text_delta")) System.out.print(ev.text());
    }
}

Tool add = functionTool("add", "Adds two numbers", Map.of("type", "object", "properties", Map.of("a", Map.of("type", "number"), "b", Map.of("type", "number"))));
List<Message> history = new ArrayList<>(List.of(user("what is 2+2?")));
r = p.chat(history, ChatOptions.builder().tools(add).build());
if (r.stop().reason().equals("tool_call")) {
    Block call = toolCallsOf(r).get(0);
    history.add(assistant(r));
    history.add(tool(toolResult(call.id(), "4")));
    r = p.chat(history, ChatOptions.builder().tools(add).build());
}
```

| Need | Write |
|---|---|
| Another host, same wire | `Provider.builder("openai").baseUrl("https://api.groq.com/openai").auth(Credential.env("GROQ_API_KEY")).build()` |
| A key in code | `.auth(Credential.key("sk-ant-..."))` |
| A token fetched at request time | `.auth(Credential.resolver(() -> fetchToken()))` |
| An existing login | `Credential.session("codex")`, `Credential.session("anthropic_profile", null, null, "work", null)` (store, client id, profile, refresh) |
| Refreshing another tool's login | `Credential.session("codex", null, null, null, "memory")` or `"write_back"`, same meaning as in Python |
| Login state | `p.authStatus()` gives `source()`, `expiresAt()` and `session()` |
| Several sources in order | `.auth(Credential.session("claude_code"), Credential.env("ANTHROPIC_API_KEY"))` |
| Your own keychain | `.keychain((service, account) -> ...)` and `.keychainWrite((service, account, value) -> true)`; the defaults read and write macOS Keychain, Linux Secret Service and Windows Credential Manager |
| Cancel | `stream.close()`, or `p.chatAsync(msgs, options).cancel(true)` |
| Options and vendor fields | `ChatOptions.builder().options(Options.builder().reasoning("high").build()).vendor(Map.of("openai", Map.of("temperature", 0.2))).build()` |
| Images | `user(text("what is this?"), image(pngBytes, "image/png"))` or `imageUrl("https://...")` |
| Embeddings | `Provider.builder("openai").defaultEmbeddingModel("text-embedding-3-small").build().embed(List.of("a", "b")).embeddings()` |
| Discovery | `p.models(false)`, `p.modelInfo("gpt-5")`, `p.capabilities("gpt-5")` |
| Your own transport | `.client(HttpClient.newBuilder()...build())`, or `.transport((request, headersTimeout) -> ...)` for anything that is not `java.net.http` |
| One compiled core for many providers | `Core core = Core.bundled()` then `.core(core)` |

Every error is a `ProviderException` with `kind()`, `code()`, `status()`, `retryable()` and the full `error()` record; `e.is(ErrorKind.RATE_LIMIT)` matches by kind. Argument mistakes throw `IllegalArgumentException` before any request. Types are records generated from the JSON schema into [`bindings/java/src/main/java/io/github/adrian729/ducktape/types`](bindings/java/src/main/java/io/github/adrian729/ducktape/types), each with a builder; unions are one record with a `type` field that says which fields apply, and open enums are `String` fields with the known values as constants (`Role.USER`, `StopReason.TOOL_CALL`).

## .NET

Not on NuGet yet. Build the native core, put it where the package expects it, then reference `bindings/dotnet/Ducktape.Provider` (.NET 8+):

```
cargo build --release -p ducktape-abi && mkdir -p bindings/dotnet/runtimes/linux-x64/native && cp target/release/libducktape_abi.so bindings/dotnet/runtimes/linux-x64/native/
```

On macOS copy `libducktape_abi.dylib` under `osx-arm64` or `osx-x64`, on Windows `ducktape_abi.dll` under `win-x64`.

```csharp
using Ducktape;
using Ducktape.Types;
using static Ducktape.Messages;

var p = new Provider("anthropic", new ProviderOptions { DefaultModel = "claude-opus-5-5" });   // key from ANTHROPIC_API_KEY
var r = await p.ChatAsync([User("hello")]);
Console.WriteLine($"{r.Text()} {r.Usage} {r.Stop?.Reason}");

await foreach (var ev in p.Stream([User("tell me a story")]))
{
    if (ev.Type == "text_delta") Console.Write(ev.Text);
}

var add = Tools.Function("add", "Adds two numbers", """{"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}""");
var history = new List<Message> { User("what is 2+2?") };
r = await p.ChatAsync(history, new ChatOptions { Tools = [add] });
if (r.Stop?.Reason == StopReason.ToolCall)
{
    var call = r.ToolCalls()[0];
    history.Add(Assistant(r));
    history.Add(ToolMessage(ToolResult(call.Id!, "4")));
    r = await p.ChatAsync(history, new ChatOptions { Tools = [add] });
}
```

| Need | Write |
|---|---|
| Another host, same wire | `new Provider("openai", new ProviderOptions { BaseUrl = "https://api.groq.com/openai", Auth = [Credential.Env("GROQ_API_KEY")] })` |
| A key in code | `Auth = [Credential.Key("sk-ant-...")]` |
| A token fetched at request time | `Auth = [Credential.Resolver(async ct => await FetchToken(ct))]` |
| An existing login | `Credential.Session("codex")`, `Credential.Session("anthropic_profile", profile: "work")` |
| Refreshing another tool's login | `Credential.Session("codex", refresh: "memory")` or `refresh: "write_back"`, same meaning as in Python |
| Login state | `await p.AuthStatusAsync()` gives `Source`, `ExpiresAt` and `Session` |
| Several sources in order | `Auth = [Credential.Session("claude_code"), Credential.Env("ANTHROPIC_API_KEY")]` |
| Your own keychain | `Keychain = (service, account) => ...` and `KeychainWrite = (service, account, value) => true`; the defaults read and write macOS Keychain, Linux Secret Service and Windows Credential Manager |
| Cancel | a `CancellationToken` on any call, `stream.Cancel()`, or leaving the `await foreach` early; all end in `OperationCanceledException` |
| Blocking calls | `p.Chat(...)`, `p.Embed(...)`, `p.Models()`, `foreach (var ev in p.Stream(...).ToBlockingEnumerable())` |
| Options and vendor fields | `new ChatOptions { Options = new Options { Reasoning = ReasoningLevel.High }, Vendor = new() { ["openai"] = new() { ["temperature"] = 0.2 } } }` |
| Images | `User("what is this?", Image(pngBytes, "image/png"))` or `ImageUrl("https://...")` |
| Embeddings | `new Provider("openai", new ProviderOptions { DefaultEmbeddingModel = "text-embedding-3-small" })` then `(await p.EmbedAsync(["a", "b"])).Embeddings` |
| Discovery | `p.ModelsAsync()`, `p.ModelInfoAsync("gpt-5")`, `p.CapabilitiesAsync("gpt-5")` |
| Your own transport | `Client = new HttpClient(handler)` |
| Your own `libducktape_abi` location | `Core.Load("/opt/lib/libducktape_abi.so")` before the first call, or the `DUCKTAPE_NATIVE` environment variable |

Every error is a `ProviderException` with `Kind`, `Code`, `Status`, `Retryable` and the full `Error` record; `e.Is(ErrorKind.RateLimit)` matches by kind. Argument mistakes throw `ArgumentException` before any request. Types are records generated from the JSON schema into [`bindings/dotnet/Ducktape.Provider/Types`](bindings/dotnet/Ducktape.Provider/Types) with init-only properties; unions are one record with a `Type` field that says which fields apply, and open enums are `string` properties with the known values as constants (`Role.User`, `StopReason.ToolCall`). The schema's `Timeout` is called `Timeouts` in C#.

## Rust

Not on crates.io yet. Depend on the crate by path (`ducktape-provider = { path = "crates/provider" }`); the default `reqwest` feature brings the HTTP transport, and every call needs a tokio runtime:

```rust
use ducktape_provider::*;

let p = Provider::builder("anthropic").default_model("claude-opus-5-5").build()?;   // key from ANTHROPIC_API_KEY
let r = p.chat([user("hello")]).await?;
println!("{} {:?} {:?}", r.text(), r.usage, r.stop.reason);

let mut s = p.chat([user("tell me a story")]).stream()?;
while let Some(ev) = s.next().await {
    if let Event::TextDelta { text, .. } = ev? {
        print!("{text}");
    }
}

let add = function_tool("add", "Adds two numbers", serde_json::json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}));
let mut history = vec![user("what is 2+2?")];
let mut r = p.chat(history.clone()).tools([add.clone()]).await?;
if r.stop.reason == StopReason::ToolCall {
    let call = r.tool_calls()[0].clone();
    history.push(assistant(&r));
    history.push(tool(tool_result(&call.id, "4")));
    r = p.chat(history.clone()).tools([add]).await?;
}
```

| Need | Write |
|---|---|
| Another host, same wire | `Provider::builder("openai").base_url("https://api.groq.com/openai").auth([Credential::env(["GROQ_API_KEY"])]).build()?` |
| A key in code | `.auth([Credential::key("sk-ant-...")])` |
| A token fetched at request time | `.auth([Credential::resolver(fetch_token)])` where `fetch_token` is any `async fn() -> Result<String, E>` with a displayable `E` |
| An existing login | `Credential::session("codex")`, `Credential::session("anthropic_profile").profile("work")` |
| Refreshing another tool's login | `Credential::session("codex").refresh("memory")` or `.refresh("write_back")`, same meaning as in Python |
| Login state | `p.auth_status().await?` gives `source`, `expires_at` and `session` |
| Several sources in order | `.auth([Credential::session("claude_code"), Credential::env(["ANTHROPIC_API_KEY"])])` |
| Your own keychain | `.keychain(reader)` and `.keychain_write(writer)`, a `keychain::Reader` and a `keychain::Writer` (`Arc` of a closure over service and account); the defaults read and write macOS Keychain, Linux Secret Service and Windows Credential Manager |
| Cancel | drop the future or the stream, or `s.cancel()` |
| Options and vendor fields | `p.chat(msgs).options(Options { reasoning: Some(ReasoningLevel::High), ..Default::default() }).vendor(vendor)` |
| Images | `user([text("what is this?"), image(&png_bytes, "image/png")])` or `image_url("https://...")` |
| Embeddings | `Provider::builder("openai").default_embedding_model("text-embedding-3-small").build()?.embed(["a", "b"]).await?.embeddings` |
| Discovery | `p.models(false).await?`, `p.model_info("gpt-5").await?`, `p.capabilities("gpt-5").await?` |
| Your own `reqwest` client | `.transport(Arc::new(ReqwestTransport::new(client)))` |
| Your own transport | implement `transport::Transport` and `transport::Body`, then `.transport(Arc::new(mine))`; `default-features = false` drops `reqwest` |

Every error is an `Error`: `Error::Argument(String)` for argument mistakes caught before any request, `Error::Provider(ProviderError)` for everything else, with `kind`, `status`, `code`, `retryable`, `retry_after`, `attempts`, `body`, `request_id` and `origin`; `e.is(ErrorKind::RateLimit)` matches by kind. Types are the core's own Rust types, re-exported at the crate root and documented by [`schema/provider.schema.json`](schema/provider.schema.json); open enums are enums with an `Other(String)` variant.

## C

Not released yet. Build the libraries, then point CMake at them; the header is [`bindings/c/include/ducktape.h`](bindings/c/include/ducktape.h):

```
cargo build --release -p ducktape-abi && cmake -S bindings/c -B build/c && cmake --build build/c && ctest --test-dir build/c && cmake --install build/c --prefix /usr/local
```

The C surface is the core itself: JSON in, JSON out, no I/O. Your program runs the effect loop: start an operation with a `Start` document, read the `Step` it returns, answer the effect (send the HTTP request, sleep, read a file, store a secret) with a `Resume` document, and repeat until the effect is `done` or `failed`. The documents are in [`schema/effects.schema.json`](schema/effects.schema.json), the requests and results in [`schema/provider.schema.json`](schema/provider.schema.json), and [`bindings/c/test/smoke.c`](bindings/c/test/smoke.c) is a whole chat in one file.

| Need | Write |
|---|---|
| Start | `op = ducktape_op_start(start_json, len, &out, &out_len)`; a null handle means the document was invalid and `out` holds a `failed` step |
| Answer an effect | `ducktape_op_resume(op, resume_json, len, &out, &out_len)` with `{"clock": {...}, "input": {...}}` |
| Deliver body bytes | `ducktape_op_feed(op, bytes, len, clock_json, clock_len, &out, &out_len)` while the effect is `receive`, then resume with `http_end` |
| Provide a secret | `ducktape_op_set_secret(op, ref, ref_len, bytes, len)`, then resume with `{"type": "secret", "secret_ref": ref, "found": true}` |
| Cancel | resume with `{"type": "cancelled"}` at any point; the core answers with a `cancelled` error |
| Memory | every `out` buffer goes back through `ducktape_free(out, out_len)`; `ducktape_op_free(op)` releases the operation |
| Link | `pkg-config --cflags --libs ducktape`, or `find_package(ducktape)` and `ducktape::ducktape` / `ducktape::ducktape_static` in CMake |
| Version | `ducktape_abi_version()` must equal `DUCKTAPE_ABI_VERSION` from the header |

## Development

```
cargo fmt --all --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --profile wasm --target wasm32-unknown-unknown -p ducktape-abi
cargo test -p ducktape-provider --no-default-features
pip install jsonschema referencing && python scripts/check_fixtures.py .
python scripts/gen_dts.py . --check
python scripts/gen_go_types.py . --check
python scripts/gen_java_types.py . --check
python scripts/gen_cs_types.py . --check
cd bindings/python && uvx maturin develop --uv && .venv/bin/python -m pytest
cd bindings/js && npm ci && npm run check:types && npm run build:wasm && npm test && npm run test:browser
cp target/wasm32-unknown-unknown/wasm/ducktape_abi.wasm bindings/go/core.wasm && cd bindings/go && go vet ./... && go test ./...
cp target/wasm32-unknown-unknown/wasm/ducktape_abi.wasm bindings/java/core.wasm && cd bindings/java && mvn -B verify
cargo build --release -p ducktape-abi && mkdir -p bindings/dotnet/runtimes/linux-x64/native && cp target/release/libducktape_abi.so bindings/dotnet/runtimes/linux-x64/native/ && cd bindings/dotnet && dotnet test
cbindgen --config crates/abi/cbindgen.toml --crate ducktape-abi --output bindings/c/include/ducktape.h && cmake -S bindings/c -B build/c && cmake --build build/c && ctest --test-dir build/c
```

`npm run test:browser` replays every fixture and runs a chat and a stream inside headless Chrome (`CHROME_BIN` picks the binary; `npm run serve:browser` serves the same page for any browser). `npm run build:wasm` runs `wasm-opt -O3` on `core.wasm` when [binaryen](https://github.com/WebAssembly/binaryen) is on the PATH. Wheels, the sdist and the npm tarball are built by the Package workflow on every push and uploaded as artifacts.

## License

[MIT](LICENSE)
