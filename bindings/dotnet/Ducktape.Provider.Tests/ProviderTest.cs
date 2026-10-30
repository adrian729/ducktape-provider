using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json.Nodes;
using Ducktape.Types;
using Xunit;
using static Ducktape.Messages;

namespace Ducktape.Tests;

public class ProviderTest
{
    private static JsonNode Anthropic(string name) => Fixtures.Load("anthropic.messages", name);

    private static JsonNode OpenAi(string name) => Fixtures.Load("openai.responses", name);

    private static ChatRequest RequestOf(JsonNode fixture) => Json.As<ChatRequest>(fixture["start"]!["operation"]!["request"]);

    private static Response DoneResponse(JsonNode fixture)
    {
        var exchange = fixture["exchange"]!.AsArray();
        return Json.As<Response>(exchange[^1]!["step"]!["effect"]!["result"]!["response"]);
    }

    private static string LastText(JsonNode fixture) => DoneResponse(fixture).Content![^1].Text!;

    private static Provider ProviderOf(JsonNode fixture, Scripted server)
    {
        var config = fixture["start"]!["provider"]!;
        return new Provider(config["vendor"]!.GetValue<string>(), new ProviderOptions { BaseUrl = config["base_url"]?.GetValue<string>(), Client = server.Client(), Auth = [Credential.Key("sk-ant-test-key")] });
    }

    private static string Jwt(JsonObject claims)
    {
        static string B64(string s) => Convert.ToBase64String(Encoding.UTF8.GetBytes(s)).TrimEnd('=').Replace('+', '-').Replace('/', '_');
        return B64("{\"alg\":\"RS256\"}") + "." + B64(claims.ToJsonString()) + ".sig";
    }

    private static string Jwt(long expSeconds)
    {
        static string B64(string s) => Convert.ToBase64String(Encoding.UTF8.GetBytes(s)).TrimEnd('=').Replace('+', '-').Replace('/', '_');
        return B64("{\"alg\":\"RS256\"}") + "." + B64("{\"exp\":" + expSeconds + "}") + ".sig";
    }

    private static string TempDir()
    {
        var dir = Path.Combine(Path.GetTempPath(), "ducktape-dotnet-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    private static void SessionStore(JsonNode fixture, string dir, string prefix)
    {
        foreach (var entry in fixture["files"]!.AsObject())
        {
            var target = Path.Combine(dir, entry.Key[(prefix.Length + 1)..]);
            Directory.CreateDirectory(Path.GetDirectoryName(target)!);
            File.WriteAllText(target, entry.Value!.GetValue<string>());
        }
    }

    [Fact]
    public async Task ChatMatchesFixture()
    {
        var fixture = Anthropic("chat_text_reasoning");
        var server = new Scripted(fixture);
        var p = ProviderOf(fixture, server);
        var req = RequestOf(fixture);
        var r = await p.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model, Options = req.Options });
        var expected = DoneResponse(fixture);
        Assert.Equal(expected.Content!.Count, r.Content!.Count);
        for (var i = 0; i < r.Content.Count; i++)
        {
            Assert.Equal(expected.Content[i].Type, r.Content[i].Type);
            Assert.Equal(expected.Content[i].Text, r.Content[i].Text);
        }
        Assert.Equal(expected.Stop!.Reason, r.Stop!.Reason);
        Assert.Equal(expected.Usage!.Input, r.Usage!.Input);
        Assert.Equal(expected.Usage.Output, r.Usage.Output);
        Assert.NotEmpty(r.Text());
        Assert.Equal("sk-ant-test-key", server.Requests[0].Headers["x-api-key"]);
        Assert.Equal(req.Model, server.Requests[0].Body!["model"]!.GetValue<string>());
        var again = ProviderOf(fixture, new Scripted(fixture));
        Assert.Equal(r.Text(), again.Chat(req.Messages!, new ChatOptions { Model = req.Model }).Text());
    }

    [Fact]
    public async Task StreamEndsWithFinish()
    {
        var fixture = Anthropic("stream_text_reasoning_chunked");
        var server = new Scripted(fixture);
        var p = ProviderOf(fixture, server);
        var req = RequestOf(fixture);
        var events = new List<Event>();
        var s = p.Stream(req.Messages!, new ChatOptions { Model = req.Model });
        await foreach (var ev in s)
        {
            events.Add(ev);
        }
        Assert.Equal("block_start", events[0].Type);
        Assert.Equal("finish", events[^1].Type);
        var text = string.Concat(events.Where(e => e.Type == "text_delta").Select(e => e.Text));
        Assert.NotEmpty(text);
        Assert.Equal(text, (await s.GetResponseAsync())!.Text());
        var sync = ProviderOf(fixture, new Scripted(fixture)).Stream(req.Messages!, new ChatOptions { Model = req.Model });
        Assert.Equal(text, string.Concat(sync.ToBlockingEnumerable().Where(e => e.Type == "text_delta").Select(e => e.Text)));
        Assert.Equal(text, sync.GetResponse()!.Text());
    }

    [Fact]
    public async Task ToolRoundTripBuildsHistory()
    {
        var fixture = Anthropic("chat_tool_call");
        var server = new Scripted(fixture);
        var p = ProviderOf(fixture, server);
        var req = RequestOf(fixture);
        var r = await p.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model, Tools = req.Tools });
        Assert.Equal(StopReason.ToolCall, r.Stop!.Reason);
        var call = r.ToolCalls()[0];
        var history = new List<Message>(req.Messages!) { Assistant(r), ToolMessage(ToolResult(call.Id!, "4")) };
        Assert.Equal(Role.Assistant, history[^2].Role);
        Assert.Equal(call.Id, history[^1].Content![0].ToolCallId);
    }

    [Fact]
    public async Task ErrorsCarryKinds()
    {
        var fixture = Anthropic("chat_context_overflow");
        var server = new Scripted(fixture);
        var p = ProviderOf(fixture, server);
        var req = RequestOf(fixture);
        var e = await Assert.ThrowsAsync<ProviderException>(() => p.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model }));
        Assert.True(e.Is(ErrorKind.ContextOverflow));
        Assert.Equal(400, e.Status);
        Assert.Contains("context_overflow", e.Message);
    }

    [Fact]
    public async Task NoCredentialIsAuthError()
    {
        var fixture = Anthropic("chat_no_credential");
        var server = new Scripted(fixture);
        var p = new Provider("anthropic", new ProviderOptions { Client = server.Client(), Auth = [Credential.Env("DUCKTAPE_TEST_UNSET_KEY")] });
        var e = await Assert.ThrowsAsync<ProviderException>(() => p.ChatAsync([User("hi")], new ChatOptions { Model = "m" }));
        Assert.True(e.Is(ErrorKind.Auth));
        Assert.Empty(server.Requests);
    }

    [Fact]
    public async Task ResolverCredential()
    {
        var fixture = Anthropic("chat_text_reasoning");
        var req = RequestOf(fixture);
        {
            var server = new Scripted(fixture);
            var calls = 0;
            var p = new Provider("anthropic", new ProviderOptions { Client = server.Client(), Auth = [Credential.Resolver(() => { calls++; return "sk-resolved"; })] });
            await p.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model });
            Assert.Equal(1, calls);
            Assert.Equal("sk-resolved", server.Requests[0].Headers["x-api-key"]);
        }
        {
            var server = new Scripted(fixture);
            var failing = new Provider("anthropic", new ProviderOptions { Client = server.Client(), Auth = [Credential.Resolver(_ => Task.FromException<string>(new InvalidOperationException("vault down")))] });
            var e = await Assert.ThrowsAsync<ProviderException>(() => failing.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model }));
            Assert.True(e.Is(ErrorKind.Auth));
            Assert.Contains("vault down", e.Message);
        }
    }

    [Fact]
    public async Task ArgumentErrorsComeFirst()
    {
        var fixture = Anthropic("chat_text_reasoning");
        var server = new Scripted(fixture);
        var p = ProviderOf(fixture, server);
        await Assert.ThrowsAsync<ArgumentException>(() => p.ChatAsync([User("hi")]));
        Assert.Throws<ArgumentException>(() => new Provider("nope"));
        await Assert.ThrowsAsync<ArgumentException>(() => p.EmbedAsync([]));
        Assert.Throws<ArgumentException>(() => new Provider("openai", new ProviderOptions { DefaultSystem = "a", DefaultSystemBlocks = [Text("b")] }));
        Assert.Empty(server.Requests);
    }

    [Fact]
    public async Task ConnectionFailureRetriesThenSucceeds()
    {
        var fixture = Anthropic("chat_connection_reset_retried");
        var server = new Scripted(fixture);
        var p = ProviderOf(fixture, server);
        var req = RequestOf(fixture);
        var r = await p.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model });
        Assert.Equal(2, server.Requests.Count);
        Assert.NotEmpty(r.Text());
    }

    [Fact]
    public async Task ModelsAreCached()
    {
        var fixture = Anthropic("models_two_pages");
        var server = new Scripted(fixture);
        var p = ProviderOf(fixture, server);
        var models = await p.GetModelsAsync();
        var again = await p.GetModelsAsync();
        Assert.NotEmpty(models);
        Assert.Equal(models.Count, again.Count);
        Assert.Equal(2, server.Requests.Count);
    }

    [Fact]
    public async Task CapabilitiesShipped()
    {
        var fixture = Anthropic("capabilities_shipped");
        var server = new Scripted(fixture);
        var p = ProviderOf(fixture, server);
        var caps = await p.GetCapabilitiesAsync(fixture["start"]!["operation"]!["model"]!.GetValue<string>());
        Assert.Equal("shipped", caps.Source);
        Assert.Empty(server.Requests);
    }

    [Fact]
    public async Task StreamCancelStops()
    {
        var fixture = Anthropic("stream_text_reasoning_chunked");
        var server = new Scripted(fixture) { DelayMs = 30 };
        var p = ProviderOf(fixture, server);
        var req = RequestOf(fixture);
        var s = p.Stream(req.Messages!, new ChatOptions { Model = req.Model });
        var e = await Assert.ThrowsAnyAsync<OperationCanceledException>(async () =>
        {
            await foreach (var ev in s)
            {
                s.Cancel();
            }
        });
        Assert.IsType<ProviderException>(e.InnerException);
        Assert.True(((ProviderException)e.InnerException!).Is(ErrorKind.Cancelled));
    }

    [Fact]
    public async Task LeavingTheLoopStopsTheStream()
    {
        var fixture = Anthropic("stream_text_reasoning_chunked");
        var server = new Scripted(fixture) { DelayMs = 30 };
        var p = ProviderOf(fixture, server);
        var req = RequestOf(fixture);
        var s = p.Stream(req.Messages!, new ChatOptions { Model = req.Model });
        await foreach (var ev in s)
        {
            break;
        }
        var e = await Assert.ThrowsAnyAsync<OperationCanceledException>(() => s.GetResponseAsync());
        Assert.True(((ProviderException)e.InnerException!).Is(ErrorKind.Cancelled));
        s.Cancel();
        s.Dispose();
    }

    [Fact]
    public async Task CancelDuringTheResolverIsCancelled()
    {
        var fixture = Anthropic("chat_text_reasoning");
        var server = new Scripted(fixture);
        using var cts = new CancellationTokenSource();
        var p = new Provider("anthropic", new ProviderOptions
        {
            Client = server.Client(),
            Auth = [Credential.Resolver(async ct =>
            {
                cts.Cancel();
                await Task.Delay(Timeout.Infinite, ct);
                return "never";
            })],
        });
        var req = RequestOf(fixture);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => p.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model }, cts.Token));
        Assert.Empty(server.Requests);
    }

    [Fact]
    public async Task OddTimeoutsAreIgnored()
    {
        var fixture = Anthropic("chat_text_reasoning");
        var p = ProviderOf(fixture, new Scripted(fixture));
        var req = RequestOf(fixture);
        var o = new ChatOptions { Model = req.Model, Timeout = new Timeouts { Read = 0, Connect = -1, Total = 1e12 } };
        var r = await p.ChatAsync(req.Messages!, o);
        Assert.NotEmpty(r.Text());
    }

    [Fact]
    public void BlockingCallsWorkUnderASynchronizationContext()
    {
        var fixture = Anthropic("chat_text_reasoning");
        var p = ProviderOf(fixture, new Scripted(fixture));
        var req = RequestOf(fixture);
        var previous = SynchronizationContext.Current;
        SynchronizationContext.SetSynchronizationContext(new SynchronizationContext());
        try
        {
            var r = p.Chat(req.Messages!, new ChatOptions { Model = req.Model });
            Assert.NotEmpty(r.Text());
        }
        finally
        {
            SynchronizationContext.SetSynchronizationContext(previous);
        }
    }

    [Fact]
    public void TimeoutsInsideResetsAreTimeouts()
    {
        Assert.Equal("timeout", Driver.Classify(new IOException("read", new TimeoutException())));
        Assert.Equal("reset", Driver.Classify(new IOException("read", new SocketException((int)SocketError.ConnectionReset))));
    }

    [Fact]
    public async Task LoadAfterTheFirstCallThrows()
    {
        var fixture = Anthropic("chat_text_reasoning");
        var p = ProviderOf(fixture, new Scripted(fixture));
        var req = RequestOf(fixture);
        await p.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model });
        Assert.Throws<InvalidOperationException>(() => Core.Load("/nowhere/libducktape_abi.so"));
    }

    [Fact]
    public async Task TokenCancelsChat()
    {
        var fixture = Anthropic("chat_text_reasoning");
        var server = new Scripted(fixture) { DelayMs = 500 };
        var p = ProviderOf(fixture, server);
        var req = RequestOf(fixture);
        using var cts = new CancellationTokenSource(50);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => p.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model }, cts.Token));
    }

    [Fact]
    public async Task ReadTimeoutMapsToTimeout()
    {
        var fixture = Anthropic("chat_text_reasoning");
        var server = new Scripted(fixture) { DelayMs = 400 };
        var p = ProviderOf(fixture, server);
        var req = RequestOf(fixture);
        var o = new ChatOptions { Model = req.Model, Timeout = new Timeouts { Read = 0.05 }, Options = new Options { Retries = 0 } };
        var e = await Assert.ThrowsAsync<ProviderException>(() => p.ChatAsync(req.Messages!, o));
        Assert.True(e.Is(ErrorKind.Timeout), e.Message);
    }

    [Fact]
    public async Task Embed()
    {
        var fixture = OpenAi("embed_two_inputs");
        var server = new Scripted(fixture);
        var p = new Provider("openai", new ProviderOptions { Client = server.Client(), Auth = [Credential.Key("sk-test")] });
        var req = fixture["start"]!["operation"]!["request"]!;
        var input = req["input"]!.AsArray().Select(s => s!.GetValue<string>()).ToList();
        var r = await p.EmbedAsync(input, new EmbedOptions { Model = req["model"]!.GetValue<string>() });
        Assert.Equal(2, r.Embeddings!.Count);
        Assert.Equal(0.1, r.Embeddings[0][0]);
    }

    [Fact]
    public void ImageBytesBecomeBase64()
    {
        byte[] png = [0x89, 0x50, 0x4e, 0x47];
        var m = User("what is this?", Image(png, "image/png"));
        Assert.Equal(Convert.ToBase64String(png), m.Content![1].Data);
        Assert.Equal("what is this?", m.Content[0].Text);
        Assert.Equal(Role.User, m.Role);
    }

    [Fact]
    public async Task ProfileRefreshesAndWritesBack()
    {
        var fixture = Anthropic("session_profile_refresh_write_back");
        var dir = TempDir();
        SessionStore(fixture, dir, "/home/u/.config/anthropic");
        var server = new Scripted(fixture);
        var p = new Provider("anthropic", new ProviderOptions { Client = server.Client(), Auth = [Credential.Session("anthropic_profile", store: dir, profile: "default", refresh: "write_back")] });
        var req = RequestOf(fixture);
        var r = await p.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model });
        Assert.Equal(LastText(fixture), r.Text());
        Assert.Equal("https://api.anthropic.com/v1/oauth/token", server.Requests[0].Url);
        Assert.Equal("Bearer sk-ant-oat01-new", server.Requests[1].Headers["authorization"]);
        var written = JsonNode.Parse(File.ReadAllText(Path.Combine(dir, "credentials", "default.json")))!;
        Assert.Equal("sk-ant-oat01-new", written["access_token"]!.GetValue<string>());
        Assert.Equal("kept", written["extra"]!.GetValue<string>());
        Assert.Equal("session:anthropic_profile", (await p.GetAuthStatusAsync()).Source);
    }

    [Fact]
    public async Task WriteConflictKeepsTheRefreshedToken()
    {
        var fixture = Anthropic("session_profile_write_conflict_keeps_token");
        var dir = TempDir();
        SessionStore(fixture, dir, "/home/u/.config/anthropic");
        var creds = Path.Combine(dir, "credentials", "default.json");
        var server = new Scripted(fixture)
        {
            OnRequest = url =>
            {
                if (url.EndsWith("/v1/oauth/token"))
                {
                    File.WriteAllText(creds, "{\"access_token\": \"sk-ant-oat01-other-process\", \"expires_at\": 4102444800}");
                }
            },
        };
        var p = new Provider("anthropic", new ProviderOptions { Client = server.Client(), Auth = [Credential.Session("anthropic_profile", store: dir, profile: "default", refresh: "write_back")] });
        var req = RequestOf(fixture);
        var r = await p.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model });
        Assert.Equal(LastText(fixture), r.Text());
        Assert.Contains("other-process", File.ReadAllText(creds));
    }

    [Fact]
    public async Task ExpiredBorrowedSessionNamesTheOwner()
    {
        var fixture = Anthropic("session_claude_code_expired");
        var dir = TempDir();
        SessionStore(fixture, dir, "/home/u/.claude");
        var server = new Scripted(fixture);
        var p = new Provider("anthropic", new ProviderOptions { Client = server.Client(), Auth = [Credential.Session("claude_code", store: Path.Combine(dir, ".credentials.json"))] });
        var e = await Assert.ThrowsAsync<ProviderException>(() => p.ChatAsync([User("hi")], new ChatOptions { Model = "m" }));
        Assert.Equal("credential_expired", e.Code);
        Assert.Contains("claude", e.Message);
        var status = await p.GetAuthStatusAsync();
        Assert.Equal("session:claude_code", status.Source);
        Assert.NotNull(status.ExpiresAt);
    }

    [Fact]
    public async Task CodexKeychainWriteBackUsesTheWriter()
    {
        var fixture = OpenAi("session_codex_keychain_write_back").DeepClone();
        var keychain = fixture["keychain"]!.AsObject();
        var service = keychain.First().Key.Split('|', 2)[0];
        var doc = JsonNode.Parse(keychain.First().Value!.GetValue<string>())!;
        var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        doc["tokens"]!["access_token"] = Jwt(now + 60);
        var fresh = JsonNode.Parse(fixture["http"]![0]!["body"]!.GetValue<string>())!;
        fresh["access_token"] = Jwt(now + 86400);
        fixture["http"]![0]!["body"] = fresh.ToJsonString();
        var written = new List<(string Service, string Account, string Value)>();
        KeychainReader reader = (s, a, ct) => Task.FromResult(s == service ? Encoding.UTF8.GetBytes(doc.ToJsonString()) : null);
        KeychainWriter writer = (s, a, v, ct) =>
        {
            written.Add((s, a, Encoding.UTF8.GetString(v)));
            return Task.FromResult(true);
        };
        var req = RequestOf(fixture);
        {
            var server = new Scripted(fixture);
            var p = new Provider("openai", new ProviderOptions { Client = server.Client(), Auth = [Credential.Session("codex", refresh: "write_back")], Keychain = reader, KeychainWrite = writer });
            var r = await p.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model });
            Assert.Equal(LastText(fixture), r.Text());
            Assert.Equal("Codex Auth", written[0].Service);
            Assert.StartsWith("cli|", written[0].Account);
            Assert.Contains("rt_d", written[0].Value);
            Assert.Equal("acct_1", server.Requests[1].Headers["chatgpt-account-id"]);
        }
        {
            var server = new Scripted(fixture);
            var readerOnly = new Provider("openai", new ProviderOptions { Client = server.Client(), Auth = [Credential.Session("codex", refresh: "write_back")], Keychain = reader });
            Assert.Equal(LastText(fixture), (await readerOnly.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model })).Text());
        }
    }

    [Fact]
    public async Task LoopbackLoginStoresTheSessionAndTimesOut()
    {
        var fixture = OpenAi("login_chatgpt_plan_new_registration");
        var dir = TempDir();
        var store = Path.Combine(dir, "auth.json");
        var server = new Scripted(fixture);
        var p = new Provider("openai", new ProviderOptions { Client = server.Client(), Auth = [Credential.Session("chatgpt_plan", store: store)] });
        Task<HttpStatusCode>? visit = null;
        var status = await p.LoginAsync((action, _) =>
        {
            if (action.Type == "open_url")
            {
                var query = System.Web.HttpUtility.ParseQueryString(new Uri(action.Url!).Query);
                var now = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
                var claims = new JsonObject { ["iss"] = "https://auth.openai.com", ["aud"] = "oaiapp_test", ["exp"] = now + 3600, ["iat"] = now, ["sub"] = "user_1", ["email"] = "u@example.com", ["nonce"] = query["nonce"] };
                var reply = JsonNode.Parse(fixture["http"]![0]!["body"]!.GetValue<string>())!;
                reply["id_token"] = Jwt(claims);
                fixture["http"]![0]!["body"] = reply.ToJsonString();
                var target = query["redirect_uri"] + "?code=code_1&state=" + query["state"] + "&client_id=oaiapp_test";
                visit = Task.Run(async () =>
                {
                    using var http = new HttpClient();
                    using var answer = await http.GetAsync(target);
                    return answer.StatusCode;
                });
            }
            return Task.FromResult<string?>(null);
        }, appName: "Example App");
        Assert.Equal("session:chatgpt_plan", status.Source);
        Assert.Equal("chatgpt_plan", status.Session);
        Assert.Equal(HttpStatusCode.OK, await visit!);
        Assert.Contains("oaiapp_test", File.ReadAllText(store));
        if (!OperatingSystem.IsWindows())
        {
            Assert.Equal(UnixFileMode.UserRead | UnixFileMode.UserWrite, File.GetUnixFileMode(store));
        }
        var e = await Assert.ThrowsAsync<ProviderException>(() => p.LoginAsync((_, _) => Task.FromResult<string?>(null), timeout: TimeSpan.FromMilliseconds(300)));
        Assert.Equal("timeout", e.Kind);
        Assert.Equal("login_timeout", e.Code);
        var probe = new TcpListener(IPAddress.Loopback, 1455);
        probe.Start();
        probe.Stop();
    }

    [Fact]
    public async Task DefaultKeychainReaderNeverThrows()
    {
        Assert.Null(await Keychain.ReadAsync("ducktape-provider-test", "nobody"));
    }

    [Fact]
    public async Task ObserverSeesHooks()
    {
        var fixture = Anthropic("chat_text_reasoning");
        var server = new Scripted(fixture);
        var hooks = new List<string>();
        var config = fixture["start"]!["provider"]!;
        var p = new Provider("anthropic", new ProviderOptions { BaseUrl = config["base_url"]?.GetValue<string>(), Client = server.Client(), Auth = [Credential.Key("k")], Observer = new Observer { OnRequest = _ => hooks.Add("request"), OnResponse = _ => hooks.Add("response") } });
        var req = RequestOf(fixture);
        await p.ChatAsync(req.Messages!, new ChatOptions { Model = req.Model });
        Assert.Contains("request", hooks);
    }

    [Fact]
    public void AbiVersionMatches()
    {
        Assert.Equal(1u, Core.AbiVersion);
    }
}
