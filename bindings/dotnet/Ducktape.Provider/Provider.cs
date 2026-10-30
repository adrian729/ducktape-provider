using System.Collections.Concurrent;
using System.Text.Json.Nodes;
using Ducktape.Types;

namespace Ducktape;

/// <summary>One endpoint of one vendor.</summary>
public sealed class Provider
{
    private static readonly HashSet<string> Vendors = ["anthropic", "openai", "google", "ollama"];
    private static readonly Lazy<HttpClient> SharedClient = new(() => new HttpClient(new SocketsHttpHandler { AllowAutoRedirect = false }) { Timeout = System.Threading.Timeout.InfiniteTimeSpan });

    private readonly ProviderConfig config;
    private readonly Driver.Host host;
    private readonly ConcurrentDictionary<string, ModelInfo> infos = new();
    private readonly ConcurrentDictionary<string, Capabilities> caps = new();
    private volatile List<ModelInfo>? models;

    /// <summary>Configures a provider for a vendor; the key comes from the environment unless options say otherwise.</summary>
    public Provider(string vendor, ProviderOptions? options = null)
    {
        if (!Vendors.Contains(vendor))
        {
            throw new ArgumentException("unknown vendor " + vendor + "; one of " + string.Join(", ", Vendors), nameof(vendor));
        }
        options ??= new ProviderOptions();
        if (options.DefaultSystem is not null && options.DefaultSystemBlocks is not null)
        {
            throw new ArgumentException("set DefaultSystem or DefaultSystemBlocks, not both");
        }
        Vendor = vendor;
        var secrets = new Dictionary<string, string>();
        var resolvers = new Dictionary<string, Func<CancellationToken, Task<string>>>();
        Auth? auth = null;
        if (options.Auth is { Count: > 0 })
        {
            var sources = new List<AuthSource>();
            for (var i = 0; i < options.Auth.Count; i++)
            {
                var credential = options.Auth[i];
                var source = credential.Source;
                if (source.Type == "inline")
                {
                    var secretRef = "inline:" + i;
                    secrets[secretRef] = credential.Secret!;
                    source = source with { SecretRef = secretRef };
                }
                else if (source.Type == "resolver")
                {
                    var id = "resolver:" + i;
                    resolvers[id] = credential.Fetch ?? throw new ArgumentException("a resolver credential needs a function");
                    source = source with { ResolverId = id };
                }
                sources.Add(source);
            }
            auth = new Auth { Sources = sources };
        }
        config = new ProviderConfig
        {
            Vendor = vendor,
            BaseUrl = options.BaseUrl?.TrimEnd('/'),
            Auth = auth,
            Headers = options.Headers,
            Timeout = options.Timeout,
            DefaultModel = options.DefaultModel,
            DefaultEmbeddingModel = options.DefaultEmbeddingModel,
            DefaultSystem = options.DefaultSystemBlocks ?? (options.DefaultSystem is null ? null : [Messages.Text(options.DefaultSystem)]),
            DefaultTools = options.DefaultTools,
            DefaultOptions = options.DefaultOptions,
            DefaultVendor = options.DefaultVendor,
            Overrides = options.Overrides,
            Strict = options.Strict,
        };
        var reader = options.Keychain ?? Keychain.Read;
        var writer = options.KeychainWrite ?? (options.Keychain is null ? Keychain.Write : null);
        host = new Driver.Host(secrets, resolvers, reader, writer, options.Client ?? SharedClient.Value, options.Observer);
    }

    /// <summary>The vendor namespace.</summary>
    public string Vendor { get; }

    private async Task<JsonNode?> RunAsync(JsonObject operation, Func<Event, Task>? onEvent, Func<LoginAction, CancellationToken, Task<string?>>? handleAction, CancellationToken ct)
    {
        var start = new JsonObject { ["clock"] = Driver.Clock(), ["provider"] = Json.ToNode(config), ["operation"] = operation };
        var (op, step) = Core.Operation.Start(start);
        using (op)
        {
            if (op is null)
            {
                var effect = step["effect"];
                throw Json.Str(effect?["type"]) == "failed" ? new ProviderException(Json.As<ProviderError>(effect!["error"])) : ProviderException.Of("other", "internal", "the core refused the operation");
            }
            try
            {
                return await new Driver(op, host, onEvent, handleAction, ct).RunAsync(start, step).ConfigureAwait(false);
            }
            catch (ProviderException e) when (e.Kind == "cancelled" && ct.IsCancellationRequested)
            {
                throw new OperationCanceledException(e.Message, e, ct);
            }
        }
    }

    private static JsonObject Operation(string type, object? request = null)
    {
        var op = new JsonObject { ["type"] = type };
        if (request is not null)
        {
            op["request"] = Json.ToNode(request);
        }
        return op;
    }

    private ChatRequest ChatRequest(IEnumerable<Message> messages, ChatOptions? o)
    {
        o ??= new ChatOptions();
        ArgumentNullException.ThrowIfNull(messages);
        if (o.System is not null && o.SystemBlocks is not null)
        {
            throw new ArgumentException("set System or SystemBlocks, not both");
        }
        var model = o.Model ?? config.DefaultModel;
        if (string.IsNullOrEmpty(model))
        {
            throw new ArgumentException("no model: set Model on the call or DefaultModel on the provider");
        }
        return new ChatRequest
        {
            Model = model,
            Messages = [.. messages],
            System = o.SystemBlocks ?? (o.System is null ? null : [Messages.Text(o.System)]),
            Tools = o.Tools,
            Options = o.Options,
            Vendor = o.Vendor,
            Timeout = o.Timeout,
            Headers = o.Headers,
            Strict = o.Strict,
            IncludeRaw = o.IncludeRaw,
        };
    }

    /// <summary>Sends the messages and returns the whole reply.</summary>
    public async Task<Response> ChatAsync(IEnumerable<Message> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var request = ChatRequest(messages, options);
        var result = await RunAsync(Operation("chat", request), null, null, cancellationToken).ConfigureAwait(false);
        return Json.As<Response>(result?["response"]);
    }

    /// <summary>Sends the messages and blocks for the whole reply.</summary>
    public Response Chat(IEnumerable<Message> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => ChatAsync(messages, options, cancellationToken), CancellationToken.None).GetAwaiter().GetResult();
    }

    /// <summary>Sends the messages and returns the events as they arrive.</summary>
    public ChatStream Stream(IEnumerable<Message> messages, ChatOptions? options = null, CancellationToken cancellationToken = default)
    {
        var request = ChatRequest(messages, options);
        var stream = new ChatStream(cancellationToken);
        _ = Task.Run(async () =>
        {
            Exception? failure = null;
            try
            {
                await RunAsync(Operation("stream", request), stream.Push, null, stream.Token).ConfigureAwait(false);
            }
            catch (Exception e)
            {
                failure = e;
            }
            finally
            {
                stream.Finish(failure);
            }
        }, CancellationToken.None);
        return stream;
    }

    /// <summary>Returns one embedding per input text.</summary>
    public async Task<EmbedResponse> EmbedAsync(IEnumerable<string> input, EmbedOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new EmbedOptions();
        var texts = input?.ToList() ?? throw new ArgumentNullException(nameof(input));
        if (texts.Count == 0)
        {
            throw new ArgumentException("embed input must not be empty", nameof(input));
        }
        var model = options.Model ?? config.DefaultEmbeddingModel;
        if (string.IsNullOrEmpty(model))
        {
            throw new ArgumentException("no embedding model: set Model on the call or DefaultEmbeddingModel on the provider");
        }
        var request = new EmbedRequest
        {
            Model = model,
            Input = texts,
            Options = options.Dimensions is null && options.Retries is null ? null : new EmbedRequestOptions { Dimensions = options.Dimensions, Retries = options.Retries },
            Vendor = options.Vendor,
            Timeout = options.Timeout,
            IncludeRaw = options.IncludeRaw,
        };
        var result = await RunAsync(Operation("embed", request), null, null, cancellationToken).ConfigureAwait(false);
        return Json.As<EmbedResponse>(result?["response"]);
    }

    /// <summary>Returns one embedding per input text, blocking.</summary>
    public EmbedResponse Embed(IEnumerable<string> input, EmbedOptions? options = null, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => EmbedAsync(input, options, cancellationToken), CancellationToken.None).GetAwaiter().GetResult();
    }

    /// <summary>Lists the models the endpoint serves; cached after the first call unless refresh is set.</summary>
    public async Task<IReadOnlyList<ModelInfo>> GetModelsAsync(bool refresh = false, CancellationToken cancellationToken = default)
    {
        var cached = models;
        if (cached is not null && !refresh)
        {
            return cached;
        }
        var op = Operation("models");
        op["refresh"] = refresh;
        var result = await RunAsync(op, null, null, cancellationToken).ConfigureAwait(false);
        var list = (result?["models"] as JsonArray ?? []).Select(m => Json.As<ModelInfo>(m)).ToList();
        models = list;
        return list;
    }

    /// <summary>Lists the models, blocking.</summary>
    public IReadOnlyList<ModelInfo> GetModels(bool refresh = false, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => GetModelsAsync(refresh, cancellationToken), CancellationToken.None).GetAwaiter().GetResult();
    }

    /// <summary>Describes one model.</summary>
    public async Task<ModelInfo> GetModelInfoAsync(string id, CancellationToken cancellationToken = default)
    {
        if (infos.TryGetValue(id, out var cached))
        {
            return cached;
        }
        var op = Operation("model_info");
        op["model"] = id;
        var result = await RunAsync(op, null, null, cancellationToken).ConfigureAwait(false);
        var info = result?["info"] ?? throw ProviderException.Of("not_found", null, "model " + id + " not found");
        var parsed = Json.As<ModelInfo>(info);
        infos[id] = parsed;
        return parsed;
    }

    /// <summary>Describes one model, blocking.</summary>
    public ModelInfo GetModelInfo(string id, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => GetModelInfoAsync(id, cancellationToken), CancellationToken.None).GetAwaiter().GetResult();
    }

    /// <summary>Tells what a model supports.</summary>
    public async Task<Capabilities> GetCapabilitiesAsync(string model, CancellationToken cancellationToken = default)
    {
        if (caps.TryGetValue(model, out var cached))
        {
            return cached;
        }
        var op = Operation("capabilities");
        op["model"] = model;
        var result = await RunAsync(op, null, null, cancellationToken).ConfigureAwait(false);
        var node = result?["capabilities"] ?? throw ProviderException.Of("not_found", null, "no capabilities known for " + model);
        var parsed = Json.As<Capabilities>(node);
        caps[model] = parsed;
        return parsed;
    }

    /// <summary>Tells what a model supports, blocking.</summary>
    public Capabilities GetCapabilities(string model, CancellationToken cancellationToken = default)
    {
        return Task.Run(() => GetCapabilitiesAsync(model, cancellationToken), CancellationToken.None).GetAwaiter().GetResult();
    }

    /// <summary>Logs into the provider's session source; the handler shows the user what to do and returns any pasted value.</summary>
    public async Task<AuthStatus> LoginAsync(Func<LoginAction, CancellationToken, Task<string?>> handler, string? callback = null, TimeSpan? timeout = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(handler);
        var op = Operation("login");
        if (callback is not null)
        {
            op["callback"] = callback;
        }
        if (timeout is { } t)
        {
            op["timeout_ms"] = (long)t.TotalMilliseconds;
        }
        var result = await RunAsync(op, null, handler, cancellationToken).ConfigureAwait(false);
        return Json.As<AuthStatus>(result?["status"]);
    }

    /// <summary>Logs in, blocking; the handler shows the user what to do and returns any pasted value.</summary>
    public AuthStatus Login(Func<LoginAction, string?> handler, string? callback = null, TimeSpan? timeout = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(handler);
        return Task.Run(() => LoginAsync((action, _) => Task.FromResult(handler(action)), callback, timeout, cancellationToken), CancellationToken.None).GetAwaiter().GetResult();
    }

    /// <summary>Forgets the provider's own session.</summary>
    public async Task LogoutAsync(CancellationToken cancellationToken = default)
    {
        await RunAsync(Operation("logout"), null, null, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Forgets the provider's own session, blocking.</summary>
    public void Logout(CancellationToken cancellationToken = default)
    {
        Task.Run(() => LogoutAsync(cancellationToken), CancellationToken.None).GetAwaiter().GetResult();
    }

    /// <summary>Reports which credential would be used and when it expires.</summary>
    public async Task<AuthStatus> GetAuthStatusAsync(CancellationToken cancellationToken = default)
    {
        var result = await RunAsync(Operation("auth_status"), null, null, cancellationToken).ConfigureAwait(false);
        var status = result?["status"];
        return status is null ? new AuthStatus() : Json.As<AuthStatus>(status);
    }

    /// <summary>Reports the credential state, blocking.</summary>
    public AuthStatus GetAuthStatus(CancellationToken cancellationToken = default)
    {
        return Task.Run(() => GetAuthStatusAsync(cancellationToken), CancellationToken.None).GetAwaiter().GetResult();
    }
}
