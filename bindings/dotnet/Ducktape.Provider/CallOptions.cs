using Ducktape.Types;

namespace Ducktape;

/// <summary>Settings for a provider.</summary>
public sealed class ProviderOptions
{
    public string? BaseUrl { get; init; }
    public IReadOnlyList<Credential>? Auth { get; init; }
    public Dictionary<string, string>? Headers { get; init; }
    public Timeouts? Timeout { get; init; }
    public string? DefaultModel { get; init; }
    public string? DefaultEmbeddingModel { get; init; }
    public string? DefaultSystem { get; init; }
    public List<Block>? DefaultSystemBlocks { get; init; }
    public List<Tool>? DefaultTools { get; init; }
    public Options? DefaultOptions { get; init; }
    public Dictionary<string, Dictionary<string, object>>? DefaultVendor { get; init; }
    public Overrides? Overrides { get; init; }
    public bool? Strict { get; init; }
    public Observer? Observer { get; init; }
    public HttpClient? Client { get; init; }
    public KeychainReader? Keychain { get; init; }
    public KeychainWriter? KeychainWrite { get; init; }
}

/// <summary>Per-call settings for chat and stream.</summary>
public sealed class ChatOptions
{
    public string? Model { get; init; }
    public string? System { get; init; }
    public List<Block>? SystemBlocks { get; init; }
    public List<Tool>? Tools { get; init; }
    public Options? Options { get; init; }
    public Dictionary<string, Dictionary<string, object>>? Vendor { get; init; }
    public Timeouts? Timeout { get; init; }
    public Dictionary<string, string>? Headers { get; init; }
    public bool? Strict { get; init; }
    public bool? IncludeRaw { get; init; }
}

/// <summary>Per-call settings for embed.</summary>
public sealed class EmbedOptions
{
    public string? Model { get; init; }
    public long? Dimensions { get; init; }
    public long? Retries { get; init; }
    public Dictionary<string, Dictionary<string, object>>? Vendor { get; init; }
    public Timeouts? Timeout { get; init; }
    public bool? IncludeRaw { get; init; }
}
