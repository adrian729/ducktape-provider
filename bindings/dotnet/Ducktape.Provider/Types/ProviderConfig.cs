using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>What a binding hands the core for one Provider</summary>
public sealed record ProviderConfig
{
    /// <summary>Vendor, company name</summary>
    [JsonPropertyName("vendor")]
    public string? Vendor { get; init; }

    [JsonPropertyName("base_url")]
    public string? BaseUrl { get; init; }

    [JsonPropertyName("auth")]
    public Auth? Auth { get; init; }

    [JsonPropertyName("headers")]
    public Dictionary<string, string>? Headers { get; init; }

    [JsonPropertyName("timeout")]
    public Timeouts? Timeout { get; init; }

    [JsonPropertyName("default_model")]
    public string? DefaultModel { get; init; }

    [JsonPropertyName("default_embedding_model")]
    public string? DefaultEmbeddingModel { get; init; }

    [JsonPropertyName("default_system")]
    public List<Block>? DefaultSystem { get; init; }

    [JsonPropertyName("default_tools")]
    public List<Tool>? DefaultTools { get; init; }

    [JsonPropertyName("default_options")]
    public Options? DefaultOptions { get; init; }

    [JsonPropertyName("default_vendor")]
    public Dictionary<string, Dictionary<string, object>>? DefaultVendor { get; init; }

    [JsonPropertyName("overrides")]
    public Overrides? Overrides { get; init; }

    [JsonPropertyName("strict")]
    public bool? Strict { get; init; }
}
