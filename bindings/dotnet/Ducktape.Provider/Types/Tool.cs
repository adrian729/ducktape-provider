using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>; the variants are merged into one record</summary>
public sealed record Tool
{
    /// <summary>One of function, provider; it selects the fields that apply</summary>
    [JsonPropertyName("type")]
    public string? Type { get; init; }

    [JsonPropertyName("name")]
    public string? Name { get; init; }

    [JsonPropertyName("description")]
    public string? Description { get; init; }

    [JsonPropertyName("input_schema")]
    public Dictionary<string, object>? InputSchema { get; init; }

    [JsonPropertyName("strict")]
    public bool? Strict { get; init; }

    [JsonPropertyName("cache")]
    public CacheMark? Cache { get; init; }

    [JsonPropertyName("vendor")]
    public Dictionary<string, Dictionary<string, object>>? Vendor { get; init; }

    [JsonPropertyName("provider_tool")]
    public string? ProviderTool { get; init; }

    [JsonPropertyName("config")]
    public Dictionary<string, object>? Config { get; init; }
}
