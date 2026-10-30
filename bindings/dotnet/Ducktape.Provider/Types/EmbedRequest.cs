using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One EmbedRequest document.</summary>
public sealed record EmbedRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("input")]
    public List<string>? Input { get; init; }

    [JsonPropertyName("options")]
    public EmbedRequestOptions? Options { get; init; }

    [JsonPropertyName("vendor")]
    public Dictionary<string, Dictionary<string, object>>? Vendor { get; init; }

    [JsonPropertyName("timeout")]
    public Timeouts? Timeout { get; init; }

    [JsonPropertyName("include_raw")]
    public bool? IncludeRaw { get; init; }
}
