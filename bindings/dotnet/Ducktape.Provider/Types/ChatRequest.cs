using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One ChatRequest document.</summary>
public sealed record ChatRequest
{
    [JsonPropertyName("model")]
    public string? Model { get; init; }

    [JsonPropertyName("messages")]
    public List<Message>? Messages { get; init; }

    [JsonPropertyName("system")]
    public List<Block>? System { get; init; }

    [JsonPropertyName("tools")]
    public List<Tool>? Tools { get; init; }

    [JsonPropertyName("options")]
    public Options? Options { get; init; }

    [JsonPropertyName("vendor")]
    public Dictionary<string, Dictionary<string, object>>? Vendor { get; init; }

    [JsonPropertyName("timeout")]
    public Timeouts? Timeout { get; init; }

    [JsonPropertyName("headers")]
    public Dictionary<string, string>? Headers { get; init; }

    [JsonPropertyName("strict")]
    public bool? Strict { get; init; }

    [JsonPropertyName("include_raw")]
    public bool? IncludeRaw { get; init; }
}
