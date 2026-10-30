using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One Response document.</summary>
public sealed record Response
{
    [JsonPropertyName("content")]
    public List<Block>? Content { get; init; }

    [JsonPropertyName("stop")]
    public Stop? Stop { get; init; }

    [JsonPropertyName("usage")]
    public Usage? Usage { get; init; }

    [JsonPropertyName("warnings")]
    public List<Warning>? Warnings { get; init; }

    [JsonPropertyName("origin")]
    public Origin? Origin { get; init; }

    [JsonPropertyName("latency_ms")]
    public double? LatencyMs { get; init; }

    [JsonPropertyName("first_token_ms")]
    public double? FirstTokenMs { get; init; }

    /// <summary>Vendor reply body; only with include_raw</summary>
    [JsonPropertyName("raw")]
    public JsonElement? Raw { get; init; }
}
