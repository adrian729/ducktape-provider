using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One EmbedResponse document.</summary>
public sealed record EmbedResponse
{
    [JsonPropertyName("embeddings")]
    public List<List<double>>? Embeddings { get; init; }

    [JsonPropertyName("usage")]
    public EmbedResponseUsage? Usage { get; init; }

    [JsonPropertyName("warnings")]
    public List<Warning>? Warnings { get; init; }

    [JsonPropertyName("origin")]
    public Origin? Origin { get; init; }

    [JsonPropertyName("latency_ms")]
    public double? LatencyMs { get; init; }

    [JsonPropertyName("raw")]
    public JsonElement? Raw { get; init; }
}
