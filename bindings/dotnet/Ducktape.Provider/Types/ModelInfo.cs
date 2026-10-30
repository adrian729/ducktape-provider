using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One ModelInfo document.</summary>
public sealed record ModelInfo
{
    [JsonPropertyName("id")]
    public string? Id { get; init; }

    [JsonPropertyName("kind")]
    public string? Kind { get; init; }

    [JsonPropertyName("context_window")]
    public long? ContextWindow { get; init; }

    [JsonPropertyName("max_output_tokens")]
    public long? MaxOutputTokens { get; init; }

    [JsonPropertyName("capabilities")]
    public Capabilities? Capabilities { get; init; }

    [JsonPropertyName("raw")]
    public JsonElement? Raw { get; init; }
}
