using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>A nested object</summary>
public sealed record CapabilitiesLimits
{
    [JsonPropertyName("context_window")]
    public long? ContextWindow { get; init; }

    [JsonPropertyName("max_output_tokens")]
    public long? MaxOutputTokens { get; init; }
}
