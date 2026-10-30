using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>A nested object</summary>
public sealed record CapabilitiesTools
{
    /// <summary>true, false or null = unknown</summary>
    [JsonPropertyName("functions")]
    public bool? Functions { get; init; }

    [JsonPropertyName("hosted")]
    public List<string>? Hosted { get; init; }
}
