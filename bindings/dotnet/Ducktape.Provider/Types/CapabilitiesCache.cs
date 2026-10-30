using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>A nested object</summary>
public sealed record CapabilitiesCache
{
    /// <summary>true, false or null = unknown</summary>
    [JsonPropertyName("breakpoints")]
    public bool? Breakpoints { get; init; }

    [JsonPropertyName("ttls")]
    public List<string>? Ttls { get; init; }
}
