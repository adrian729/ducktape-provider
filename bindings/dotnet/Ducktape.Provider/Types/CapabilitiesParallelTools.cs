using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>A nested object</summary>
public sealed record CapabilitiesParallelTools
{
    /// <summary>true, false or null = unknown</summary>
    [JsonPropertyName("supported")]
    public bool? Supported { get; init; }

    /// <summary>true, false or null = unknown</summary>
    [JsonPropertyName("toggle")]
    public bool? Toggle { get; init; }
}
