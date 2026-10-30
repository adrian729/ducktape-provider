using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>A nested object</summary>
public sealed record CapabilitiesOutput
{
    /// <summary>true, false or null = unknown</summary>
    [JsonPropertyName("json_schema")]
    public bool? JsonSchema { get; init; }
}
