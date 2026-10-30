using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>A nested object</summary>
public sealed record CapabilitiesReasoning
{
    [JsonPropertyName("levels")]
    public List<string>? Levels { get; init; }

    /// <summary>true, false or null = unknown</summary>
    [JsonPropertyName("can_disable")]
    public bool? CanDisable { get; init; }
}
