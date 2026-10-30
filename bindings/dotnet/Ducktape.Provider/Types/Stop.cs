using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One Stop document.</summary>
public sealed record Stop
{
    [JsonPropertyName("reason")]
    public string? Reason { get; init; }

    /// <summary>Vendor's own value</summary>
    [JsonPropertyName("raw")]
    public JsonElement? Raw { get; init; }
}
