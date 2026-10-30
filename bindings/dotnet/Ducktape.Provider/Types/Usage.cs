using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One Usage document.</summary>
public sealed record Usage
{
    /// <summary>total includes cached tokens</summary>
    [JsonPropertyName("input")]
    public UsageInput? Input { get; init; }

    [JsonPropertyName("output")]
    public UsageOutput? Output { get; init; }

    [JsonPropertyName("raw")]
    public JsonElement? Raw { get; init; }
}
