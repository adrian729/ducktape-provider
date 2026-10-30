using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One Overrides document.</summary>
public sealed record Overrides
{
    [JsonPropertyName("api_version")]
    public string? ApiVersion { get; init; }

    [JsonPropertyName("capabilities")]
    public Capabilities? Capabilities { get; init; }

    [JsonPropertyName("dialect")]
    public Dictionary<string, object>? Dialect { get; init; }
}
