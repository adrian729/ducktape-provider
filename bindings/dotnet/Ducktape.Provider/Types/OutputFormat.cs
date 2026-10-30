using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One OutputFormat document.</summary>
public sealed record OutputFormat
{
    [JsonPropertyName("json_schema")]
    public Dictionary<string, object>? JsonSchema { get; init; }

    [JsonPropertyName("name")]
    public string? Name { get; init; }
}
