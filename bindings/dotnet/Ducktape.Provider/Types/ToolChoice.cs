using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>Canonical form; bindings accept the strings auto / none / required and {name} as sugar</summary>
public sealed record ToolChoice
{
    [JsonPropertyName("mode")]
    public string? Mode { get; init; }

    [JsonPropertyName("name")]
    public string? Name { get; init; }
}
