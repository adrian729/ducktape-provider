using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>A nested object</summary>
public sealed record EmbedResponseUsage
{
    [JsonPropertyName("input")]
    public EmbedResponseUsageInput? Input { get; init; }
}
