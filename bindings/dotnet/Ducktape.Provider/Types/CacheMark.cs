using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>Cache breakpoint on a block or tool list</summary>
public sealed record CacheMark
{
    [JsonPropertyName("ttl")]
    public string? Ttl { get; init; }
}
