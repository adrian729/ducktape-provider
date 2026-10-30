using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>total includes cached tokens</summary>
public sealed record UsageInput
{
    [JsonPropertyName("total")]
    public long? Total { get; init; }

    [JsonPropertyName("cache_read")]
    public long? CacheRead { get; init; }

    [JsonPropertyName("cache_write")]
    public long? CacheWrite { get; init; }
}
