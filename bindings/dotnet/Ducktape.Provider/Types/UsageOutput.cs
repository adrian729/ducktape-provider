using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>A nested object</summary>
public sealed record UsageOutput
{
    [JsonPropertyName("total")]
    public long? Total { get; init; }

    [JsonPropertyName("reasoning")]
    public long? Reasoning { get; init; }
}
