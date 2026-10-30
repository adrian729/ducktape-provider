using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>A nested object</summary>
public sealed record EmbedRequestOptions
{
    [JsonPropertyName("dimensions")]
    public long? Dimensions { get; init; }

    [JsonPropertyName("retries")]
    public long? Retries { get; init; }
}
