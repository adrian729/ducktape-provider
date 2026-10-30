using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>Where a response or an opaque block came from; opaque state is replayed when wire and base_url match the call</summary>
public sealed record Origin
{
    [JsonPropertyName("wire")]
    public string? Wire { get; init; }

    /// <summary>Vendor namespace, company name</summary>
    [JsonPropertyName("ns")]
    public string? Ns { get; init; }

    [JsonPropertyName("base_url")]
    public string? BaseUrl { get; init; }

    [JsonPropertyName("model")]
    public string? Model { get; init; }

    /// <summary>env:NAME, session:PRESET, resolver, inline</summary>
    [JsonPropertyName("auth_source")]
    public string? AuthSource { get; init; }
}
