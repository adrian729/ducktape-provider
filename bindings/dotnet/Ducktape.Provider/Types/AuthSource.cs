using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>; the variants are merged into one record</summary>
public sealed record AuthSource
{
    /// <summary>One of env, exchange, inline, resolver, session; it selects the fields that apply</summary>
    [JsonPropertyName("type")]
    public string? Type { get; init; }

    /// <summary>Handle filled through set_secret; never the secret</summary>
    [JsonPropertyName("secret_ref")]
    public string? SecretRef { get; init; }

    [JsonPropertyName("kind")]
    public string? Kind { get; init; }

    [JsonPropertyName("placement")]
    public string? Placement { get; init; }

    [JsonPropertyName("names")]
    public List<string>? Names { get; init; }

    [JsonPropertyName("resolver_id")]
    public string? ResolverId { get; init; }

    [JsonPropertyName("preset")]
    public string? Preset { get; init; }

    [JsonPropertyName("store")]
    public string? Store { get; init; }

    [JsonPropertyName("client_id")]
    public string? ClientId { get; init; }

    [JsonPropertyName("profile")]
    public string? Profile { get; init; }

    [JsonPropertyName("refresh")]
    public string? Refresh { get; init; }

    [JsonPropertyName("grant")]
    public string? Grant { get; init; }

    [JsonPropertyName("params")]
    public Dictionary<string, object>? Params { get; init; }
}
