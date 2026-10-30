using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One ProviderError document.</summary>
public sealed record ProviderError
{
    [JsonPropertyName("kind")]
    public string? Kind { get; init; }

    [JsonPropertyName("message")]
    public string? Message { get; init; }

    [JsonPropertyName("status")]
    public long? Status { get; init; }

    [JsonPropertyName("code")]
    public string? Code { get; init; }

    [JsonPropertyName("retryable")]
    public bool? Retryable { get; init; }

    [JsonPropertyName("retry_after")]
    public double? RetryAfter { get; init; }

    [JsonPropertyName("attempts")]
    public long? Attempts { get; init; }

    /// <summary>Redacted vendor error body</summary>
    [JsonPropertyName("body")]
    public JsonElement? Body { get; init; }

    [JsonPropertyName("request_id")]
    public string? RequestId { get; init; }

    [JsonPropertyName("origin")]
    public Origin? Origin { get; init; }
}
