using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>; the variants are merged into one record</summary>
public sealed record LoginAction
{
    /// <summary>One of open_url, paste, show_code; it selects the fields that apply</summary>
    [JsonPropertyName("type")]
    public string? Type { get; init; }

    [JsonPropertyName("url")]
    public string? Url { get; init; }

    [JsonPropertyName("instructions")]
    public string? Instructions { get; init; }

    [JsonPropertyName("code")]
    public string? Code { get; init; }

    [JsonPropertyName("prompt")]
    public string? Prompt { get; init; }
}
