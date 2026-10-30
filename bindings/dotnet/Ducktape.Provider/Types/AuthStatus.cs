using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One AuthStatus document.</summary>
public sealed record AuthStatus
{
    [JsonPropertyName("source")]
    public string? Source { get; init; }

    [JsonPropertyName("expires_at")]
    public string? ExpiresAt { get; init; }

    [JsonPropertyName("session")]
    public string? Session { get; init; }
}
