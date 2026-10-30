using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One Warning document.</summary>
public sealed record Warning
{
    [JsonPropertyName("kind")]
    public string? Kind { get; init; }

    [JsonPropertyName("feature")]
    public string? Feature { get; init; }

    [JsonPropertyName("detail")]
    public string? Detail { get; init; }
}
