using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>Portable options; everything else goes through vendor</summary>
public sealed record Options
{
    [JsonPropertyName("max_output_tokens")]
    public long? MaxOutputTokens { get; init; }

    [JsonPropertyName("reasoning")]
    public string? Reasoning { get; init; }

    [JsonPropertyName("reasoning_summary")]
    public string? ReasoningSummary { get; init; }

    [JsonPropertyName("tool_choice")]
    public ToolChoice? ToolChoice { get; init; }

    [JsonPropertyName("parallel_tools")]
    public bool? ParallelTools { get; init; }

    [JsonPropertyName("output")]
    public OutputFormat? Output { get; init; }

    [JsonPropertyName("stop_sequences")]
    public List<string>? StopSequences { get; init; }

    [JsonPropertyName("cache")]
    public string? Cache { get; init; }

    [JsonPropertyName("retries")]
    public long? Retries { get; init; }

    [JsonPropertyName("user_id")]
    public string? UserId { get; init; }
}
