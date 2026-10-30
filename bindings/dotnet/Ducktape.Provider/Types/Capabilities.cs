using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>One Capabilities document.</summary>
public sealed record Capabilities
{
    [JsonPropertyName("reasoning")]
    public CapabilitiesReasoning? Reasoning { get; init; }

    [JsonPropertyName("tool_choice")]
    public List<string>? ToolChoice { get; init; }

    [JsonPropertyName("parallel_tools")]
    public CapabilitiesParallelTools? ParallelTools { get; init; }

    [JsonPropertyName("output")]
    public CapabilitiesOutput? Output { get; init; }

    [JsonPropertyName("cache")]
    public CapabilitiesCache? Cache { get; init; }

    [JsonPropertyName("inputs")]
    public List<string>? Inputs { get; init; }

    [JsonPropertyName("tools")]
    public CapabilitiesTools? Tools { get; init; }

    [JsonPropertyName("limits")]
    public CapabilitiesLimits? Limits { get; init; }

    /// <summary>true, false or null = unknown</summary>
    [JsonPropertyName("embeddings")]
    public bool? Embeddings { get; init; }

    [JsonPropertyName("source")]
    public string? Source { get; init; }
}
