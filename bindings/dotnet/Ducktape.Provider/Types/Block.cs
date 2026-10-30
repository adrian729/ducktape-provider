using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>Unknown types decode as CustomBlock; the variants are merged into one record</summary>
public sealed record Block
{
    /// <summary>One of compaction, custom, document, image, reasoning, text, tool_call, tool_result; it selects the fields that apply</summary>
    [JsonPropertyName("type")]
    public string? Type { get; init; }

    [JsonPropertyName("id")]
    public string? Id { get; init; }

    [JsonPropertyName("cache")]
    public CacheMark? Cache { get; init; }

    [JsonPropertyName("vendor")]
    public Dictionary<string, Dictionary<string, object>>? Vendor { get; init; }

    [JsonPropertyName("provider_executed")]
    public bool? ProviderExecuted { get; init; }

    [JsonPropertyName("text")]
    public string? Text { get; init; }

    [JsonPropertyName("data")]
    public string? Data { get; init; }

    [JsonPropertyName("url")]
    public string? Url { get; init; }

    [JsonPropertyName("file_id")]
    public string? FileId { get; init; }

    [JsonPropertyName("media_type")]
    public string? MediaType { get; init; }

    [JsonPropertyName("name")]
    public string? Name { get; init; }

    [JsonPropertyName("arguments")]
    public Dictionary<string, object>? Arguments { get; init; }

    [JsonPropertyName("truncated")]
    public bool? Truncated { get; init; }

    /// <summary>Raw arguments text when parsing failed or was cut</summary>
    [JsonPropertyName("arguments_json")]
    public string? ArgumentsJson { get; init; }

    /// <summary>Reserved for agents</summary>
    [JsonPropertyName("status")]
    public string? Status { get; init; }

    [JsonPropertyName("tool_call_id")]
    public string? ToolCallId { get; init; }

    [JsonPropertyName("content")]
    public List<Block>? Content { get; init; }

    [JsonPropertyName("is_error")]
    public bool? IsError { get; init; }

    /// <summary>Opaque</summary>
    [JsonPropertyName("vendor_state")]
    public JsonElement? VendorState { get; init; }

    [JsonPropertyName("origin")]
    public Origin? Origin { get; init; }

    /// <summary>&lt;ns&gt;.&lt;type&gt;</summary>
    [JsonPropertyName("kind")]
    public string? Kind { get; init; }
}
