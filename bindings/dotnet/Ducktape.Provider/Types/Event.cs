using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>Unknown event types are delivered as raw; the variants are merged into one record</summary>
public sealed record Event
{
    /// <summary>One of block_start, block_stop, finish, raw, reasoning_delta, text_delta, tool_call_delta; it selects the fields that apply</summary>
    [JsonPropertyName("type")]
    public string? Type { get; init; }

    /// <summary>Reserved for agents</summary>
    [JsonPropertyName("parent_id")]
    public string? ParentId { get; init; }

    [JsonPropertyName("index")]
    public long? Index { get; init; }

    [JsonPropertyName("kind")]
    public string? Kind { get; init; }

    [JsonPropertyName("id")]
    public string? Id { get; init; }

    [JsonPropertyName("name")]
    public string? Name { get; init; }

    [JsonPropertyName("text")]
    public string? Text { get; init; }

    [JsonPropertyName("arguments_json")]
    public string? ArgumentsJson { get; init; }

    [JsonPropertyName("block")]
    public Block? Block { get; init; }

    [JsonPropertyName("response")]
    public Response? Response { get; init; }

    [JsonPropertyName("event")]
    public JsonElement? EventValue { get; init; }
}
