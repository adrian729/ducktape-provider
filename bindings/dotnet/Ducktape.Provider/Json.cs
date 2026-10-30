using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

namespace Ducktape;

internal static class Json
{
    public static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.General)
    {
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    };

    public static T As<T>(JsonNode? node)
    {
        return node is null ? throw new ProviderException(ProviderException.Build("malformed_response", null, "the core returned no " + typeof(T).Name)) : node.Deserialize<T>(Options)!;
    }

    public static JsonNode? ToNode<T>(T value)
    {
        return JsonSerializer.SerializeToNode(value, Options);
    }

    public static JsonElement ToElement(JsonNode? node)
    {
        return node is null ? default : node.Deserialize<JsonElement>(Options);
    }

    public static string? Str(JsonNode? node)
    {
        return node is JsonValue v && v.TryGetValue<string>(out var s) ? s : null;
    }
}
