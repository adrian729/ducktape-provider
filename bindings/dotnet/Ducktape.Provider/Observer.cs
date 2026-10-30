using System.Text.Json;

namespace Ducktape;

/// <summary>Receives request, response, event and retry notifications as redacted JSON.</summary>
public sealed class Observer
{
    public Action<JsonElement>? OnRequest { get; init; }
    public Action<JsonElement>? OnResponse { get; init; }
    public Action<JsonElement>? OnEvent { get; init; }
    public Action<JsonElement>? OnRetry { get; init; }
}
