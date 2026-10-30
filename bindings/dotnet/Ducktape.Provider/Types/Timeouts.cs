using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>Seconds; bindings accept a plain number as total</summary>
public sealed record Timeouts
{
    [JsonPropertyName("connect")]
    public double? Connect { get; init; }

    [JsonPropertyName("read")]
    public double? Read { get; init; }

    [JsonPropertyName("total")]
    public double? Total { get; init; }
}
