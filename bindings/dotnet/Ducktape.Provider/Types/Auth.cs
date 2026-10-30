using System.Text.Json;
using System.Text.Json.Serialization;

namespace Ducktape.Types;

/// <summary>See Auth model note</summary>
public sealed record Auth
{
    /// <summary>Tried in literal order</summary>
    [JsonPropertyName("sources")]
    public List<AuthSource>? Sources { get; init; }

    /// <summary>Where credentials may be sent; default = base_url origin</summary>
    [JsonPropertyName("origins")]
    public List<string>? Origins { get; init; }

    [JsonPropertyName("extra_headers")]
    public Dictionary<string, string>? ExtraHeaders { get; init; }

    [JsonPropertyName("dangerously_allow_browser")]
    public bool? DangerouslyAllowBrowser { get; init; }
}
