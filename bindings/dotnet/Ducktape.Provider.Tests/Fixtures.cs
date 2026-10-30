using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Ducktape.Tests;

internal static class Fixtures
{
    public static readonly string Wire = FindWire();
    public static readonly string Entropy = Convert.ToBase64String(Enumerable.Repeat((byte)128, 32).ToArray());

    private static string FindWire()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            var candidate = Path.Combine(dir.FullName, "fixtures", "wire");
            if (Directory.Exists(candidate))
            {
                return candidate;
            }
        }
        throw new InvalidOperationException("fixtures/wire not found above " + AppContext.BaseDirectory);
    }

    public static JsonObject Clock(long tick)
    {
        return new JsonObject { ["now_ms"] = 1_761_696_000_000L + tick * 100, ["entropy"] = Entropy };
    }

    public static JsonNode Load(string path)
    {
        return JsonNode.Parse(File.ReadAllBytes(path))!;
    }

    public static JsonNode Load(string wire, string name)
    {
        return Load(Path.Combine(Wire, wire, name + ".json"));
    }

    public static List<string> All()
    {
        return Directory.GetDirectories(Wire).Order(StringComparer.Ordinal).SelectMany(dir => Directory.GetFiles(dir, "*.json").Order(StringComparer.Ordinal)).ToList();
    }

    public static string Encode(string text)
    {
        return Convert.ToBase64String(Encoding.UTF8.GetBytes(text));
    }

    private static JsonObject Input(string type)
    {
        return new JsonObject { ["type"] = type };
    }

    private static string? Str(JsonNode? node)
    {
        return node is JsonValue v && v.TryGetValue<string>(out var s) ? s : null;
    }

    /// <summary>Replays a fixture's script through the core and returns the exchange it produced.</summary>
    public static List<(JsonNode Step, JsonNode? Input)> Replay(JsonNode fixture)
    {
        var name = Str(fixture["name"]);
        var output = new List<(JsonNode, JsonNode?)>();
        var start = fixture["start"]!.DeepClone();
        start["clock"] = Clock(0);
        var (op, step) = Core.Operation.Start(start);
        using (op)
        {
            if (op is null)
            {
                output.Add((step, null));
                return output;
            }
            var http = 0;
            var writes = 0;
            var userActions = 0;
            var callbacks = 0;
            var queue = new List<JsonObject>();
            for (long tick = 1; tick < 10_000; tick++)
            {
                var effect = step["effect"]!;
                var type = Str(effect["type"]);
                JsonObject input;
                switch (type)
                {
                    case "done":
                    case "failed":
                        output.Add((step, null));
                        return output;
                    case "need_secret":
                        var secretRef = Str(effect["secret_ref"])!;
                        var secret = Str(fixture["secrets"]?[secretRef]);
                        if (secret is not null)
                        {
                            op.SetSecret(secretRef, Encoding.UTF8.GetBytes(secret));
                        }
                        input = new JsonObject { ["type"] = "secret", ["secret_ref"] = secretRef, ["found"] = secret is not null };
                        break;
                    case "need_http":
                        var h = (fixture["http"] as JsonArray)?.ElementAtOrDefault(http++) ?? throw new InvalidOperationException(name + ": script has no HTTP reply for request");
                        if (h["error"] is not null)
                        {
                            input = new JsonObject { ["type"] = "http_error", ["error"] = h["error"]!.DeepClone() };
                        }
                        else
                        {
                            var chunks = new List<string>();
                            if (h["chunks"] is JsonArray { Count: > 0 } array)
                            {
                                chunks.AddRange(array.Select(c => Str(c)!));
                            }
                            else if (h["body"] is not null)
                            {
                                chunks.Add(Str(h["body"])!);
                            }
                            queue.Clear();
                            queue.Add(Input("http_end"));
                            for (var i = chunks.Count - 1; i >= 0; i--)
                            {
                                queue.Add(new JsonObject { ["type"] = "http_chunk", ["bytes"] = Encode(chunks[i]) });
                            }
                            input = new JsonObject { ["type"] = "http_response", ["status"] = h["status"]?.GetValue<int>() ?? 200, ["headers"] = h["headers"]?.DeepClone() ?? new JsonObject() };
                        }
                        break;
                    case "receive":
                        if (queue.Count == 0)
                        {
                            input = Input("http_end");
                        }
                        else
                        {
                            input = queue[^1];
                            queue.RemoveAt(queue.Count - 1);
                        }
                        break;
                    case "need_sleep":
                        input = Input("slept");
                        break;
                    case "need_read":
                        var loc = effect["locator"]!;
                        var locType = Str(loc["type"]);
                        var value = locType switch
                        {
                            "file" => Str(fixture["files"]?[Str(loc["path"])!]),
                            "env" => Str(fixture["env"]?[Str(loc["name"])!]),
                            _ => Str(fixture["keychain"]?[Str(loc["service"]) + "|" + Str(loc["account"])]),
                        };
                        input = new JsonObject { ["type"] = "read", ["found"] = value is not null };
                        if (value is not null)
                        {
                            input["bytes"] = Encode(value);
                            if (locType == "file")
                            {
                                input["etag"] = Driver.Etag(Encoding.UTF8.GetBytes(value));
                            }
                        }
                        break;
                    case "need_write":
                        var w = (fixture["writes"] as JsonArray)?.ElementAtOrDefault(writes++);
                        input = new JsonObject { ["type"] = "written", ["ok"] = w?["ok"]?.GetValue<bool>() ?? true, ["conflict"] = w?["conflict"]?.GetValue<bool>() ?? false };
                        break;
                    case "need_user_action":
                        var v = (fixture["user_actions"] as JsonArray)?.ElementAtOrDefault(userActions++);
                        input = new JsonObject { ["type"] = "user_action" };
                        if (v is not null)
                        {
                            input["value"] = v.GetValue<string>();
                        }
                        break;
                    case "need_callback":
                        var url = (fixture["callbacks"] as JsonArray)?.ElementAtOrDefault(callbacks++) ?? throw new InvalidOperationException(name + ": script has no callback for the login");
                        input = new JsonObject { ["type"] = "callback", ["url"] = url.GetValue<string>() };
                        break;
                    default:
                        throw new InvalidOperationException(name + ": unsupported effect in replay: " + type);
                }
                output.Add((step, input));
                step = op.Resume(new JsonObject { ["clock"] = Clock(tick), ["input"] = input.DeepClone() });
            }
        }
        throw new InvalidOperationException(name + ": replay did not finish");
    }

    private static JsonNode? Sorted(JsonNode? node)
    {
        switch (node)
        {
            case JsonObject o:
                var sorted = new JsonObject();
                foreach (var key in o.Select(p => p.Key).Order(StringComparer.Ordinal))
                {
                    sorted[key] = Sorted(o[key]);
                }
                return sorted;
            case JsonArray a:
                return new JsonArray(a.Select(Sorted).ToArray());
            default:
                return node?.DeepClone();
        }
    }

    /// <summary>A step with its request body parsed and keys sorted, so JSON key order does not matter.</summary>
    public static string Canonical(JsonNode? node)
    {
        if (node is null)
        {
            return "null";
        }
        var copy = node.DeepClone();
        if (copy["effect"]?["request"] is JsonObject request && Str(request["body"]) is { } body)
        {
            try
            {
                request["body"] = JsonNode.Parse(body);
            }
            catch (JsonException)
            {
            }
        }
        return Sorted(copy)!.ToJsonString();
    }
}
