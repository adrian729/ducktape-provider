using Ducktape.Types;

namespace Ducktape;

/// <summary>Builders for messages and blocks.</summary>
public static class Messages
{
    /// <summary>A text block.</summary>
    public static Block Text(string text, string? cache = null)
    {
        return new Block { Type = "text", Text = text, Cache = cache is null ? null : new CacheMark { Ttl = cache } };
    }

    /// <summary>An image from bytes.</summary>
    public static Block Image(byte[] data, string mediaType)
    {
        return new Block { Type = "image", Data = Convert.ToBase64String(data), MediaType = mediaType };
    }

    /// <summary>An image at a URL.</summary>
    public static Block ImageUrl(string url)
    {
        return new Block { Type = "image", Url = url };
    }

    /// <summary>A document from bytes.</summary>
    public static Block Document(byte[] data, string mediaType, string? name = null)
    {
        return new Block { Type = "document", Data = Convert.ToBase64String(data), MediaType = mediaType, Name = name };
    }

    /// <summary>A document at a URL.</summary>
    public static Block DocumentUrl(string url, string? name = null)
    {
        return new Block { Type = "document", Url = url, Name = name };
    }

    /// <summary>A tool result with text content.</summary>
    public static Block ToolResult(string toolCallId, string text, bool isError = false)
    {
        return new Block { Type = "tool_result", ToolCallId = toolCallId, Content = [Text(text)], IsError = isError ? true : null };
    }

    /// <summary>A tool result with block content.</summary>
    public static Block ToolResult(string toolCallId, IEnumerable<Block> content, bool isError = false)
    {
        return new Block { Type = "tool_result", ToolCallId = toolCallId, Content = [.. content], IsError = isError ? true : null };
    }

    private static List<Block> Blocks(object[] parts)
    {
        var blocks = new List<Block>(parts.Length);
        foreach (var part in parts)
        {
            blocks.Add(part switch
            {
                string s => Text(s),
                Block b => b,
                _ => throw new ArgumentException("a message part must be a string or a Block, not " + part?.GetType().Name),
            });
        }
        return blocks;
    }

    /// <summary>A user message from strings and blocks.</summary>
    public static Message User(params object[] parts)
    {
        return new Message { Role = Role.User, Content = Blocks(parts) };
    }

    /// <summary>A system message from strings and blocks.</summary>
    public static Message SystemMessage(params object[] parts)
    {
        return new Message { Role = Role.System, Content = Blocks(parts) };
    }

    /// <summary>An assistant message from strings and blocks.</summary>
    public static Message Assistant(params object[] parts)
    {
        return new Message { Role = Role.Assistant, Content = Blocks(parts) };
    }

    /// <summary>An assistant message carrying a reply as it was.</summary>
    public static Message Assistant(Response response)
    {
        return new Message { Role = Role.Assistant, Content = [.. response.Content ?? []] };
    }

    /// <summary>A tool message carrying tool results.</summary>
    public static Message ToolMessage(params Block[] results)
    {
        return new Message { Role = Role.Tool, Content = [.. results] };
    }
}

/// <summary>Builders for tools.</summary>
public static class Tools
{
    /// <summary>A function tool with a JSON schema for its input.</summary>
    public static Tool Function(string name, string description, Dictionary<string, object> inputSchema, bool? strict = null)
    {
        return new Tool { Type = "function", Name = name, Description = description, InputSchema = inputSchema, Strict = strict };
    }

    /// <summary>A function tool with its input schema given as JSON text.</summary>
    public static Tool Function(string name, string description, string inputSchemaJson, bool? strict = null)
    {
        var schema = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(inputSchemaJson, Json.Options) ?? throw new ArgumentException("the input schema must be a JSON object", nameof(inputSchemaJson));
        return Function(name, description, schema, strict);
    }
}

/// <summary>Views over a reply.</summary>
public static class ResponseExtensions
{
    /// <summary>The text blocks joined.</summary>
    public static string Text(this Response response)
    {
        return string.Concat((response.Content ?? []).Where(b => b.Type == "text").Select(b => b.Text));
    }

    /// <summary>The reasoning blocks joined.</summary>
    public static string Reasoning(this Response response)
    {
        return string.Concat((response.Content ?? []).Where(b => b.Type == "reasoning").Select(b => b.Text));
    }

    /// <summary>The tool calls in order.</summary>
    public static List<Block> ToolCalls(this Response response)
    {
        return (response.Content ?? []).Where(b => b.Type == "tool_call").ToList();
    }
}
