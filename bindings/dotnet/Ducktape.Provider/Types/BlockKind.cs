namespace Ducktape.Types;

/// <summary>Known values of BlockKind; other strings are allowed.</summary>
public static class BlockKind
{
    public const string Text = "text";
    public const string Reasoning = "reasoning";
    public const string ToolCall = "tool_call";
    public const string Compaction = "compaction";
    public const string Custom = "custom";
}
