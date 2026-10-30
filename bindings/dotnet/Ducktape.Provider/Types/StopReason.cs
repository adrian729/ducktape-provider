namespace Ducktape.Types;

/// <summary>Known values of StopReason; other strings are allowed.</summary>
public static class StopReason
{
    public const string Complete = "complete";
    public const string ToolCall = "tool_call";
    public const string MaxOutputTokens = "max_output_tokens";
    public const string StopSequence = "stop_sequence";
    public const string ContextWindowExceeded = "context_window_exceeded";
    public const string Refusal = "refusal";
    public const string ContentFiltered = "content_filtered";
    public const string ContinueRequired = "continue_required";
    public const string Compaction = "compaction";
    public const string MalformedOutput = "malformed_output";
    public const string Other = "other";
}
