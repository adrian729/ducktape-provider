namespace Ducktape.Types;

/// <summary>Known values of Block.status; other strings are allowed.</summary>
public static class BlockStatus
{
    public const string Pending = "pending";
    public const string InProgress = "in_progress";
    public const string Completed = "completed";
    public const string Failed = "failed";
    public const string Cancelled = "cancelled";
}
