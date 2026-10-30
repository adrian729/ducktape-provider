namespace Ducktape.Types;

/// <summary>Known values of ErrorKind; other strings are allowed.</summary>
public static class ErrorKind
{
    public const string Auth = "auth";
    public const string RateLimit = "rate_limit";
    public const string Quota = "quota";
    public const string Server = "server";
    public const string Connection = "connection";
    public const string Timeout = "timeout";
    public const string Cancelled = "cancelled";
    public const string NotFound = "not_found";
    public const string ContextOverflow = "context_overflow";
    public const string RequestTooLarge = "request_too_large";
    public const string BadRequest = "bad_request";
    public const string MalformedResponse = "malformed_response";
    public const string Unsupported = "unsupported";
}
