using Ducktape.Types;

namespace Ducktape;

/// <summary>Any failure reported by the core, with the normalized error attached.</summary>
public sealed class ProviderException : Exception
{
    public ProviderException(ProviderError error) : base("[" + error.Kind + (error.Code is null ? "" : " " + error.Code) + "] " + error.Message)
    {
        Error = error;
    }

    public ProviderException(ProviderError error, Exception? inner) : base("[" + error.Kind + (error.Code is null ? "" : " " + error.Code) + "] " + error.Message, inner)
    {
        Error = error;
    }

    /// <summary>The normalized error.</summary>
    public ProviderError Error { get; }

    /// <summary>The error kind, the stable discriminator.</summary>
    public string Kind => Error.Kind ?? "other";

    /// <summary>The vendor-specific code, if any.</summary>
    public string? Code => Error.Code;

    /// <summary>The HTTP status, if any.</summary>
    public long? Status => Error.Status;

    /// <summary>Whether a retry may succeed.</summary>
    public bool Retryable => Error.Retryable == true;

    /// <summary>Whether this error is of the given kind.</summary>
    public bool Is(string kind)
    {
        return Kind == kind;
    }

    internal static ProviderError Build(string kind, string? code, string message)
    {
        return new ProviderError { Kind = kind, Code = code, Message = message, Retryable = false, Attempts = 0 };
    }

    internal static ProviderException Of(string kind, string? code, string message)
    {
        return new ProviderException(Build(kind, code, message));
    }
}
