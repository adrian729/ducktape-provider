namespace Ducktape.Types;

/// <summary>Known values of AuthSource.kind; other strings are allowed.</summary>
public static class AuthSourceKind
{
    public const string None = "none";
    public const string ApiKey = "api_key";
    public const string Bearer = "bearer";
    public const string Sigv4 = "sigv4";
}
