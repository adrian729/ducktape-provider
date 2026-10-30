namespace Ducktape.Types;

/// <summary>Known values of AuthSource.refresh; other strings are allowed.</summary>
public static class AuthSourceRefresh
{
    public const string Never = "never";
    public const string Memory = "memory";
    public const string WriteBack = "write_back";
}
