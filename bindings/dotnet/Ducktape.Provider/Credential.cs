using Ducktape.Types;

namespace Ducktape;

/// <summary>One place a credential comes from.</summary>
public sealed class Credential
{
    internal readonly AuthSource Source;
    internal readonly string? Secret;
    internal readonly Func<CancellationToken, Task<string>>? Fetch;

    private Credential(AuthSource source, string? secret, Func<CancellationToken, Task<string>>? fetch)
    {
        Source = source;
        Secret = secret;
        Fetch = fetch;
    }

    /// <summary>A key given in code.</summary>
    public static Credential Key(string secret)
    {
        return new Credential(new AuthSource { Type = "inline" }, secret ?? throw new ArgumentNullException(nameof(secret)), null);
    }

    /// <summary>The first non-empty environment variable among the names.</summary>
    public static Credential Env(params string[] names)
    {
        if (names is null || names.Length == 0)
        {
            throw new ArgumentException("env needs at least one variable name", nameof(names));
        }
        return new Credential(new AuthSource { Type = "env", Names = [.. names] }, null, null);
    }

    /// <summary>An existing login of another tool, or the library's own session.</summary>
    public static Credential Session(string preset, string? store = null, string? clientId = null, string? profile = null, string? refresh = null)
    {
        return new Credential(new AuthSource { Type = "session", Preset = preset, Store = store, ClientId = clientId, Profile = profile, Refresh = refresh }, null, null);
    }

    /// <summary>A token fetched at request time.</summary>
    public static Credential Resolver(Func<CancellationToken, Task<string>> fetch, string? kind = null, string? placement = null)
    {
        return new Credential(new AuthSource { Type = "resolver", Kind = kind, Placement = placement }, null, fetch ?? throw new ArgumentNullException(nameof(fetch)));
    }

    /// <summary>A token fetched synchronously at request time.</summary>
    public static Credential Resolver(Func<string> fetch, string? kind = null, string? placement = null)
    {
        ArgumentNullException.ThrowIfNull(fetch);
        return Resolver(_ => Task.FromResult(fetch()), kind, placement);
    }

    /// <summary>A source in the schema's own form.</summary>
    public static Credential From(AuthSource source)
    {
        return new Credential(source ?? throw new ArgumentNullException(nameof(source)), null, null);
    }

    /// <summary>The same credential sent as another header kind.</summary>
    public Credential WithKind(string kind, string? placement = null)
    {
        return new Credential(Source with { Kind = kind, Placement = placement ?? Source.Placement }, Secret, Fetch);
    }
}
