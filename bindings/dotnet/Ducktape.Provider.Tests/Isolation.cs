using System.Runtime.CompilerServices;

namespace Ducktape.Tests;

internal static class Isolation
{
    [ModuleInitializer]
    internal static void Init()
    {
        var home = Path.Combine(Path.GetTempPath(), "ducktape-dotnet-test-home-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(home);
        Environment.SetEnvironmentVariable("HOME", home);
        Environment.SetEnvironmentVariable("USERPROFILE", home);
        Environment.SetEnvironmentVariable("CODEX_HOME", Path.Combine(home, ".codex"));
        Environment.SetEnvironmentVariable("ANTHROPIC_CONFIG_DIR", Path.Combine(home, ".config", "anthropic"));
        Environment.SetEnvironmentVariable("CLAUDE_CONFIG_DIR", Path.Combine(home, ".claude"));
    }
}
