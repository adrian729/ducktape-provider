using System.Reflection;
using System.Runtime.InteropServices;

namespace Ducktape;

internal static unsafe partial class Native
{
    private const string Lib = "ducktape_abi";

    internal static string? LibraryPath;

    static Native()
    {
        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolve);
    }

    private static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (name != Lib)
        {
            return IntPtr.Zero;
        }
        foreach (var candidate in Candidates())
        {
            if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out var handle))
            {
                return handle;
            }
        }
        return IntPtr.Zero;
    }

    private static IEnumerable<string> Candidates()
    {
        if (LibraryPath is not null)
        {
            yield return LibraryPath;
        }
        var env = Environment.GetEnvironmentVariable("DUCKTAPE_NATIVE");
        if (!string.IsNullOrEmpty(env))
        {
            yield return env;
        }
        var file = OperatingSystem.IsWindows() ? "ducktape_abi.dll" : OperatingSystem.IsMacOS() ? "libducktape_abi.dylib" : "libducktape_abi.so";
        yield return Path.Combine(AppContext.BaseDirectory, "runtimes", RuntimeInformation.RuntimeIdentifier, "native", file);
        yield return Path.Combine(AppContext.BaseDirectory, file);
    }

    [LibraryImport(Lib, EntryPoint = "ducktape_abi_version")]
    internal static partial uint AbiVersion();

    [LibraryImport(Lib, EntryPoint = "ducktape_free")]
    internal static partial void Free(byte* ptr, nuint len);

    [LibraryImport(Lib, EntryPoint = "ducktape_op_start")]
    internal static partial IntPtr OpStart(byte* json, nuint len, byte** output, nuint* outputLen);

    [LibraryImport(Lib, EntryPoint = "ducktape_op_resume")]
    internal static partial int OpResume(IntPtr op, byte* json, nuint len, byte** output, nuint* outputLen);

    [LibraryImport(Lib, EntryPoint = "ducktape_op_feed")]
    internal static partial int OpFeed(IntPtr op, byte* bytes, nuint bytesLen, byte* clock, nuint clockLen, byte** output, nuint* outputLen);

    [LibraryImport(Lib, EntryPoint = "ducktape_op_set_secret")]
    internal static partial int OpSetSecret(IntPtr op, byte* secretRef, nuint refLen, byte* bytes, nuint len);

    [LibraryImport(Lib, EntryPoint = "ducktape_op_free")]
    internal static partial void OpFree(IntPtr op);
}
