using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Ducktape;

internal static unsafe partial class Native
{
    private const string Lib = "ducktape_abi";

    private static readonly object Gate = new();
    private static string? libraryPath;
    private static bool loaded;

    static Native()
    {
        NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, Resolve);
    }

    internal static void Load(string path)
    {
        lock (Gate)
        {
            if (loaded)
            {
                throw new InvalidOperationException("the native core is already loaded; call Core.Load before the first operation");
            }
            libraryPath = Path.GetFullPath(path);
        }
    }

    private static IntPtr Resolve(string name, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (name != Lib)
        {
            return IntPtr.Zero;
        }
        lock (Gate)
        {
            if (libraryPath is not null)
            {
                var handle = NativeLibrary.Load(libraryPath);
                loaded = true;
                return handle;
            }
            foreach (var candidate in Candidates())
            {
                if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out var handle))
                {
                    loaded = true;
                    return handle;
                }
            }
            return IntPtr.Zero;
        }
    }

    private static IEnumerable<string> Candidates()
    {
        var env = Environment.GetEnvironmentVariable("DUCKTAPE_NATIVE");
        if (!string.IsNullOrEmpty(env))
        {
            yield return env;
        }
        var file = OperatingSystem.IsWindows() ? "ducktape_abi.dll" : OperatingSystem.IsMacOS() ? "libducktape_abi.dylib" : "libducktape_abi.so";
        var os = OperatingSystem.IsWindows() ? "win" : OperatingSystem.IsMacOS() ? "osx" : "linux";
        var portable = os + "-" + RuntimeInformation.ProcessArchitecture.ToString().ToLowerInvariant();
        foreach (var rid in new[] { RuntimeInformation.RuntimeIdentifier, portable }.Distinct())
        {
            yield return Path.Combine(AppContext.BaseDirectory, "runtimes", rid, "native", file);
        }
        yield return Path.Combine(AppContext.BaseDirectory, file);
    }

    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    [LibraryImport(Lib, EntryPoint = "ducktape_abi_version")]
    internal static partial uint AbiVersion();

    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    [LibraryImport(Lib, EntryPoint = "ducktape_free")]
    internal static partial void Free(byte* ptr, nuint len);

    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    [LibraryImport(Lib, EntryPoint = "ducktape_op_start")]
    internal static partial IntPtr OpStart(byte* json, nuint len, byte** output, nuint* outputLen);

    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    [LibraryImport(Lib, EntryPoint = "ducktape_op_resume")]
    internal static partial int OpResume(IntPtr op, byte* json, nuint len, byte** output, nuint* outputLen);

    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    [LibraryImport(Lib, EntryPoint = "ducktape_op_feed")]
    internal static partial int OpFeed(IntPtr op, byte* bytes, nuint bytesLen, byte* clock, nuint clockLen, byte** output, nuint* outputLen);

    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    [LibraryImport(Lib, EntryPoint = "ducktape_op_set_secret")]
    internal static partial int OpSetSecret(IntPtr op, byte* secretRef, nuint refLen, byte* bytes, nuint len);

    [UnmanagedCallConv(CallConvs = [typeof(CallConvCdecl)])]
    [LibraryImport(Lib, EntryPoint = "ducktape_op_free")]
    internal static partial void OpFree(IntPtr op);
}
