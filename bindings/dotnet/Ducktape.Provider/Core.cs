using System.Text;
using System.Text.Json.Nodes;

namespace Ducktape;

/// <summary>The native core library: where it is loaded from and which ABI it speaks.</summary>
public static class Core
{
    internal const uint Abi = 1;

    /// <summary>Loads the core from an explicit path instead of the package's runtimes folder; only before the first call.</summary>
    public static void Load(string path)
    {
        Native.Load(path);
    }

    /// <summary>The ABI version of the loaded core.</summary>
    public static uint AbiVersion => Native.AbiVersion();

    internal static void Check()
    {
        var version = Native.AbiVersion();
        if (version != Abi)
        {
            throw new InvalidOperationException($"the native core speaks ABI {version}, this package needs {Abi}");
        }
    }

    /// <summary>One operation in the core, driven step by step.</summary>
    internal sealed unsafe class Operation : IDisposable
    {
        private IntPtr handle;

        private Operation(IntPtr handle)
        {
            this.handle = handle;
        }

        private static JsonNode Take(byte* ptr, nuint len)
        {
            if (ptr is null || len == 0)
            {
                throw new InvalidOperationException("the core returned no step");
            }
            try
            {
                return JsonNode.Parse(new ReadOnlySpan<byte>(ptr, checked((int)len))) ?? throw new InvalidOperationException("the core returned an empty step");
            }
            finally
            {
                Native.Free(ptr, len);
            }
        }

        public static (Operation? Op, JsonNode Step) Start(JsonNode start)
        {
            Check();
            var json = Encoding.UTF8.GetBytes(start.ToJsonString());
            byte* output;
            nuint outputLen;
            IntPtr handle;
            fixed (byte* p = json)
            {
                handle = Native.OpStart(p, (nuint)json.Length, &output, &outputLen);
            }
            var step = Take(output, outputLen);
            return (handle == IntPtr.Zero ? null : new Operation(handle), step);
        }

        private IntPtr Handle => handle == IntPtr.Zero ? throw new ObjectDisposedException(nameof(Operation)) : handle;

        public JsonNode Resume(JsonNode resume)
        {
            var json = Encoding.UTF8.GetBytes(resume.ToJsonString());
            byte* output;
            nuint outputLen;
            fixed (byte* p = json)
            {
                Native.OpResume(Handle, p, (nuint)json.Length, &output, &outputLen);
            }
            return Take(output, outputLen);
        }

        public JsonNode Feed(ReadOnlySpan<byte> bytes, JsonNode clock)
        {
            var json = Encoding.UTF8.GetBytes(clock.ToJsonString());
            byte* output;
            nuint outputLen;
            fixed (byte* b = bytes)
            fixed (byte* c = json)
            {
                Native.OpFeed(Handle, b, (nuint)bytes.Length, c, (nuint)json.Length, &output, &outputLen);
            }
            return Take(output, outputLen);
        }

        public void SetSecret(string secretRef, ReadOnlySpan<byte> value)
        {
            var name = Encoding.UTF8.GetBytes(secretRef);
            fixed (byte* r = name)
            fixed (byte* v = value)
            {
                Native.OpSetSecret(Handle, r, (nuint)name.Length, v, (nuint)value.Length);
            }
        }

        public void Dispose()
        {
            if (handle != IntPtr.Zero)
            {
                Native.OpFree(handle);
                handle = IntPtr.Zero;
            }
        }
    }
}
