using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;

namespace Ducktape;

/// <summary>Returns the secret stored under a service and account, or null when absent.</summary>
public delegate byte[]? KeychainReader(string service, string account);

/// <summary>Stores a secret under a service and account; false when the store is unavailable.</summary>
public delegate bool KeychainWriter(string service, string account, byte[] value);

/// <summary>Reads and writes another tool's keychain entry through the platform's own store.</summary>
public static class Keychain
{
    /// <summary>The default reader: macOS Keychain, Linux Secret Service or Windows Credential Manager.</summary>
    public static byte[]? Read(string service, string account)
    {
        if (OperatingSystem.IsMacOS())
        {
            var output = Run(["security", "find-generic-password", "-s", service, "-a", account, "-w"], null);
            if (output is { Length: > 0 } && output[^1] == (byte)'\n')
            {
                output = output[..^1];
            }
            return Present(output);
        }
        if (OperatingSystem.IsLinux() || OperatingSystem.IsFreeBSD())
        {
            return Present(Run(["secret-tool", "lookup", "service", service, "username", account], null));
        }
        if (OperatingSystem.IsWindows())
        {
            return Present(Windows.Read(account + "." + service));
        }
        return null;
    }

    /// <summary>The default writer for the same three stores.</summary>
    public static bool Write(string service, string account, byte[] value)
    {
        if (OperatingSystem.IsMacOS())
        {
            var script = "add-generic-password -U -s " + Quote(service) + " -a " + Quote(account) + " -w " + Quote(Encoding.UTF8.GetString(value)) + "\n";
            return Run(["security", "-i"], Encoding.UTF8.GetBytes(script)) is not null;
        }
        if (OperatingSystem.IsLinux() || OperatingSystem.IsFreeBSD())
        {
            return Run(["secret-tool", "store", "--label=" + service, "service", service, "username", account], value) is not null;
        }
        if (OperatingSystem.IsWindows())
        {
            return Windows.Write(account + "." + service, account, value);
        }
        return false;
    }

    private static string Quote(string s)
    {
        return "\"" + s.Replace("\\", "\\\\").Replace("\"", "\\\"") + "\"";
    }

    private static byte[]? Present(byte[]? output)
    {
        return output is { Length: > 0 } ? output : null;
    }

    private static byte[]? Run(string[] argv, byte[]? stdin)
    {
        try
        {
            var info = new ProcessStartInfo(argv[0]) { RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true, UseShellExecute = false };
            foreach (var arg in argv.Skip(1))
            {
                info.ArgumentList.Add(arg);
            }
            using var process = Process.Start(info);
            if (process is null)
            {
                return null;
            }
            using (var input = process.StandardInput.BaseStream)
            {
                if (stdin is not null)
                {
                    input.Write(stdin);
                }
            }
            var collected = new MemoryStream();
            var copy = process.StandardOutput.BaseStream.CopyToAsync(collected);
            process.StandardError.BaseStream.CopyToAsync(Stream.Null);
            if (!process.WaitForExit(10_000))
            {
                process.Kill(true);
                return null;
            }
            if (!copy.Wait(1_000) || process.ExitCode != 0)
            {
                return null;
            }
            return collected.ToArray();
        }
        catch (Exception)
        {
            return null;
        }
    }

    private static class Windows
    {
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        private struct CredentialW
        {
            public uint Flags;
            public uint Type;
            public IntPtr TargetName;
            public IntPtr Comment;
            public long LastWritten;
            public uint CredentialBlobSize;
            public IntPtr CredentialBlob;
            public uint Persist;
            public uint AttributeCount;
            public IntPtr Attributes;
            public IntPtr TargetAlias;
            public IntPtr UserName;
        }

        [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool CredReadW(string target, uint type, uint flags, out IntPtr credential);

        [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
        private static extern bool CredWriteW(ref CredentialW credential, uint flags);

        [DllImport("advapi32.dll")]
        private static extern void CredFree(IntPtr credential);

        public static byte[]? Read(string target)
        {
            if (!OperatingSystem.IsWindows() || !CredReadW(target, 1, 0, out var ptr))
            {
                return null;
            }
            try
            {
                var cred = Marshal.PtrToStructure<CredentialW>(ptr);
                var blob = new byte[cred.CredentialBlobSize];
                Marshal.Copy(cred.CredentialBlob, blob, 0, blob.Length);
                return Encoding.UTF8.GetBytes(Encoding.Unicode.GetString(blob));
            }
            finally
            {
                CredFree(ptr);
            }
        }

        public static bool Write(string target, string user, byte[] value)
        {
            if (!OperatingSystem.IsWindows())
            {
                return false;
            }
            var blob = Encoding.Unicode.GetBytes(Encoding.UTF8.GetString(value));
            var blobPtr = Marshal.AllocHGlobal(blob.Length);
            var targetPtr = Marshal.StringToHGlobalUni(target);
            var userPtr = Marshal.StringToHGlobalUni(user);
            try
            {
                Marshal.Copy(blob, 0, blobPtr, blob.Length);
                var cred = new CredentialW { Type = 1, TargetName = targetPtr, CredentialBlobSize = (uint)blob.Length, CredentialBlob = blobPtr, Persist = 3, UserName = userPtr };
                return CredWriteW(ref cred, 0);
            }
            finally
            {
                Marshal.FreeHGlobal(blobPtr);
                Marshal.FreeHGlobal(targetPtr);
                Marshal.FreeHGlobal(userPtr);
            }
        }
    }
}
