package io.github.adrian729.ducktape;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/** Reads and writes another tool's keychain entry through the platform's own tools. */
public final class Keychain {
    private Keychain() {}

    /** Returns the secret stored under a service and account, or null when absent. */
    @FunctionalInterface
    public interface Reader {
        byte[] read(String service, String account) throws Exception;
    }

    /** Stores a secret under a service and account; false when the store is unavailable. */
    @FunctionalInterface
    public interface Writer {
        boolean write(String service, String account, byte[] value) throws Exception;
    }

    private static final String WINDOWS_READER = String.join("\n",
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8",
        "Add-Type -TypeDefinition @\"",
        "using System;",
        "using System.Runtime.InteropServices;",
        "public static class DucktapeCred {",
        "  [DllImport(\"advapi32.dll\", CharSet = CharSet.Unicode, SetLastError = true)]",
        "  static extern bool CredReadW(string target, uint type, uint flags, out IntPtr credential);",
        "  [DllImport(\"advapi32.dll\")]",
        "  static extern void CredFree(IntPtr credential);",
        "  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]",
        "  struct CREDENTIAL {",
        "    public uint Flags; public uint Type; public string TargetName; public string Comment;",
        "    public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;",
        "    public uint CredentialBlobSize; public IntPtr CredentialBlob; public uint Persist; public uint AttributeCount;",
        "    public IntPtr Attributes; public string TargetAlias; public string UserName;",
        "  }",
        "  public static string Read(string target) {",
        "    IntPtr p;",
        "    if (!CredReadW(target, 1, 0, out p)) return null;",
        "    try {",
        "      CREDENTIAL c = (CREDENTIAL)Marshal.PtrToStructure(p, typeof(CREDENTIAL));",
        "      byte[] b = new byte[c.CredentialBlobSize];",
        "      Marshal.Copy(c.CredentialBlob, b, 0, b.Length);",
        "      return System.Text.Encoding.Unicode.GetString(b);",
        "    } finally { CredFree(p); }",
        "  }",
        "}",
        "\"@",
        "[Console]::Out.Write([DucktapeCred]::Read($env:DUCKTAPE_CRED_TARGET))");

    private static final String WINDOWS_WRITER = String.join("\n",
        "Add-Type -TypeDefinition @\"",
        "using System;",
        "using System.Runtime.InteropServices;",
        "public static class DucktapeCredW {",
        "  [DllImport(\"advapi32.dll\", CharSet = CharSet.Unicode, SetLastError = true)]",
        "  static extern bool CredWriteW(ref CREDENTIAL credential, uint flags);",
        "  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]",
        "  struct CREDENTIAL {",
        "    public uint Flags; public uint Type; public string TargetName; public string Comment;",
        "    public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;",
        "    public uint CredentialBlobSize; public IntPtr CredentialBlob; public uint Persist; public uint AttributeCount;",
        "    public IntPtr Attributes; public string TargetAlias; public string UserName;",
        "  }",
        "  public static bool Write(string target, string user, string value) {",
        "    byte[] b = System.Text.Encoding.Unicode.GetBytes(value);",
        "    IntPtr p = Marshal.AllocHGlobal(b.Length);",
        "    try {",
        "      Marshal.Copy(b, 0, p, b.Length);",
        "      CREDENTIAL c = new CREDENTIAL();",
        "      c.Type = 1; c.TargetName = target; c.CredentialBlobSize = (uint)b.Length; c.CredentialBlob = p; c.Persist = 3; c.UserName = user;",
        "      return CredWriteW(ref c, 0);",
        "    } finally { Marshal.FreeHGlobal(p); }",
        "  }",
        "}",
        "\"@",
        "$value = [Console]::In.ReadToEnd()",
        "if ([DucktapeCredW]::Write($env:DUCKTAPE_CRED_TARGET, $env:DUCKTAPE_CRED_USER, $value)) { exit 0 } else { exit 1 }");

    private static String os() {
        return System.getProperty("os.name", "").toLowerCase(Locale.ROOT);
    }

    /** The default reader: macOS Keychain, Linux Secret Service or Windows Credential Manager. */
    public static byte[] read(String service, String account) {
        String os = os();
        if (os.contains("mac")) {
            byte[] out = run(List.of("security", "find-generic-password", "-s", service, "-a", account, "-w"), Map.of(), null);
            if (out != null && out.length > 0 && out[out.length - 1] == '\n') {
                out = java.util.Arrays.copyOf(out, out.length - 1);
            }
            return present(out);
        }
        if (os.contains("linux") || os.contains("bsd")) {
            return present(run(List.of("secret-tool", "lookup", "service", service, "username", account), Map.of(), null));
        }
        if (os.contains("win")) {
            List<String> argv = List.of("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_READER);
            return present(run(argv, Map.of("DUCKTAPE_CRED_TARGET", account + "." + service), null));
        }
        return null;
    }

    /** The default writer for the same three stores. */
    public static boolean write(String service, String account, byte[] value) {
        String os = os();
        if (os.contains("mac")) {
            String script = "add-generic-password -U -s " + quote(service) + " -a " + quote(account) + " -w " + quote(new String(value, StandardCharsets.UTF_8)) + "\n";
            return run(List.of("security", "-i"), Map.of(), script.getBytes(StandardCharsets.UTF_8)) != null;
        }
        if (os.contains("linux") || os.contains("bsd")) {
            return run(List.of("secret-tool", "store", "--label=" + service, "service", service, "username", account), Map.of(), value) != null;
        }
        if (os.contains("win")) {
            List<String> argv = List.of("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_WRITER);
            return run(argv, Map.of("DUCKTAPE_CRED_TARGET", account + "." + service, "DUCKTAPE_CRED_USER", account), value) != null;
        }
        return false;
    }

    private static String quote(String s) {
        return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
    }

    private static byte[] present(byte[] out) {
        return out == null || out.length == 0 ? null : out;
    }

    private static byte[] run(List<String> argv, Map<String, String> env, byte[] stdin) {
        try {
            ProcessBuilder pb = new ProcessBuilder(argv).redirectErrorStream(false);
            pb.environment().putAll(env);
            Process p = pb.start();
            try (var in = p.getOutputStream()) {
                if (stdin != null) {
                    in.write(stdin);
                }
            }
            byte[] out = readAll(p.getInputStream());
            if (!p.waitFor(10, TimeUnit.SECONDS)) {
                p.destroyForcibly();
                return null;
            }
            return p.exitValue() == 0 ? out : null;
        } catch (IOException | InterruptedException | RuntimeException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            return null;
        }
    }

    private static byte[] readAll(InputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) > 0) {
            out.write(buf, 0, n);
        }
        return out.toByteArray();
    }
}
