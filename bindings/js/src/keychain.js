import { isNode } from "./platform.js";

const WINDOWS_READER = `
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class DucktapeCred {
  [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  static extern bool CredReadW(string target, uint type, uint flags, out IntPtr credential);
  [DllImport("advapi32.dll")]
  static extern void CredFree(IntPtr credential);
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  struct CREDENTIAL {
    public uint Flags; public uint Type; public string TargetName; public string Comment;
    public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
    public uint CredentialBlobSize; public IntPtr CredentialBlob; public uint Persist; public uint AttributeCount;
    public IntPtr Attributes; public string TargetAlias; public string UserName;
  }
  public static string Read(string target) {
    IntPtr p;
    if (!CredReadW(target, 1, 0, out p)) return null;
    try {
      CREDENTIAL c = (CREDENTIAL)Marshal.PtrToStructure(p, typeof(CREDENTIAL));
      byte[] b = new byte[c.CredentialBlobSize];
      Marshal.Copy(c.CredentialBlob, b, 0, b.Length);
      return System.Text.Encoding.Unicode.GetString(b);
    } finally { CredFree(p); }
  }
}
"@
[Console]::Out.Write([DucktapeCred]::Read($env:DUCKTAPE_CRED_TARGET))
`;

async function run(file, args, env) {
  const { execFile } = await import("node:child_process");
  return new Promise((resolve) => {
    execFile(file, args, { encoding: "buffer", timeout: 10_000, windowsHide: true, env: { ...process.env, ...env } }, (error, stdout) => {
      resolve(error || !stdout?.length ? null : new Uint8Array(stdout));
    });
  });
}

/** Reads another tool's keychain entry through the platform's own tools. */
export async function readKeychain(service, account) {
  if (!isNode) return null;
  const platform = process.platform;
  if (platform === "darwin") {
    const out = await run("security", ["find-generic-password", "-s", service, "-a", account, "-w"]);
    return out && out.at(-1) === 10 ? out.subarray(0, -1) : out;
  }
  if (platform === "linux" || platform.endsWith("bsd")) {
    return run("secret-tool", ["lookup", "service", service, "username", account]);
  }
  if (platform === "win32") {
    return run("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", WINDOWS_READER], { DUCKTAPE_CRED_TARGET: `${account}.${service}` });
  }
  return null;
}
