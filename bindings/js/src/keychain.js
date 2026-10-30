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

const WINDOWS_WRITER = `
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class DucktapeCredW {
  [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  static extern bool CredWriteW(ref CREDENTIAL credential, uint flags);
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  struct CREDENTIAL {
    public uint Flags; public uint Type; public string TargetName; public string Comment;
    public System.Runtime.InteropServices.ComTypes.FILETIME LastWritten;
    public uint CredentialBlobSize; public IntPtr CredentialBlob; public uint Persist; public uint AttributeCount;
    public IntPtr Attributes; public string TargetAlias; public string UserName;
  }
  public static bool Write(string target, string user, string value) {
    byte[] b = System.Text.Encoding.Unicode.GetBytes(value);
    IntPtr p = Marshal.AllocHGlobal(b.Length);
    try {
      Marshal.Copy(b, 0, p, b.Length);
      CREDENTIAL c = new CREDENTIAL();
      c.Type = 1; c.TargetName = target; c.CredentialBlobSize = (uint)b.Length; c.CredentialBlob = p; c.Persist = 3; c.UserName = user;
      return CredWriteW(ref c, 0);
    } finally { Marshal.FreeHGlobal(p); }
  }
}
"@
$value = [Console]::In.ReadToEnd()
if ([DucktapeCredW]::Write($env:DUCKTAPE_CRED_TARGET, $env:DUCKTAPE_CRED_USER, $value)) { exit 0 } else { exit 1 }
`;

async function run(file, args, env, stdin) {
  const { execFile } = await import("node:child_process");
  return new Promise((resolve) => {
    const child = execFile(file, args, { encoding: "buffer", timeout: 10_000, windowsHide: true, env: { ...process.env, ...env } }, (error, stdout) => {
      resolve(error ? null : new Uint8Array(stdout ?? []));
    });
    if (stdin !== undefined) child.stdin.end(stdin);
    else child.stdin.end();
  });
}

const text = (bytes) => new TextDecoder().decode(bytes);

/** Reads another tool's keychain entry through the platform's own tools. */
export async function readKeychain(service, account) {
  if (!isNode) return null;
  const platform = process.platform;
  const present = (out) => (out && out.length ? out : null);
  if (platform === "darwin") {
    const out = await run("security", ["find-generic-password", "-s", service, "-a", account, "-w"]);
    return present(out && out.at(-1) === 10 ? out.subarray(0, -1) : out);
  }
  if (platform === "linux" || platform.endsWith("bsd")) {
    return present(await run("secret-tool", ["lookup", "service", service, "username", account]));
  }
  if (platform === "win32") {
    return present(await run("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", WINDOWS_READER], { DUCKTAPE_CRED_TARGET: `${account}.${service}` }));
  }
  return null;
}

/** Stores a keychain entry through the platform's own tools; false when unavailable. */
export async function writeKeychain(service, account, bytes) {
  if (!isNode) return false;
  const platform = process.platform;
  if (platform === "darwin") {
    return (await run("security", ["add-generic-password", "-U", "-s", service, "-a", account, "-w", text(bytes)])) !== null;
  }
  if (platform === "linux" || platform.endsWith("bsd")) {
    return (await run("secret-tool", ["store", `--label=${service}`, "service", service, "username", account], undefined, bytes)) !== null;
  }
  if (platform === "win32") {
    const env = { DUCKTAPE_CRED_TARGET: `${account}.${service}`, DUCKTAPE_CRED_USER: account };
    return (await run("powershell.exe", ["-NoProfile", "-NonInteractive", "-Command", WINDOWS_WRITER], env, bytes)) !== null;
  }
  return false;
}
