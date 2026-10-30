//! Reads and writes another tool's keychain entry through the platform's own store.
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Returns the secret stored under a service and account, or none when absent.
pub type Reader = Arc<dyn Fn(&str, &str) -> Option<Vec<u8>> + Send + Sync>;

/// Stores a secret under a service and account; false when the store is unavailable.
pub type Writer = Arc<dyn Fn(&str, &str, &[u8]) -> bool + Send + Sync>;

const WINDOWS_READER: &str = r#"[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
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
[Console]::Out.Write([DucktapeCred]::Read($env:DUCKTAPE_CRED_TARGET))"#;

const WINDOWS_WRITER: &str = r#"Add-Type -TypeDefinition @"
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
$stdin = [Console]::OpenStandardInput()
$buffer = New-Object System.IO.MemoryStream
$stdin.CopyTo($buffer)
$value = [System.Text.Encoding]::UTF8.GetString($buffer.ToArray())
if ([DucktapeCredW]::Write($env:DUCKTAPE_CRED_TARGET, $env:DUCKTAPE_CRED_USER, $value)) { exit 0 } else { exit 1 }"#;

/// The default reader: macOS Keychain, Linux Secret Service or Windows Credential Manager.
pub fn read(service: &str, account: &str) -> Option<Vec<u8>> {
    let output = if cfg!(target_os = "macos") {
        let mut out = run(
            &[
                "security",
                "find-generic-password",
                "-s",
                service,
                "-a",
                account,
                "-w",
            ],
            &[],
            None,
        )?;
        if out.last() == Some(&b'\n') {
            out.pop();
        }
        out
    } else if cfg!(target_os = "windows") {
        run(
            &[
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                WINDOWS_READER,
            ],
            &[("DUCKTAPE_CRED_TARGET", &format!("{account}.{service}"))],
            None,
        )?
    } else {
        run(
            &[
                "secret-tool",
                "lookup",
                "service",
                service,
                "username",
                account,
            ],
            &[],
            None,
        )?
    };
    (!output.is_empty()).then_some(output)
}

/// The default writer for the same three stores.
pub fn write(service: &str, account: &str, value: &[u8]) -> bool {
    let line_break = |s: &[u8]| s.iter().any(|b| *b == b'\n' || *b == b'\r');
    if line_break(service.as_bytes()) || line_break(account.as_bytes()) || line_break(value) {
        return false;
    }
    if cfg!(target_os = "macos") {
        let script = format!(
            "add-generic-password -U -s {} -a {} -w {}\n",
            quote(service),
            quote(account),
            quote(&String::from_utf8_lossy(value))
        );
        run(&["security", "-i"], &[], Some(script.as_bytes())).is_some()
    } else if cfg!(target_os = "windows") {
        run(
            &[
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                WINDOWS_WRITER,
            ],
            &[
                ("DUCKTAPE_CRED_TARGET", &format!("{account}.{service}")),
                ("DUCKTAPE_CRED_USER", account),
            ],
            Some(value),
        )
        .is_some()
    } else {
        run(
            &[
                "secret-tool",
                "store",
                &format!("--label={service}"),
                "service",
                service,
                "username",
                account,
            ],
            &[],
            Some(value),
        )
        .is_some()
    }
}

fn quote(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn run(argv: &[&str], env: &[(&str, &str)], stdin: Option<&[u8]>) -> Option<Vec<u8>> {
    let mut command = Command::new(argv[0]);
    command
        .args(&argv[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    for (k, v) in env {
        command.env(k, v);
    }
    let mut child = command.spawn().ok()?;
    let input = child.stdin.take();
    let bytes = stdin.map(<[u8]>::to_vec);
    std::thread::spawn(move || {
        if let (Some(mut input), Some(bytes)) = (input, bytes) {
            let _ = input.write_all(&bytes);
        }
    });
    let mut stdout = child.stdout.take()?;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut out = Vec::new();
        let _ = tx.send(stdout.read_to_end(&mut out).ok().map(|_| out));
    });
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let left = deadline.saturating_duration_since(Instant::now());
                let out = rx
                    .recv_timeout(left.max(Duration::from_secs(1)))
                    .ok()
                    .flatten()?;
                return status.success().then_some(out);
            }
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}
