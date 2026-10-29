"""Reads another tool's keychain entry through the platform's own tools."""
from __future__ import annotations

import subprocess
import sys
from typing import Optional


def read_keychain(service: str, account: str) -> Optional[bytes]:
    """The secret stored under `service` and `account`, or None."""
    if sys.platform == "darwin":
        return _run(["security", "find-generic-password", "-s", service, "-a", account, "-w"], strip=True)
    if sys.platform.startswith("linux") or "bsd" in sys.platform:
        return _run(["secret-tool", "lookup", "service", service, "username", account], strip=False)
    if sys.platform == "win32":
        return _windows(f"{account}.{service}")
    return None


def _run(argv: list[str], *, strip: bool) -> Optional[bytes]:
    try:
        done = subprocess.run(argv, capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0 or not done.stdout:
        return None
    out = done.stdout
    if strip and out.endswith(b"\n"):
        out = out[:-1]
    return out or None


def _windows(target: str) -> Optional[bytes]:
    import ctypes
    from ctypes import wintypes

    class Credential(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    try:
        advapi = ctypes.WinDLL("advapi32", use_last_error=True)
        advapi.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(Credential))]
        advapi.CredReadW.restype = wintypes.BOOL
        advapi.CredFree.argtypes = [ctypes.c_void_p]
        cred = ctypes.POINTER(Credential)()
        if not advapi.CredReadW(target, 1, 0, ctypes.byref(cred)):
            return None
        try:
            blob = ctypes.string_at(cred.contents.CredentialBlob, cred.contents.CredentialBlobSize)
        finally:
            advapi.CredFree(cred)
        return blob.decode("utf-16-le").encode() or None
    except (OSError, AttributeError, UnicodeDecodeError):
        return None
