package ducktape

import (
	"bytes"
	"context"
	"os/exec"
	"runtime"
	"time"
)

// KeychainReader returns the secret stored under a service and account, or nil when absent.
type KeychainReader func(ctx context.Context, service, account string) ([]byte, error)

// ReadKeychain reads another tool's keychain entry through the platform's own tools.
func ReadKeychain(ctx context.Context, service, account string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	switch runtime.GOOS {
	case "darwin":
		out, err := exec.CommandContext(ctx, "security", "find-generic-password", "-s", service, "-a", account, "-w").Output()
		if err != nil || len(out) == 0 {
			return nil, nil
		}
		return bytes.TrimSuffix(out, []byte("\n")), nil
	case "linux", "freebsd", "openbsd", "netbsd":
		out, err := exec.CommandContext(ctx, "secret-tool", "lookup", "service", service, "username", account).Output()
		if err != nil || len(out) == 0 {
			return nil, nil
		}
		return out, nil
	case "windows":
		return readCredential(account + "." + service)
	}
	return nil, nil
}
