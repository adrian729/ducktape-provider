package ducktape

import (
	"bytes"
	"context"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

// KeychainReader returns the secret stored under a service and account, or nil when absent.
type KeychainReader func(ctx context.Context, service, account string) ([]byte, error)

// KeychainWriter stores a secret under a service and account; false when the store is unavailable.
type KeychainWriter func(ctx context.Context, service, account string, value []byte) (bool, error)

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

func securityQuote(s string) string {
	return "\"" + strings.NewReplacer("\\", "\\\\", "\"", "\\\"").Replace(s) + "\""
}

func securityAdd(service, account, value string) string {
	return "add-generic-password -U -s " + securityQuote(service) + " -a " + securityQuote(account) + " -w " + securityQuote(value) + "\n"
}

// WriteKeychain stores a keychain entry through the platform's own tools.
func WriteKeychain(ctx context.Context, service, account string, value []byte) (bool, error) {
	if strings.ContainsAny(service, "\r\n") || strings.ContainsAny(account, "\r\n") || strings.ContainsAny(string(value), "\r\n") {
		return false, nil
	}
	ctx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	switch runtime.GOOS {
	case "darwin":
		cmd := exec.CommandContext(ctx, "security", "-i")
		cmd.Stdin = strings.NewReader(securityAdd(service, account, string(value)))
		return cmd.Run() == nil, nil
	case "linux", "freebsd", "openbsd", "netbsd":
		cmd := exec.CommandContext(ctx, "secret-tool", "store", "--label="+service, "service", service, "username", account)
		cmd.Stdin = bytes.NewReader(value)
		return cmd.Run() == nil, nil
	case "windows":
		return writeCredential(account+"."+service, account, value)
	}
	return false, nil
}
