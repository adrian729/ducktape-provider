//go:build !windows

package ducktape

func readCredential(string) ([]byte, error) {
	return nil, nil
}
