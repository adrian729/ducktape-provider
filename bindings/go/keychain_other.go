//go:build !windows

package ducktape

func readCredential(string) ([]byte, error) {
	return nil, nil
}

func writeCredential(string, string, []byte) (bool, error) {
	return false, nil
}
