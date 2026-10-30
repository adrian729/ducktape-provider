//go:build windows

package ducktape

import (
	"syscall"
	"unicode/utf16"
	"unsafe"
)

var (
	advapi32  = syscall.NewLazyDLL("advapi32.dll")
	credReadW = advapi32.NewProc("CredReadW")
	credFree  = advapi32.NewProc("CredFree")
)

type credential struct {
	Flags              uint32
	Type               uint32
	TargetName         *uint16
	Comment            *uint16
	LastWritten        syscall.Filetime
	CredentialBlobSize uint32
	CredentialBlob     *byte
	Persist            uint32
	AttributeCount     uint32
	Attributes         uintptr
	TargetAlias        *uint16
	UserName           *uint16
}

func readCredential(target string) ([]byte, error) {
	name, err := syscall.UTF16PtrFromString(target)
	if err != nil {
		return nil, nil
	}
	var cred *credential
	ok, _, _ := credReadW.Call(uintptr(unsafe.Pointer(name)), 1, 0, uintptr(unsafe.Pointer(&cred)))
	if ok == 0 || cred == nil {
		return nil, nil
	}
	defer credFree.Call(uintptr(unsafe.Pointer(cred)))
	blob := unsafe.Slice(cred.CredentialBlob, cred.CredentialBlobSize)
	units := make([]uint16, len(blob)/2)
	for i := range units {
		units[i] = uint16(blob[2*i]) | uint16(blob[2*i+1])<<8
	}
	return []byte(string(utf16.Decode(units))), nil
}
