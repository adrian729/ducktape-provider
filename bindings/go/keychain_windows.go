//go:build windows

package ducktape

import (
	"syscall"
	"unicode/utf16"
	"unsafe"
)

var (
	advapi32   = syscall.NewLazyDLL("advapi32.dll")
	credReadW  = advapi32.NewProc("CredReadW")
	credWriteW = advapi32.NewProc("CredWriteW")
	credFree   = advapi32.NewProc("CredFree")
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

func writeCredential(target, user string, value []byte) (bool, error) {
	name, err := syscall.UTF16PtrFromString(target)
	if err != nil {
		return false, nil
	}
	userName, err := syscall.UTF16PtrFromString(user)
	if err != nil {
		return false, nil
	}
	units := utf16.Encode([]rune(string(value)))
	blob := make([]byte, 2*len(units))
	for i, u := range units {
		blob[2*i] = byte(u)
		blob[2*i+1] = byte(u >> 8)
	}
	if len(blob) == 0 {
		blob = []byte{0}
	}
	cred := credential{Type: 1, TargetName: name, CredentialBlobSize: uint32(2 * len(units)), CredentialBlob: &blob[0], Persist: 3, UserName: userName}
	ok, _, _ := credWriteW.Call(uintptr(unsafe.Pointer(&cred)), 0)
	return ok != 0, nil
}
