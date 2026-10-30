package ducktape

import (
	"errors"
	"fmt"
)

// Error returns the kind and message.
func (e *ProviderError) Error() string {
	code := ""
	if e.Code != "" {
		code = " " + e.Code
	}
	return fmt.Sprintf("ducktape: [%s%s] %s", e.Kind, code, e.Message)
}

// Is reports whether target is the sentinel of this error's kind.
func (e *ProviderError) Is(target error) bool {
	k, ok := target.(kindError)
	return ok && ErrorKind(k) == e.Kind
}

type kindError ErrorKind

func (k kindError) Error() string {
	return "ducktape: " + string(k)
}

// Sentinels for errors.Is, one per error kind.
var (
	ErrAuth              error = kindError(ErrorKindAuth)
	ErrRateLimit         error = kindError(ErrorKindRateLimit)
	ErrQuota             error = kindError(ErrorKindQuota)
	ErrServer            error = kindError(ErrorKindServer)
	ErrConnection        error = kindError(ErrorKindConnection)
	ErrTimeout           error = kindError(ErrorKindTimeout)
	ErrCancelled         error = kindError(ErrorKindCancelled)
	ErrNotFound          error = kindError(ErrorKindNotFound)
	ErrContextOverflow   error = kindError(ErrorKindContextOverflow)
	ErrRequestTooLarge   error = kindError(ErrorKindRequestTooLarge)
	ErrBadRequest        error = kindError(ErrorKindBadRequest)
	ErrMalformedResponse error = kindError(ErrorKindMalformedResponse)
	ErrUnsupported       error = kindError(ErrorKindUnsupported)
)

// AsProviderError returns the ProviderError inside err, if any.
func AsProviderError(err error) (*ProviderError, bool) {
	var pe *ProviderError
	if errors.As(err, &pe) {
		return pe, true
	}
	return nil, false
}

func unsupported(message string) *ProviderError {
	return &ProviderError{Kind: ErrorKindUnsupported, Message: message}
}

func authFailure(code, message string) *ProviderError {
	return &ProviderError{Kind: ErrorKindAuth, Code: code, Message: message}
}
