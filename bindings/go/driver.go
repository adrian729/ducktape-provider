package ducktape

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

type clockDoc struct {
	NowMs   int64  `json:"now_ms"`
	Entropy string `json:"entropy"`
}

func clock() clockDoc {
	entropy := make([]byte, 32)
	_, _ = rand.Read(entropy)
	return clockDoc{NowMs: time.Now().UnixMilli(), Entropy: base64.StdEncoding.EncodeToString(entropy)}
}

type step struct {
	Notify []notification `json:"notify"`
	Effect effect         `json:"effect"`
	raw    []byte
}

type notification struct {
	Type  string          `json:"type"`
	Event *Event          `json:"event,omitempty"`
	Hook  string          `json:"hook,omitempty"`
	Data  json.RawMessage `json:"data,omitempty"`
}

type effect struct {
	Type      string         `json:"type"`
	Request   *httpRequest   `json:"request,omitempty"`
	Ms        int64          `json:"ms,omitempty"`
	SecretRef string         `json:"secret_ref,omitempty"`
	Source    *AuthSource    `json:"source,omitempty"`
	Locator   *locator       `json:"locator,omitempty"`
	Bytes     string         `json:"bytes,omitempty"`
	Expect    string         `json:"expect,omitempty"`
	Private   *bool          `json:"private,omitempty"`
	Action    *LoginAction   `json:"action,omitempty"`
	Result    *result        `json:"result,omitempty"`
	Error     *ProviderError `json:"error,omitempty"`
}

type httpRequest struct {
	Method  string            `json:"method"`
	URL     string            `json:"url"`
	Headers map[string]string `json:"headers"`
	Body    string            `json:"body,omitempty"`
	Stream  bool              `json:"stream,omitempty"`
	Timeout *Timeout          `json:"timeout,omitempty"`
	Purpose string            `json:"purpose"`
}

type locator struct {
	Type    string `json:"type"`
	Path    string `json:"path,omitempty"`
	Service string `json:"service,omitempty"`
	Account string `json:"account,omitempty"`
	Name    string `json:"name,omitempty"`
}

type result struct {
	Type         string          `json:"type"`
	Response     json.RawMessage `json:"response,omitempty"`
	Models       []ModelInfo     `json:"models,omitempty"`
	Info         *ModelInfo      `json:"info,omitempty"`
	Capabilities *Capabilities   `json:"capabilities,omitempty"`
	Status       *AuthStatus     `json:"status,omitempty"`
}

type transportError struct {
	Kind    string `json:"kind"`
	Message string `json:"message"`
}

type input struct {
	Type      string            `json:"type"`
	Status    int               `json:"status,omitempty"`
	Headers   map[string]string `json:"headers,omitzero"`
	Bytes     string            `json:"bytes,omitempty"`
	Error     *transportError   `json:"error,omitempty"`
	SecretRef string            `json:"secret_ref,omitempty"`
	Found     *bool             `json:"found,omitempty"`
	Ok        *bool             `json:"ok,omitempty"`
	Conflict  *bool             `json:"conflict,omitempty"`
	Value     string            `json:"value,omitempty"`
	URL       string            `json:"url,omitempty"`
}

type resumeDoc struct {
	Clock clockDoc `json:"clock"`
	Input input    `json:"input"`
}

type host struct {
	secrets   map[string]string
	resolvers map[string]func(context.Context) (string, error)
	keychain  KeychainReader
	client    *http.Client
	observer  Observer
}

// Observer receives request, response, event and retry notifications as raw JSON.
type Observer struct {
	OnRequest  func(json.RawMessage)
	OnResponse func(json.RawMessage)
	OnEvent    func(json.RawMessage)
	OnRetry    func(json.RawMessage)
}

type runner struct {
	ctx          context.Context
	in           *instance
	host         *host
	onEvent      func(Event) error
	handleAction func(LoginAction) (string, error)
	body         io.ReadCloser
	cancelBody   context.CancelFunc
	readTimeout  time.Duration
	readTimer    *time.Timer
	buf          []byte
}

func (r *runner) closeBody() {
	if r.readTimer != nil {
		r.readTimer.Stop()
		r.readTimer = nil
	}
	if r.body != nil {
		_ = r.body.Close()
		r.body = nil
	}
	if r.cancelBody != nil {
		r.cancelBody()
		r.cancelBody = nil
	}
}

func boolPtr(b bool) *bool { return &b }

func (r *runner) run(start any) (*result, error) {
	defer r.closeBody()
	s, err := r.in.start(r.ctx, start)
	if err != nil {
		return nil, err
	}
	for {
		for _, n := range s.Notify {
			if n.Type == "event" && n.Event != nil && r.onEvent != nil {
				if err := r.onEvent(*n.Event); err != nil {
					return nil, err
				}
			} else if n.Type == "observer" {
				r.host.observer.call(n.Hook, n.Data)
			}
		}
		e := s.Effect
		switch e.Type {
		case "done":
			return e.Result, nil
		case "failed":
			if e.Error == nil {
				return nil, &ProviderError{Kind: "other", Message: "the core failed without an error"}
			}
			return nil, e.Error
		}
		if r.ctx.Err() != nil && e.Type != "receive" {
			s, err = r.resume(input{Type: "cancelled"})
			if err != nil {
				return nil, err
			}
			continue
		}
		var in input
		switch e.Type {
		case "need_secret":
			in, err = r.secret(e)
		case "need_http":
			r.closeBody()
			in = r.http(e.Request)
		case "receive":
			var next *step
			next, in, err = r.receive()
			if err != nil {
				return nil, err
			}
			if next != nil {
				s = *next
				continue
			}
		case "need_sleep":
			r.closeBody()
			select {
			case <-time.After(time.Duration(e.Ms) * time.Millisecond):
				in = input{Type: "slept"}
			case <-r.ctx.Done():
				in = input{Type: "cancelled"}
			}
		case "need_read":
			in = r.read(e.Locator)
		case "need_write":
			in = r.write(e)
		case "need_user_action":
			if r.handleAction == nil {
				return nil, unsupported("this call cannot ask the user for anything")
			}
			value, actErr := r.handleAction(*e.Action)
			if actErr != nil {
				return nil, &ProviderError{Kind: ErrorKindAuth, Code: "user_action_failed", Message: "the login handler failed: " + actErr.Error()}
			}
			in = input{Type: "user_action", Value: value}
		case "need_callback":
			return nil, unsupported("loopback callbacks are not supported by this binding; use callback paste")
		default:
			return nil, unsupported("unknown effect " + e.Type)
		}
		if err != nil {
			return nil, err
		}
		s, err = r.resume(in)
		if err != nil {
			return nil, err
		}
	}
}

func (r *runner) resume(in input) (step, error) {
	return r.in.resume(r.ctx, resumeDoc{Clock: clock(), Input: in})
}

func (r *runner) secret(e effect) (input, error) {
	ref := e.SecretRef
	value := ""
	if e.Source != nil {
		switch e.Source.Type {
		case "inline":
			value = r.host.secrets[ref]
		case "env":
			for _, name := range e.Source.Names {
				if v := os.Getenv(name); v != "" {
					value = v
					break
				}
			}
		case "resolver":
			if fn := r.host.resolvers[e.Source.ResolverID]; fn != nil {
				v, err := fn(r.ctx)
				if err != nil {
					return input{}, &ProviderError{Kind: ErrorKindAuth, Code: "resolver_failed", Message: "the credential resolver failed: " + err.Error()}
				}
				value = v
			}
		}
	}
	if value != "" {
		if err := r.in.setSecret(r.ctx, ref, []byte(value)); err != nil {
			return input{}, err
		}
	}
	return input{Type: "secret", SecretRef: ref, Found: boolPtr(value != "")}, nil
}

func (r *runner) http(req *httpRequest) input {
	ctx := r.ctx
	var cancel context.CancelFunc
	if req.Timeout != nil && req.Timeout.Total != nil {
		ctx, cancel = context.WithTimeout(ctx, time.Duration(*req.Timeout.Total*float64(time.Second)))
	} else {
		ctx, cancel = context.WithCancel(ctx)
	}
	var body io.Reader
	if req.Body != "" {
		body = strings.NewReader(req.Body)
	}
	hr, err := http.NewRequestWithContext(ctx, req.Method, req.URL, body)
	if err != nil {
		cancel()
		return input{Type: "http_error", Error: &transportError{Kind: "other", Message: err.Error()}}
	}
	for k, v := range req.Headers {
		hr.Header.Set(k, v)
	}
	resp, err := r.host.client.Do(hr)
	if err != nil {
		cancel()
		if r.ctx.Err() != nil {
			return input{Type: "cancelled"}
		}
		return input{Type: "http_error", Error: classify(err)}
	}
	r.body = resp.Body
	r.cancelBody = cancel
	r.readTimeout = 0
	if req.Timeout != nil && req.Timeout.Read != nil {
		r.readTimeout = time.Duration(*req.Timeout.Read * float64(time.Second))
	}
	headers := make(map[string]string, len(resp.Header))
	for k, v := range resp.Header {
		if len(v) > 0 {
			headers[strings.ToLower(k)] = v[0]
		}
	}
	return input{Type: "http_response", Status: resp.StatusCode, Headers: headers}
}

func (r *runner) receive() (*step, input, error) {
	if r.body == nil {
		return nil, input{Type: "http_end"}, nil
	}
	if r.buf == nil {
		r.buf = make([]byte, 64*1024)
	}
	if r.readTimeout > 0 {
		cancel := r.cancelBody
		if r.readTimer == nil {
			r.readTimer = time.AfterFunc(r.readTimeout, cancel)
		} else {
			r.readTimer.Reset(r.readTimeout)
		}
	}
	n, err := r.body.Read(r.buf)
	if r.readTimer != nil {
		r.readTimer.Stop()
	}
	if n > 0 {
		next, ferr := r.in.feed(r.ctx, r.buf[:n], clock())
		if ferr != nil {
			return nil, input{}, ferr
		}
		return &next, input{}, nil
	}
	if errors.Is(err, io.EOF) {
		r.closeBody()
		return nil, input{Type: "http_end"}, nil
	}
	r.closeBody()
	if err == nil {
		return nil, input{Type: "http_end"}, nil
	}
	if r.ctx.Err() != nil {
		return nil, input{Type: "cancelled"}, nil
	}
	te := classify(err)
	if te.Kind == "cancelled" {
		te = &transportError{Kind: "timeout", Message: "read timed out"}
	}
	return nil, input{Type: "http_error", Error: te}, nil
}

func classify(err error) *transportError {
	msg := err.Error()
	var dnsErr *net.DNSError
	var certErr x509.UnknownAuthorityError
	var hostErr x509.HostnameError
	var recErr tls.RecordHeaderError
	var netErr net.Error
	switch {
	case errors.Is(err, context.Canceled):
		return &transportError{Kind: "cancelled", Message: msg}
	case errors.Is(err, context.DeadlineExceeded):
		return &transportError{Kind: "timeout", Message: msg}
	case errors.As(err, &dnsErr):
		return &transportError{Kind: "dns", Message: msg}
	case errors.As(err, &certErr), errors.As(err, &hostErr), errors.As(err, &recErr), strings.Contains(msg, "tls:"), strings.Contains(msg, "x509:"):
		return &transportError{Kind: "tls", Message: msg}
	case errors.Is(err, syscall.ECONNREFUSED):
		return &transportError{Kind: "connect", Message: msg}
	case errors.Is(err, syscall.ECONNRESET), errors.Is(err, io.ErrUnexpectedEOF), errors.Is(err, syscall.EPIPE):
		return &transportError{Kind: "reset", Message: msg}
	case errors.As(err, &netErr) && netErr.Timeout():
		return &transportError{Kind: "timeout", Message: msg}
	}
	return &transportError{Kind: "other", Message: msg}
}

func (r *runner) read(loc *locator) input {
	if loc == nil {
		return input{Type: "read", Found: boolPtr(false)}
	}
	var data []byte
	switch loc.Type {
	case "file":
		b, err := os.ReadFile(loc.Path)
		if err != nil {
			return input{Type: "read", Found: boolPtr(false)}
		}
		data = b
	case "env":
		v, ok := os.LookupEnv(loc.Name)
		if !ok {
			return input{Type: "read", Found: boolPtr(false)}
		}
		data = []byte(v)
	case "keychain":
		if r.host.keychain == nil {
			return input{Type: "read", Found: boolPtr(false)}
		}
		b, err := r.host.keychain(r.ctx, loc.Service, loc.Account)
		if err != nil || len(b) == 0 {
			return input{Type: "read", Found: boolPtr(false)}
		}
		data = b
	default:
		return input{Type: "read", Found: boolPtr(false)}
	}
	return input{Type: "read", Found: boolPtr(true), Bytes: base64.StdEncoding.EncodeToString(data)}
}

func (r *runner) write(e effect) input {
	if e.Locator == nil || e.Locator.Type != "file" {
		return input{Type: "written", Ok: boolPtr(false)}
	}
	data, err := base64.StdEncoding.DecodeString(e.Bytes)
	if err != nil {
		return input{Type: "written", Ok: boolPtr(false)}
	}
	path := e.Locator.Path
	if e.Expect != "" {
		current, readErr := os.ReadFile(path)
		if readErr == nil {
			sum := sha256.Sum256(current)
			if hex.EncodeToString(sum[:]) != e.Expect {
				return input{Type: "written", Ok: boolPtr(false), Conflict: boolPtr(true)}
			}
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return input{Type: "written", Ok: boolPtr(false)}
	}
	mode := os.FileMode(0o644)
	if e.Private == nil || *e.Private {
		mode = 0o600
	}
	tmp := path + ".tmp" + strconv.Itoa(os.Getpid())
	if err := os.WriteFile(tmp, data, mode); err != nil {
		return input{Type: "written", Ok: boolPtr(false)}
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return input{Type: "written", Ok: boolPtr(false)}
	}
	return input{Type: "written", Ok: boolPtr(true)}
}

func (o Observer) call(hook string, data json.RawMessage) {
	switch hook {
	case "on_request":
		if o.OnRequest != nil {
			o.OnRequest(data)
		}
	case "on_response":
		if o.OnResponse != nil {
			o.OnResponse(data)
		}
	case "on_event":
		if o.OnEvent != nil {
			o.OnEvent(data)
		}
	case "on_retry":
		if o.OnRetry != nil {
			o.OnRetry(data)
		}
	}
}
