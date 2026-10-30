package ducktape

import (
	"bufio"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

type clockDoc struct {
	NowMs   int64  `json:"now_ms"`
	Entropy string `json:"entropy"`
}

var clock = func() clockDoc {
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
	Type        string         `json:"type"`
	Request     *httpRequest   `json:"request,omitempty"`
	Ms          int64          `json:"ms,omitempty"`
	SecretRef   string         `json:"secret_ref,omitempty"`
	Source      *AuthSource    `json:"source,omitempty"`
	Locator     *locator       `json:"locator,omitempty"`
	Bytes       string         `json:"bytes,omitempty"`
	Expect      string         `json:"expect,omitempty"`
	Private     *bool          `json:"private,omitempty"`
	Action      *LoginAction   `json:"action,omitempty"`
	RedirectURI string         `json:"redirect_uri,omitempty"`
	TimeoutMs   int64          `json:"timeout_ms,omitempty"`
	Result      *result        `json:"result,omitempty"`
	Error       *ProviderError `json:"error,omitempty"`
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
	Etag      string            `json:"etag,omitempty"`
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
	secrets       map[string]string
	resolvers     map[string]func(context.Context) (string, error)
	keychain      KeychainReader
	keychainWrite KeychainWriter
	client        *http.Client
	observer      Observer
}

func etag(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
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
			if e.Action == nil {
				return nil, &ProviderError{Kind: ErrorKindMalformedResponse, Message: "the core asked for a user action without one"}
			}
			value, actErr := r.handleAction(*e.Action)
			if actErr != nil {
				return nil, authFailure("user_action_failed", "the login handler failed: "+actErr.Error())
			}
			in = input{Type: "user_action", Value: value}
		case "need_callback":
			in, err = r.callback(e)
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
					return input{}, authFailure("resolver_failed", "the credential resolver failed: "+err.Error())
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

func seconds(s *float64) time.Duration {
	if s == nil {
		return 0
	}
	return time.Duration(*s * float64(time.Second))
}

func (r *runner) http(req *httpRequest) input {
	ctx := r.ctx
	var cancel context.CancelFunc
	if total := seconds(timeoutOf(req).Total); total > 0 {
		ctx, cancel = context.WithTimeout(ctx, total)
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
	connectTimedOut := false
	var connectTimer *time.Timer
	if connect := seconds(timeoutOf(req).Connect); connect > 0 {
		connectTimer = time.AfterFunc(connect+seconds(timeoutOf(req).Read), func() {
			connectTimedOut = true
			cancel()
		})
	}
	resp, err := r.host.client.Do(hr)
	if connectTimer != nil && !connectTimer.Stop() && err == nil {
		err = context.DeadlineExceeded
	}
	if err != nil {
		cancel()
		if r.ctx.Err() != nil {
			return input{Type: "cancelled"}
		}
		if connectTimedOut {
			return input{Type: "http_error", Error: &transportError{Kind: "timeout", Message: "no response headers within the connect and read timeouts"}}
		}
		return input{Type: "http_error", Error: classify(err)}
	}
	r.body = resp.Body
	r.cancelBody = cancel
	r.readTimeout = seconds(timeoutOf(req).Read)
	headers := make(map[string]string, len(resp.Header))
	for k, v := range resp.Header {
		if len(v) > 0 {
			headers[strings.ToLower(k)] = strings.Join(v, ", ")
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

func timeoutOf(req *httpRequest) Timeout {
	if req.Timeout == nil {
		return Timeout{}
	}
	return *req.Timeout
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
	tag := ""
	switch loc.Type {
	case "file":
		b, err := os.ReadFile(loc.Path)
		if err != nil {
			return input{Type: "read", Found: boolPtr(false)}
		}
		data = b
		tag = etag(b)
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
	return input{Type: "read", Found: boolPtr(true), Bytes: base64.StdEncoding.EncodeToString(data), Etag: tag}
}

func (r *runner) write(e effect) input {
	data, err := base64.StdEncoding.DecodeString(e.Bytes)
	if err != nil || e.Locator == nil {
		return input{Type: "written", Ok: boolPtr(false)}
	}
	if e.Locator.Type == "keychain" {
		ok := false
		if r.host.keychainWrite != nil {
			ok, _ = r.host.keychainWrite(r.ctx, e.Locator.Service, e.Locator.Account, data)
		}
		return input{Type: "written", Ok: boolPtr(ok)}
	}
	if e.Locator.Type != "file" {
		return input{Type: "written", Ok: boolPtr(false)}
	}
	path := e.Locator.Path
	if e.Expect != "" {
		current, readErr := os.ReadFile(path)
		if readErr != nil || etag(current) != e.Expect {
			return input{Type: "written", Ok: boolPtr(false), Conflict: boolPtr(true)}
		}
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return input{Type: "written", Ok: boolPtr(false)}
	}
	mode := os.FileMode(0o644)
	if e.Private == nil || *e.Private {
		mode = 0o600
	}
	tmp, err := os.CreateTemp(filepath.Dir(path), filepath.Base(path)+".tmp*")
	if err != nil {
		return input{Type: "written", Ok: boolPtr(false)}
	}
	_, writeErr := tmp.Write(data)
	closeErr := tmp.Close()
	if writeErr != nil || closeErr != nil || os.Chmod(tmp.Name(), mode) != nil || os.Rename(tmp.Name(), path) != nil {
		_ = os.Remove(tmp.Name())
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

func (r *runner) callback(e effect) (input, error) {
	target, err := url.Parse(e.RedirectURI)
	if err != nil || target.Hostname() == "" {
		return input{}, unsupported("the login callback address is invalid")
	}
	host := target.Hostname()
	if host != "127.0.0.1" && host != "::1" {
		return input{}, unsupported("the login callback must be a loopback address")
	}
	addr := net.JoinHostPort(host, target.Port())
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return input{}, &ProviderError{Kind: ErrorKindConnection, Code: "callback_port_busy", Message: addr + " is in use; close the other login or use callback paste"}
	}
	defer listener.Close()
	if r.handleAction == nil {
		return input{}, unsupported("this call cannot ask the user for anything")
	}
	if e.Action == nil {
		return input{}, &ProviderError{Kind: ErrorKindMalformedResponse, Message: "the core asked for a callback without an action"}
	}
	if _, actErr := r.handleAction(*e.Action); actErr != nil {
		return input{}, authFailure("user_action_failed", "the login handler failed: "+actErr.Error())
	}
	timeout := time.Duration(e.TimeoutMs) * time.Millisecond
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}
	deadline := time.Now().Add(timeout)
	stop := context.AfterFunc(r.ctx, func() { listener.Close() })
	defer stop()
	origin := target.Scheme + "://" + target.Host
	for {
		if tcp, ok := listener.(*net.TCPListener); ok {
			tcp.SetDeadline(deadline)
		}
		conn, err := listener.Accept()
		if err != nil {
			if r.ctx.Err() != nil {
				return input{Type: "cancelled"}, nil
			}
			return input{}, &ProviderError{Kind: ErrorKindTimeout, Code: "login_timeout", Message: fmt.Sprintf("no sign-in callback arrived within %d seconds", int64(timeout/time.Second))}
		}
		requestTarget, matched := serveCallback(conn, target.Path)
		if matched {
			return input{Type: "callback", URL: origin + requestTarget}, nil
		}
	}
}

func serveCallback(conn net.Conn, path string) (string, bool) {
	defer conn.Close()
	conn.SetDeadline(time.Now().Add(5 * time.Second))
	reader := bufio.NewReader(io.LimitReader(conn, 8<<10))
	line, err := reader.ReadString('\n')
	if err != nil {
		return "", false
	}
	for {
		header, err := reader.ReadString('\n')
		if err != nil || strings.TrimRight(header, "\r\n") == "" {
			break
		}
	}
	parts := strings.Fields(line)
	if len(parts) < 2 {
		return "", false
	}
	requestTarget := parts[1]
	requestPath, _, _ := strings.Cut(requestTarget, "?")
	status, body := "404 Not Found", "Not found"
	matched := requestPath == path
	if matched {
		status, body = "200 OK", "<!doctype html><html><body><p>Sign-in finished. You can close this window.</p></body></html>"
	}
	fmt.Fprintf(conn, "HTTP/1.1 %s\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: %d\r\nConnection: close\r\n\r\n%s", status, len(body), body)
	return requestTarget, matched
}
