package ducktape

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
	"sync"
	"time"
)

var vendors = []string{"anthropic", "openai", "google", "ollama"}

// Config sets up one Provider; every field is optional.
type Config struct {
	BaseURL               string
	Auth                  []Credential
	Headers               map[string]string
	Timeout               *Timeout
	DefaultModel          string
	DefaultEmbeddingModel string
	DefaultSystem         []Block
	DefaultTools          []Tool
	DefaultOptions        *Options
	DefaultVendor         VendorMap
	Overrides             *Overrides
	Strict                *bool
	Observer              Observer
	Client                *http.Client
	Core                  *Core
	Keychain              KeychainReader
	KeychainWrite         KeychainWriter
}

// Provider is one endpoint of one vendor.
type Provider struct {
	vendor string
	config ProviderConfig
	host   *host
	core   *Core
	mu     sync.Mutex
	models []ModelInfo
	infos  map[string]*ModelInfo
	caps   map[string]*Capabilities
}

var defaultClient = &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }, Transport: &http.Transport{
	Proxy:                 http.ProxyFromEnvironment,
	DialContext:           (&net.Dialer{Timeout: 10 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
	ForceAttemptHTTP2:     true,
	MaxIdleConns:          100,
	IdleConnTimeout:       90 * time.Second,
	TLSHandshakeTimeout:   10 * time.Second,
	ExpectContinueTimeout: time.Second,
}}

// New makes a provider for a vendor.
func New(vendor string, cfg Config) (*Provider, error) {
	known := false
	for _, n := range vendors {
		known = known || n == vendor
	}
	if !known {
		return nil, fmt.Errorf("ducktape: unknown vendor %q; one of %s", vendor, strings.Join(vendors, ", "))
	}
	p := &Provider{
		vendor: vendor,
		host:   &host{secrets: map[string]string{}, resolvers: map[string]func(context.Context) (string, error){}, keychain: cfg.Keychain, keychainWrite: cfg.KeychainWrite, client: cfg.Client, observer: cfg.Observer},
		core:   cfg.Core,
		infos:  map[string]*ModelInfo{},
		caps:   map[string]*Capabilities{},
	}
	if p.host.client == nil {
		p.host.client = defaultClient
	}
	if p.host.keychain == nil {
		p.host.keychain = ReadKeychain
		if p.host.keychainWrite == nil {
			p.host.keychainWrite = WriteKeychain
		}
	}
	c := ProviderConfig{Vendor: ProviderConfigVendor(vendor)}
	if cfg.BaseURL != "" {
		c.BaseURL = strings.TrimRight(cfg.BaseURL, "/")
	}
	if len(cfg.Auth) > 0 {
		sources := make([]AuthSource, 0, len(cfg.Auth))
		for i, cred := range cfg.Auth {
			src := cred.source
			switch src.Type {
			case "inline":
				src.SecretRef = fmt.Sprintf("inline:%d", i)
				p.host.secrets[src.SecretRef] = cred.key
			case "resolver":
				src.ResolverID = fmt.Sprintf("resolver:%d", i)
				if cred.resolver == nil {
					return nil, errors.New("ducktape: a resolver credential needs a function")
				}
				p.host.resolvers[src.ResolverID] = cred.resolver
			}
			sources = append(sources, src)
		}
		c.Auth = &Auth{Sources: sources}
	}
	c.Headers = cfg.Headers
	c.Timeout = cfg.Timeout
	c.DefaultModel = cfg.DefaultModel
	c.DefaultEmbeddingModel = cfg.DefaultEmbeddingModel
	c.DefaultSystem = cfg.DefaultSystem
	c.DefaultTools = cfg.DefaultTools
	c.DefaultOptions = cfg.DefaultOptions
	c.DefaultVendor = cfg.DefaultVendor
	c.Overrides = cfg.Overrides
	c.Strict = cfg.Strict
	p.config = c
	return p, nil
}

// Vendor is the vendor the provider speaks to.
func (p *Provider) Vendor() string { return p.vendor }

type operation map[string]any

type startDoc struct {
	Clock     clockDoc       `json:"clock"`
	Provider  ProviderConfig `json:"provider"`
	Operation operation      `json:"operation"`
}

func (p *Provider) run(ctx context.Context, op operation, onEvent func(Event) error, handleAction func(LoginAction) (string, error)) (*result, error) {
	core := p.core
	if core == nil {
		c, err := DefaultCore(ctx)
		if err != nil {
			return nil, err
		}
		core = c
	}
	in, err := core.instantiate(ctx)
	if err != nil {
		return nil, err
	}
	defer in.close(context.WithoutCancel(ctx))
	r := &runner{ctx: ctx, in: in, host: p.host, onEvent: onEvent, handleAction: handleAction}
	return r.run(startDoc{Clock: clock(), Provider: p.config, Operation: op})
}

// ChatOptions tune one chat or stream call.
type ChatOptions struct {
	Model      string
	System     []Block
	Tools      []Tool
	Options    *Options
	Vendor     VendorMap
	Timeout    *Timeout
	Headers    map[string]string
	Strict     *bool
	IncludeRaw *bool
}

func (p *Provider) chatRequest(messages []Message, o *ChatOptions) (ChatRequest, error) {
	if o == nil {
		o = &ChatOptions{}
	}
	model := o.Model
	if model == "" {
		model = p.config.DefaultModel
	}
	if model == "" {
		return ChatRequest{}, errors.New("ducktape: no model: set Model or DefaultModel")
	}
	if messages == nil {
		messages = []Message{}
	}
	return ChatRequest{Model: model, Messages: messages, System: o.System, Tools: o.Tools, Options: o.Options, Vendor: o.Vendor, Timeout: o.Timeout, Headers: o.Headers, Strict: o.Strict, IncludeRaw: o.IncludeRaw}, nil
}

// Chat sends the messages and returns the whole reply.
func (p *Provider) Chat(ctx context.Context, messages []Message, options *ChatOptions) (*Response, error) {
	req, err := p.chatRequest(messages, options)
	if err != nil {
		return nil, err
	}
	res, err := p.run(ctx, operation{"type": "chat", "request": req}, nil, nil)
	if err != nil {
		return nil, err
	}
	var out Response
	if err := json.Unmarshal(res.Response, &out); err != nil {
		return nil, fmt.Errorf("ducktape: decode response: %w", err)
	}
	return &out, nil
}

// Stream is a streamed chat reply: call Next until it returns false, then check Err; Close when stopping early.
type Stream struct {
	events   chan Event
	done     chan struct{}
	cancel   context.CancelFunc
	current  Event
	err      error
	response *Response
	once     sync.Once
}

// Stream sends the messages and returns the events as they arrive.
func (p *Provider) Stream(ctx context.Context, messages []Message, options *ChatOptions) (*Stream, error) {
	req, err := p.chatRequest(messages, options)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(ctx)
	s := &Stream{events: make(chan Event, 16), done: make(chan struct{}), cancel: cancel}
	go func() {
		defer close(s.done)
		defer close(s.events)
		_, runErr := p.run(ctx, operation{"type": "stream", "request": req}, func(ev Event) error {
			if ev.Type == "finish" && ev.Response != nil {
				s.response = ev.Response
			}
			select {
			case s.events <- ev:
				return nil
			case <-ctx.Done():
				return ctx.Err()
			}
		}, nil)
		if runErr != nil && s.err == nil {
			s.err = runErr
		}
	}()
	return s, nil
}

// Next waits for the next event; false at the end or on error.
func (s *Stream) Next() bool {
	ev, ok := <-s.events
	if !ok {
		<-s.done
		return false
	}
	s.current = ev
	return true
}

// Event is the event Next returned.
func (s *Stream) Event() Event { return s.current }

// Err is the error that ended the stream, if any.
func (s *Stream) Err() error {
	<-s.done
	if s.err != nil {
		if errors.Is(s.err, context.Canceled) {
			return &ProviderError{Kind: ErrorKindCancelled, Message: "the stream was cancelled"}
		}
		return s.err
	}
	return nil
}

// Response is the final response once the stream finished.
func (s *Stream) Response() *Response {
	<-s.done
	return s.response
}

// Close stops the stream and releases the connection.
func (s *Stream) Close() {
	s.once.Do(func() {
		s.cancel()
		go func() {
			for range s.events {
			}
		}()
	})
}

// EmbedOptions tune one embed call.
type EmbedOptions struct {
	Model      string
	Dimensions *int64
	Retries    *int64
	Vendor     VendorMap
	Timeout    *Timeout
	IncludeRaw *bool
}

// Embed returns one embedding per input text.
func (p *Provider) Embed(ctx context.Context, input []string, options *EmbedOptions) (*EmbedResponse, error) {
	if options == nil {
		options = &EmbedOptions{}
	}
	if len(input) == 0 {
		return nil, errors.New("ducktape: embed input must not be empty")
	}
	model := options.Model
	if model == "" {
		model = p.config.DefaultEmbeddingModel
	}
	if model == "" {
		return nil, errors.New("ducktape: no embedding model: set Model or DefaultEmbeddingModel")
	}
	req := EmbedRequest{Model: model, Input: input, Vendor: options.Vendor, Timeout: options.Timeout, IncludeRaw: options.IncludeRaw}
	if options.Dimensions != nil || options.Retries != nil {
		req.Options = &EmbedRequestOptions{Dimensions: options.Dimensions, Retries: options.Retries}
	}
	res, err := p.run(ctx, operation{"type": "embed", "request": req}, nil, nil)
	if err != nil {
		return nil, err
	}
	var out EmbedResponse
	if err := json.Unmarshal(res.Response, &out); err != nil {
		return nil, fmt.Errorf("ducktape: decode embed response: %w", err)
	}
	return &out, nil
}

// Models lists the models the endpoint serves; cached after the first call unless refresh is set.
func (p *Provider) Models(ctx context.Context, refresh bool) ([]ModelInfo, error) {
	p.mu.Lock()
	cached := p.models
	p.mu.Unlock()
	if cached != nil && !refresh {
		return cached, nil
	}
	res, err := p.run(ctx, operation{"type": "models", "refresh": refresh}, nil, nil)
	if err != nil {
		return nil, err
	}
	models := res.Models
	if models == nil {
		models = []ModelInfo{}
	}
	p.mu.Lock()
	p.models = models
	p.mu.Unlock()
	return models, nil
}

// ModelInfo describes one model.
func (p *Provider) ModelInfo(ctx context.Context, id string) (*ModelInfo, error) {
	p.mu.Lock()
	cached := p.infos[id]
	p.mu.Unlock()
	if cached != nil {
		return cached, nil
	}
	res, err := p.run(ctx, operation{"type": "model_info", "model": id}, nil, nil)
	if err != nil {
		return nil, err
	}
	if res.Info == nil {
		return nil, &ProviderError{Kind: ErrorKindNotFound, Message: "model " + id + " not found"}
	}
	p.mu.Lock()
	p.infos[id] = res.Info
	p.mu.Unlock()
	return res.Info, nil
}

// Capabilities tells what a model supports.
func (p *Provider) Capabilities(ctx context.Context, model string) (*Capabilities, error) {
	p.mu.Lock()
	cached := p.caps[model]
	p.mu.Unlock()
	if cached != nil {
		return cached, nil
	}
	res, err := p.run(ctx, operation{"type": "capabilities", "model": model}, nil, nil)
	if err != nil {
		return nil, err
	}
	if res.Capabilities == nil {
		return nil, &ProviderError{Kind: ErrorKindNotFound, Message: "no capabilities known for " + model}
	}
	p.mu.Lock()
	p.caps[model] = res.Capabilities
	p.mu.Unlock()
	return res.Capabilities, nil
}

// LoginOptions tune a login.
type LoginOptions struct {
	Callback string
	AppName  string
	Timeout  time.Duration
}

// Login logs into the provider's session source; handle shows the user what to do.
func (p *Provider) Login(ctx context.Context, handle func(LoginAction) (string, error), options *LoginOptions) (*AuthStatus, error) {
	op := operation{"type": "login"}
	if options != nil {
		if options.Callback != "" {
			op["callback"] = options.Callback
		}
		if options.AppName != "" {
			op["app_name"] = options.AppName
		}
		if options.Timeout > 0 {
			op["timeout_ms"] = options.Timeout.Milliseconds()
		}
	}
	res, err := p.run(ctx, op, nil, handle)
	if err != nil {
		return nil, err
	}
	return res.Status, nil
}

// Logout forgets the provider's own session.
func (p *Provider) Logout(ctx context.Context) error {
	_, err := p.run(ctx, operation{"type": "logout"}, nil, nil)
	return err
}

// AuthStatus reports which credential would be used and when it expires.
func (p *Provider) AuthStatus(ctx context.Context) (*AuthStatus, error) {
	res, err := p.run(ctx, operation{"type": "auth_status"}, nil, nil)
	if err != nil {
		return nil, err
	}
	if res.Status == nil {
		return &AuthStatus{}, nil
	}
	return res.Status, nil
}
