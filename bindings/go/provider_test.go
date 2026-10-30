package ducktape

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

type seenRequest struct {
	URL     string
	Headers http.Header
	Body    map[string]any
}

type scripted struct {
	mu     sync.Mutex
	script []scriptedHTTP
	seen   []seenRequest
	delay  time.Duration
}

type slowReader struct {
	chunks [][]byte
	delay  time.Duration
	ctx    context.Context
}

func (s *slowReader) Read(p []byte) (int, error) {
	if len(s.chunks) == 0 {
		return 0, io.EOF
	}
	if s.delay > 0 {
		select {
		case <-time.After(s.delay):
		case <-s.ctx.Done():
			return 0, s.ctx.Err()
		}
	}
	n := copy(p, s.chunks[0])
	if n < len(s.chunks[0]) {
		s.chunks[0] = s.chunks[0][n:]
	} else {
		s.chunks = s.chunks[1:]
	}
	return n, nil
}

func (s *scripted) RoundTrip(req *http.Request) (*http.Response, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	seen := seenRequest{URL: req.URL.String(), Headers: req.Header}
	if req.Body != nil {
		b, _ := io.ReadAll(req.Body)
		_ = json.Unmarshal(b, &seen.Body)
	}
	s.seen = append(s.seen, seen)
	if len(s.script) == 0 {
		return nil, errors.New("script exhausted")
	}
	h := s.script[0]
	s.script = s.script[1:]
	if h.Error != nil {
		return nil, errors.New(h.Error.Message)
	}
	chunks := h.Chunks
	if len(chunks) == 0 && h.Body != nil {
		chunks = []string{*h.Body}
	}
	var raw [][]byte
	for _, c := range chunks {
		raw = append(raw, []byte(c))
	}
	status := h.Status
	if status == 0 {
		status = 200
	}
	header := http.Header{}
	for k, v := range h.Headers {
		header.Set(k, v)
	}
	return &http.Response{StatusCode: status, Header: header, Body: io.NopCloser(&slowReader{chunks: raw, delay: s.delay, ctx: req.Context()}), Request: req}, nil
}

func anthropicFixture(t *testing.T, name string) wireFixture {
	t.Helper()
	return loadFixture(t, filepath.Join(fixturesDir(t), "anthropic.messages", name+".json"))
}

func openaiFixture(t *testing.T, name string) wireFixture {
	t.Helper()
	return loadFixture(t, filepath.Join(fixturesDir(t), "openai.responses", name+".json"))
}

type startRequest struct {
	Provider  ProviderConfig `json:"provider"`
	Operation struct {
		Request ChatRequest `json:"request"`
	} `json:"operation"`
}

func requestOf(t *testing.T, f wireFixture) (ProviderConfig, ChatRequest) {
	t.Helper()
	var s startRequest
	if err := json.Unmarshal(f.Start, &s); err != nil {
		t.Fatal(err)
	}
	return s.Provider, s.Operation.Request
}

func doneResponse(t *testing.T, f wireFixture) Response {
	t.Helper()
	var last struct {
		Effect struct {
			Result struct {
				Response Response `json:"response"`
			} `json:"result"`
		} `json:"effect"`
	}
	if err := json.Unmarshal(f.Exchange[len(f.Exchange)-1].Step, &last); err != nil {
		t.Fatal(err)
	}
	return last.Effect.Result.Response
}

func newProvider(t *testing.T, f wireFixture, cfg Config) (*Provider, *scripted) {
	t.Helper()
	tr := &scripted{script: f.HTTP}
	conf, _ := requestOf(t, f)
	cfg.BaseURL = conf.BaseURL
	cfg.Client = &http.Client{Transport: tr}
	if cfg.Auth == nil {
		cfg.Auth = []Credential{Key("sk-ant-test-key")}
	}
	p, err := New(string(conf.NS), cfg)
	if err != nil {
		t.Fatal(err)
	}
	return p, tr
}

func withoutAuthSource(content []Block) []Block {
	out := make([]Block, len(content))
	for i, b := range content {
		if b.Origin != nil {
			o := *b.Origin
			o.AuthSource = ""
			b.Origin = &o
		}
		out[i] = b
	}
	return out
}

func mustJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestChatMatchesFixture(t *testing.T) {
	f := anthropicFixture(t, "chat_text_reasoning")
	p, tr := newProvider(t, f, Config{})
	_, req := requestOf(t, f)
	r, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model, Options: req.Options})
	if err != nil {
		t.Fatal(err)
	}
	expected := doneResponse(t, f)
	if got, want := mustJSON(t, withoutAuthSource(r.Content)), mustJSON(t, withoutAuthSource(expected.Content)); got != want {
		t.Fatalf("content %s != %s", got, want)
	}
	if r.Stop.Reason != expected.Stop.Reason || mustJSON(t, r.Usage) != mustJSON(t, expected.Usage) {
		t.Fatalf("stop or usage differ: %+v", r)
	}
	if r.Text() == "" || r.Reasoning() == "" {
		t.Fatal("text views are empty")
	}
	if tr.seen[0].Headers.Get("x-api-key") != "sk-ant-test-key" || tr.seen[0].Body["model"] != req.Model {
		t.Fatalf("request not as expected: %+v", tr.seen[0])
	}
}

func TestStreamEndsWithFinish(t *testing.T) {
	f := anthropicFixture(t, "stream_text_reasoning_chunked")
	p, _ := newProvider(t, f, Config{})
	_, req := requestOf(t, f)
	s, err := p.Stream(context.Background(), req.Messages, &ChatOptions{Model: req.Model})
	if err != nil {
		t.Fatal(err)
	}
	var events []Event
	for s.Next() {
		events = append(events, s.Event())
	}
	if err := s.Err(); err != nil {
		t.Fatal(err)
	}
	if events[0].Type != "block_start" || events[len(events)-1].Type != "finish" {
		t.Fatalf("unexpected event order: %s", mustJSON(t, events))
	}
	text := ""
	for _, e := range events {
		if e.Type == "text_delta" {
			text += e.Text
		}
	}
	if text == "" || text != s.Response().Text() {
		t.Fatalf("text %q != response %q", text, s.Response().Text())
	}
}

func TestToolRoundTripBuildsHistory(t *testing.T) {
	f := anthropicFixture(t, "chat_tool_call")
	p, _ := newProvider(t, f, Config{})
	_, req := requestOf(t, f)
	r, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model, Tools: req.Tools})
	if err != nil {
		t.Fatal(err)
	}
	if r.Stop.Reason != StopReasonToolCall {
		t.Fatalf("stop %s", r.Stop.Reason)
	}
	call := r.ToolCalls()[0]
	history := append(append([]Message{}, req.Messages...), AssistantFrom(r), ToolMessage(ToolResultText(call.ID, "4")))
	if history[len(history)-2].Role != RoleAssistant || history[len(history)-1].Content[0].ToolCallID != call.ID {
		t.Fatalf("history %s", mustJSON(t, history))
	}
}

func TestErrorsMapToKinds(t *testing.T) {
	f := anthropicFixture(t, "chat_context_overflow")
	p, _ := newProvider(t, f, Config{})
	_, req := requestOf(t, f)
	_, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model})
	if !errors.Is(err, ErrContextOverflow) {
		t.Fatalf("got %v", err)
	}
	pe, ok := AsProviderError(err)
	if !ok || pe.Status == nil || *pe.Status != 400 || !strings.Contains(err.Error(), "context_overflow") {
		t.Fatalf("got %+v", pe)
	}
}

func TestNoCredentialIsAuthError(t *testing.T) {
	f := anthropicFixture(t, "chat_no_credential")
	p, tr := newProvider(t, f, Config{Auth: []Credential{Env("DUCKTAPE_TEST_UNSET_KEY")}})
	_, err := p.Chat(context.Background(), []Message{UserText("hi")}, &ChatOptions{Model: "m"})
	if !errors.Is(err, ErrAuth) || len(tr.seen) != 0 {
		t.Fatalf("got %v after %d requests", err, len(tr.seen))
	}
}

func TestEnvCredential(t *testing.T) {
	f := anthropicFixture(t, "chat_text_reasoning")
	t.Setenv("DUCKTAPE_TEST_KEY", "sk-from-env")
	p, tr := newProvider(t, f, Config{Auth: []Credential{Env("DUCKTAPE_TEST_MISSING", "DUCKTAPE_TEST_KEY")}})
	_, req := requestOf(t, f)
	if _, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model}); err != nil {
		t.Fatal(err)
	}
	if tr.seen[0].Headers.Get("x-api-key") != "sk-from-env" {
		t.Fatalf("headers %v", tr.seen[0].Headers)
	}
}

func TestResolverCredential(t *testing.T) {
	f := anthropicFixture(t, "chat_text_reasoning")
	calls := 0
	p, tr := newProvider(t, f, Config{Auth: []Credential{Resolver(func(context.Context) (string, error) { calls++; return "sk-resolved", nil })}})
	_, req := requestOf(t, f)
	if _, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model}); err != nil {
		t.Fatal(err)
	}
	if calls != 1 || tr.seen[0].Headers.Get("x-api-key") != "sk-resolved" {
		t.Fatalf("calls %d headers %v", calls, tr.seen[0].Headers)
	}
	failing, _ := newProvider(t, f, Config{Auth: []Credential{Resolver(func(context.Context) (string, error) { return "", errors.New("vault down") })}})
	_, err := failing.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model})
	if !errors.Is(err, ErrAuth) || !strings.Contains(err.Error(), "vault down") {
		t.Fatalf("got %v", err)
	}
}

func TestArgumentErrorsComeFirst(t *testing.T) {
	f := anthropicFixture(t, "chat_text_reasoning")
	p, tr := newProvider(t, f, Config{})
	if _, err := p.Chat(context.Background(), []Message{UserText("hi")}, nil); err == nil || len(tr.seen) != 0 {
		t.Fatalf("expected a model error before any request, got %v", err)
	}
	if _, err := New("nope", Config{}); err == nil {
		t.Fatal("unknown namespace accepted")
	}
	if _, err := p.Embed(context.Background(), nil, nil); err == nil {
		t.Fatal("empty embed accepted")
	}
}

func TestConnectionFailureRetriesThenSucceeds(t *testing.T) {
	f := anthropicFixture(t, "chat_connection_reset_retried")
	p, tr := newProvider(t, f, Config{})
	_, req := requestOf(t, f)
	r, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model})
	if err != nil {
		t.Fatal(err)
	}
	if len(tr.seen) != 2 || r.Text() == "" {
		t.Fatalf("seen %d", len(tr.seen))
	}
	exhausted, _ := newProvider(t, f, Config{Client: nil})
	exhausted.host.client = &http.Client{Transport: &scripted{script: []scriptedHTTP{{Error: &transportError{Kind: "connect", Message: "connection refused"}}, {Error: &transportError{Kind: "connect", Message: "connection refused"}}, {Error: &transportError{Kind: "connect", Message: "connection refused"}}}}}
	_, err = exhausted.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model, Options: &Options{Retries: new(int64)}})
	if !errors.Is(err, ErrConnection) {
		t.Fatalf("got %v", err)
	}
}

func TestModelsAreCached(t *testing.T) {
	f := anthropicFixture(t, "models_two_pages")
	p, tr := newProvider(t, f, Config{})
	models, err := p.Models(context.Background(), false)
	if err != nil {
		t.Fatal(err)
	}
	again, err := p.Models(context.Background(), false)
	if err != nil {
		t.Fatal(err)
	}
	if len(models) == 0 || len(again) != len(models) || len(tr.seen) != 2 {
		t.Fatalf("models %d seen %d", len(models), len(tr.seen))
	}
}

func TestCapabilitiesShipped(t *testing.T) {
	f := anthropicFixture(t, "capabilities_shipped")
	p, tr := newProvider(t, f, Config{})
	var op struct {
		Operation struct {
			Model string `json:"model"`
		} `json:"operation"`
	}
	_ = json.Unmarshal(f.Start, &op)
	caps, err := p.Capabilities(context.Background(), op.Operation.Model)
	if err != nil {
		t.Fatal(err)
	}
	if caps.Source != CapabilitiesSourceShipped || len(tr.seen) != 0 {
		t.Fatalf("caps %+v seen %d", caps, len(tr.seen))
	}
}

func TestStreamCloseCancels(t *testing.T) {
	f := anthropicFixture(t, "stream_text_reasoning_chunked")
	p, tr := newProvider(t, f, Config{})
	tr.delay = 20 * time.Millisecond
	_, req := requestOf(t, f)
	s, err := p.Stream(context.Background(), req.Messages, &ChatOptions{Model: req.Model})
	if err != nil {
		t.Fatal(err)
	}
	if !s.Next() {
		t.Fatal("no first event")
	}
	s.Close()
	if err := s.Err(); !errors.Is(err, ErrCancelled) {
		t.Fatalf("got %v", err)
	}
}

func TestContextCancelsChat(t *testing.T) {
	f := anthropicFixture(t, "chat_text_reasoning")
	p, tr := newProvider(t, f, Config{})
	tr.delay = 200 * time.Millisecond
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(30 * time.Millisecond)
		cancel()
	}()
	_, req := requestOf(t, f)
	_, err := p.Chat(ctx, req.Messages, &ChatOptions{Model: req.Model})
	if !errors.Is(err, ErrCancelled) {
		t.Fatalf("got %v", err)
	}
}

func TestTimeoutMapsToTimeout(t *testing.T) {
	f := anthropicFixture(t, "chat_text_reasoning")
	p, tr := newProvider(t, f, Config{})
	tr.delay = 300 * time.Millisecond
	_, req := requestOf(t, f)
	read := 0.05
	_, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model, Timeout: &Timeout{Read: &read}, Options: &Options{Retries: new(int64)}})
	if !errors.Is(err, ErrTimeout) {
		t.Fatalf("got %v", err)
	}
}

func TestEmbed(t *testing.T) {
	f := openaiFixture(t, "embed_two_inputs")
	p, _ := newProvider(t, f, Config{Auth: []Credential{Key("sk-test")}})
	var op struct {
		Operation struct {
			Request EmbedRequest `json:"request"`
		} `json:"operation"`
	}
	_ = json.Unmarshal(f.Start, &op)
	r, err := p.Embed(context.Background(), op.Operation.Request.Input, &EmbedOptions{Model: op.Operation.Request.Model})
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Embeddings) != 2 || r.Embeddings[0][0] != 0.1 {
		t.Fatalf("embeddings %v", r.Embeddings)
	}
}

func TestImageBytesBecomeBase64(t *testing.T) {
	png := []byte{0x89, 0x50, 0x4e, 0x47}
	m := User(Text("what is this?"), Image(png, "image/png"))
	if m.Content[1].Data != base64.StdEncoding.EncodeToString(png) || m.Content[0].Text != "what is this?" {
		t.Fatalf("%s", mustJSON(t, m))
	}
}

func b64url(v any) string {
	b, _ := json.Marshal(v)
	return base64.RawURLEncoding.EncodeToString(b)
}

func freshJWT(exp time.Time) string {
	return b64url(map[string]string{"alg": "RS256"}) + "." + b64url(map[string]int64{"exp": exp.Unix()}) + ".sig"
}

func sessionStore(t *testing.T, f wireFixture, dir, prefix string) {
	t.Helper()
	for path, text := range f.Files {
		target := filepath.Join(dir, strings.TrimPrefix(path, prefix))
		if err := os.MkdirAll(filepath.Dir(target), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(target, []byte(text), 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

func TestCodexSessionRefreshesAndWritesBack(t *testing.T) {
	f := openaiFixture(t, "session_codex_refresh_write_back")
	var token map[string]any
	_ = json.Unmarshal([]byte(*f.HTTP[0].Body), &token)
	token["access_token"] = freshJWT(time.Now().Add(24 * time.Hour))
	body := mustJSON(t, token)
	f.HTTP[0].Body = &body
	dir := t.TempDir()
	sessionStore(t, f, dir, "/home/u/.codex")
	t.Setenv("CODEX_HOME", dir)
	p, tr := newProvider(t, f, Config{Auth: []Credential{Session("codex", SessionOptions{Refresh: AuthSourceRefreshWriteBack})}})
	_, req := requestOf(t, f)
	r, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model})
	if err != nil {
		t.Fatal(err)
	}
	expected := doneResponse(t, f)
	if r.Text() != expected.Content[len(expected.Content)-1].Text {
		t.Fatalf("text %q", r.Text())
	}
	if tr.seen[0].URL != "https://auth.openai.com/oauth/token" || tr.seen[1].URL != "https://chatgpt.com/backend-api/codex/responses" || tr.seen[1].Headers.Get("chatgpt-account-id") != "acct_1" {
		t.Fatalf("requests %+v", tr.seen)
	}
	written, err := os.ReadFile(filepath.Join(dir, "auth.json"))
	if err != nil {
		t.Fatal(err)
	}
	var doc struct {
		Tokens struct {
			RefreshToken string `json:"refresh_token"`
		} `json:"tokens"`
	}
	_ = json.Unmarshal(written, &doc)
	if doc.Tokens.RefreshToken != "rt_d" {
		t.Fatalf("written %s", written)
	}
	info, _ := os.Stat(filepath.Join(dir, "auth.json"))
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("mode %v", info.Mode())
	}
	status, err := p.AuthStatus(context.Background())
	if err != nil || status.Session != "codex" {
		t.Fatalf("status %+v %v", status, err)
	}
}

func TestExpiredBorrowedSessionNamesTheOwner(t *testing.T) {
	f := openaiFixture(t, "session_codex_expired_never")
	dir := t.TempDir()
	sessionStore(t, f, dir, "/home/u/.codex")
	t.Setenv("CODEX_HOME", dir)
	p, _ := newProvider(t, f, Config{Auth: []Credential{Session("codex")}})
	_, err := p.Chat(context.Background(), []Message{UserText("hi")}, &ChatOptions{Model: "m"})
	pe, ok := AsProviderError(err)
	if !ok || !errors.Is(err, ErrAuth) || pe.Code != "credential_expired" || !strings.Contains(pe.Message, "codex login") {
		t.Fatalf("got %v", err)
	}
	status, err := p.AuthStatus(context.Background())
	if err != nil || status.Source == nil || *status.Source != "session:codex" || status.ExpiresAt != "2025-10-28T23:59:55Z" {
		t.Fatalf("status %+v %v", status, err)
	}
}

func TestCodexKeychainFallback(t *testing.T) {
	f := openaiFixture(t, "session_codex_keychain_fallback")
	t.Setenv("CODEX_HOME", t.TempDir())
	var service, account, stored string
	for k, v := range f.Keychain {
		parts := strings.SplitN(k, "|", 2)
		service, account, stored = parts[0], parts[1], v
	}
	var doc map[string]any
	_ = json.Unmarshal([]byte(stored), &doc)
	doc["tokens"].(map[string]any)["access_token"] = freshJWT(time.Now().Add(time.Hour))
	var asked [][2]string
	keychain := func(ctx context.Context, s, a string) ([]byte, error) {
		asked = append(asked, [2]string{s, a})
		if s == service {
			return []byte(mustJSON(t, doc)), nil
		}
		return nil, nil
	}
	p, tr := newProvider(t, f, Config{Auth: []Credential{Session("codex")}, Keychain: keychain})
	_, req := requestOf(t, f)
	r, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model})
	if err != nil {
		t.Fatal(err)
	}
	expected := doneResponse(t, f)
	if r.Text() != expected.Content[len(expected.Content)-1].Text {
		t.Fatalf("text %q", r.Text())
	}
	if asked[0][0] != "Codex Auth" || !strings.HasPrefix(asked[0][1], "cli|") || asked[0][1] == account {
		t.Fatalf("asked %v", asked)
	}
	if tr.seen[0].Headers.Get("chatgpt-account-id") != "acct_1" {
		t.Fatalf("headers %v", tr.seen[0].Headers)
	}
}

func TestDefaultKeychainReaderNeverFails(t *testing.T) {
	value, err := ReadKeychain(context.Background(), "ducktape-provider-test", "nobody")
	if err != nil || len(value) != 0 {
		t.Fatalf("value %q err %v", value, err)
	}
}

func TestSharedCore(t *testing.T) {
	core, err := NewCore(context.Background(), coreWasm)
	if err != nil {
		t.Fatal(err)
	}
	defer core.Close(context.Background())
	f := anthropicFixture(t, "chat_text_reasoning")
	_, req := requestOf(t, f)
	for i := 0; i < 2; i++ {
		p, _ := newProvider(t, f, Config{Core: core})
		if _, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model}); err != nil {
			t.Fatal(err)
		}
	}
}

func TestObserverSeesEvents(t *testing.T) {
	f := anthropicFixture(t, "chat_text_reasoning")
	var hooks []string
	obs := Observer{OnRequest: func(json.RawMessage) { hooks = append(hooks, "request") }, OnResponse: func(json.RawMessage) { hooks = append(hooks, "response") }}
	p, _ := newProvider(t, f, Config{Observer: obs})
	_, req := requestOf(t, f)
	if _, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model}); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains([]byte(strings.Join(hooks, ",")), []byte("request")) {
		t.Fatalf("hooks %v", hooks)
	}
}

func TestWriteConflictKeepsRefreshedToken(t *testing.T) {
	f := anthropicFixture(t, "session_profile_write_conflict_keeps_token")
	dir := t.TempDir()
	sessionStore(t, f, dir, "/home/u/.config/anthropic")
	t.Setenv("ANTHROPIC_CONFIG_DIR", dir)
	t.Setenv("ANTHROPIC_PROFILE", "")
	creds := filepath.Join(dir, "credentials", "default.json")
	tr := &scripted{script: f.HTTP}
	rt := roundTripFunc(func(req *http.Request) (*http.Response, error) {
		if strings.HasSuffix(req.URL.Path, "/v1/oauth/token") {
			_ = os.WriteFile(creds, []byte(`{"access_token": "sk-ant-oat01-other-process", "expires_at": 4102444800}`), 0o600)
		}
		return tr.RoundTrip(req)
	})
	p, err := New("anthropic", Config{Auth: []Credential{Session("anthropic_profile")}, Client: &http.Client{Transport: rt}})
	if err != nil {
		t.Fatal(err)
	}
	_, req := requestOf(t, f)
	r, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model})
	if err != nil {
		t.Fatal(err)
	}
	expected := doneResponse(t, f)
	if r.Text() != expected.Content[len(expected.Content)-1].Text {
		t.Fatalf("text %q", r.Text())
	}
	after, _ := os.ReadFile(creds)
	if !strings.Contains(string(after), "other-process") {
		t.Fatalf("the conflicting write was overwritten: %s", after)
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(req *http.Request) (*http.Response, error) { return f(req) }

func TestCodexKeychainWriteBack(t *testing.T) {
	f := openaiFixture(t, "session_codex_keychain_write_back")
	t.Setenv("CODEX_HOME", t.TempDir())
	var service, stored string
	for k, v := range f.Keychain {
		service, stored = strings.SplitN(k, "|", 2)[0], v
	}
	var doc map[string]any
	_ = json.Unmarshal([]byte(stored), &doc)
	doc["tokens"].(map[string]any)["access_token"] = freshJWT(time.Now().Add(time.Minute))
	var token map[string]any
	_ = json.Unmarshal([]byte(*f.HTTP[0].Body), &token)
	token["access_token"] = freshJWT(time.Now().Add(24 * time.Hour))
	body := mustJSON(t, token)
	f.HTTP[0].Body = &body
	var written [][]string
	reader := func(ctx context.Context, s, a string) ([]byte, error) {
		if s == service {
			return []byte(mustJSON(t, doc)), nil
		}
		return nil, nil
	}
	writer := func(ctx context.Context, s, a string, value []byte) (bool, error) {
		written = append(written, []string{s, a, string(value)})
		return true, nil
	}
	p, tr := newProvider(t, f, Config{Auth: []Credential{Session("codex", SessionOptions{Refresh: AuthSourceRefreshWriteBack})}, Keychain: reader, KeychainWrite: writer})
	_, req := requestOf(t, f)
	r, err := p.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model})
	if err != nil {
		t.Fatal(err)
	}
	expected := doneResponse(t, f)
	if r.Text() != expected.Content[len(expected.Content)-1].Text || len(tr.seen) != 2 {
		t.Fatalf("text %q seen %d", r.Text(), len(tr.seen))
	}
	if len(written) != 1 || written[0][0] != "Codex Auth" || !strings.HasPrefix(written[0][1], "cli|") || !strings.Contains(written[0][2], "rt_d") {
		t.Fatalf("written %v", written)
	}
	readerOnly, _ := newProvider(t, f, Config{Auth: []Credential{Session("codex", SessionOptions{Refresh: AuthSourceRefreshWriteBack})}, Keychain: reader})
	if readerOnly.host.keychainWrite != nil {
		t.Fatal("a custom reader must not get the default writer")
	}
	again, err := readerOnly.Chat(context.Background(), req.Messages, &ChatOptions{Model: req.Model})
	if err != nil || again.Text() != r.Text() {
		t.Fatalf("reader-only chat: %v", err)
	}
}
