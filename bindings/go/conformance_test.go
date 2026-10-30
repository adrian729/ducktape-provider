package ducktape

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"testing"
)

type scriptedHTTP struct {
	Status  int               `json:"status"`
	Headers map[string]string `json:"headers"`
	Body    *string           `json:"body"`
	Chunks  []string          `json:"chunks"`
	Error   *transportError   `json:"error"`
}

type exchange struct {
	Step  json.RawMessage `json:"step"`
	Input json.RawMessage `json:"input,omitempty"`
}

type scriptedWrite struct {
	Ok       bool `json:"ok"`
	Conflict bool `json:"conflict"`
}

type wireFixture struct {
	Name        string            `json:"name"`
	Start       json.RawMessage   `json:"start"`
	Secrets     map[string]string `json:"secrets"`
	Env         map[string]string `json:"env"`
	Files       map[string]string `json:"files"`
	Keychain    map[string]string `json:"keychain"`
	HTTP        []scriptedHTTP    `json:"http"`
	Writes      []scriptedWrite   `json:"writes"`
	UserActions []*string         `json:"user_actions"`
	Callbacks   []string          `json:"callbacks"`
	Exchange    []exchange        `json:"exchange"`
}

var entropy = base64.StdEncoding.EncodeToString(bytes.Repeat([]byte{128}, 32))

func fixtureClock(tick int64) clockDoc {
	return clockDoc{NowMs: 1_761_696_000_000 + tick*100, Entropy: entropy}
}

func fixturesDir(t *testing.T) string {
	t.Helper()
	dir, err := filepath.Abs(filepath.Join("..", "..", "fixtures", "wire"))
	if err != nil {
		t.Fatal(err)
	}
	return dir
}

func loadFixture(t *testing.T, path string) wireFixture {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var f wireFixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("%s: %v", path, err)
	}
	return f
}

func wireFixtures(t *testing.T) []string {
	t.Helper()
	paths, err := filepath.Glob(filepath.Join(fixturesDir(t), "*", "*.json"))
	if err != nil {
		t.Fatal(err)
	}
	sort.Strings(paths)
	if len(paths) == 0 {
		t.Fatal("no wire fixtures found")
	}
	return paths
}

type actual struct {
	step  []byte
	input any
}

func replay(t *testing.T, core *Core, f wireFixture) []actual {
	t.Helper()
	ctx := context.Background()
	in, err := core.instantiate(ctx)
	if err != nil {
		t.Fatal(err)
	}
	defer in.close(ctx)
	var start map[string]any
	if err := json.Unmarshal(f.Start, &start); err != nil {
		t.Fatal(err)
	}
	start["clock"] = fixtureClock(0)
	s, err := in.start(ctx, start)
	if err != nil {
		t.Fatal(err)
	}
	var out []actual
	var queue []input
	http := f.HTTP
	writes := f.Writes
	userActions := f.UserActions
	callbacks := f.Callbacks
	var tick int64
	for i := 0; i < 10_000; i++ {
		e := s.Effect
		var next input
		switch e.Type {
		case "done", "failed":
			return append(out, actual{step: s.raw})
		case "need_secret":
			value, ok := f.Secrets[e.SecretRef]
			if ok {
				if err := in.setSecret(ctx, e.SecretRef, []byte(value)); err != nil {
					t.Fatal(err)
				}
			}
			next = input{Type: "secret", SecretRef: e.SecretRef, Found: boolPtr(ok)}
		case "need_http":
			if len(http) == 0 {
				t.Fatalf("%s: script has no HTTP reply for request", f.Name)
			}
			h := http[0]
			http = http[1:]
			if h.Error != nil {
				next = input{Type: "http_error", Error: h.Error}
			} else {
				chunks := h.Chunks
				if len(chunks) == 0 && h.Body != nil {
					chunks = []string{*h.Body}
				}
				queue = queue[:0]
				queue = append(queue, input{Type: "http_end"})
				for i := len(chunks) - 1; i >= 0; i-- {
					queue = append(queue, input{Type: "http_chunk", Bytes: base64.StdEncoding.EncodeToString([]byte(chunks[i]))})
				}
				status := h.Status
				if status == 0 {
					status = 200
				}
				headers := h.Headers
				if headers == nil {
					headers = map[string]string{}
				}
				next = input{Type: "http_response", Status: status, Headers: headers}
			}
		case "receive":
			if len(queue) == 0 {
				next = input{Type: "http_end"}
			} else {
				next = queue[len(queue)-1]
				queue = queue[:len(queue)-1]
			}
		case "need_sleep":
			next = input{Type: "slept"}
		case "need_read":
			var value string
			var ok bool
			switch e.Locator.Type {
			case "file":
				value, ok = f.Files[e.Locator.Path]
			case "env":
				value, ok = f.Env[e.Locator.Name]
			case "keychain":
				value, ok = f.Keychain[e.Locator.Service+"|"+e.Locator.Account]
			}
			next = input{Type: "read", Found: boolPtr(ok)}
			if ok {
				next.Bytes = base64.StdEncoding.EncodeToString([]byte(value))
				if e.Locator.Type == "file" {
					next.Etag = etag([]byte(value))
				}
			}
		case "need_write":
			w := scriptedWrite{Ok: true}
			if len(writes) > 0 {
				w, writes = writes[0], writes[1:]
			}
			next = input{Type: "written", Ok: boolPtr(w.Ok), Conflict: boolPtr(w.Conflict)}
		case "need_user_action":
			next = input{Type: "user_action"}
			if len(userActions) > 0 {
				if userActions[0] != nil {
					next.Value = *userActions[0]
				}
				userActions = userActions[1:]
			}
		case "need_callback":
			if len(callbacks) == 0 {
				t.Fatalf("%s: script has no callback for the login", f.Name)
			}
			next = input{Type: "callback", URL: callbacks[0]}
			callbacks = callbacks[1:]
		default:
			t.Fatalf("%s: unsupported effect in replay: %s", f.Name, e.Type)
		}
		out = append(out, actual{step: s.raw, input: next})
		tick++
		s, err = in.resume(ctx, resumeDoc{Clock: fixtureClock(tick), Input: next})
		if err != nil {
			t.Fatal(err)
		}
	}
	t.Fatalf("%s: replay did not finish", f.Name)
	return nil
}

func canonical(t *testing.T, raw []byte) string {
	t.Helper()
	if len(raw) == 0 || string(raw) == "null" {
		return "null"
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		t.Fatal(err)
	}
	if m, ok := v.(map[string]any); ok {
		if effect, ok := m["effect"].(map[string]any); ok {
			if request, ok := effect["request"].(map[string]any); ok {
				if body, ok := request["body"].(string); ok {
					bd := json.NewDecoder(bytes.NewReader([]byte(body)))
					bd.UseNumber()
					var parsed any
					if bd.Decode(&parsed) == nil {
						request["body"] = parsed
					}
				}
			}
		}
	}
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestConformance(t *testing.T) {
	core, err := DefaultCore(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for _, path := range wireFixtures(t) {
		f := loadFixture(t, path)
		name := filepath.Base(filepath.Dir(path)) + "/" + f.Name
		t.Run(name, func(t *testing.T) {
			got := replay(t, core, f)
			if len(got) != len(f.Exchange) {
				t.Fatalf("length %d != %d; last %s", len(got), len(f.Exchange), got[len(got)-1].step)
			}
			for i := range got {
				if a, e := canonical(t, got[i].step), canonical(t, f.Exchange[i].Step); a != e {
					t.Fatalf("step %d:\n%s\n!=\n%s", i, a, e)
				}
				var ai []byte
				if got[i].input != nil {
					ai, _ = json.Marshal(got[i].input)
				}
				if a, e := canonical(t, ai), canonical(t, f.Exchange[i].Input); a != e {
					t.Fatalf("input %d:\n%s\n!=\n%s", i, a, e)
				}
			}
		})
	}
}
