package ducktape

import (
	"context"
	"encoding/base64"
)

// Text makes a text block.
func Text(text string) Block {
	return Block{Type: "text", Text: text}
}

// Image makes an image block from bytes.
func Image(data []byte, mediaType string) Block {
	return Block{Type: "image", Data: base64.StdEncoding.EncodeToString(data), MediaType: mediaType}
}

// ImageURL makes an image block the vendor fetches.
func ImageURL(url string) Block {
	return Block{Type: "image", URL: url}
}

// Document makes a document block from bytes.
func Document(data []byte, mediaType, name string) Block {
	return Block{Type: "document", Data: base64.StdEncoding.EncodeToString(data), MediaType: mediaType, Name: name}
}

// ToolResult makes a tool result block for a call.
func ToolResult(toolCallID string, content ...Block) Block {
	if content == nil {
		content = []Block{}
	}
	return Block{Type: "tool_result", ToolCallID: toolCallID, Content: content}
}

// ToolResultText makes a tool result block holding one text block.
func ToolResultText(toolCallID, text string) Block {
	return ToolResult(toolCallID, Text(text))
}

// User makes a user message.
func User(content ...Block) Message {
	return Message{Role: RoleUser, Content: content}
}

// UserText makes a user message with one text block.
func UserText(text string) Message {
	return User(Text(text))
}

// System makes a system message.
func System(content ...Block) Message {
	return Message{Role: RoleSystem, Content: content}
}

// Assistant makes an assistant message.
func Assistant(content ...Block) Message {
	return Message{Role: RoleAssistant, Content: content}
}

// AssistantFrom makes the assistant message that appends a response to the history.
func AssistantFrom(r *Response) Message {
	return Message{Role: RoleAssistant, Content: r.Content}
}

// ToolMessage makes a tool message from result blocks.
func ToolMessage(results ...Block) Message {
	return Message{Role: RoleTool, Content: results}
}

// FunctionTool makes a function tool definition.
func FunctionTool(name, description string, inputSchema map[string]any) Tool {
	return Tool{Type: "function", Name: name, Description: description, InputSchema: inputSchema}
}

// Text joins the text blocks.
func (r *Response) Text() string {
	out := ""
	for _, b := range r.Content {
		if b.Type == "text" {
			out += b.Text
		}
	}
	return out
}

// Reasoning joins the reasoning blocks.
func (r *Response) Reasoning() string {
	out := ""
	for _, b := range r.Content {
		if b.Type == "reasoning" {
			out += b.Text
		}
	}
	return out
}

// ToolCalls returns the tool call blocks in order.
func (r *Response) ToolCalls() []Block {
	var out []Block
	for _, b := range r.Content {
		if b.Type == "tool_call" {
			out = append(out, b)
		}
	}
	return out
}

// Credential is one way to obtain a key: inline, from the environment, from a login session or from a resolver.
type Credential struct {
	source   AuthSource
	key      string
	resolver func(context.Context) (string, error)
}

// Key uses a key given in code.
func Key(secret string) Credential {
	return Credential{source: AuthSource{Type: "inline"}, key: secret}
}

// Env reads the key from the first of these environment variables that is set.
func Env(names ...string) Credential {
	return Credential{source: AuthSource{Type: "env", Names: names}}
}

// SessionOptions tune a login session source.
type SessionOptions struct {
	Store    string
	ClientID string
	Profile  string
	Refresh  AuthSourceRefresh
}

// Session uses a login session of the given preset.
func Session(preset string, options ...SessionOptions) Credential {
	src := AuthSource{Type: "session", Preset: AuthSourcePreset(preset)}
	for _, o := range options {
		src.Store, src.ClientID, src.Profile, src.Refresh = o.Store, o.ClientID, o.Profile, o.Refresh
	}
	return Credential{source: src}
}

// Resolver calls fn for the key at request time.
func Resolver(fn func(context.Context) (string, error)) Credential {
	return Credential{source: AuthSource{Type: "resolver"}, resolver: fn}
}

// Source uses any auth source document as it is.
func Source(src AuthSource) Credential {
	return Credential{source: src}
}

// WithKind sets how the credential is sent.
func (c Credential) WithKind(kind AuthSourceKind, placement AuthSourcePlacement) Credential {
	c.source.Kind, c.source.Placement = kind, placement
	return c
}
