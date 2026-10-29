export type Ns = "anthropic" | "openai" | "google" | "ollama";
export type Role = "user" | "assistant" | "system" | "tool" | (string & {});
export type CacheTtl = "short" | "long" | (string & {});
export type ReasoningLevel = "auto" | "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | (string & {});
export type StopReason =
  | "complete"
  | "tool_call"
  | "max_output_tokens"
  | "stop_sequence"
  | "context_window_exceeded"
  | "refusal"
  | "content_filtered"
  | "continue_required"
  | "compaction"
  | "malformed_output"
  | "other"
  | (string & {});
export type ErrorKind =
  | "auth"
  | "rate_limit"
  | "quota"
  | "server"
  | "connection"
  | "timeout"
  | "cancelled"
  | "not_found"
  | "context_overflow"
  | "request_too_large"
  | "bad_request"
  | "malformed_response"
  | "unsupported"
  | (string & {});

export interface Origin {
  wire: string;
  ns: Ns | (string & {});
  base_url: string;
  model: string;
  auth_source?: string;
}

export interface CacheMark {
  ttl?: CacheTtl;
}

export type VendorMap = Record<string, Record<string, unknown>>;

interface BlockMeta {
  id?: string;
  cache?: CacheMark;
  vendor?: VendorMap;
  provider_executed?: boolean;
}

export interface TextBlock extends BlockMeta {
  type: "text";
  text: string;
}
export interface ImageBlock extends BlockMeta {
  type: "image";
  data?: string;
  url?: string;
  file_id?: string;
  media_type?: string;
}
export interface DocumentBlock extends BlockMeta {
  type: "document";
  data?: string;
  url?: string;
  file_id?: string;
  text?: string;
  media_type?: string;
  name?: string;
}
export interface ToolCallBlock extends BlockMeta {
  type: "tool_call";
  id: string;
  name: string;
  arguments: unknown;
  truncated?: boolean;
  arguments_json?: string;
  status?: string;
}
export interface ToolResultBlock extends BlockMeta {
  type: "tool_result";
  tool_call_id: string;
  content: Block[];
  is_error?: boolean;
}
export interface ReasoningBlock extends BlockMeta {
  type: "reasoning";
  text: string;
  vendor_state?: unknown;
  origin?: Origin;
}
export interface CompactionBlock extends BlockMeta {
  type: "compaction";
  vendor_state: unknown;
  text?: string;
  origin: Origin;
}
export interface CustomBlock extends BlockMeta {
  type: "custom";
  kind: string;
  data?: unknown;
  origin?: Origin;
}
export type Block =
  | TextBlock
  | ImageBlock
  | DocumentBlock
  | ToolCallBlock
  | ToolResultBlock
  | ReasoningBlock
  | CompactionBlock
  | CustomBlock;

export interface Message {
  role: Role;
  content: Block[];
}

export interface ToolDef {
  type: "function";
  name: string;
  description: string;
  input_schema: unknown;
  strict?: boolean;
  cache?: CacheMark;
  vendor?: VendorMap;
}
export interface ProviderTool {
  type: "provider";
  provider_tool: string;
  config?: unknown;
}
export type Tool = ToolDef | ProviderTool;

export interface ToolChoice {
  mode: "auto" | "none" | "required" | "named" | (string & {});
  name?: string;
}
export interface OutputFormat {
  json_schema: unknown;
  name?: string;
}
export interface Timeout {
  connect?: number;
  read?: number;
  total?: number;
}
export interface Options {
  max_output_tokens?: number;
  reasoning?: ReasoningLevel;
  reasoning_summary?: "none" | "auto" | (string & {});
  tool_choice?: ToolChoice | "auto" | "none" | "required" | { name: string };
  parallel_tools?: boolean;
  output?: OutputFormat;
  stop_sequences?: string[];
  cache?: "none" | "auto" | "explicit" | (string & {});
  retries?: number;
  user_id?: string;
}

export interface Usage {
  input: { total: number; cache_read?: number; cache_write?: number };
  output: { total: number; reasoning?: number };
  raw?: unknown;
}
export interface Warning {
  kind: "unsupported" | "compatibility" | "deprecated" | (string & {});
  feature: string;
  detail?: string;
}
export interface Response {
  content: Block[];
  stop: { reason: StopReason; raw?: unknown };
  usage: Usage | null;
  warnings: Warning[];
  origin: Origin;
  latency_ms: number;
  first_token_ms?: number;
  raw?: unknown;
  readonly text: string;
  readonly reasoning: string;
  readonly toolCalls: ToolCallBlock[];
}

export type Event =
  | { type: "block_start"; index: number; kind: string; id?: string; name?: string; parent_id?: string }
  | { type: "text_delta"; index: number; text: string; parent_id?: string }
  | { type: "reasoning_delta"; index: number; text: string; parent_id?: string }
  | { type: "tool_call_delta"; index: number; arguments_json: string; parent_id?: string }
  | { type: "block_stop"; index: number; block: Block; parent_id?: string }
  | { type: "finish"; response: Response; parent_id?: string }
  | { type: "raw"; event: unknown; parent_id?: string };

export interface EmbedResponse {
  embeddings: number[][];
  usage: { input: { total: number } } | null;
  warnings: Warning[];
  origin: Origin;
  latency_ms: number;
  raw?: unknown;
}

export interface Limits {
  context_window?: number;
  max_output_tokens?: number;
}
export interface Capabilities {
  reasoning?: { levels?: ReasoningLevel[] | null; can_disable?: boolean | null };
  tool_choice?: string[] | null;
  parallel_tools?: { supported?: boolean | null; toggle?: boolean | null };
  output?: { json_schema?: boolean | null };
  cache?: { breakpoints?: boolean | null; ttls?: CacheTtl[] | null };
  inputs?: string[] | null;
  tools?: { functions?: boolean | null; hosted?: string[] | null };
  limits?: Limits;
  embeddings?: boolean | null;
  source: "shipped" | "vendor" | "user" | (string & {});
}
export interface ModelInfo {
  id: string;
  kind: "chat" | "embedding" | "unknown" | (string & {});
  context_window?: number;
  max_output_tokens?: number;
  capabilities?: Capabilities;
  raw?: unknown;
}
export interface AuthStatus {
  source: string | null;
  expires_at?: string;
  session?: string;
}
export type LoginAction =
  | { type: "open_url"; url: string; instructions: string }
  | { type: "show_code"; url: string; code: string; instructions: string }
  | { type: "paste"; prompt: string };

export interface EnvAuth {
  type: "env";
  names: string[];
  kind?: string;
  placement?: string;
}
export interface SessionAuth {
  type: "session";
  preset: string;
  store?: string;
  client_id?: string;
}
export interface ResolverAuth {
  type: "resolver";
  fn: () => string | { secret: string } | Promise<string | { secret: string }>;
  kind?: string;
  placement?: string;
}
export type AuthSource = string | EnvAuth | SessionAuth | ResolverAuth | { type: string; [key: string]: unknown };

export interface Observer {
  on_request?(data: unknown): void;
  on_response?(data: unknown): void;
  on_event?(data: unknown): void;
  on_retry?(data: unknown): void;
}

export type Fetch = (url: string, init: RequestInit) => Promise<globalThis.Response>;

export interface ProviderOptions {
  base_url?: string;
  auth?: AuthSource | AuthSource[];
  headers?: Record<string, string>;
  timeout?: number | Timeout;
  default_model?: string;
  default_embedding_model?: string;
  default_system?: string | Array<string | TextBlock>;
  default_tools?: Tool[];
  default_options?: Options;
  default_vendor?: VendorMap;
  overrides?: { api_version?: string; capabilities?: Capabilities; dialect?: unknown };
  strict?: boolean;
  observer?: Observer;
  dangerously_allow_browser?: boolean;
  fetch?: Fetch;
  core?: Core | Promise<Core> | string | URL | Uint8Array | WebAssembly.Module;
}

export interface ChatOptions {
  model?: string;
  system?: string | Array<string | TextBlock>;
  tools?: Tool[];
  options?: Options;
  vendor?: VendorMap;
  timeout?: number | Timeout;
  headers?: Record<string, string>;
  strict?: boolean;
  include_raw?: boolean;
  signal?: AbortSignal;
}

export interface EmbedOptions {
  model?: string;
  options?: { dimensions?: number; retries?: number };
  vendor?: VendorMap;
  timeout?: number | Timeout;
  include_raw?: boolean;
  signal?: AbortSignal;
}

export declare class Stream implements AsyncIterableIterator<Event> {
  response: Response | null;
  [Symbol.asyncIterator](): AsyncIterableIterator<Event>;
  next(): Promise<IteratorResult<Event>>;
  return(): Promise<IteratorResult<Event>>;
  cancel(): void;
}

export declare class Provider {
  constructor(ns: Ns, options?: ProviderOptions);
  readonly ns: Ns;
  chat(messages: Message[], options?: ChatOptions): Promise<Response>;
  stream(messages: Message[], options?: ChatOptions): Stream;
  embed(input: string | string[], options?: EmbedOptions): Promise<EmbedResponse>;
  models(options?: { refresh?: boolean; signal?: AbortSignal }): Promise<ModelInfo[]>;
  modelInfo(id: string, options?: { signal?: AbortSignal }): Promise<ModelInfo>;
  capabilities(model: string, options?: { signal?: AbortSignal }): Promise<Capabilities>;
  login(
    handleAction: (action: LoginAction) => string | undefined | void | Promise<string | undefined | void>,
    options?: { callback?: "loopback" | "paste"; timeout?: number; signal?: AbortSignal },
  ): Promise<AuthStatus>;
  logout(options?: { signal?: AbortSignal }): Promise<void>;
  authStatus(options?: { signal?: AbortSignal }): Promise<AuthStatus>;
}

export declare class Core {}
export declare function loadCore(source?: string | URL | Uint8Array | ArrayBuffer | WebAssembly.Module): Promise<Core>;

export type Part = string | Uint8Array | ArrayBuffer | Block;
export declare function text(value: string, options?: { cache?: CacheTtl }): TextBlock;
export declare function image(
  data?: Uint8Array | ArrayBuffer | string | null,
  options?: { url?: string; file_id?: string; media_type?: string },
): ImageBlock;
export declare function document(
  data?: Uint8Array | ArrayBuffer | string | null,
  options?: { url?: string; file_id?: string; text?: string; media_type?: string; name?: string },
): DocumentBlock;
export declare function toolResult(tool_call_id: string, content: Part | Part[], options?: { is_error?: boolean }): ToolResultBlock;
export declare function user(...parts: Part[]): Message;
export declare function assistant(...parts: Array<Part | Response>): Message;
export declare function system(...parts: Part[]): Message;
export declare function tool(...results: ToolResultBlock[]): Message;
export declare function env(...names: string[]): EnvAuth;
export declare function session(preset: string, options?: { store?: string; client_id?: string }): SessionAuth;
export declare function resolver(fn: ResolverAuth["fn"], options?: { kind?: string; placement?: string }): ResolverAuth;

export declare class ProviderError extends Error {
  kind: ErrorKind;
  status: number | null;
  code: string | null;
  retryable: boolean;
  retry_after: number | null;
  attempts: number;
  body: unknown;
  request_id: string | null;
  origin: Origin | null;
}
export declare class AuthError extends ProviderError {}
export declare class RateLimitError extends ProviderError {}
export declare class QuotaError extends ProviderError {}
export declare class ServerError extends ProviderError {}
export declare class NetworkError extends ProviderError {}
export declare class TimeoutError extends ProviderError {}
export declare class CancelledError extends ProviderError {}
export declare class NotFoundError extends ProviderError {}
export declare class ContextOverflowError extends ProviderError {}
export declare class RequestTooLargeError extends ProviderError {}
export declare class BadRequestError extends ProviderError {}
export declare class MalformedResponseError extends ProviderError {}
export declare class UnsupportedError extends ProviderError {}
