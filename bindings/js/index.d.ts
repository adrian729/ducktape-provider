export * from "./schema.js";
import type {
  AuthSource as ConfigAuthSource,
  Block,
  DocumentBlock,
  ImageBlock,
  CacheMark,
  Capabilities,
  EmbedResponse,
  Event,
  LoginAction,
  Message,
  ModelInfo,
  Options,
  Response,
  TextBlock,
  Timeout,
  Tool,
  ToolCallBlock,
  ToolChoice,
  ToolResultBlock,
  VendorMap,
  AuthStatus,
  ErrorKind,
  Origin,
} from "./schema.js";

declare module "./schema.js" {
  interface Response {
    /** The text blocks joined. */
    readonly text: string;
    /** The reasoning blocks joined. */
    readonly reasoning: string;
    /** The tool calls in order. */
    readonly toolCalls: ToolCallBlock[];
  }
}

export type Ns = "anthropic" | "openai" | "google" | "ollama" | (string & {});
export type CacheTtl = NonNullable<CacheMark["ttl"]>;
export type Limits = NonNullable<Capabilities["limits"]>;
export type EnvAuth = Extract<ConfigAuthSource, { type: "env" }>;
export type SessionAuth = Extract<ConfigAuthSource, { type: "session" }>;
/** Options as the binding accepts them: tool_choice also takes auto, none, required or { name }. */
export type OptionsInput = Omit<Options, "tool_choice"> & { tool_choice?: ToolChoice | "auto" | "none" | "required" | { name: string } };

export interface ResolverAuth {
  type: "resolver";
  fn: () => string | { secret: string } | Promise<string | { secret: string }>;
  kind?: string;
  placement?: string;
}
export type AuthInput = string | EnvAuth | SessionAuth | ResolverAuth | ConfigAuthSource;

export interface Observer {
  on_request?(data: unknown): void;
  on_response?(data: unknown): void;
  on_event?(data: unknown): void;
  on_retry?(data: unknown): void;
}

export type Fetch = (url: string, init: RequestInit) => Promise<globalThis.Response>;

export interface ProviderOptions {
  base_url?: string;
  auth?: AuthInput | AuthInput[];
  headers?: Record<string, string>;
  timeout?: number | Timeout;
  default_model?: string;
  default_embedding_model?: string;
  default_system?: string | Array<string | TextBlock>;
  default_tools?: Tool[];
  default_options?: OptionsInput;
  default_vendor?: VendorMap;
  overrides?: { api_version?: string; capabilities?: Capabilities; dialect?: unknown };
  strict?: boolean;
  observer?: Observer;
  dangerously_allow_browser?: boolean;
  fetch?: Fetch;
  core?: Core | Promise<Core> | string | URL | Uint8Array | WebAssembly.Module;
  keychain?: KeychainReader;
}
export type KeychainReader = (service: string, account: string) => string | Uint8Array | null | undefined | Promise<string | Uint8Array | null | undefined>;

export interface ChatOptions {
  model?: string;
  system?: string | Array<string | TextBlock>;
  tools?: Tool[];
  options?: OptionsInput;
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
export declare function session(preset: SessionAuth["preset"], options?: { store?: string; client_id?: string; profile?: string; refresh?: "never" | "memory" | "write_back" }): SessionAuth;
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
