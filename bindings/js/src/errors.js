/** Errors raised by every call. */
export class ProviderError extends Error {
  constructor(message, fields = {}) {
    super(message);
    this.name = new.target.name;
    this.kind = fields.kind ?? new.target.kind ?? "other";
    this.status = fields.status ?? null;
    this.code = fields.code ?? null;
    this.retryable = fields.retryable ?? false;
    this.retry_after = fields.retry_after ?? null;
    this.attempts = fields.attempts ?? 1;
    this.body = fields.body ?? null;
    this.request_id = fields.request_id ?? null;
    this.origin = fields.origin ?? null;
    if (fields.cause) this.cause = fields.cause;
  }
}

export class AuthError extends ProviderError {}
AuthError.kind = "auth";
export class RateLimitError extends ProviderError {}
RateLimitError.kind = "rate_limit";
export class QuotaError extends ProviderError {}
QuotaError.kind = "quota";
export class ServerError extends ProviderError {}
ServerError.kind = "server";
export class NetworkError extends ProviderError {}
NetworkError.kind = "connection";
export class TimeoutError extends ProviderError {}
TimeoutError.kind = "timeout";
export class CancelledError extends ProviderError {}
CancelledError.kind = "cancelled";
export class NotFoundError extends ProviderError {}
NotFoundError.kind = "not_found";
export class ContextOverflowError extends ProviderError {}
ContextOverflowError.kind = "context_overflow";
export class RequestTooLargeError extends ProviderError {}
RequestTooLargeError.kind = "request_too_large";
export class BadRequestError extends ProviderError {}
BadRequestError.kind = "bad_request";
export class MalformedResponseError extends ProviderError {}
MalformedResponseError.kind = "malformed_response";
export class UnsupportedError extends ProviderError {}
UnsupportedError.kind = "unsupported";

const byKind = new Map(
  [
    AuthError,
    RateLimitError,
    QuotaError,
    ServerError,
    NetworkError,
    TimeoutError,
    CancelledError,
    NotFoundError,
    ContextOverflowError,
    RequestTooLargeError,
    BadRequestError,
    MalformedResponseError,
    UnsupportedError,
  ].map((cls) => [cls.kind, cls]),
);

/** Builds the matching error class from the core's error data. */
export function errorFromData(data) {
  const cls = byKind.get(data.kind) ?? ProviderError;
  return new cls(data.message ?? "", data);
}
