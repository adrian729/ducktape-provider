package io.github.adrian729.ducktape;

import io.github.adrian729.ducktape.types.ProviderError;

/** Any failure reported by the core, with the normalized error attached. */
public class ProviderException extends RuntimeException {
    private final ProviderError error;

    public ProviderException(ProviderError error) {
        super("[" + error.kind() + (error.code() == null ? "" : " " + error.code()) + "] " + error.message());
        this.error = error;
    }

    public ProviderException(ProviderError error, Throwable cause) {
        this(error);
        initCause(cause);
    }

    /** The normalized error. */
    public ProviderError error() {
        return error;
    }

    /** The error kind, the stable discriminator. */
    public String kind() {
        return error.kind();
    }

    /** The vendor-specific code, if any. */
    public String code() {
        return error.code();
    }

    /** The HTTP status, if any. */
    public Long status() {
        return error.status();
    }

    /** Whether a retry may succeed. */
    public boolean retryable() {
        return Boolean.TRUE.equals(error.retryable());
    }

    /** Whether this error is of the given kind. */
    public boolean is(String kind) {
        return kind.equals(error.kind());
    }

    static ProviderException of(String kind, String code, String message) {
        return new ProviderException(ProviderError.builder().kind(kind).code(code).message(message).retryable(false).attempts(0L).build());
    }
}
