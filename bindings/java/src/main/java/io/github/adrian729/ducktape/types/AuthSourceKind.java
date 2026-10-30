package io.github.adrian729.ducktape.types;

/** Known values of AuthSource.kind; other strings are allowed. */
public final class AuthSourceKind {
    private AuthSourceKind() {}
    public static final String NONE = "none";
    public static final String API_KEY = "api_key";
    public static final String BEARER = "bearer";
    public static final String SIGV4 = "sigv4";
}
