package io.github.adrian729.ducktape.types;

/** Known values of AuthSource.placement; other strings are allowed. */
public final class AuthSourcePlacement {
    private AuthSourcePlacement() {}
    public static final String X_API_KEY = "x-api-key";
    public static final String AUTHORIZATION = "authorization";
    public static final String X_GOOG_API_KEY = "x-goog-api-key";
    public static final String API_KEY = "api-key";
}
