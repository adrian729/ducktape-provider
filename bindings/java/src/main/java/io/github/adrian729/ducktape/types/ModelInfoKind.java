package io.github.adrian729.ducktape.types;

/** Known values of ModelInfo.kind; other strings are allowed. */
public final class ModelInfoKind {
    private ModelInfoKind() {}
    public static final String CHAT = "chat";
    public static final String EMBEDDING = "embedding";
    public static final String UNKNOWN = "unknown";
}
