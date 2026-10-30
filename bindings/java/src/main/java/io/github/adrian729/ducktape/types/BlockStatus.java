package io.github.adrian729.ducktape.types;

/** Known values of Block.status; other strings are allowed. */
public final class BlockStatus {
    private BlockStatus() {}
    public static final String PENDING = "pending";
    public static final String IN_PROGRESS = "in_progress";
    public static final String COMPLETED = "completed";
    public static final String FAILED = "failed";
    public static final String CANCELLED = "cancelled";
}
