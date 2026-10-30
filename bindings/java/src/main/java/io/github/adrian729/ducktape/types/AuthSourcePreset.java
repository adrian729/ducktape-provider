package io.github.adrian729.ducktape.types;

/** Known values of AuthSource.preset; other strings are allowed. */
public final class AuthSourcePreset {
    private AuthSourcePreset() {}
    public static final String ANTHROPIC_PROFILE = "anthropic_profile";
    public static final String CLAUDE_CODE = "claude_code";
    public static final String CLAUDE_CODE_TOKEN_ENV = "claude_code_token_env";
    public static final String CODEX = "codex";
    public static final String CHATGPT_PLAN = "chatgpt_plan";
}
