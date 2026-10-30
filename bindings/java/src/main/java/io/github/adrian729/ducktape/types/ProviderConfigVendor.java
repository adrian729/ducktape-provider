package io.github.adrian729.ducktape.types;

/** Known values of ProviderConfig.vendor; other strings are allowed. */
public final class ProviderConfigVendor {
    private ProviderConfigVendor() {}
    public static final String ANTHROPIC = "anthropic";
    public static final String OPENAI = "openai";
    public static final String GOOGLE = "google";
    public static final String OLLAMA = "ollama";
}
