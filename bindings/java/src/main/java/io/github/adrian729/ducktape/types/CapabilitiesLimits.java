package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** A nested object */
public record CapabilitiesLimits(
    @JsonProperty("context_window") Long contextWindow,
    @JsonProperty("max_output_tokens") Long maxOutputTokens
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a CapabilitiesLimits field by field. */
    public static final class Builder {
        private Long contextWindow;
        private Long maxOutputTokens;
        public Builder contextWindow(Long value) { this.contextWindow = value; return this; }
        public Builder maxOutputTokens(Long value) { this.maxOutputTokens = value; return this; }
        public CapabilitiesLimits build() { return new CapabilitiesLimits(contextWindow, maxOutputTokens); }
    }
}
