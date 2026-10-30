package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** One Overrides document. */
public record Overrides(
    @JsonProperty("api_version") String apiVersion,
    @JsonProperty("capabilities") Capabilities capabilities,
    @JsonProperty("dialect") Map<String, Object> dialect
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a Overrides field by field. */
    public static final class Builder {
        private String apiVersion;
        private Capabilities capabilities;
        private Map<String, Object> dialect;
        public Builder apiVersion(String value) { this.apiVersion = value; return this; }
        public Builder capabilities(Capabilities value) { this.capabilities = value; return this; }
        public Builder dialect(Map<String, Object> value) { this.dialect = value; return this; }
        public Overrides build() { return new Overrides(apiVersion, capabilities, dialect); }
    }
}
