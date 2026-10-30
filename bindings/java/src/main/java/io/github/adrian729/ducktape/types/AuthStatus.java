package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** One AuthStatus document. */
public record AuthStatus(
    @JsonProperty("source") String source,
    @JsonProperty("expires_at") String expiresAt,
    @JsonProperty("session") String session
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a AuthStatus field by field. */
    public static final class Builder {
        private String source;
        private String expiresAt;
        private String session;
        public Builder source(String value) { this.source = value; return this; }
        public Builder expiresAt(String value) { this.expiresAt = value; return this; }
        public Builder session(String value) { this.session = value; return this; }
        public AuthStatus build() { return new AuthStatus(source, expiresAt, session); }
    }
}
