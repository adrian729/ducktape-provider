package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** Cache breakpoint on a block or tool list */
public record CacheMark(
    @JsonProperty("ttl") String ttl
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a CacheMark field by field. */
    public static final class Builder {
        private String ttl;
        public Builder ttl(String value) { this.ttl = value; return this; }
        public CacheMark build() { return new CacheMark(ttl); }
    }
}
