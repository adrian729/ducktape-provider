package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** A nested object */
public record EmbedResponseUsageInput(
    @JsonProperty("total") Long total
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a EmbedResponseUsageInput field by field. */
    public static final class Builder {
        private Long total;
        public Builder total(Long value) { this.total = value; return this; }
        public EmbedResponseUsageInput build() { return new EmbedResponseUsageInput(total); }
    }
}
