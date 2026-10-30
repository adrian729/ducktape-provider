package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** A nested object */
public record UsageOutput(
    @JsonProperty("total") Long total,
    @JsonProperty("reasoning") Long reasoning
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a UsageOutput field by field. */
    public static final class Builder {
        private Long total;
        private Long reasoning;
        public Builder total(Long value) { this.total = value; return this; }
        public Builder reasoning(Long value) { this.reasoning = value; return this; }
        public UsageOutput build() { return new UsageOutput(total, reasoning); }
    }
}
