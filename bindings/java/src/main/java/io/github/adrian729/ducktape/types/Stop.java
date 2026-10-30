package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** One Stop document. */
public record Stop(
    @JsonProperty("reason") String reason,
    @JsonProperty("raw") JsonNode raw
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a Stop field by field. */
    public static final class Builder {
        private String reason;
        private JsonNode raw;
        public Builder reason(String value) { this.reason = value; return this; }
        /** Vendor's own value */
        public Builder raw(JsonNode value) { this.raw = value; return this; }
        public Stop build() { return new Stop(reason, raw); }
    }
}
