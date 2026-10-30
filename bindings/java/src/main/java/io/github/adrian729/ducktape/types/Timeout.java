package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** Seconds; bindings accept a plain number as total */
public record Timeout(
    @JsonProperty("connect") Double connect,
    @JsonProperty("read") Double read,
    @JsonProperty("total") Double total
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a Timeout field by field. */
    public static final class Builder {
        private Double connect;
        private Double read;
        private Double total;
        public Builder connect(Double value) { this.connect = value; return this; }
        public Builder read(Double value) { this.read = value; return this; }
        public Builder total(Double value) { this.total = value; return this; }
        public Timeout build() { return new Timeout(connect, read, total); }
    }
}
