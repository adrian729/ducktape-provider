package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** total includes cached tokens */
public record UsageInput(
    @JsonProperty("total") Long total,
    @JsonProperty("cache_read") Long cacheRead,
    @JsonProperty("cache_write") Long cacheWrite
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a UsageInput field by field. */
    public static final class Builder {
        private Long total;
        private Long cacheRead;
        private Long cacheWrite;
        public Builder total(Long value) { this.total = value; return this; }
        public Builder cacheRead(Long value) { this.cacheRead = value; return this; }
        public Builder cacheWrite(Long value) { this.cacheWrite = value; return this; }
        public UsageInput build() { return new UsageInput(total, cacheRead, cacheWrite); }
    }
}
