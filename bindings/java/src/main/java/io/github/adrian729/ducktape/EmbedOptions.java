package io.github.adrian729.ducktape;

import io.github.adrian729.ducktape.types.Timeout;
import java.util.Map;

/** Per-call settings for embed. */
public record EmbedOptions(String model, Long dimensions, Long retries, Map<String, Map<String, Object>> vendor, Timeout timeout, Boolean includeRaw) {
    public static Builder builder() {
        return new Builder();
    }

    /** Builds EmbedOptions field by field. */
    public static final class Builder {
        private String model;
        private Long dimensions;
        private Long retries;
        private Map<String, Map<String, Object>> vendor;
        private Timeout timeout;
        private Boolean includeRaw;

        public Builder model(String v) { model = v; return this; }
        public Builder dimensions(Long v) { dimensions = v; return this; }
        public Builder retries(Long v) { retries = v; return this; }
        public Builder vendor(Map<String, Map<String, Object>> v) { vendor = v; return this; }
        public Builder timeout(Timeout v) { timeout = v; return this; }
        public Builder includeRaw(Boolean v) { includeRaw = v; return this; }
        public EmbedOptions build() { return new EmbedOptions(model, dimensions, retries, vendor, timeout, includeRaw); }
    }
}
