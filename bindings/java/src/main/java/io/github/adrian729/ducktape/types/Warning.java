package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** One Warning document. */
public record Warning(
    @JsonProperty("kind") String kind,
    @JsonProperty("feature") String feature,
    @JsonProperty("detail") String detail
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a Warning field by field. */
    public static final class Builder {
        private String kind;
        private String feature;
        private String detail;
        public Builder kind(String value) { this.kind = value; return this; }
        public Builder feature(String value) { this.feature = value; return this; }
        public Builder detail(String value) { this.detail = value; return this; }
        public Warning build() { return new Warning(kind, feature, detail); }
    }
}
