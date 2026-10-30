package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** A nested object */
public record CapabilitiesParallelTools(
    @JsonProperty("supported") Boolean supported,
    @JsonProperty("toggle") Boolean toggle
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a CapabilitiesParallelTools field by field. */
    public static final class Builder {
        private Boolean supported;
        private Boolean toggle;
        /** true, false or null = unknown */
        public Builder supported(Boolean value) { this.supported = value; return this; }
        /** true, false or null = unknown */
        public Builder toggle(Boolean value) { this.toggle = value; return this; }
        public CapabilitiesParallelTools build() { return new CapabilitiesParallelTools(supported, toggle); }
    }
}
