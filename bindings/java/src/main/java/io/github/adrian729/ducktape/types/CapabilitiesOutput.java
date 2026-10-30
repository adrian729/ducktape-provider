package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** A nested object */
public record CapabilitiesOutput(
    @JsonProperty("json_schema") Boolean jsonSchema
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a CapabilitiesOutput field by field. */
    public static final class Builder {
        private Boolean jsonSchema;
        /** true, false or null = unknown */
        public Builder jsonSchema(Boolean value) { this.jsonSchema = value; return this; }
        public CapabilitiesOutput build() { return new CapabilitiesOutput(jsonSchema); }
    }
}
