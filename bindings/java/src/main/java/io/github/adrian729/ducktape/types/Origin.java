package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** Where a response or an opaque block came from; opaque state is replayed when wire and base_url match the call */
public record Origin(
    @JsonProperty("wire") String wire,
    @JsonProperty("vendor") String vendor,
    @JsonProperty("base_url") String baseUrl,
    @JsonProperty("model") String model,
    @JsonProperty("auth_source") String authSource
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a Origin field by field. */
    public static final class Builder {
        private String wire;
        private String vendor;
        private String baseUrl;
        private String model;
        private String authSource;
        public Builder wire(String value) { this.wire = value; return this; }
        /** Vendor, company name */
        public Builder vendor(String value) { this.vendor = value; return this; }
        public Builder baseUrl(String value) { this.baseUrl = value; return this; }
        public Builder model(String value) { this.model = value; return this; }
        /** env:NAME, session:PRESET, resolver, inline */
        public Builder authSource(String value) { this.authSource = value; return this; }
        public Origin build() { return new Origin(wire, vendor, baseUrl, model, authSource); }
    }
}
