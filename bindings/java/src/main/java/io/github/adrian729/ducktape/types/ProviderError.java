package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** One ProviderError document. */
public record ProviderError(
    @JsonProperty("kind") String kind,
    @JsonProperty("message") String message,
    @JsonProperty("status") Long status,
    @JsonProperty("code") String code,
    @JsonProperty("retryable") Boolean retryable,
    @JsonProperty("retry_after") Double retryAfter,
    @JsonProperty("attempts") Long attempts,
    @JsonProperty("body") JsonNode body,
    @JsonProperty("request_id") String requestId,
    @JsonProperty("origin") Origin origin
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a ProviderError field by field. */
    public static final class Builder {
        private String kind;
        private String message;
        private Long status;
        private String code;
        private Boolean retryable;
        private Double retryAfter;
        private Long attempts;
        private JsonNode body;
        private String requestId;
        private Origin origin;
        public Builder kind(String value) { this.kind = value; return this; }
        public Builder message(String value) { this.message = value; return this; }
        public Builder status(Long value) { this.status = value; return this; }
        public Builder code(String value) { this.code = value; return this; }
        public Builder retryable(Boolean value) { this.retryable = value; return this; }
        public Builder retryAfter(Double value) { this.retryAfter = value; return this; }
        public Builder attempts(Long value) { this.attempts = value; return this; }
        /** Redacted vendor error body */
        public Builder body(JsonNode value) { this.body = value; return this; }
        public Builder requestId(String value) { this.requestId = value; return this; }
        public Builder origin(Origin value) { this.origin = value; return this; }
        public ProviderError build() { return new ProviderError(kind, message, status, code, retryable, retryAfter, attempts, body, requestId, origin); }
    }
}
