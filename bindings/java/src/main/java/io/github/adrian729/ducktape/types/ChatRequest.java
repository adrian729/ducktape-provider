package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** One ChatRequest document. */
public record ChatRequest(
    @JsonProperty("model") String model,
    @JsonProperty("messages") List<Message> messages,
    @JsonProperty("system") List<Block> system,
    @JsonProperty("tools") List<Tool> tools,
    @JsonProperty("options") Options options,
    @JsonProperty("vendor") Map<String, Map<String, Object>> vendor,
    @JsonProperty("timeout") Timeout timeout,
    @JsonProperty("headers") Map<String, String> headers,
    @JsonProperty("strict") Boolean strict,
    @JsonProperty("include_raw") Boolean includeRaw
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a ChatRequest field by field. */
    public static final class Builder {
        private String model;
        private List<Message> messages;
        private List<Block> system;
        private List<Tool> tools;
        private Options options;
        private Map<String, Map<String, Object>> vendor;
        private Timeout timeout;
        private Map<String, String> headers;
        private Boolean strict;
        private Boolean includeRaw;
        public Builder model(String value) { this.model = value; return this; }
        public Builder messages(List<Message> value) { this.messages = value; return this; }
        public Builder system(List<Block> value) { this.system = value; return this; }
        public Builder tools(List<Tool> value) { this.tools = value; return this; }
        public Builder options(Options value) { this.options = value; return this; }
        public Builder vendor(Map<String, Map<String, Object>> value) { this.vendor = value; return this; }
        public Builder timeout(Timeout value) { this.timeout = value; return this; }
        public Builder headers(Map<String, String> value) { this.headers = value; return this; }
        public Builder strict(Boolean value) { this.strict = value; return this; }
        public Builder includeRaw(Boolean value) { this.includeRaw = value; return this; }
        public ChatRequest build() { return new ChatRequest(model, messages, system, tools, options, vendor, timeout, headers, strict, includeRaw); }
    }
}
