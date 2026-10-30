package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** Unknown types decode as CustomBlock; the variants are merged into one record */
public record Block(
    @JsonProperty("type") String type,
    @JsonProperty("id") String id,
    @JsonProperty("cache") CacheMark cache,
    @JsonProperty("vendor") Map<String, Map<String, Object>> vendor,
    @JsonProperty("provider_executed") Boolean providerExecuted,
    @JsonProperty("text") String text,
    @JsonProperty("data") String data,
    @JsonProperty("url") String url,
    @JsonProperty("file_id") String fileId,
    @JsonProperty("media_type") String mediaType,
    @JsonProperty("name") String name,
    @JsonProperty("arguments") Map<String, Object> arguments,
    @JsonProperty("truncated") Boolean truncated,
    @JsonProperty("arguments_json") String argumentsJson,
    @JsonProperty("status") String status,
    @JsonProperty("tool_call_id") String toolCallId,
    @JsonProperty("content") List<Block> content,
    @JsonProperty("is_error") Boolean isError,
    @JsonProperty("vendor_state") JsonNode vendorState,
    @JsonProperty("origin") Origin origin,
    @JsonProperty("kind") String kind
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a Block field by field. */
    public static final class Builder {
        private String type;
        private String id;
        private CacheMark cache;
        private Map<String, Map<String, Object>> vendor;
        private Boolean providerExecuted;
        private String text;
        private String data;
        private String url;
        private String fileId;
        private String mediaType;
        private String name;
        private Map<String, Object> arguments;
        private Boolean truncated;
        private String argumentsJson;
        private String status;
        private String toolCallId;
        private List<Block> content;
        private Boolean isError;
        private JsonNode vendorState;
        private Origin origin;
        private String kind;
        /** One of compaction, custom, document, image, reasoning, text, tool_call, tool_result; it selects the fields that apply */
        public Builder type(String value) { this.type = value; return this; }
        public Builder id(String value) { this.id = value; return this; }
        public Builder cache(CacheMark value) { this.cache = value; return this; }
        public Builder vendor(Map<String, Map<String, Object>> value) { this.vendor = value; return this; }
        public Builder providerExecuted(Boolean value) { this.providerExecuted = value; return this; }
        public Builder text(String value) { this.text = value; return this; }
        public Builder data(String value) { this.data = value; return this; }
        public Builder url(String value) { this.url = value; return this; }
        public Builder fileId(String value) { this.fileId = value; return this; }
        public Builder mediaType(String value) { this.mediaType = value; return this; }
        public Builder name(String value) { this.name = value; return this; }
        public Builder arguments(Map<String, Object> value) { this.arguments = value; return this; }
        public Builder truncated(Boolean value) { this.truncated = value; return this; }
        /** Raw arguments text when parsing failed or was cut */
        public Builder argumentsJson(String value) { this.argumentsJson = value; return this; }
        /** Reserved for agents */
        public Builder status(String value) { this.status = value; return this; }
        public Builder toolCallId(String value) { this.toolCallId = value; return this; }
        public Builder content(List<Block> value) { this.content = value; return this; }
        public Builder isError(Boolean value) { this.isError = value; return this; }
        /** Opaque */
        public Builder vendorState(JsonNode value) { this.vendorState = value; return this; }
        public Builder origin(Origin value) { this.origin = value; return this; }
        /** <ns>.<type> */
        public Builder kind(String value) { this.kind = value; return this; }
        public Block build() { return new Block(type, id, cache, vendor, providerExecuted, text, data, url, fileId, mediaType, name, arguments, truncated, argumentsJson, status, toolCallId, content, isError, vendorState, origin, kind); }
    }
}
