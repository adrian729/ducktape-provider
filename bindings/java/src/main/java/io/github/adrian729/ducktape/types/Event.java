package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** Unknown event types are delivered as raw; the variants are merged into one record */
public record Event(
    @JsonProperty("type") String type,
    @JsonProperty("parent_id") String parentId,
    @JsonProperty("index") Long index,
    @JsonProperty("kind") String kind,
    @JsonProperty("id") String id,
    @JsonProperty("name") String name,
    @JsonProperty("text") String text,
    @JsonProperty("arguments_json") String argumentsJson,
    @JsonProperty("block") Block block,
    @JsonProperty("response") Response response,
    @JsonProperty("event") JsonNode event
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a Event field by field. */
    public static final class Builder {
        private String type;
        private String parentId;
        private Long index;
        private String kind;
        private String id;
        private String name;
        private String text;
        private String argumentsJson;
        private Block block;
        private Response response;
        private JsonNode event;
        /** One of block_start, block_stop, finish, raw, reasoning_delta, text_delta, tool_call_delta; it selects the fields that apply */
        public Builder type(String value) { this.type = value; return this; }
        /** Reserved for agents */
        public Builder parentId(String value) { this.parentId = value; return this; }
        public Builder index(Long value) { this.index = value; return this; }
        public Builder kind(String value) { this.kind = value; return this; }
        public Builder id(String value) { this.id = value; return this; }
        public Builder name(String value) { this.name = value; return this; }
        public Builder text(String value) { this.text = value; return this; }
        public Builder argumentsJson(String value) { this.argumentsJson = value; return this; }
        public Builder block(Block value) { this.block = value; return this; }
        public Builder response(Response value) { this.response = value; return this; }
        public Builder event(JsonNode value) { this.event = value; return this; }
        public Event build() { return new Event(type, parentId, index, kind, id, name, text, argumentsJson, block, response, event); }
    }
}
