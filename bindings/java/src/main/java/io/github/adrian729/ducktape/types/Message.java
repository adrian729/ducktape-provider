package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** One Message document. */
public record Message(
    @JsonProperty("role") String role,
    @JsonProperty("content") List<Block> content
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a Message field by field. */
    public static final class Builder {
        private String role;
        private List<Block> content;
        public Builder role(String value) { this.role = value; return this; }
        public Builder content(List<Block> value) { this.content = value; return this; }
        public Message build() { return new Message(role, content); }
    }
}
