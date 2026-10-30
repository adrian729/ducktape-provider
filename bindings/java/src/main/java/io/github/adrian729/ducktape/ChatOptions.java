package io.github.adrian729.ducktape;

import io.github.adrian729.ducktape.types.Block;
import io.github.adrian729.ducktape.types.Options;
import io.github.adrian729.ducktape.types.Timeout;
import io.github.adrian729.ducktape.types.Tool;
import java.util.List;
import java.util.Map;

/** Per-call settings for chat and stream. */
public record ChatOptions(
    String model,
    List<Block> system,
    List<Tool> tools,
    Options options,
    Map<String, Map<String, Object>> vendor,
    Timeout timeout,
    Map<String, String> headers,
    Boolean strict,
    Boolean includeRaw
) {
    public static Builder builder() {
        return new Builder();
    }

    /** Builds ChatOptions field by field. */
    public static final class Builder {
        private String model;
        private List<Block> system;
        private List<Tool> tools;
        private Options options;
        private Map<String, Map<String, Object>> vendor;
        private Timeout timeout;
        private Map<String, String> headers;
        private Boolean strict;
        private Boolean includeRaw;

        public Builder model(String v) { model = v; return this; }
        public Builder system(List<Block> v) { system = v; return this; }
        public Builder system(String text) { system = List.of(Ducktape.text(text)); return this; }
        public Builder tools(List<Tool> v) { tools = v; return this; }
        public Builder tools(Tool... v) { tools = List.of(v); return this; }
        public Builder options(Options v) { options = v; return this; }
        public Builder vendor(Map<String, Map<String, Object>> v) { vendor = v; return this; }
        public Builder timeout(Timeout v) { timeout = v; return this; }
        public Builder headers(Map<String, String> v) { headers = v; return this; }
        public Builder strict(Boolean v) { strict = v; return this; }
        public Builder includeRaw(Boolean v) { includeRaw = v; return this; }
        public ChatOptions build() { return new ChatOptions(model, system, tools, options, vendor, timeout, headers, strict, includeRaw); }
    }
}
