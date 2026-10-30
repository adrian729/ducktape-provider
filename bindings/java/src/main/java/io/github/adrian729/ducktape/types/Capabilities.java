package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** One Capabilities document. */
public record Capabilities(
    @JsonProperty("reasoning") CapabilitiesReasoning reasoning,
    @JsonProperty("tool_choice") List<String> toolChoice,
    @JsonProperty("parallel_tools") CapabilitiesParallelTools parallelTools,
    @JsonProperty("output") CapabilitiesOutput output,
    @JsonProperty("cache") CapabilitiesCache cache,
    @JsonProperty("inputs") List<String> inputs,
    @JsonProperty("tools") CapabilitiesTools tools,
    @JsonProperty("limits") CapabilitiesLimits limits,
    @JsonProperty("embeddings") Boolean embeddings,
    @JsonProperty("source") String source
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a Capabilities field by field. */
    public static final class Builder {
        private CapabilitiesReasoning reasoning;
        private List<String> toolChoice;
        private CapabilitiesParallelTools parallelTools;
        private CapabilitiesOutput output;
        private CapabilitiesCache cache;
        private List<String> inputs;
        private CapabilitiesTools tools;
        private CapabilitiesLimits limits;
        private Boolean embeddings;
        private String source;
        public Builder reasoning(CapabilitiesReasoning value) { this.reasoning = value; return this; }
        public Builder toolChoice(List<String> value) { this.toolChoice = value; return this; }
        public Builder parallelTools(CapabilitiesParallelTools value) { this.parallelTools = value; return this; }
        public Builder output(CapabilitiesOutput value) { this.output = value; return this; }
        public Builder cache(CapabilitiesCache value) { this.cache = value; return this; }
        public Builder inputs(List<String> value) { this.inputs = value; return this; }
        public Builder tools(CapabilitiesTools value) { this.tools = value; return this; }
        public Builder limits(CapabilitiesLimits value) { this.limits = value; return this; }
        /** true, false or null = unknown */
        public Builder embeddings(Boolean value) { this.embeddings = value; return this; }
        public Builder source(String value) { this.source = value; return this; }
        public Capabilities build() { return new Capabilities(reasoning, toolChoice, parallelTools, output, cache, inputs, tools, limits, embeddings, source); }
    }
}
