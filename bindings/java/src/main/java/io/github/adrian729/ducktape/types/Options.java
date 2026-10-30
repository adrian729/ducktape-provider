package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** Portable options; everything else goes through vendor */
public record Options(
    @JsonProperty("max_output_tokens") Long maxOutputTokens,
    @JsonProperty("reasoning") String reasoning,
    @JsonProperty("reasoning_summary") String reasoningSummary,
    @JsonProperty("tool_choice") ToolChoice toolChoice,
    @JsonProperty("parallel_tools") Boolean parallelTools,
    @JsonProperty("output") OutputFormat output,
    @JsonProperty("stop_sequences") List<String> stopSequences,
    @JsonProperty("cache") String cache,
    @JsonProperty("retries") Long retries,
    @JsonProperty("user_id") String userId
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a Options field by field. */
    public static final class Builder {
        private Long maxOutputTokens;
        private String reasoning;
        private String reasoningSummary;
        private ToolChoice toolChoice;
        private Boolean parallelTools;
        private OutputFormat output;
        private List<String> stopSequences;
        private String cache;
        private Long retries;
        private String userId;
        public Builder maxOutputTokens(Long value) { this.maxOutputTokens = value; return this; }
        public Builder reasoning(String value) { this.reasoning = value; return this; }
        public Builder reasoningSummary(String value) { this.reasoningSummary = value; return this; }
        public Builder toolChoice(ToolChoice value) { this.toolChoice = value; return this; }
        public Builder parallelTools(Boolean value) { this.parallelTools = value; return this; }
        public Builder output(OutputFormat value) { this.output = value; return this; }
        public Builder stopSequences(List<String> value) { this.stopSequences = value; return this; }
        public Builder cache(String value) { this.cache = value; return this; }
        public Builder retries(Long value) { this.retries = value; return this; }
        public Builder userId(String value) { this.userId = value; return this; }
        public Options build() { return new Options(maxOutputTokens, reasoning, reasoningSummary, toolChoice, parallelTools, output, stopSequences, cache, retries, userId); }
    }
}
