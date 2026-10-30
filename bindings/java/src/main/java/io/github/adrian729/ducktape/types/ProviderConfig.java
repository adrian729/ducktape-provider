package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** What a binding hands the core for one Provider */
public record ProviderConfig(
    @JsonProperty("ns") String ns,
    @JsonProperty("base_url") String baseUrl,
    @JsonProperty("auth") Auth auth,
    @JsonProperty("headers") Map<String, String> headers,
    @JsonProperty("timeout") Timeout timeout,
    @JsonProperty("default_model") String defaultModel,
    @JsonProperty("default_embedding_model") String defaultEmbeddingModel,
    @JsonProperty("default_system") List<Block> defaultSystem,
    @JsonProperty("default_tools") List<Tool> defaultTools,
    @JsonProperty("default_options") Options defaultOptions,
    @JsonProperty("default_vendor") Map<String, Map<String, Object>> defaultVendor,
    @JsonProperty("overrides") Overrides overrides,
    @JsonProperty("strict") Boolean strict
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a ProviderConfig field by field. */
    public static final class Builder {
        private String ns;
        private String baseUrl;
        private Auth auth;
        private Map<String, String> headers;
        private Timeout timeout;
        private String defaultModel;
        private String defaultEmbeddingModel;
        private List<Block> defaultSystem;
        private List<Tool> defaultTools;
        private Options defaultOptions;
        private Map<String, Map<String, Object>> defaultVendor;
        private Overrides overrides;
        private Boolean strict;
        /** Vendor namespace, company name */
        public Builder ns(String value) { this.ns = value; return this; }
        public Builder baseUrl(String value) { this.baseUrl = value; return this; }
        public Builder auth(Auth value) { this.auth = value; return this; }
        public Builder headers(Map<String, String> value) { this.headers = value; return this; }
        public Builder timeout(Timeout value) { this.timeout = value; return this; }
        public Builder defaultModel(String value) { this.defaultModel = value; return this; }
        public Builder defaultEmbeddingModel(String value) { this.defaultEmbeddingModel = value; return this; }
        public Builder defaultSystem(List<Block> value) { this.defaultSystem = value; return this; }
        public Builder defaultTools(List<Tool> value) { this.defaultTools = value; return this; }
        public Builder defaultOptions(Options value) { this.defaultOptions = value; return this; }
        public Builder defaultVendor(Map<String, Map<String, Object>> value) { this.defaultVendor = value; return this; }
        public Builder overrides(Overrides value) { this.overrides = value; return this; }
        public Builder strict(Boolean value) { this.strict = value; return this; }
        public ProviderConfig build() { return new ProviderConfig(ns, baseUrl, auth, headers, timeout, defaultModel, defaultEmbeddingModel, defaultSystem, defaultTools, defaultOptions, defaultVendor, overrides, strict); }
    }
}
