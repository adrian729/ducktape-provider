package io.github.adrian729.ducktape.types;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import java.util.List;
import java.util.Map;

/** ; the variants are merged into one record */
public record AuthSource(
    @JsonProperty("type") String type,
    @JsonProperty("secret_ref") String secretRef,
    @JsonProperty("kind") String kind,
    @JsonProperty("placement") String placement,
    @JsonProperty("names") List<String> names,
    @JsonProperty("resolver_id") String resolverId,
    @JsonProperty("preset") String preset,
    @JsonProperty("store") String store,
    @JsonProperty("client_id") String clientId,
    @JsonProperty("profile") String profile,
    @JsonProperty("refresh") String refresh,
    @JsonProperty("grant") String grant,
    @JsonProperty("params") Map<String, Object> params
) {
    /** A builder with every field unset. */
    public static Builder builder() { return new Builder(); }

    /** Builds a AuthSource field by field. */
    public static final class Builder {
        private String type;
        private String secretRef;
        private String kind;
        private String placement;
        private List<String> names;
        private String resolverId;
        private String preset;
        private String store;
        private String clientId;
        private String profile;
        private String refresh;
        private String grant;
        private Map<String, Object> params;
        /** One of env, exchange, inline, resolver, session; it selects the fields that apply */
        public Builder type(String value) { this.type = value; return this; }
        /** Handle filled through set_secret; never the secret */
        public Builder secretRef(String value) { this.secretRef = value; return this; }
        public Builder kind(String value) { this.kind = value; return this; }
        public Builder placement(String value) { this.placement = value; return this; }
        public Builder names(List<String> value) { this.names = value; return this; }
        public Builder resolverId(String value) { this.resolverId = value; return this; }
        public Builder preset(String value) { this.preset = value; return this; }
        public Builder store(String value) { this.store = value; return this; }
        public Builder clientId(String value) { this.clientId = value; return this; }
        public Builder profile(String value) { this.profile = value; return this; }
        public Builder refresh(String value) { this.refresh = value; return this; }
        public Builder grant(String value) { this.grant = value; return this; }
        public Builder params(Map<String, Object> value) { this.params = value; return this; }
        public AuthSource build() { return new AuthSource(type, secretRef, kind, placement, names, resolverId, preset, store, clientId, profile, refresh, grant, params); }
    }
}
