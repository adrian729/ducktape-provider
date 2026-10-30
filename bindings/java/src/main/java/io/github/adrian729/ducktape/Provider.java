package io.github.adrian729.ducktape;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.github.adrian729.ducktape.types.Auth;
import io.github.adrian729.ducktape.types.AuthSource;
import io.github.adrian729.ducktape.types.AuthStatus;
import io.github.adrian729.ducktape.types.Block;
import io.github.adrian729.ducktape.types.Capabilities;
import io.github.adrian729.ducktape.types.ChatRequest;
import io.github.adrian729.ducktape.types.EmbedRequest;
import io.github.adrian729.ducktape.types.EmbedRequestOptions;
import io.github.adrian729.ducktape.types.EmbedResponse;
import io.github.adrian729.ducktape.types.Event;
import io.github.adrian729.ducktape.types.LoginAction;
import io.github.adrian729.ducktape.types.Message;
import io.github.adrian729.ducktape.types.ModelInfo;
import io.github.adrian729.ducktape.types.Options;
import io.github.adrian729.ducktape.types.Overrides;
import io.github.adrian729.ducktape.types.ProviderConfig;
import io.github.adrian729.ducktape.types.Response;
import io.github.adrian729.ducktape.types.Timeout;
import io.github.adrian729.ducktape.types.Tool;
import java.net.http.HttpClient;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;

/** One endpoint of one vendor. */
public final class Provider {
    private static final Set<String> VENDORS = Set.of("anthropic", "openai", "google", "ollama");
    private static final ExecutorService WORKERS = Executors.newCachedThreadPool(r -> {
        Thread t = new Thread(r, "ducktape-worker");
        t.setDaemon(true);
        return t;
    });
    private static volatile Core defaultCore;
    private static volatile Transport defaultTransport;

    private final String vendor;
    private final ProviderConfig config;
    private final Driver.Host host;
    private final Core core;
    private final Map<String, ModelInfo> infos = new java.util.concurrent.ConcurrentHashMap<>();
    private final Map<String, Capabilities> caps = new java.util.concurrent.ConcurrentHashMap<>();
    private volatile List<ModelInfo> models;

    private Provider(String vendor, ProviderConfig config, Driver.Host host, Core core) {
        this.vendor = vendor;
        this.config = config;
        this.host = host;
        this.core = core;
    }

    /** Starts configuring a provider for a vendor. */
    public static Builder builder(String vendor) {
        return new Builder(vendor);
    }

    /** A provider with every default: key from the environment, no default model. */
    public static Provider of(String vendor) {
        return builder(vendor).build();
    }

    /** The vendor the provider speaks to. */
    public String vendor() {
        return vendor;
    }

    static Core sharedCore() {
        Core c = defaultCore;
        if (c == null) {
            synchronized (Provider.class) {
                c = defaultCore;
                if (c == null) {
                    c = defaultCore = Core.bundled();
                }
            }
        }
        return c;
    }

    private static Transport sharedTransport() {
        Transport t = defaultTransport;
        if (t == null) {
            synchronized (Provider.class) {
                t = defaultTransport;
                if (t == null) {
                    t = defaultTransport = Transport.of(HttpClient.newBuilder().followRedirects(HttpClient.Redirect.NEVER).connectTimeout(Duration.ofSeconds(10)).build());
                }
            }
        }
        return t;
    }

    /** Configures a Provider. */
    public static final class Builder {
        private final String vendor;
        private String baseUrl;
        private List<Credential> auth;
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
        private Observer observer;
        private Transport transport;
        private Core core;
        private Keychain.Reader keychain;
        private Keychain.Writer keychainWrite;

        private Builder(String vendor) {
            if (!VENDORS.contains(vendor)) {
                throw new IllegalArgumentException("unknown vendor " + vendor + "; one of " + String.join(", ", VENDORS));
            }
            this.vendor = vendor;
        }

        public Builder baseUrl(String v) { baseUrl = v; return this; }
        public Builder auth(Credential... v) { auth = List.of(v); return this; }
        public Builder auth(List<Credential> v) { auth = v; return this; }
        public Builder headers(Map<String, String> v) { headers = v; return this; }
        public Builder timeout(Timeout v) { timeout = v; return this; }
        public Builder defaultModel(String v) { defaultModel = v; return this; }
        public Builder defaultEmbeddingModel(String v) { defaultEmbeddingModel = v; return this; }
        public Builder defaultSystem(List<Block> v) { defaultSystem = v; return this; }
        public Builder defaultSystem(String text) { defaultSystem = List.of(Ducktape.text(text)); return this; }
        public Builder defaultTools(List<Tool> v) { defaultTools = v; return this; }
        public Builder defaultOptions(Options v) { defaultOptions = v; return this; }
        public Builder defaultVendor(Map<String, Map<String, Object>> v) { defaultVendor = v; return this; }
        public Builder overrides(Overrides v) { overrides = v; return this; }
        public Builder strict(Boolean v) { strict = v; return this; }
        public Builder observer(Observer v) { observer = v; return this; }
        public Builder client(HttpClient v) { transport = Transport.of(v); return this; }
        public Builder transport(Transport v) { transport = v; return this; }
        public Builder core(Core v) { core = v; return this; }
        public Builder keychain(Keychain.Reader v) { keychain = v; return this; }
        public Builder keychainWrite(Keychain.Writer v) { keychainWrite = v; return this; }

        public Provider build() {
            Map<String, String> secrets = new HashMap<>();
            Map<String, Supplier<String>> resolvers = new HashMap<>();
            Auth authDoc = null;
            if (auth != null && !auth.isEmpty()) {
                List<AuthSource> sources = new ArrayList<>();
                for (int i = 0; i < auth.size(); i++) {
                    Credential cred = auth.get(i);
                    AuthSource src = cred.source;
                    switch (src.type()) {
                        case "inline" -> {
                            String ref = "inline:" + i;
                            secrets.put(ref, cred.key);
                            src = new AuthSource(src.type(), ref, src.kind(), src.placement(), src.names(), src.resolverId(), src.preset(), src.store(), src.clientId(), src.profile(), src.refresh(), src.grant(), src.params());
                        }
                        case "resolver" -> {
                            if (cred.resolver == null) {
                                throw new IllegalArgumentException("a resolver credential needs a function");
                            }
                            String id = "resolver:" + i;
                            resolvers.put(id, cred.resolver);
                            src = new AuthSource(src.type(), src.secretRef(), src.kind(), src.placement(), src.names(), id, src.preset(), src.store(), src.clientId(), src.profile(), src.refresh(), src.grant(), src.params());
                        }
                        default -> {
                        }
                    }
                    sources.add(src);
                }
                authDoc = Auth.builder().sources(sources).build();
            }
            ProviderConfig config = ProviderConfig.builder()
                .vendor(vendor)
                .baseUrl(baseUrl == null ? null : baseUrl.replaceAll("/+$", ""))
                .auth(authDoc)
                .headers(headers)
                .timeout(timeout)
                .defaultModel(defaultModel)
                .defaultEmbeddingModel(defaultEmbeddingModel)
                .defaultSystem(defaultSystem)
                .defaultTools(defaultTools)
                .defaultOptions(defaultOptions)
                .defaultVendor(defaultVendor)
                .overrides(overrides)
                .strict(strict)
                .build();
            Keychain.Reader reader = keychain == null ? Keychain::read : keychain;
            Keychain.Writer writer = keychainWrite != null ? keychainWrite : keychain == null ? Keychain::write : null;
            Driver.Host host = new Driver.Host(secrets, resolvers, reader, writer, transport == null ? sharedTransport() : transport, observer);
            return new Provider(vendor, config, host, core);
        }
    }

    private JsonNode run(ObjectNode operation, Consumer<Event> onEvent, Function<LoginAction, String> handleAction, Driver.Cancel cancel) {
        Core c = core == null ? sharedCore() : core;
        ObjectNode start = Json.object();
        start.set("clock", Driver.clock());
        start.set("provider", Json.MAPPER.valueToTree(config));
        start.set("operation", operation);
        try (Core.Operation op = c.operation()) {
            return new Driver(op, host, onEvent, handleAction, cancel).run(start);
        }
    }

    private ChatRequest chatRequest(List<Message> messages, ChatOptions o) {
        if (o == null) {
            o = ChatOptions.builder().build();
        }
        String model = o.model() != null ? o.model() : config.defaultModel();
        if (model == null || model.isEmpty()) {
            throw new IllegalArgumentException("no model: set model on the call or defaultModel on the provider");
        }
        return new ChatRequest(model, messages == null ? List.of() : messages, o.system(), o.tools(), o.options(), o.vendor(), o.timeout(), o.headers(), o.strict(), o.includeRaw());
    }

    private static ObjectNode operation(String type, Object request) {
        ObjectNode op = Json.object().put("type", type);
        if (request != null) {
            op.set("request", Json.MAPPER.valueToTree(request));
        }
        return op;
    }

    /** Sends the messages and returns the whole reply. */
    public Response chat(List<Message> messages) {
        return chat(messages, null);
    }

    /** Sends the messages with per-call settings and returns the whole reply. */
    public Response chat(List<Message> messages, ChatOptions options) {
        return chat(messages, options, null);
    }

    private Response chat(List<Message> messages, ChatOptions options, Driver.Cancel cancel) {
        JsonNode result = run(operation("chat", chatRequest(messages, options)), null, null, cancel);
        return Json.as(result.get("response"), Response.class);
    }

    /** Sends the messages on a worker thread; cancelling the future cancels the call. */
    public CompletableFuture<Response> chatAsync(List<Message> messages, ChatOptions options) {
        ChatRequest request = chatRequest(messages, options);
        Driver.Cancel cancel = new Driver.Cancel();
        CompletableFuture<Response> future = new CompletableFuture<>();
        WORKERS.submit(() -> {
            try {
                future.complete(Json.as(run(operation("chat", request), null, null, cancel).get("response"), Response.class));
            } catch (Throwable e) {
                future.completeExceptionally(e);
            }
        });
        future.whenComplete((r, e) -> {
            if (future.isCancelled()) {
                cancel.cancel();
            }
        });
        return future;
    }

    /** Sends the messages and returns the events as they arrive. */
    public Stream stream(List<Message> messages) {
        return stream(messages, null);
    }

    /** Sends the messages with per-call settings and returns the events as they arrive. */
    public Stream stream(List<Message> messages, ChatOptions options) {
        ChatRequest request = chatRequest(messages, options);
        Stream s = new Stream();
        WORKERS.submit(() -> {
            RuntimeException failure = null;
            try {
                run(operation("stream", request), s::push, null, s.cancel());
            } catch (RuntimeException e) {
                failure = e;
            } catch (Throwable e) {
                failure = new IllegalStateException(e);
            } finally {
                s.finish(failure);
            }
        });
        return s;
    }

    /** Returns one embedding per input text. */
    public EmbedResponse embed(List<String> input) {
        return embed(input, null);
    }

    /** Returns one embedding per input text, with per-call settings. */
    public EmbedResponse embed(List<String> input, EmbedOptions o) {
        if (o == null) {
            o = EmbedOptions.builder().build();
        }
        if (input == null || input.isEmpty()) {
            throw new IllegalArgumentException("embed input must not be empty");
        }
        String model = o.model() != null ? o.model() : config.defaultEmbeddingModel();
        if (model == null || model.isEmpty()) {
            throw new IllegalArgumentException("no embedding model: set model on the call or defaultEmbeddingModel on the provider");
        }
        EmbedRequestOptions eo = o.dimensions() == null && o.retries() == null ? null : new EmbedRequestOptions(o.dimensions(), o.retries());
        EmbedRequest request = new EmbedRequest(model, input, eo, o.vendor(), o.timeout(), o.includeRaw());
        JsonNode result = run(operation("embed", request), null, null, null);
        return Json.as(result.get("response"), EmbedResponse.class);
    }

    /** Lists the models the endpoint serves; cached after the first call unless refresh is set. */
    public List<ModelInfo> models(boolean refresh) {
        List<ModelInfo> cached = models;
        if (cached != null && !refresh) {
            return cached;
        }
        JsonNode result = run(operation("models", null).put("refresh", refresh), null, null, null);
        List<ModelInfo> out = new ArrayList<>();
        for (JsonNode m : result.path("models")) {
            out.add(Json.as(m, ModelInfo.class));
        }
        models = List.copyOf(out);
        return models;
    }

    /** Describes one model. */
    public ModelInfo modelInfo(String id) {
        ModelInfo cached = infos.get(id);
        if (cached != null) {
            return cached;
        }
        JsonNode result = run(operation("model_info", null).put("model", id), null, null, null);
        JsonNode info = result.get("info");
        if (info == null || info.isNull()) {
            throw ProviderException.of("not_found", null, "model " + id + " not found");
        }
        ModelInfo out = Json.as(info, ModelInfo.class);
        infos.put(id, out);
        return out;
    }

    /** Tells what a model supports. */
    public Capabilities capabilities(String model) {
        Capabilities cached = caps.get(model);
        if (cached != null) {
            return cached;
        }
        JsonNode result = run(operation("capabilities", null).put("model", model), null, null, null);
        JsonNode c = result.get("capabilities");
        if (c == null || c.isNull()) {
            throw ProviderException.of("not_found", null, "no capabilities known for " + model);
        }
        Capabilities out = Json.as(c, Capabilities.class);
        caps.put(model, out);
        return out;
    }

    /** Logs into the provider's session source; the handler shows the user what to do and returns any pasted value. */
    public AuthStatus login(Function<LoginAction, String> handler, String callback, Duration timeout) {
        return login(handler, callback, timeout, null);
    }

    /** Logs in as the named application. */
    public AuthStatus login(Function<LoginAction, String> handler, String callback, Duration timeout, String appName) {
        ObjectNode op = operation("login", null);
        if (callback != null) {
            op.put("callback", callback);
        }
        if (timeout != null) {
            op.put("timeout_ms", timeout.toMillis());
        }
        if (appName != null) {
            op.put("app_name", appName);
        }
        return Json.as(run(op, null, handler, null).get("status"), AuthStatus.class);
    }

    /** Forgets the provider's own session. */
    public void logout() {
        run(operation("logout", null), null, null, null);
    }

    /** Reports which credential would be used and when it expires. */
    public AuthStatus authStatus() {
        JsonNode status = run(operation("auth_status", null), null, null, null).get("status");
        return status == null || status.isNull() ? new AuthStatus(null, null, null) : Json.as(status, AuthStatus.class);
    }
}
