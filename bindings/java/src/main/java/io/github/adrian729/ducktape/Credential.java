package io.github.adrian729.ducktape;

import io.github.adrian729.ducktape.types.AuthSource;
import java.util.List;
import java.util.function.Supplier;

/** One way to obtain a key: in code, from the environment, from a login session or from a resolver. */
public final class Credential {
    final AuthSource source;
    final String key;
    final Supplier<String> resolver;

    private Credential(AuthSource source, String key, Supplier<String> resolver) {
        this.source = source;
        this.key = key;
        this.resolver = resolver;
    }

    /** A key given in code. */
    public static Credential key(String secret) {
        return new Credential(AuthSource.builder().type("inline").build(), secret, null);
    }

    /** The first of these environment variables that is set. */
    public static Credential env(String... names) {
        return new Credential(AuthSource.builder().type("env").names(List.of(names)).build(), null, null);
    }

    /** A login session of the given preset. */
    public static Credential session(String preset) {
        return new Credential(AuthSource.builder().type("session").preset(preset).build(), null, null);
    }

    /** A login session with its store, client id, profile or refresh mode set. */
    public static Credential session(String preset, String store, String clientId, String profile, String refresh) {
        return new Credential(AuthSource.builder().type("session").preset(preset).store(store).clientId(clientId).profile(profile).refresh(refresh).build(), null, null);
    }

    /** A key fetched at request time. */
    public static Credential resolver(Supplier<String> fn) {
        return new Credential(AuthSource.builder().type("resolver").build(), null, fn);
    }

    /** Any auth source document as it is. */
    public static Credential source(AuthSource source) {
        return new Credential(source, null, null);
    }

    /** The same credential sent with another kind and placement. */
    public Credential withKind(String kind, String placement) {
        AuthSource s = source;
        AuthSource next = new AuthSource(s.type(), s.secretRef(), kind, placement, s.names(), s.resolverId(), s.preset(), s.store(), s.clientId(), s.profile(), s.refresh(), s.grant(), s.params());
        return new Credential(next, key, resolver);
    }
}
