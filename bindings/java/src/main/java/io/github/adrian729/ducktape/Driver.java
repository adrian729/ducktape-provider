package io.github.adrian729.ducktape;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.github.adrian729.ducktape.types.Event;
import io.github.adrian729.ducktape.types.LoginAction;
import io.github.adrian729.ducktape.types.ProviderError;
import java.io.IOException;
import java.net.ConnectException;
import java.net.UnknownHostException;
import java.net.http.HttpConnectTimeoutException;
import java.net.http.HttpTimeoutException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermissions;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.time.Duration;
import java.util.Base64;
import java.util.HexFormat;
import java.util.Iterator;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.Function;
import java.util.function.Supplier;
import javax.net.ssl.SSLException;

/** The effect loop: answers what the core asks for with the host's HTTP, files, environment and keychain. */
final class Driver {
    private static final SecureRandom RANDOM = new SecureRandom();

    record Host(Map<String, String> secrets, Map<String, Supplier<String>> resolvers, Keychain.Reader keychain, Keychain.Writer keychainWrite, Transport transport, Observer observer) {}

    /** A cancellation flag shared with the caller; closing the body unblocks a read. */
    static final class Cancel {
        volatile boolean flag;
        volatile Transport.Body body;

        void cancel() {
            flag = true;
            Transport.Body b = body;
            if (b != null) {
                b.close();
            }
        }
    }

    static ObjectNode clock() {
        byte[] entropy = new byte[32];
        RANDOM.nextBytes(entropy);
        ObjectNode c = Json.object();
        c.put("now_ms", System.currentTimeMillis());
        c.put("entropy", Base64.getEncoder().encodeToString(entropy));
        return c;
    }

    static String etag(byte[] bytes) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
        } catch (java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException(e);
        }
    }

    private final Core.Operation op;
    private final Host host;
    private final Consumer<Event> onEvent;
    private final Function<LoginAction, String> handleAction;
    private final Cancel cancel;
    private Transport.Body body;
    private Duration readTimeout;
    private long deadline;
    private byte[] buffer = new byte[64 * 1024];
    private ObjectNode pendingInput;

    Driver(Core.Operation op, Host host, Consumer<Event> onEvent, Function<LoginAction, String> handleAction, Cancel cancel) {
        this.op = op;
        this.host = host;
        this.onEvent = onEvent;
        this.handleAction = handleAction;
        this.cancel = cancel == null ? new Cancel() : cancel;
    }

    private void closeBody() {
        if (body != null) {
            body.close();
            body = null;
            cancel.body = null;
        }
    }

    JsonNode run(Object start) {
        try {
            JsonNode step = op.start(start);
            for (int i = 0; i < 1_000_000; i++) {
                for (JsonNode note : step.path("notify")) {
                    String type = note.path("type").asText();
                    if ("event".equals(type) && onEvent != null) {
                        onEvent.accept(Json.as(note.get("event"), Event.class));
                    } else if ("observer".equals(type) && host.observer() != null) {
                        observe(note.path("hook").asText(), note.get("data"));
                    }
                }
                JsonNode effect = step.path("effect");
                String type = effect.path("type").asText();
                if ("done".equals(type)) {
                    return effect.get("result");
                }
                if ("failed".equals(type)) {
                    throw new ProviderException(Json.as(effect.get("error"), ProviderError.class));
                }
                ObjectNode input;
                if (cancel.flag && !"receive".equals(type)) {
                    input = Json.object().put("type", "cancelled");
                } else {
                    switch (type) {
                        case "need_secret" -> input = secret(effect);
                        case "need_http" -> {
                            closeBody();
                            input = http(effect.get("request"));
                        }
                        case "receive" -> {
                            JsonNode fed = receive();
                            if (fed != null) {
                                step = fed;
                                continue;
                            }
                            input = pendingInput;
                        }
                        case "need_sleep" -> {
                            closeBody();
                            input = sleep(effect.path("ms").asLong());
                        }
                        case "need_read" -> input = read(effect.get("locator"));
                        case "need_write" -> input = write(effect);
                        case "need_user_action" -> input = userAction(effect.get("action"));
                        case "need_callback" -> throw ProviderException.of("unsupported", null, "the loopback login callback is not implemented in the Java binding yet");
                        default -> throw ProviderException.of("unsupported", null, "the core asked for an unknown effect " + type);
                    }
                }
                step = op.resume(resume(input));
            }
            throw ProviderException.of("other", "internal", "the operation did not finish");
        } finally {
            closeBody();
        }
    }

    private static ObjectNode resume(ObjectNode input) {
        ObjectNode doc = Json.object();
        doc.set("clock", clock());
        doc.set("input", input);
        return doc;
    }

    private void observe(String hook, JsonNode data) {
        Observer o = host.observer();
        switch (hook) {
            case "on_request" -> o.onRequest(data);
            case "on_response" -> o.onResponse(data);
            case "on_event" -> o.onEvent(data);
            case "on_retry" -> o.onRetry(data);
            default -> {
            }
        }
    }

    private ObjectNode secret(JsonNode effect) {
        String ref = effect.path("secret_ref").asText();
        JsonNode source = effect.path("source");
        String value = null;
        switch (source.path("type").asText()) {
            case "inline" -> value = host.secrets().get(ref);
            case "env" -> {
                for (JsonNode name : source.path("names")) {
                    String v = System.getenv(name.asText());
                    if (v != null && !v.isEmpty()) {
                        value = v;
                        break;
                    }
                }
            }
            case "resolver" -> {
                Supplier<String> fn = host.resolvers().get(source.path("resolver_id").asText());
                if (fn != null) {
                    try {
                        value = fn.get();
                    } catch (RuntimeException e) {
                        throw new ProviderException(ProviderError.builder().kind("auth").code("resolver_failed").message("the credential resolver failed: " + e.getMessage()).retryable(false).attempts(0L).build(), e);
                    }
                }
            }
            default -> {
            }
        }
        if (value != null && !value.isEmpty()) {
            op.setSecret(ref, value.getBytes(StandardCharsets.UTF_8));
        }
        return Json.object().put("type", "secret").put("secret_ref", ref).put("found", value != null && !value.isEmpty());
    }

    private static Duration seconds(JsonNode node) {
        return node == null || node.isNull() || node.isMissingNode() ? null : Duration.ofMillis((long) (node.asDouble() * 1000));
    }

    private ObjectNode transportError(String kind, String message) {
        ObjectNode error = Json.object().put("kind", kind).put("message", message == null ? kind : message);
        ObjectNode in = Json.object().put("type", "http_error");
        in.set("error", error);
        return in;
    }

    private ObjectNode http(JsonNode req) {
        JsonNode timeout = req.path("timeout");
        Duration connect = seconds(timeout.get("connect"));
        Duration total = seconds(timeout.get("total"));
        readTimeout = seconds(timeout.get("read"));
        deadline = total == null ? 0 : System.currentTimeMillis() + total.toMillis();
        Map<String, String> headers = new java.util.LinkedHashMap<>();
        for (Iterator<Map.Entry<String, JsonNode>> it = req.path("headers").fields(); it.hasNext(); ) {
            Map.Entry<String, JsonNode> h = it.next();
            headers.put(h.getKey(), h.getValue().asText());
        }
        JsonNode bodyNode = req.get("body");
        Transport.Request request = new Transport.Request(req.path("method").asText("GET"), req.path("url").asText(), headers, bodyNode == null || bodyNode.isNull() ? null : bodyNode.asText());
        Duration headersTimeout = connect;
        if (total != null && (headersTimeout == null || total.compareTo(headersTimeout) < 0)) {
            headersTimeout = total;
        }
        Transport.Reply reply;
        try {
            reply = host.transport().send(request, headersTimeout);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return Json.object().put("type", "cancelled");
        } catch (IOException | RuntimeException e) {
            if (cancel.flag) {
                return Json.object().put("type", "cancelled");
            }
            return transportError(classify(e), e.getMessage());
        }
        body = reply.body();
        cancel.body = body;
        ObjectNode headerNode = Json.object();
        reply.headers().forEach((k, v) -> headerNode.put(k.toLowerCase(java.util.Locale.ROOT), v));
        ObjectNode in = Json.object().put("type", "http_response").put("status", reply.status());
        in.set("headers", headerNode);
        return in;
    }

    private JsonNode receive() {
        pendingInput = null;
        if (body == null) {
            pendingInput = Json.object().put("type", "http_end");
            return null;
        }
        Duration wait = readTimeout;
        if (deadline != 0) {
            Duration left = Duration.ofMillis(Math.max(0, deadline - System.currentTimeMillis()));
            wait = wait == null || left.compareTo(wait) < 0 ? left : wait;
        }
        java.nio.ByteBuffer chunk;
        try {
            chunk = body.next(wait);
        } catch (IOException e) {
            closeBody();
            pendingInput = cancel.flag ? Json.object().put("type", "cancelled") : transportError(classify(e), e.getMessage());
            return null;
        }
        if (cancel.flag) {
            closeBody();
            pendingInput = Json.object().put("type", "cancelled");
            return null;
        }
        if (chunk == null) {
            closeBody();
            pendingInput = Json.object().put("type", "http_end");
            return null;
        }
        int n = chunk.remaining();
        if (n == 0) {
            return receive();
        }
        if (buffer.length < n) {
            buffer = new byte[n];
        }
        chunk.get(buffer, 0, n);
        return op.feed(buffer, n, clock());
    }

    static String classify(Throwable e) {
        for (Throwable t = e; t != null; t = t.getCause()) {
            if (t instanceof HttpConnectTimeoutException || t instanceof HttpTimeoutException) {
                return "timeout";
            }
            if (t instanceof UnknownHostException) {
                return "dns";
            }
            if (t instanceof SSLException) {
                return "tls";
            }
            if (t instanceof ConnectException) {
                return "connect";
            }
            String m = t.getMessage() == null ? "" : t.getMessage().toLowerCase(java.util.Locale.ROOT);
            if (m.contains("reset") || m.contains("eof") || m.contains("broken pipe")) {
                return "reset";
            }
        }
        return "other";
    }

    private ObjectNode sleep(long ms) {
        try {
            long end = System.currentTimeMillis() + ms;
            while (!cancel.flag) {
                long left = end - System.currentTimeMillis();
                if (left <= 0) {
                    return Json.object().put("type", "slept");
                }
                Thread.sleep(Math.min(left, 50));
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        return Json.object().put("type", "cancelled");
    }

    private ObjectNode read(JsonNode locator) {
        ObjectNode missing = Json.object().put("type", "read").put("found", false);
        if (locator == null) {
            return missing;
        }
        byte[] data;
        String tag = null;
        switch (locator.path("type").asText()) {
            case "file" -> {
                try {
                    data = Files.readAllBytes(Path.of(locator.path("path").asText()));
                } catch (IOException | RuntimeException e) {
                    return missing;
                }
                tag = etag(data);
            }
            case "env" -> {
                String v = System.getenv(locator.path("name").asText());
                if (v == null) {
                    return missing;
                }
                data = v.getBytes(StandardCharsets.UTF_8);
            }
            case "keychain" -> {
                if (host.keychain() == null) {
                    return missing;
                }
                try {
                    data = host.keychain().read(locator.path("service").asText(), locator.path("account").asText());
                } catch (Exception e) {
                    data = null;
                }
                if (data == null || data.length == 0) {
                    return missing;
                }
            }
            default -> {
                return missing;
            }
        }
        ObjectNode in = Json.object().put("type", "read").put("found", true).put("bytes", Base64.getEncoder().encodeToString(data));
        if (tag != null) {
            in.put("etag", tag);
        }
        return in;
    }

    private ObjectNode write(JsonNode effect) {
        JsonNode locator = effect.path("locator");
        byte[] data = Base64.getDecoder().decode(effect.path("bytes").asText());
        ObjectNode failed = Json.object().put("type", "written").put("ok", false);
        if ("keychain".equals(locator.path("type").asText())) {
            boolean ok = false;
            if (host.keychainWrite() != null) {
                try {
                    ok = host.keychainWrite().write(locator.path("service").asText(), locator.path("account").asText(), data);
                } catch (Exception e) {
                    ok = false;
                }
            }
            return Json.object().put("type", "written").put("ok", ok);
        }
        if (!"file".equals(locator.path("type").asText())) {
            return failed;
        }
        Path path = Path.of(locator.path("path").asText());
        JsonNode expect = effect.get("expect");
        if (expect != null && !expect.isNull()) {
            String current = null;
            try {
                current = etag(Files.readAllBytes(path));
            } catch (IOException | RuntimeException ignored) {
            }
            if (!expect.asText().equals(current)) {
                return failed.put("conflict", true);
            }
        }
        boolean isPrivate = effect.path("private").asBoolean(true);
        try {
            Path dir = path.toAbsolutePath().getParent();
            Files.createDirectories(dir);
            Path tmp = Files.createTempFile(dir, path.getFileName().toString(), ".tmp");
            try {
                Files.write(tmp, data);
                if (isPrivate) {
                    try {
                        Files.setPosixFilePermissions(tmp, PosixFilePermissions.fromString("rw-------"));
                    } catch (UnsupportedOperationException ignored) {
                    }
                }
                Files.move(tmp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } finally {
                Files.deleteIfExists(tmp);
            }
            return Json.object().put("type", "written").put("ok", true);
        } catch (IOException | RuntimeException e) {
            return failed;
        }
    }

    private ObjectNode userAction(JsonNode action) {
        if (handleAction == null) {
            throw ProviderException.of("unsupported", null, "this call cannot ask the user for anything");
        }
        if (action == null) {
            throw ProviderException.of("malformed_response", null, "the core asked for a user action without one");
        }
        String value;
        try {
            value = handleAction.apply(Json.as(action, LoginAction.class));
        } catch (RuntimeException e) {
            throw new ProviderException(ProviderError.builder().kind("auth").code("user_action_failed").message("the login handler failed: " + e.getMessage()).retryable(false).attempts(0L).build(), e);
        }
        ObjectNode in = Json.object().put("type", "user_action");
        if (value != null) {
            in.put("value", value);
        }
        return in;
    }
}
