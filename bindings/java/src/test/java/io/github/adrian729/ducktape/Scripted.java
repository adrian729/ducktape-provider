package io.github.adrian729.ducktape;

import com.fasterxml.jackson.databind.JsonNode;
import java.io.IOException;
import java.net.ConnectException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;

/** A transport that answers with a fixture's scripted replies in order. */
final class Scripted implements Transport {
    record Seen(String url, Map<String, String> headers, JsonNode body) {}

    private final List<JsonNode> script = new ArrayList<>();
    final List<Seen> seen = new ArrayList<>();
    volatile long delayMs;
    volatile Consumer<String> onRequest;

    Scripted(JsonNode fixture) {
        for (JsonNode h : fixture.path("http")) {
            script.add(h);
        }
    }

    @Override
    public synchronized Reply send(Request request, Duration headersTimeout) throws IOException {
        JsonNode parsed = null;
        try {
            parsed = request.body() == null ? null : Json.MAPPER.readTree(request.body());
        } catch (IOException ignored) {
        }
        Map<String, String> headers = new HashMap<>();
        request.headers().forEach((k, v) -> headers.put(k.toLowerCase(java.util.Locale.ROOT), v));
        seen.add(new Seen(request.url(), headers, parsed));
        if (onRequest != null) {
            onRequest.accept(request.url());
        }
        if (script.isEmpty()) {
            throw new IOException("script exhausted");
        }
        JsonNode h = script.remove(0);
        if (h.has("error")) {
            throw new ConnectException(h.path("error").path("message").asText());
        }
        List<byte[]> chunks = new ArrayList<>();
        if (h.has("chunks") && h.get("chunks").size() > 0) {
            for (JsonNode c : h.get("chunks")) {
                chunks.add(c.asText().getBytes(StandardCharsets.UTF_8));
            }
        } else if (h.has("body")) {
            chunks.add(h.get("body").asText().getBytes(StandardCharsets.UTF_8));
        }
        Map<String, String> replyHeaders = new HashMap<>();
        for (var it = h.path("headers").fields(); it.hasNext(); ) {
            var e = it.next();
            replyHeaders.put(e.getKey(), e.getValue().asText());
        }
        long delay = delayMs;
        Body body = new Body() {
            private int index;
            private boolean closed;

            @Override
            public ByteBuffer next(Duration timeout) throws IOException {
                if (closed) {
                    throw new IOException("closed");
                }
                if (index >= chunks.size()) {
                    return null;
                }
                if (delay > 0) {
                    if (timeout != null && timeout.toMillis() < delay) {
                        try {
                            Thread.sleep(timeout.toMillis());
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                        throw new java.net.http.HttpTimeoutException("read timed out");
                    }
                    try {
                        Thread.sleep(delay);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    }
                    if (closed) {
                        throw new IOException("closed");
                    }
                }
                return ByteBuffer.wrap(chunks.get(index++));
            }

            @Override
            public void close() {
                closed = true;
            }
        };
        return new Reply(h.path("status").asInt(200), replyHeaders, body);
    }
}
