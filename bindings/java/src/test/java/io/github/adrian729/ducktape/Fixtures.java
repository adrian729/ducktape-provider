package io.github.adrian729.ducktape;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.stream.Stream;

final class Fixtures {
    private Fixtures() {}

    static final Path WIRE = Path.of("..", "..", "fixtures", "wire").toAbsolutePath().normalize();
    static final String ENTROPY = Base64.getEncoder().encodeToString(repeat((byte) 128, 32));

    private static byte[] repeat(byte b, int n) {
        byte[] out = new byte[n];
        java.util.Arrays.fill(out, b);
        return out;
    }

    static ObjectNode clock(long tick) {
        return Json.object().put("now_ms", 1_761_696_000_000L + tick * 100).put("entropy", ENTROPY);
    }

    static JsonNode load(Path path) {
        try {
            return Json.MAPPER.readTree(Files.readAllBytes(path));
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }

    static JsonNode load(String wire, String name) {
        return load(WIRE.resolve(wire).resolve(name + ".json"));
    }

    static List<Path> all() {
        List<Path> out = new ArrayList<>();
        try (Stream<Path> dirs = Files.list(WIRE)) {
            for (Path dir : dirs.sorted().toList()) {
                if (!Files.isDirectory(dir)) {
                    continue;
                }
                try (Stream<Path> files = Files.list(dir)) {
                    for (Path f : files.sorted().toList()) {
                        if (f.toString().endsWith(".json")) {
                            out.add(f);
                        }
                    }
                }
            }
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
        return out;
    }

    static String encode(String text) {
        return Base64.getEncoder().encodeToString(text.getBytes(StandardCharsets.UTF_8));
    }

    /** Replays a fixture's script through the core and returns the exchange it produced. */
    static List<JsonNode[]> replay(Core core, JsonNode fixture) {
        List<JsonNode[]> out = new ArrayList<>();
        try (Core.Operation op = core.operation()) {
            ObjectNode start = fixture.get("start").deepCopy();
            start.set("clock", clock(0));
            JsonNode step = op.start(start);
            int http = 0;
            int writes = 0;
            int userActions = 0;
            int callbacks = 0;
            List<ObjectNode> queue = new ArrayList<>();
            for (long tick = 1; tick < 10_000; tick++) {
                JsonNode effect = step.path("effect");
                String type = effect.path("type").asText();
                ObjectNode input;
                switch (type) {
                    case "done", "failed" -> {
                        out.add(new JsonNode[] {step, null});
                        return out;
                    }
                    case "need_secret" -> {
                        String ref = effect.path("secret_ref").asText();
                        JsonNode value = fixture.path("secrets").get(ref);
                        if (value != null) {
                            op.setSecret(ref, value.asText().getBytes(StandardCharsets.UTF_8));
                        }
                        input = Json.object().put("type", "secret").put("secret_ref", ref).put("found", value != null);
                    }
                    case "need_http" -> {
                        JsonNode h = fixture.path("http").get(http++);
                        if (h == null) {
                            throw new AssertionError(fixture.path("name").asText() + ": script has no HTTP reply for request");
                        }
                        if (h.has("error")) {
                            input = Json.object().put("type", "http_error");
                            input.set("error", h.get("error"));
                        } else {
                            List<String> chunks = new ArrayList<>();
                            if (h.has("chunks") && h.get("chunks").size() > 0) {
                                for (JsonNode c : h.get("chunks")) {
                                    chunks.add(c.asText());
                                }
                            } else if (h.has("body")) {
                                chunks.add(h.get("body").asText());
                            }
                            queue.clear();
                            queue.add(Json.object().put("type", "http_end"));
                            for (int i = chunks.size() - 1; i >= 0; i--) {
                                queue.add(Json.object().put("type", "http_chunk").put("bytes", encode(chunks.get(i))));
                            }
                            input = Json.object().put("type", "http_response").put("status", h.path("status").asInt(200));
                            input.set("headers", h.has("headers") ? h.get("headers") : Json.object());
                        }
                    }
                    case "receive" -> input = queue.isEmpty() ? Json.object().put("type", "http_end") : queue.remove(queue.size() - 1);
                    case "need_sleep" -> input = Json.object().put("type", "slept");
                    case "need_read" -> {
                        JsonNode loc = effect.get("locator");
                        JsonNode value = switch (loc.path("type").asText()) {
                            case "file" -> fixture.path("files").get(loc.path("path").asText());
                            case "env" -> fixture.path("env").get(loc.path("name").asText());
                            default -> fixture.path("keychain").get(loc.path("service").asText() + "|" + loc.path("account").asText());
                        };
                        input = Json.object().put("type", "read").put("found", value != null);
                        if (value != null) {
                            input.put("bytes", encode(value.asText()));
                            if ("file".equals(loc.path("type").asText())) {
                                input.put("etag", Driver.etag(value.asText().getBytes(StandardCharsets.UTF_8)));
                            }
                        }
                    }
                    case "need_write" -> {
                        JsonNode w = fixture.path("writes").get(writes++);
                        input = Json.object().put("type", "written").put("ok", w == null || w.path("ok").asBoolean(true)).put("conflict", w != null && w.path("conflict").asBoolean(false));
                    }
                    case "need_user_action" -> {
                        JsonNode v = fixture.path("user_actions").get(userActions++);
                        input = Json.object().put("type", "user_action");
                        if (v != null && v.isTextual()) {
                            input.put("value", v.asText());
                        }
                    }
                    case "need_callback" -> {
                        JsonNode url = fixture.path("callbacks").get(callbacks++);
                        if (url == null) {
                            throw new AssertionError(fixture.path("name").asText() + ": script has no callback for the login");
                        }
                        input = Json.object().put("type", "callback").put("url", url.asText());
                    }
                    default -> throw new AssertionError(fixture.path("name").asText() + ": unsupported effect in replay: " + type);
                }
                out.add(new JsonNode[] {step, input});
                ObjectNode resume = Json.object();
                resume.set("clock", clock(tick));
                resume.set("input", input);
                step = op.resume(resume);
            }
        }
        throw new AssertionError(fixture.path("name").asText() + ": replay did not finish");
    }

    /** A step with its request body parsed and keys sorted, so JSON key order does not matter. */
    static String canonical(JsonNode node) {
        if (node == null || node.isNull()) {
            return "null";
        }
        JsonNode copy = node.deepCopy();
        JsonNode body = copy.path("effect").path("request").path("body");
        if (body.isTextual()) {
            try {
                ((ObjectNode) copy.get("effect").get("request")).set("body", Json.MAPPER.readTree(body.asText()));
            } catch (IOException ignored) {
            }
        }
        try {
            Object plain = Json.MAPPER.treeToValue(copy, Object.class);
            return Json.MAPPER.writer().with(com.fasterxml.jackson.databind.SerializationFeature.ORDER_MAP_ENTRIES_BY_KEYS).writeValueAsString(plain);
        } catch (IOException e) {
            throw new IllegalStateException(e);
        }
    }
}
