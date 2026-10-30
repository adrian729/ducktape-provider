package io.github.adrian729.ducktape;

import static io.github.adrian729.ducktape.Ducktape.assistant;
import static io.github.adrian729.ducktape.Ducktape.text;
import static io.github.adrian729.ducktape.Ducktape.textOf;
import static io.github.adrian729.ducktape.Ducktape.tool;
import static io.github.adrian729.ducktape.Ducktape.toolCallsOf;
import static io.github.adrian729.ducktape.Ducktape.toolResult;
import static io.github.adrian729.ducktape.Ducktape.user;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import io.github.adrian729.ducktape.types.Block;
import io.github.adrian729.ducktape.types.ChatRequest;
import io.github.adrian729.ducktape.types.Event;
import io.github.adrian729.ducktape.types.Message;
import io.github.adrian729.ducktape.types.Options;
import io.github.adrian729.ducktape.types.Response;
import io.github.adrian729.ducktape.types.Timeout;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ProviderTest {
    private static JsonNode anthropic(String name) {
        return Fixtures.load("anthropic.messages", name);
    }

    private static JsonNode openai(String name) {
        return Fixtures.load("openai.responses", name);
    }

    private static ChatRequest requestOf(JsonNode fixture) {
        return Json.as(fixture.path("start").path("operation").path("request"), ChatRequest.class);
    }

    private static Response doneResponse(JsonNode fixture) {
        JsonNode exchange = fixture.get("exchange");
        return Json.as(exchange.get(exchange.size() - 1).path("step").path("effect").path("result").path("response"), Response.class);
    }

    private static String lastText(JsonNode fixture) {
        List<Block> content = doneResponse(fixture).content();
        return content.get(content.size() - 1).text();
    }

    private static Provider.Builder provider(JsonNode fixture, Scripted server) {
        JsonNode config = fixture.path("start").path("provider");
        return Provider.builder(config.path("vendor").asText()).baseUrl(config.has("base_url") ? config.get("base_url").asText() : null).transport(server).auth(Credential.key("sk-ant-test-key"));
    }

    private static String jwt(long expSeconds) {
        var enc = Base64.getUrlEncoder().withoutPadding();
        return enc.encodeToString("{\"alg\":\"RS256\"}".getBytes(StandardCharsets.UTF_8)) + "." + enc.encodeToString(("{\"exp\":" + expSeconds + "}").getBytes(StandardCharsets.UTF_8)) + ".sig";
    }

    private static void sessionStore(JsonNode fixture, Path dir, String prefix) throws Exception {
        for (var it = fixture.path("files").fields(); it.hasNext(); ) {
            var e = it.next();
            Path target = dir.resolve(e.getKey().substring(prefix.length() + 1));
            Files.createDirectories(target.getParent());
            Files.writeString(target, e.getValue().asText());
        }
    }

    @Test
    void chatMatchesFixture() throws Exception {
        JsonNode fixture = anthropic("chat_text_reasoning");
        {
            Scripted server = new Scripted(fixture);
            Provider p = provider(fixture, server).build();
            ChatRequest req = requestOf(fixture);
            Response r = p.chat(req.messages(), ChatOptions.builder().model(req.model()).options(req.options()).build());
            Response expected = doneResponse(fixture);
            assertEquals(expected.content().size(), r.content().size());
            for (int i = 0; i < r.content().size(); i++) {
                assertEquals(expected.content().get(i).type(), r.content().get(i).type());
                assertEquals(expected.content().get(i).text(), r.content().get(i).text());
            }
            assertEquals(expected.stop().reason(), r.stop().reason());
            assertEquals(expected.usage(), r.usage());
            assertFalse(textOf(r).isEmpty());
            assertEquals("sk-ant-test-key", server.seen.get(0).headers().get("x-api-key"));
            assertEquals(req.model(), server.seen.get(0).body().path("model").asText());
        }
    }

    @Test
    void streamEndsWithFinish() throws Exception {
        JsonNode fixture = anthropic("stream_text_reasoning_chunked");
        {
            Scripted server = new Scripted(fixture);
            Provider p = provider(fixture, server).build();
            ChatRequest req = requestOf(fixture);
            List<Event> events = new ArrayList<>();
            Stream s = p.stream(req.messages(), ChatOptions.builder().model(req.model()).build());
            for (Event ev : s) {
                events.add(ev);
            }
            assertEquals("block_start", events.get(0).type());
            assertEquals("finish", events.get(events.size() - 1).type());
            StringBuilder text = new StringBuilder();
            for (Event ev : events) {
                if ("text_delta".equals(ev.type())) {
                    text.append(ev.text());
                }
            }
            assertEquals(text.toString(), textOf(s.response()));
            assertFalse(text.isEmpty());
        }
    }

    @Test
    void toolRoundTripBuildsHistory() throws Exception {
        JsonNode fixture = anthropic("chat_tool_call");
        {
            Scripted server = new Scripted(fixture);
            Provider p = provider(fixture, server).build();
            ChatRequest req = requestOf(fixture);
            Response r = p.chat(req.messages(), ChatOptions.builder().model(req.model()).tools(req.tools()).build());
            assertEquals("tool_call", r.stop().reason());
            Block call = toolCallsOf(r).get(0);
            List<Message> history = new ArrayList<>(req.messages());
            history.add(assistant(r));
            history.add(tool(toolResult(call.id(), "4")));
            assertEquals("assistant", history.get(history.size() - 2).role());
            assertEquals(call.id(), history.get(history.size() - 1).content().get(0).toolCallId());
        }
    }

    @Test
    void errorsCarryKinds() throws Exception {
        JsonNode fixture = anthropic("chat_context_overflow");
        {
            Scripted server = new Scripted(fixture);
            Provider p = provider(fixture, server).build();
            ChatRequest req = requestOf(fixture);
            ProviderException e = assertThrows(ProviderException.class, () -> p.chat(req.messages(), ChatOptions.builder().model(req.model()).build()));
            assertTrue(e.is("context_overflow"));
            assertEquals(400L, e.status());
            assertTrue(e.getMessage().contains("context_overflow"));
        }
    }

    @Test
    void noCredentialIsAuthError() throws Exception {
        JsonNode fixture = anthropic("chat_no_credential");
        {
            Scripted server = new Scripted(fixture);
            Provider p = Provider.builder("anthropic").transport(server).auth(Credential.env("DUCKTAPE_TEST_UNSET_KEY")).build();
            ProviderException e = assertThrows(ProviderException.class, () -> p.chat(List.of(user("hi")), ChatOptions.builder().model("m").build()));
            assertTrue(e.is("auth"));
            assertTrue(server.seen.isEmpty());
        }
    }

    @Test
    void resolverCredential() throws Exception {
        JsonNode fixture = anthropic("chat_text_reasoning");
        {
            Scripted server = new Scripted(fixture);
            int[] calls = {0};
            Provider p = Provider.builder("anthropic").transport(server).auth(Credential.resolver(() -> { calls[0]++; return "sk-resolved"; })).build();
            ChatRequest req = requestOf(fixture);
            p.chat(req.messages(), ChatOptions.builder().model(req.model()).build());
            assertEquals(1, calls[0]);
            assertEquals("sk-resolved", server.seen.get(0).headers().get("x-api-key"));
        }
        {
            Scripted server = new Scripted(fixture);
            Provider failing = Provider.builder("anthropic").transport(server).auth(Credential.resolver(() -> { throw new IllegalStateException("vault down"); })).build();
            ChatRequest req = requestOf(fixture);
            ProviderException e = assertThrows(ProviderException.class, () -> failing.chat(req.messages(), ChatOptions.builder().model(req.model()).build()));
            assertTrue(e.is("auth") && e.getMessage().contains("vault down"));
        }
    }

    @Test
    void argumentErrorsComeFirst() throws Exception {
        JsonNode fixture = anthropic("chat_text_reasoning");
        {
            Scripted server = new Scripted(fixture);
            Provider p = provider(fixture, server).build();
            assertThrows(IllegalArgumentException.class, () -> p.chat(List.of(user("hi"))));
            assertThrows(IllegalArgumentException.class, () -> Provider.builder("nope"));
            assertThrows(IllegalArgumentException.class, () -> p.embed(List.of()));
            assertTrue(server.seen.isEmpty());
        }
    }

    @Test
    void connectionFailureRetriesThenSucceeds() throws Exception {
        JsonNode fixture = anthropic("chat_connection_reset_retried");
        {
            Scripted server = new Scripted(fixture);
            Provider p = provider(fixture, server).build();
            ChatRequest req = requestOf(fixture);
            Response r = p.chat(req.messages(), ChatOptions.builder().model(req.model()).build());
            assertEquals(2, server.seen.size());
            assertFalse(textOf(r).isEmpty());
        }
    }

    @Test
    void modelsAreCached() throws Exception {
        JsonNode fixture = anthropic("models_two_pages");
        {
            Scripted server = new Scripted(fixture);
            Provider p = provider(fixture, server).build();
            var models = p.models(false);
            var again = p.models(false);
            assertFalse(models.isEmpty());
            assertEquals(models.size(), again.size());
            assertEquals(2, server.seen.size());
        }
    }

    @Test
    void capabilitiesShipped() throws Exception {
        JsonNode fixture = anthropic("capabilities_shipped");
        {
            Scripted server = new Scripted(fixture);
            Provider p = provider(fixture, server).build();
            var caps = p.capabilities(fixture.path("start").path("operation").path("model").asText());
            assertEquals("shipped", caps.source());
            assertTrue(server.seen.isEmpty());
        }
    }

    @Test
    void streamCloseCancels() throws Exception {
        JsonNode fixture = anthropic("stream_text_reasoning_chunked");
        {
            Scripted server = new Scripted(fixture);
            server.delayMs = 30;
            Provider p = provider(fixture, server).build();
            ChatRequest req = requestOf(fixture);
            Stream s = p.stream(req.messages(), ChatOptions.builder().model(req.model()).build());
            assertTrue(s.hasNext());
            s.next();
            s.close();
            ProviderException e = assertThrows(ProviderException.class, () -> { while (s.hasNext()) { s.next(); } });
            assertTrue(e.is("cancelled"));
        }
    }

    @Test
    void futureCancelCancelsChat() throws Exception {
        JsonNode fixture = anthropic("chat_text_reasoning");
        {
            Scripted server = new Scripted(fixture);
            server.delayMs = 500;
            Provider p = provider(fixture, server).build();
            ChatRequest req = requestOf(fixture);
            CompletableFuture<Response> f = p.chatAsync(req.messages(), ChatOptions.builder().model(req.model()).build());
            Thread.sleep(50);
            assertTrue(f.cancel(true));
            assertThrows(CancellationException.class, () -> f.get(5, TimeUnit.SECONDS));
        }
    }

    @Test
    void readTimeoutMapsToTimeout() throws Exception {
        JsonNode fixture = anthropic("chat_text_reasoning");
        {
            Scripted server = new Scripted(fixture);
            server.delayMs = 400;
            Provider p = provider(fixture, server).build();
            ChatRequest req = requestOf(fixture);
            ChatOptions o = ChatOptions.builder().model(req.model()).timeout(new Timeout(null, 0.05, null)).options(Options.builder().retries(0L).build()).build();
            ProviderException e = assertThrows(ProviderException.class, () -> p.chat(req.messages(), o));
            assertTrue(e.is("timeout"), e.getMessage());
        }
    }

    @Test
    void embed() throws Exception {
        JsonNode fixture = openai("embed_two_inputs");
        {
            Scripted server = new Scripted(fixture);
            Provider p = Provider.builder("openai").transport(server).auth(Credential.key("sk-test")).build();
            JsonNode req = fixture.path("start").path("operation").path("request");
            List<String> input = new ArrayList<>();
            for (JsonNode s : req.get("input")) {
                input.add(s.asText());
            }
            var r = p.embed(input, EmbedOptions.builder().model(req.path("model").asText()).build());
            assertEquals(2, r.embeddings().size());
            assertEquals(0.1, r.embeddings().get(0).get(0));
        }
    }

    @Test
    void imageBytesBecomeBase64() {
        byte[] png = {(byte) 0x89, 0x50, 0x4e, 0x47};
        Message m = user(text("what is this?"), Ducktape.image(png, "image/png"));
        assertEquals(Base64.getEncoder().encodeToString(png), m.content().get(1).data());
        assertEquals("what is this?", m.content().get(0).text());
    }

    @Test
    void profileRefreshesAndWritesBack(@TempDir Path dir) throws Exception {
        JsonNode fixture = anthropic("session_profile_refresh_write_back");
        sessionStore(fixture, dir, "/home/u/.config/anthropic");
        {
            Scripted server = new Scripted(fixture);
            Provider p = Provider.builder("anthropic").transport(server).auth(Credential.session("anthropic_profile", dir.toString(), null, "default", "write_back")).build();
            ChatRequest req = requestOf(fixture);
            Response r = p.chat(req.messages(), ChatOptions.builder().model(req.model()).build());
            assertEquals(lastText(fixture), textOf(r));
            assertEquals("https://api.anthropic.com/v1/oauth/token", server.seen.get(0).url());
            assertEquals("Bearer sk-ant-oat01-new", server.seen.get(1).headers().get("authorization"));
            JsonNode written = Json.MAPPER.readTree(Files.readAllBytes(dir.resolve("credentials").resolve("default.json")));
            assertEquals("sk-ant-oat01-new", written.path("access_token").asText());
            assertEquals("kept", written.path("extra").asText());
            assertEquals("session:anthropic_profile", p.authStatus().source());
        }
    }

    @Test
    void writeConflictKeepsTheRefreshedToken(@TempDir Path dir) throws Exception {
        JsonNode fixture = anthropic("session_profile_write_conflict_keeps_token");
        sessionStore(fixture, dir, "/home/u/.config/anthropic");
        Path creds = dir.resolve("credentials").resolve("default.json");
        {
            Scripted server = new Scripted(fixture);
            server.onRequest = url -> {
                if (url.endsWith("/v1/oauth/token")) {
                    try {
                        Files.writeString(creds, "{\"access_token\": \"sk-ant-oat01-other-process\", \"expires_at\": 4102444800}");
                    } catch (java.io.IOException e) {
                        throw new IllegalStateException(e);
                    }
                }
            };
            Provider p = Provider.builder("anthropic").transport(server).auth(Credential.session("anthropic_profile", dir.toString(), null, "default", "write_back")).build();
            ChatRequest req = requestOf(fixture);
            Response r = p.chat(req.messages(), ChatOptions.builder().model(req.model()).build());
            assertEquals(lastText(fixture), textOf(r));
            assertTrue(Files.readString(creds).contains("other-process"));
        }
    }

    @Test
    void expiredBorrowedSessionNamesTheOwner(@TempDir Path dir) throws Exception {
        JsonNode fixture = anthropic("session_claude_code_expired");
        sessionStore(fixture, dir, "/home/u/.claude");
        {
            Scripted server = new Scripted(fixture);
            Provider p = Provider.builder("anthropic").transport(server).auth(Credential.session("claude_code", dir.resolve(".credentials.json").toString(), null, null, null)).build();
            ProviderException e = assertThrows(ProviderException.class, () -> p.chat(List.of(user("hi")), ChatOptions.builder().model("m").build()));
            assertEquals("credential_expired", e.code());
            assertTrue(e.getMessage().contains("claude"));
            var status = p.authStatus();
            assertEquals("session:claude_code", status.source());
            assertNotNull(status.expiresAt());
        }
    }

    @Test
    void codexKeychainWriteBackUsesTheWriter() throws Exception {
        JsonNode fixture = openai("session_codex_keychain_write_back").deepCopy();
        String service = fixture.path("keychain").fieldNames().next().split("\\|", 2)[0];
        ObjectNode doc = (ObjectNode) Json.MAPPER.readTree(fixture.path("keychain").elements().next().asText());
        ((ObjectNode) doc.get("tokens")).put("access_token", jwt(System.currentTimeMillis() / 1000 + 60));
        ObjectNode fresh = (ObjectNode) Json.MAPPER.readTree(fixture.path("http").get(0).path("body").asText());
        fresh.put("access_token", jwt(System.currentTimeMillis() / 1000 + 86400));
        ((ObjectNode) fixture.path("http").get(0)).put("body", fresh.toString());
        List<String[]> written = new ArrayList<>();
        Keychain.Reader reader = (s, a) -> s.equals(service) ? doc.toString().getBytes(StandardCharsets.UTF_8) : null;
        Keychain.Writer writer = (s, a, v) -> { written.add(new String[] {s, a, new String(v, StandardCharsets.UTF_8)}); return true; };
        {
            Scripted server = new Scripted(fixture);
            Provider p = Provider.builder("openai").transport(server).auth(Credential.session("codex", null, null, null, "write_back")).keychain(reader).keychainWrite(writer).build();
            ChatRequest req = requestOf(fixture);
            Response r = p.chat(req.messages(), ChatOptions.builder().model(req.model()).build());
            assertEquals(lastText(fixture), textOf(r));
            assertEquals("Codex Auth", written.get(0)[0]);
            assertTrue(written.get(0)[1].startsWith("cli|"));
            assertTrue(written.get(0)[2].contains("rt_d"));
            assertEquals("acct_1", server.seen.get(1).headers().get("chatgpt-account-id"));
        }
        {
            Scripted server = new Scripted(fixture);
            Provider readerOnly = Provider.builder("openai").transport(server).auth(Credential.session("codex", null, null, null, "write_back")).keychain(reader).build();
            ChatRequest req = requestOf(fixture);
            assertEquals(lastText(fixture), textOf(readerOnly.chat(req.messages(), ChatOptions.builder().model(req.model()).build())));
        }
    }

    @Test
    void defaultKeychainReaderNeverThrows() {
        assertNull(Keychain.read("ducktape-provider-test", "nobody"));
    }

    @Test
    void observerSeesHooks() throws Exception {
        JsonNode fixture = anthropic("chat_text_reasoning");
        {
            Scripted server = new Scripted(fixture);
            List<String> hooks = new ArrayList<>();
            Observer obs = new Observer() {
                @Override
                public void onRequest(JsonNode data) { hooks.add("request"); }

                @Override
                public void onResponse(JsonNode data) { hooks.add("response"); }
            };
            Provider p = provider(fixture, server).observer(obs).build();
            ChatRequest req = requestOf(fixture);
            p.chat(req.messages(), ChatOptions.builder().model(req.model()).build());
            assertTrue(hooks.contains("request"));
        }
    }
}
