package io.github.adrian729.ducktape;

import io.github.adrian729.ducktape.types.Block;
import io.github.adrian729.ducktape.types.Message;
import io.github.adrian729.ducktape.types.Response;
import io.github.adrian729.ducktape.types.Role;
import io.github.adrian729.ducktape.types.Tool;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;

/** Static builders for messages, blocks, tools and response views. */
public final class Ducktape {
    private Ducktape() {}

    /** A text block. */
    public static Block text(String text) {
        return Block.builder().type("text").text(text).build();
    }

    /** An image block from bytes. */
    public static Block image(byte[] data, String mediaType) {
        return Block.builder().type("image").data(Base64.getEncoder().encodeToString(data)).mediaType(mediaType).build();
    }

    /** An image block the vendor fetches. */
    public static Block imageUrl(String url) {
        return Block.builder().type("image").url(url).build();
    }

    /** A document block from bytes. */
    public static Block document(byte[] data, String mediaType, String name) {
        return Block.builder().type("document").data(Base64.getEncoder().encodeToString(data)).mediaType(mediaType).name(name).build();
    }

    /** A tool result block for a call. */
    public static Block toolResult(String toolCallId, Block... content) {
        return Block.builder().type("tool_result").toolCallId(toolCallId).content(List.of(content)).build();
    }

    /** A tool result block holding one text block. */
    public static Block toolResult(String toolCallId, String text) {
        return toolResult(toolCallId, text(text));
    }

    /** A user message. */
    public static Message user(Block... content) {
        return new Message(Role.USER, List.of(content));
    }

    /** A user message with one text block. */
    public static Message user(String text) {
        return user(text(text));
    }

    /** A system message. */
    public static Message system(Block... content) {
        return new Message(Role.SYSTEM, List.of(content));
    }

    /** A system message with one text block. */
    public static Message system(String text) {
        return system(text(text));
    }

    /** An assistant message. */
    public static Message assistant(Block... content) {
        return new Message(Role.ASSISTANT, List.of(content));
    }

    /** The assistant message that appends a response to the history. */
    public static Message assistant(Response response) {
        return new Message(Role.ASSISTANT, response.content());
    }

    /** A tool message from result blocks. */
    public static Message tool(Block... results) {
        return new Message(Role.TOOL, List.of(results));
    }

    /** A function tool definition. */
    public static Tool functionTool(String name, String description, Map<String, Object> inputSchema) {
        return Tool.builder().type("function").name(name).description(description).inputSchema(inputSchema).build();
    }

    /** The text blocks of a response joined. */
    public static String textOf(Response response) {
        StringBuilder out = new StringBuilder();
        for (Block b : response.content()) {
            if ("text".equals(b.type()) && b.text() != null) {
                out.append(b.text());
            }
        }
        return out.toString();
    }

    /** The reasoning blocks of a response joined. */
    public static String reasoningOf(Response response) {
        StringBuilder out = new StringBuilder();
        for (Block b : response.content()) {
            if ("reasoning".equals(b.type()) && b.text() != null) {
                out.append(b.text());
            }
        }
        return out.toString();
    }

    /** The tool call blocks of a response in order. */
    public static List<Block> toolCallsOf(Response response) {
        List<Block> out = new ArrayList<>();
        for (Block b : response.content()) {
            if ("tool_call".equals(b.type())) {
                out.add(b);
            }
        }
        return out;
    }
}
