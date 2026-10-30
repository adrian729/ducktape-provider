package io.github.adrian729.ducktape;

import com.fasterxml.jackson.databind.JsonNode;

/** Receives request, response, event and retry notifications as raw JSON. */
public interface Observer {
    default void onRequest(JsonNode data) {}

    default void onResponse(JsonNode data) {}

    default void onEvent(JsonNode data) {}

    default void onRetry(JsonNode data) {}
}
