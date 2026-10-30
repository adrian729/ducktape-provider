#include "ducktape.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CLOCK "{\"now_ms\":1761696000000,\"entropy\":\"gICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIA=\"}"

static const char START[] =
    "{\"clock\":" CLOCK ","
    "\"provider\":{\"vendor\":\"anthropic\"},"
    "\"operation\":{\"type\":\"chat\",\"request\":{\"model\":\"claude-opus-5-5\","
    "\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"hi\"}]}]}}}";

static const char BODY[] =
    "event: message_start\n"
    "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_01\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-opus-5-5\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":12,\"output_tokens\":1}}}\n\n"
    "event: content_block_start\n"
    "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
    "event: content_block_delta\n"
    "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello from C\"}}\n\n"
    "event: content_block_stop\n"
    "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
    "event: message_delta\n"
    "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":4}}\n\n"
    "event: message_stop\n"
    "data: {\"type\":\"message_stop\"}\n\n";

static char *step;

static void take(uint8_t *out, size_t len)
{
    free(step);
    step = malloc(len + 1);
    if (step == NULL) {
        fprintf(stderr, "smoke: out of memory\n");
        exit(1);
    }
    memcpy(step, out, len);
    step[len] = '\0';
    ducktape_free(out, len);
}

static int field(const char *json, const char *key, char *out, size_t cap)
{
    char needle[128];
    snprintf(needle, sizeof needle, "\"%s\":\"", key);
    const char *at = strstr(json, needle);
    if (at == NULL) {
        return 0;
    }
    at += strlen(needle);
    const char *end = strchr(at, '"');
    if (end == NULL || (size_t)(end - at) >= cap) {
        return 0;
    }
    memcpy(out, at, (size_t)(end - at));
    out[end - at] = '\0';
    return 1;
}

static int effect_type(const char *json, char *out, size_t cap)
{
    const char *at = strstr(json, "\"effect\":{\"type\":\"");
    return at != NULL && field(at + 10, "type", out, cap);
}

static void resume(ducktape_Operation *op, const char *input)
{
    char doc[1024];
    uint8_t *out = NULL;
    size_t len = 0;
    snprintf(doc, sizeof doc, "{\"clock\":" CLOCK ",\"input\":%s}", input);
    ducktape_op_resume(op, (const uint8_t *)doc, strlen(doc), &out, &len);
    take(out, len);
}

static int fail(const char *what)
{
    fprintf(stderr, "smoke: %s\nlast step: %s\n", what, step == NULL ? "(none)" : step);
    return 1;
}

int main(void)
{
    uint8_t *out = NULL;
    size_t len = 0;
    if (ducktape_abi_version() != DUCKTAPE_ABI_VERSION) {
        return fail("the library speaks another ABI version than the header");
    }
    uint8_t *scratch = ducktape_alloc(16);
    if (scratch == NULL) {
        return fail("alloc returned null");
    }
    ducktape_free(scratch, 16);

    const char *bad = "{\"nope\":1}";
    ducktape_Operation *op = ducktape_op_start((const uint8_t *)bad, strlen(bad), &out, &len);
    take(out, len);
    if (op != NULL || strstr(step, "\"type\":\"failed\"") == NULL) {
        return fail("an invalid start document should give a null handle and a failed step");
    }

    op = ducktape_op_start((const uint8_t *)START, sizeof START - 1, &out, &len);
    take(out, len);
    if (op == NULL) {
        return fail("start returned a null handle");
    }
    int body_sent = 0;
    for (int i = 0; i < 100; i++) {
        char type[64];
        if (!effect_type(step, type, sizeof type)) {
            return fail("no effect type in the step");
        }
        if (strcmp(type, "done") == 0) {
            if (strstr(step, "\"text\":\"Hello from C\"") == NULL || strstr(step, "\"reason\":\"complete\"") == NULL) {
                return fail("the result does not carry the reply");
            }
            ducktape_op_free(op);
            free(step);
            puts("smoke: ok");
            return 0;
        }
        if (strcmp(type, "failed") == 0) {
            return fail("the core failed");
        }
        if (strcmp(type, "need_secret") == 0) {
            char secret_ref[64];
            char input[256];
            if (!field(step, "secret_ref", secret_ref, sizeof secret_ref)) {
                return fail("need_secret without a secret_ref");
            }
            ducktape_op_set_secret(op, (const uint8_t *)secret_ref, strlen(secret_ref), (const uint8_t *)"sk-ant-test", 11);
            snprintf(input, sizeof input, "{\"type\":\"secret\",\"secret_ref\":\"%s\",\"found\":true}", secret_ref);
            resume(op, input);
        } else if (strcmp(type, "need_http") == 0) {
            if (strstr(step, "\"url\":\"https://api.anthropic.com/v1/messages\"") == NULL) {
                return fail("unexpected request URL");
            }
            if (strstr(step, "\"x-api-key\":\"sk-ant-test\"") == NULL) {
                return fail("the secret was not placed in the request");
            }
            resume(op, "{\"type\":\"http_response\",\"status\":200,\"headers\":{\"content-type\":\"text/event-stream\"}}");
        } else if (strcmp(type, "receive") == 0) {
            if (body_sent) {
                resume(op, "{\"type\":\"http_end\"}");
            } else {
                body_sent = 1;
                ducktape_op_feed(op, (const uint8_t *)BODY, sizeof BODY - 1, (const uint8_t *)CLOCK, sizeof CLOCK - 1, &out, &len);
                take(out, len);
            }
        } else {
            return fail("unexpected effect");
        }
    }
    return fail("the operation did not finish");
}
