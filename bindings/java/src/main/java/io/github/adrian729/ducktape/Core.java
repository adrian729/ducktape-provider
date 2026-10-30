package io.github.adrian729.ducktape;

import com.dylibso.chicory.runtime.ExportFunction;
import com.dylibso.chicory.runtime.Instance;
import com.dylibso.chicory.runtime.Memory;
import com.dylibso.chicory.wasm.WasmModule;
import com.fasterxml.jackson.databind.JsonNode;
import io.github.adrian729.ducktape.wasm.CoreModule;
import java.nio.charset.StandardCharsets;

/** The compiled core; one instance is created per operation. */
public final class Core {
    private final WasmModule module;

    private Core(WasmModule module) {
        this.module = module;
    }

    /** The core compiled into this jar. */
    public static Core bundled() {
        return new Core(CoreModule.load());
    }

    Operation operation() {
        Instance instance = Instance.builder(module).withMachineFactory(CoreModule::create).build();
        return new Operation(instance);
    }

    /** One wasm instance driving one operation. */
    static final class Operation implements AutoCloseable {
        private final Instance instance;
        private final Memory memory;
        private final ExportFunction alloc;
        private final ExportFunction free;
        private final ExportFunction opStart;
        private final ExportFunction opResume;
        private final ExportFunction opFeed;
        private final ExportFunction opSecret;
        private final ExportFunction opFree;
        private final int scratch;
        private long handle;

        Operation(Instance instance) {
            this.instance = instance;
            this.memory = instance.memory();
            long version = instance.export("ducktape_abi_version").apply()[0];
            if (version != 1) {
                throw new IllegalStateException("ducktape: the core has an unknown ABI version " + version);
            }
            alloc = instance.export("ducktape_alloc");
            free = instance.export("ducktape_free");
            opStart = instance.export("ducktape_op_start");
            opResume = instance.export("ducktape_op_resume");
            opFeed = instance.export("ducktape_op_feed");
            opSecret = instance.export("ducktape_op_set_secret");
            opFree = instance.export("ducktape_op_free");
            scratch = (int) alloc.apply(8)[0];
        }

        private int put(byte[] bytes) {
            if (bytes.length == 0) {
                return 0;
            }
            int ptr = (int) alloc.apply(bytes.length)[0];
            memory.write(ptr, bytes);
            return ptr;
        }

        private void release(int ptr, int len) {
            if (ptr != 0 && len != 0) {
                free.apply(ptr, len);
            }
        }

        private JsonNode take() {
            int ptr = memory.readInt(scratch);
            int len = memory.readInt(scratch + 4);
            byte[] raw = memory.readBytes(ptr, len);
            release(ptr, len);
            return Json.parse(raw);
        }

        JsonNode start(Object doc) {
            byte[] bytes = Json.bytes(doc);
            int ptr = put(bytes);
            try {
                handle = opStart.apply(ptr, bytes.length, scratch, scratch + 4)[0];
            } finally {
                release(ptr, bytes.length);
            }
            return take();
        }

        JsonNode resume(Object doc) {
            byte[] bytes = Json.bytes(doc);
            int ptr = put(bytes);
            try {
                opResume.apply(handle, ptr, bytes.length, scratch, scratch + 4);
            } finally {
                release(ptr, bytes.length);
            }
            return take();
        }

        JsonNode feed(byte[] body, int length, Object clock) {
            byte[] clockBytes = Json.bytes(clock);
            int clockPtr = put(clockBytes);
            byte[] chunk = length == body.length ? body : java.util.Arrays.copyOf(body, length);
            int bodyPtr = put(chunk);
            try {
                opFeed.apply(handle, bodyPtr, chunk.length, clockPtr, clockBytes.length, scratch, scratch + 4);
            } finally {
                release(bodyPtr, chunk.length);
                release(clockPtr, clockBytes.length);
            }
            return take();
        }

        void setSecret(String ref, byte[] secret) {
            byte[] refBytes = ref.getBytes(StandardCharsets.UTF_8);
            int refPtr = put(refBytes);
            int secretPtr = put(secret);
            try {
                opSecret.apply(handle, refPtr, refBytes.length, secretPtr, secret.length);
            } finally {
                release(secretPtr, secret.length);
                release(refPtr, refBytes.length);
            }
        }

        @Override
        public void close() {
            if (handle != 0) {
                opFree.apply(handle);
                handle = 0;
            }
            release(scratch, 8);
        }
    }
}
