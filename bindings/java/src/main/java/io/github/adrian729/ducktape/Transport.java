package io.github.adrian729.ducktape;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Flow;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/** Sends one HTTP request and hands back the status, headers and a body read chunk by chunk. */
@FunctionalInterface
public interface Transport {
    /** One request as the core built it. */
    record Request(String method, String url, Map<String, String> headers, String body) {}

    /** A reply whose body is pulled with a per-chunk timeout. */
    record Reply(int status, Map<String, String> headers, Body body) {}

    /** The body of a reply; next returns null at the end. */
    interface Body extends AutoCloseable {
        ByteBuffer next(Duration timeout) throws IOException;

        @Override
        void close();
    }

    /** Sends the request; headersTimeout bounds the wait for the status line, null means no bound. */
    Reply send(Request request, Duration headersTimeout) throws IOException, InterruptedException;

    /** A transport over a java.net.http client. */
    static Transport of(HttpClient client) {
        return (request, headersTimeout) -> {
            HttpRequest.Builder b = HttpRequest.newBuilder(URI.create(request.url()));
            b.method(request.method(), request.body() == null ? HttpRequest.BodyPublishers.noBody() : HttpRequest.BodyPublishers.ofString(request.body()));
            request.headers().forEach((k, v) -> {
                try {
                    b.header(k, v);
                } catch (IllegalArgumentException ignored) {
                }
            });
            if (headersTimeout != null) {
                b.timeout(headersTimeout);
            }
            HttpResponse<Flow.Publisher<List<ByteBuffer>>> resp = client.send(b.build(), HttpResponse.BodyHandlers.ofPublisher());
            QueueBody body = new QueueBody();
            resp.body().subscribe(body);
            Map<String, String> headers = new java.util.HashMap<>();
            resp.headers().map().forEach((k, v) -> headers.put(k.toLowerCase(java.util.Locale.ROOT), String.join(", ", v)));
            return new Reply(resp.statusCode(), headers, body);
        };
    }

    /** A body fed by a reactive publisher and drained with timeouts. */
    final class QueueBody implements Body, Flow.Subscriber<List<ByteBuffer>> {
        private static final Object END = new Object();
        private final LinkedBlockingQueue<Object> queue = new LinkedBlockingQueue<>();
        private volatile Flow.Subscription subscription;
        private volatile boolean closed;

        @Override
        public void onSubscribe(Flow.Subscription s) {
            subscription = s;
            s.request(1);
        }

        @Override
        public void onNext(List<ByteBuffer> item) {
            for (ByteBuffer b : item) {
                queue.add(b);
            }
            if (item.isEmpty()) {
                subscription.request(1);
            }
        }

        @Override
        public void onError(Throwable t) {
            queue.add(t);
        }

        @Override
        public void onComplete() {
            queue.add(END);
        }

        @Override
        public ByteBuffer next(Duration timeout) throws IOException {
            if (closed) {
                throw new IOException("the body was closed");
            }
            Object item;
            try {
                item = timeout == null ? queue.take() : queue.poll(timeout.toMillis(), TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("interrupted", e);
            }
            if (item == null) {
                throw new java.net.http.HttpTimeoutException("read timed out");
            }
            if (item == END) {
                return null;
            }
            if (item instanceof Throwable t) {
                throw t instanceof IOException io ? io : new IOException(t);
            }
            ByteBuffer b = (ByteBuffer) item;
            if (queue.isEmpty()) {
                Flow.Subscription s = subscription;
                if (s != null) {
                    s.request(1);
                }
            }
            return b;
        }

        @Override
        public void close() {
            closed = true;
            Flow.Subscription s = subscription;
            if (s != null) {
                s.cancel();
            }
            queue.add(END);
        }
    }
}
