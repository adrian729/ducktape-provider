package io.github.adrian729.ducktape;

import io.github.adrian729.ducktape.types.Event;
import io.github.adrian729.ducktape.types.ProviderError;
import io.github.adrian729.ducktape.types.Response;
import java.util.Iterator;
import java.util.NoSuchElementException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;

/** A streamed chat reply: iterate the events, then read the response; cancel when stopping early. */
public final class Stream implements Iterator<Event>, Iterable<Event>, AutoCloseable {
    private static final Object END = new Object();
    private final LinkedBlockingQueue<Object> queue = new LinkedBlockingQueue<>();
    private final CountDownLatch done = new CountDownLatch(1);
    private final Driver.Cancel cancel = new Driver.Cancel();
    private volatile Response response;
    private volatile RuntimeException failure;
    private Object next;

    Stream() {}

    Driver.Cancel cancel() {
        return cancel;
    }

    void push(Event event) {
        if ("finish".equals(event.type()) && event.response() != null) {
            response = event.response();
        }
        queue.add(event);
    }

    void finish(RuntimeException error) {
        failure = error;
        queue.add(END);
        done.countDown();
    }

    @Override
    public Iterator<Event> iterator() {
        return this;
    }

    @Override
    public boolean hasNext() {
        if (next == null) {
            try {
                next = queue.take();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                close();
                throw new ProviderException(ProviderError.builder().kind("cancelled").message("the stream was interrupted").retryable(false).attempts(0L).build(), e);
            }
        }
        if (next == END) {
            await();
            if (failure != null) {
                RuntimeException f = failure;
                failure = null;
                throw f;
            }
            return false;
        }
        return true;
    }

    @Override
    public Event next() {
        if (!hasNext()) {
            throw new NoSuchElementException();
        }
        Event ev = (Event) next;
        next = null;
        return ev;
    }

    /** The final response once the stream finished, or null before that. */
    public Response response() {
        await();
        return response;
    }

    private void await() {
        try {
            done.await();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    /** Stops the stream and releases the connection. */
    @Override
    public void close() {
        cancel.cancel();
    }
}
