using System.Runtime.ExceptionServices;
using System.Threading.Channels;
using Ducktape.Types;

namespace Ducktape;

/// <summary>A streamed chat reply: enumerate the events, then read the response; cancel when stopping early.</summary>
public sealed class ChatStream : IAsyncEnumerable<Event>, IDisposable, IAsyncDisposable
{
    private readonly Channel<Event> channel = Channel.CreateUnbounded<Event>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
    private readonly CancellationTokenSource cts;
    private readonly TaskCompletionSource done = new(TaskCreationOptions.RunContinuationsAsynchronously);
    private readonly object gate = new();
    private bool finished;
    private Response? response;
    private Exception? failure;

    internal ChatStream(CancellationToken cancellation)
    {
        cts = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
    }

    internal CancellationToken Token => cts.Token;

    internal Task Push(Event ev)
    {
        if (ev.Type == "finish" && ev.Response is not null)
        {
            response = ev.Response;
        }
        channel.Writer.TryWrite(ev);
        return Task.CompletedTask;
    }

    internal void Finish(Exception? error)
    {
        failure = error;
        channel.Writer.TryComplete();
        lock (gate)
        {
            finished = true;
            cts.Dispose();
        }
        done.TrySetResult();
    }

    private Response? Result()
    {
        if (failure is not null)
        {
            ExceptionDispatchInfo.Throw(failure);
        }
        return response;
    }

    /// <summary>Waits for the stream to finish and returns the final response, blocking.</summary>
    public Response? GetResponse()
    {
        done.Task.GetAwaiter().GetResult();
        return Result();
    }

    /// <summary>Waits for the stream to finish and returns the final response.</summary>
    public async Task<Response?> GetResponseAsync()
    {
        await done.Task.ConfigureAwait(false);
        return Result();
    }

    public async IAsyncEnumerator<Event> GetAsyncEnumerator(CancellationToken cancellationToken = default)
    {
        using var registration = cancellationToken.Register(Cancel);
        try
        {
            while (await channel.Reader.WaitToReadAsync(CancellationToken.None).ConfigureAwait(false))
            {
                while (channel.Reader.TryRead(out var ev))
                {
                    yield return ev;
                }
            }
            await done.Task.ConfigureAwait(false);
            Result();
        }
        finally
        {
            Cancel();
        }
    }

    /// <summary>The events as a blocking sequence.</summary>
    public IEnumerable<Event> ToBlockingEnumerable()
    {
        try
        {
            while (channel.Reader.WaitToReadAsync(CancellationToken.None).AsTask().GetAwaiter().GetResult())
            {
                while (channel.Reader.TryRead(out var ev))
                {
                    yield return ev;
                }
            }
            done.Task.GetAwaiter().GetResult();
            Result();
        }
        finally
        {
            Cancel();
        }
    }

    /// <summary>Stops the stream and releases the connection.</summary>
    public void Cancel()
    {
        lock (gate)
        {
            if (!finished)
            {
                cts.Cancel();
            }
        }
    }

    public void Dispose()
    {
        Cancel();
    }

    public ValueTask DisposeAsync()
    {
        Cancel();
        return ValueTask.CompletedTask;
    }
}
