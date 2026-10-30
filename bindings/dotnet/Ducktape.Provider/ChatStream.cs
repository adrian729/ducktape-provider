using System.Collections;
using System.Threading.Channels;
using Ducktape.Types;

namespace Ducktape;

/// <summary>A streamed chat reply: enumerate the events, then read the response; cancel when stopping early.</summary>
public sealed class ChatStream : IAsyncEnumerable<Event>, IEnumerable<Event>, IDisposable
{
    private readonly Channel<Event> channel = Channel.CreateUnbounded<Event>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
    private readonly CancellationTokenSource cts;
    private readonly TaskCompletionSource done = new(TaskCreationOptions.RunContinuationsAsynchronously);
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
        done.TrySetResult();
    }

    /// <summary>The final response once the stream finished; throws the stream's failure if it had one.</summary>
    public Response? Response
    {
        get
        {
            done.Task.GetAwaiter().GetResult();
            return failure is null ? response : throw failure;
        }
    }

    /// <summary>Waits for the stream to finish and returns the final response.</summary>
    public async Task<Response?> ResponseAsync()
    {
        await done.Task.ConfigureAwait(false);
        return failure is null ? response : throw failure;
    }

    public async IAsyncEnumerator<Event> GetAsyncEnumerator(CancellationToken cancellationToken = default)
    {
        using var registration = cancellationToken.Register(Cancel);
        while (await channel.Reader.WaitToReadAsync(CancellationToken.None).ConfigureAwait(false))
        {
            while (channel.Reader.TryRead(out var ev))
            {
                yield return ev;
            }
        }
        await done.Task.ConfigureAwait(false);
        if (failure is not null)
        {
            throw failure;
        }
    }

    public IEnumerator<Event> GetEnumerator()
    {
        while (channel.Reader.WaitToReadAsync(CancellationToken.None).AsTask().GetAwaiter().GetResult())
        {
            while (channel.Reader.TryRead(out var ev))
            {
                yield return ev;
            }
        }
        done.Task.GetAwaiter().GetResult();
        if (failure is not null)
        {
            throw failure;
        }
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    /// <summary>Stops the stream and releases the connection.</summary>
    public void Cancel()
    {
        cts.Cancel();
    }

    public void Dispose()
    {
        Cancel();
    }
}
