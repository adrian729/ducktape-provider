using System.Net;
using System.Net.Sockets;
using System.Security.Authentication;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Ducktape.Tests;

/// <summary>An HTTP handler that answers with a fixture's scripted replies in order.</summary>
internal sealed class Scripted : HttpMessageHandler
{
    public sealed record Seen(string Url, Dictionary<string, string> Headers, JsonNode? Body);

    private readonly List<JsonNode> script;
    public readonly List<Seen> Requests = [];
    public int DelayMs;
    public Action<string>? OnRequest;

    public Scripted(JsonNode fixture)
    {
        script = (fixture["http"] as JsonArray)?.Select(h => h!).ToList() ?? [];
    }

    public HttpClient Client()
    {
        return new HttpClient(this) { Timeout = System.Threading.Timeout.InfiniteTimeSpan };
    }

    protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
        JsonNode? parsed = null;
        if (request.Content is not null)
        {
            var text = await request.Content.ReadAsStringAsync(cancellationToken);
            try
            {
                parsed = JsonNode.Parse(text);
            }
            catch (JsonException)
            {
            }
        }
        var headers = new Dictionary<string, string>();
        foreach (var h in request.Headers.Concat(request.Content?.Headers ?? Enumerable.Empty<KeyValuePair<string, IEnumerable<string>>>()))
        {
            headers[h.Key.ToLowerInvariant()] = string.Join(", ", h.Value);
        }
        var url = request.RequestUri!.ToString();
        lock (Requests)
        {
            Requests.Add(new Seen(url, headers, parsed));
        }
        OnRequest?.Invoke(url);
        JsonNode h0;
        lock (script)
        {
            if (script.Count == 0)
            {
                throw new HttpRequestException("script exhausted", new SocketException((int)SocketError.ConnectionRefused));
            }
            h0 = script[0];
            script.RemoveAt(0);
        }
        if (h0["error"] is { } error)
        {
            var message = error["message"]?.GetValue<string>() ?? "scripted failure";
            throw error["kind"]?.GetValue<string>() switch
            {
                "dns" => new HttpRequestException(message, new SocketException((int)SocketError.HostNotFound)),
                "tls" => new HttpRequestException(message, new AuthenticationException(message)),
                "timeout" => new TimeoutException(message),
                "reset" => new IOException(message),
                _ => new HttpRequestException(message, new SocketException((int)SocketError.ConnectionRefused)),
            };
        }
        var chunks = new List<byte[]>();
        if (h0["chunks"] is JsonArray { Count: > 0 } array)
        {
            chunks.AddRange(array.Select(c => Encoding.UTF8.GetBytes(c!.GetValue<string>())));
        }
        else if (h0["body"] is { } body)
        {
            chunks.Add(Encoding.UTF8.GetBytes(body.GetValue<string>()));
        }
        var response = new HttpResponseMessage((HttpStatusCode)(h0["status"]?.GetValue<int>() ?? 200)) { Content = new StreamContent(new ChunkStream(chunks, DelayMs)) };
        foreach (var h in h0["headers"] as JsonObject ?? [])
        {
            var value = h.Value?.GetValue<string>() ?? "";
            if (!response.Headers.TryAddWithoutValidation(h.Key, value))
            {
                response.Content.Headers.TryAddWithoutValidation(h.Key, value);
            }
        }
        return response;
    }

    private sealed class ChunkStream : Stream
    {
        private readonly List<byte[]> chunks;
        private readonly int delay;
        private int index;
        private int offset;

        public ChunkStream(List<byte[]> chunks, int delay)
        {
            this.chunks = chunks;
            this.delay = delay;
        }

        public override async ValueTask<int> ReadAsync(Memory<byte> buffer, CancellationToken cancellationToken = default)
        {
            while (index < chunks.Count && chunks[index].Length == 0)
            {
                index++;
            }
            if (index >= chunks.Count)
            {
                return 0;
            }
            if (delay > 0 && offset == 0)
            {
                await Task.Delay(delay, cancellationToken);
            }
            var chunk = chunks[index];
            var n = Math.Min(buffer.Length, chunk.Length - offset);
            chunk.AsMemory(offset, n).CopyTo(buffer);
            offset += n;
            if (offset >= chunk.Length)
            {
                index++;
                offset = 0;
            }
            return n;
        }

        public override Task<int> ReadAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken)
        {
            return ReadAsync(buffer.AsMemory(offset, count), cancellationToken).AsTask();
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            return ReadAsync(buffer, offset, count, CancellationToken.None).GetAwaiter().GetResult();
        }

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
