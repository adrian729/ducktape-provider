using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Security.Authentication;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Ducktape.Types;

namespace Ducktape;

/// <summary>The effect loop: answers what the core asks for with the host's HTTP, files, environment and keychain.</summary>
internal sealed class Driver
{
    internal sealed record Host(Dictionary<string, string> Secrets, Dictionary<string, Func<CancellationToken, Task<string>>> Resolvers, KeychainReader? Keychain, KeychainWriter? KeychainWrite, HttpClient Client, Observer? Observer);

    private readonly Core.Operation op;
    private readonly Host host;
    private readonly Func<Event, Task>? onEvent;
    private readonly Func<LoginAction, CancellationToken, Task<string?>>? handleAction;
    private readonly CancellationToken ct;
    private HttpResponseMessage? response;
    private Stream? body;
    private CancellationTokenSource? requestCts;
    private TimeSpan? readTimeout;
    private readonly byte[] buffer = new byte[64 * 1024];

    public Driver(Core.Operation op, Host host, Func<Event, Task>? onEvent, Func<LoginAction, CancellationToken, Task<string?>>? handleAction, CancellationToken ct)
    {
        this.op = op;
        this.host = host;
        this.onEvent = onEvent;
        this.handleAction = handleAction;
        this.ct = ct;
    }

    public static JsonObject Clock()
    {
        return new JsonObject { ["now_ms"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(), ["entropy"] = Convert.ToBase64String(RandomNumberGenerator.GetBytes(32)) };
    }

    public static string Etag(byte[] bytes)
    {
        return Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    }

    private static JsonObject Input(string type)
    {
        return new JsonObject { ["type"] = type };
    }

    private void CloseBody()
    {
        body?.Dispose();
        body = null;
        response?.Dispose();
        response = null;
        requestCts?.Dispose();
        requestCts = null;
    }

    public async Task<JsonNode?> RunAsync(JsonNode start, JsonNode step)
    {
        try
        {
            for (var i = 0; i < 1_000_000; i++)
            {
                foreach (var note in step["notify"] as JsonArray ?? [])
                {
                    var noteType = Json.Str(note?["type"]);
                    if (noteType == "event" && onEvent is not null)
                    {
                        await onEvent(Json.As<Event>(note!["event"])).ConfigureAwait(false);
                    }
                    else if (noteType == "observer" && host.Observer is not null)
                    {
                        Observe(Json.Str(note!["hook"]), note["data"]);
                    }
                }
                var effect = step["effect"];
                var type = Json.Str(effect?["type"]);
                if (type == "done")
                {
                    return effect!["result"];
                }
                if (type == "failed")
                {
                    throw new ProviderException(Json.As<ProviderError>(effect!["error"]));
                }
                JsonObject input;
                if (ct.IsCancellationRequested && type != "receive")
                {
                    input = Input("cancelled");
                }
                else
                {
                    switch (type)
                    {
                        case "need_secret":
                            input = await SecretAsync(effect!).ConfigureAwait(false);
                            break;
                        case "need_http":
                            CloseBody();
                            input = await HttpAsync(effect!["request"]!).ConfigureAwait(false);
                            break;
                        case "receive":
                            var (fed, pending) = await ReceiveAsync().ConfigureAwait(false);
                            if (fed is not null)
                            {
                                step = fed;
                                continue;
                            }
                            input = pending!;
                            break;
                        case "need_sleep":
                            CloseBody();
                            input = await SleepAsync(effect!["ms"]?.GetValue<long>() ?? 0).ConfigureAwait(false);
                            break;
                        case "need_read":
                            input = await ReadAsync(effect!["locator"], ct).ConfigureAwait(false);
                            break;
                        case "need_write":
                            input = await WriteAsync(effect!, ct).ConfigureAwait(false);
                            break;
                        case "need_user_action":
                            input = await UserActionAsync(effect!["action"]).ConfigureAwait(false);
                            break;
                        case "need_callback":
                            input = await CallbackAsync(effect!.AsObject()).ConfigureAwait(false);
                            break;
                        default:
                            throw ProviderException.Of("unsupported", null, "the core asked for an unknown effect " + type);
                    }
                }
                step = op.Resume(new JsonObject { ["clock"] = Clock(), ["input"] = input });
            }
            throw ProviderException.Of("other", "internal", "the operation did not finish");
        }
        finally
        {
            CloseBody();
        }
    }

    private void Observe(string? hook, JsonNode? data)
    {
        var o = host.Observer!;
        var element = Json.ToElement(data);
        switch (hook)
        {
            case "on_request":
                o.OnRequest?.Invoke(element);
                break;
            case "on_response":
                o.OnResponse?.Invoke(element);
                break;
            case "on_event":
                o.OnEvent?.Invoke(element);
                break;
            case "on_retry":
                o.OnRetry?.Invoke(element);
                break;
        }
    }

    private async Task<JsonObject> SecretAsync(JsonNode effect)
    {
        var secretRef = Json.Str(effect["secret_ref"]) ?? "";
        var source = effect["source"];
        string? value = null;
        switch (Json.Str(source?["type"]))
        {
            case "inline":
                host.Secrets.TryGetValue(secretRef, out value);
                break;
            case "env":
                foreach (var name in source!["names"] as JsonArray ?? [])
                {
                    var v = Environment.GetEnvironmentVariable(Json.Str(name) ?? "");
                    if (!string.IsNullOrEmpty(v))
                    {
                        value = v;
                        break;
                    }
                }
                break;
            case "resolver":
                if (host.Resolvers.TryGetValue(Json.Str(source!["resolver_id"]) ?? "", out var fetch))
                {
                    try
                    {
                        value = await fetch(ct).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) when (ct.IsCancellationRequested)
                    {
                        return Input("cancelled");
                    }
                    catch (Exception e)
                    {
                        throw new ProviderException(ProviderException.Build("auth", "resolver_failed", "the credential resolver failed: " + e.Message), e);
                    }
                }
                break;
        }
        var found = !string.IsNullOrEmpty(value);
        if (found)
        {
            op.SetSecret(secretRef, Encoding.UTF8.GetBytes(value!));
        }
        return new JsonObject { ["type"] = "secret", ["secret_ref"] = secretRef, ["found"] = found };
    }

    private static TimeSpan? Seconds(JsonNode? node)
    {
        if (node is not JsonValue v || !v.TryGetValue<double>(out var s) || !double.IsFinite(s) || s <= 0)
        {
            return null;
        }
        return TimeSpan.FromMilliseconds(Math.Min(s * 1000, uint.MaxValue - 1));
    }

    private static JsonObject TransportError(string kind, string? message)
    {
        return new JsonObject { ["type"] = "http_error", ["error"] = new JsonObject { ["kind"] = kind, ["message"] = message ?? kind } };
    }

    private async Task<JsonObject> HttpAsync(JsonNode req)
    {
        var timeout = req["timeout"];
        var connect = Seconds(timeout?["connect"]);
        var total = Seconds(timeout?["total"]);
        readTimeout = Seconds(timeout?["read"]);
        requestCts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        if (total is { } t)
        {
            requestCts.CancelAfter(t);
        }
        using var headersCts = CancellationTokenSource.CreateLinkedTokenSource(requestCts.Token);
        if (connect is { } c)
        {
            headersCts.CancelAfter(c);
        }
        HttpRequestMessage message;
        try
        {
            message = new HttpRequestMessage(new HttpMethod(Json.Str(req["method"]) ?? "GET"), Json.Str(req["url"]));
        }
        catch (Exception e)
        {
            return TransportError("other", e.Message);
        }
        var bodyText = Json.Str(req["body"]);
        if (bodyText is not null)
        {
            message.Content = new ByteArrayContent(Encoding.UTF8.GetBytes(bodyText));
        }
        foreach (var header in req["headers"] as JsonObject ?? [])
        {
            var value = Json.Str(header.Value) ?? "";
            if (message.Headers.TryAddWithoutValidation(header.Key, value))
            {
                continue;
            }
            if (message.Content is not null && !message.Content.Headers.TryAddWithoutValidation(header.Key, value))
            {
                message.Dispose();
                return TransportError("other", "the header " + header.Key + " was rejected");
            }
        }
        HttpResponseMessage? reply = null;
        try
        {
            reply = await host.Client.SendAsync(message, HttpCompletionOption.ResponseHeadersRead, headersCts.Token).ConfigureAwait(false);
            body = await reply.Content.ReadAsStreamAsync(requestCts.Token).ConfigureAwait(false);
        }
        catch (Exception) when (ct.IsCancellationRequested)
        {
            reply?.Dispose();
            return Input("cancelled");
        }
        catch (OperationCanceledException)
        {
            reply?.Dispose();
            return TransportError("timeout", "no response headers within the connect timeout");
        }
        catch (Exception e)
        {
            reply?.Dispose();
            return TransportError(Classify(e), e.Message);
        }
        response = reply;
        var headers = new JsonObject();
        foreach (var header in reply.Headers.Concat(reply.Content.Headers))
        {
            headers[header.Key.ToLowerInvariant()] = string.Join(", ", header.Value);
        }
        return new JsonObject { ["type"] = "http_response", ["status"] = (int)reply.StatusCode, ["headers"] = headers };
    }

    private async Task<(JsonNode? Fed, JsonObject? Pending)> ReceiveAsync()
    {
        while (true)
        {
            if (body is null || requestCts is null)
            {
                return (null, Input("http_end"));
            }
            using var readCts = CancellationTokenSource.CreateLinkedTokenSource(requestCts.Token);
            if (readTimeout is { } r)
            {
                readCts.CancelAfter(r);
            }
            int n;
            try
            {
                n = await body.ReadAsync(buffer, readCts.Token).ConfigureAwait(false);
            }
            catch (Exception) when (ct.IsCancellationRequested)
            {
                CloseBody();
                return (null, Input("cancelled"));
            }
            catch (OperationCanceledException)
            {
                CloseBody();
                return (null, TransportError("timeout", "read timed out"));
            }
            catch (Exception e)
            {
                CloseBody();
                return (null, TransportError(Classify(e), e.Message));
            }
            if (ct.IsCancellationRequested)
            {
                CloseBody();
                return (null, Input("cancelled"));
            }
            if (n == 0)
            {
                CloseBody();
                return (null, Input("http_end"));
            }
            return (op.Feed(buffer.AsSpan(0, n), Clock()), null);
        }
    }

    internal static string Classify(Exception e)
    {
        for (Exception? t = e; t is not null; t = t.InnerException)
        {
            if (t is TimeoutException or OperationCanceledException or SocketException { SocketErrorCode: SocketError.TimedOut })
            {
                return "timeout";
            }
        }
        for (Exception? t = e; t is not null; t = t.InnerException)
        {
            switch (t)
            {
                case HttpRequestException { HttpRequestError: HttpRequestError.NameResolutionError }:
                    return "dns";
                case HttpRequestException { HttpRequestError: HttpRequestError.SecureConnectionError }:
                case AuthenticationException:
                    return "tls";
                case HttpRequestException { HttpRequestError: HttpRequestError.ConnectionError }:
                case SocketException { SocketErrorCode: SocketError.ConnectionRefused or SocketError.HostUnreachable or SocketError.NetworkUnreachable }:
                    return "connect";
                case SocketException { SocketErrorCode: SocketError.HostNotFound or SocketError.NoData or SocketError.TryAgain }:
                    return "dns";
                case SocketException { SocketErrorCode: SocketError.ConnectionReset or SocketError.ConnectionAborted or SocketError.Shutdown }:
                case HttpRequestException { HttpRequestError: HttpRequestError.ResponseEnded or HttpRequestError.InvalidResponse }:
                case EndOfStreamException:
                case IOException:
                    return "reset";
            }
        }
        return "other";
    }

    private async Task<JsonObject> SleepAsync(long ms)
    {
        try
        {
            await Task.Delay(TimeSpan.FromMilliseconds(ms), ct).ConfigureAwait(false);
            return Input("slept");
        }
        catch (OperationCanceledException)
        {
            return Input("cancelled");
        }
    }

    private async Task<JsonObject> ReadAsync(JsonNode? locator, CancellationToken ct)
    {
        var missing = new JsonObject { ["type"] = "read", ["found"] = false };
        if (locator is null)
        {
            return missing;
        }
        byte[] data;
        string? tag = null;
        switch (Json.Str(locator["type"]))
        {
            case "file":
                try
                {
                    data = File.ReadAllBytes(Json.Str(locator["path"]) ?? "");
                }
                catch (Exception)
                {
                    return missing;
                }
                tag = Etag(data);
                break;
            case "env":
                var v = Environment.GetEnvironmentVariable(Json.Str(locator["name"]) ?? "");
                if (v is null)
                {
                    return missing;
                }
                data = Encoding.UTF8.GetBytes(v);
                break;
            case "keychain":
                if (host.Keychain is null)
                {
                    return missing;
                }
                byte[]? stored;
                try
                {
                    stored = await host.Keychain(Json.Str(locator["service"]) ?? "", Json.Str(locator["account"]) ?? "", ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (ct.IsCancellationRequested)
                {
                    throw;
                }
                catch (Exception)
                {
                    stored = null;
                }
                if (stored is not { Length: > 0 })
                {
                    return missing;
                }
                data = stored;
                break;
            default:
                return missing;
        }
        var input = new JsonObject { ["type"] = "read", ["found"] = true, ["bytes"] = Convert.ToBase64String(data) };
        if (tag is not null)
        {
            input["etag"] = tag;
        }
        return input;
    }

    private async Task<JsonObject> WriteAsync(JsonNode effect, CancellationToken ct)
    {
        var locator = effect["locator"];
        var failed = new JsonObject { ["type"] = "written", ["ok"] = false };
        byte[] data;
        try
        {
            data = Convert.FromBase64String(Json.Str(effect["bytes"]) ?? "");
        }
        catch (FormatException)
        {
            return failed;
        }
        var locatorType = Json.Str(locator?["type"]);
        if (locatorType == "keychain")
        {
            var ok = false;
            if (host.KeychainWrite is not null)
            {
                try
                {
                    ok = await host.KeychainWrite(Json.Str(locator!["service"]) ?? "", Json.Str(locator["account"]) ?? "", data, ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (ct.IsCancellationRequested)
                {
                    throw;
                }
                catch (Exception)
                {
                    ok = false;
                }
            }
            return new JsonObject { ["type"] = "written", ["ok"] = ok };
        }
        if (locatorType != "file")
        {
            return failed;
        }
        var path = Path.GetFullPath(Json.Str(locator!["path"]) ?? "");
        var expect = Json.Str(effect["expect"]);
        if (expect is not null)
        {
            string? current = null;
            try
            {
                current = Etag(File.ReadAllBytes(path));
            }
            catch (Exception)
            {
            }
            if (current != expect)
            {
                failed["conflict"] = true;
                return failed;
            }
        }
        var isPrivate = effect["private"]?.GetValue<bool>() ?? true;
        try
        {
            var dir = Path.GetDirectoryName(path)!;
            if (OperatingSystem.IsWindows())
            {
                Directory.CreateDirectory(dir);
            }
            else
            {
                Directory.CreateDirectory(dir, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
            }
            var tmp = Path.Combine(dir, Path.GetFileName(path) + ".tmp" + Guid.NewGuid().ToString("N"));
            try
            {
                var options = new FileStreamOptions { Mode = FileMode.CreateNew, Access = FileAccess.Write, Share = FileShare.None };
                if (!OperatingSystem.IsWindows())
                {
                    var mode = UnixFileMode.UserRead | UnixFileMode.UserWrite;
                    if (!isPrivate)
                    {
                        mode |= UnixFileMode.GroupRead | UnixFileMode.OtherRead;
                    }
                    options.UnixCreateMode = mode;
                }
                using (var file = new FileStream(tmp, options))
                {
                    file.Write(data);
                }
                File.Move(tmp, path, true);
            }
            finally
            {
                File.Delete(tmp);
            }
            return new JsonObject { ["type"] = "written", ["ok"] = true };
        }
        catch (Exception)
        {
            return failed;
        }
    }

    private async Task<JsonObject> UserActionAsync(JsonNode? action)
    {
        if (handleAction is null)
        {
            throw ProviderException.Of("unsupported", null, "this call cannot ask the user for anything");
        }
        if (action is null)
        {
            throw ProviderException.Of("malformed_response", null, "the core asked for a user action without one");
        }
        string? value;
        try
        {
            value = await handleAction(Json.As<LoginAction>(action), ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            return Input("cancelled");
        }
        catch (Exception e)
        {
            throw new ProviderException(ProviderException.Build("auth", "user_action_failed", "the login handler failed: " + e.Message), e);
        }
        var input = Input("user_action");
        if (value is not null)
        {
            input["value"] = value;
        }
        return input;
    }

    private async Task<JsonObject> CallbackAsync(JsonObject effect)
    {
        var redirect = Json.Str(effect["redirect_uri"]);
        if (redirect is null || !Uri.TryCreate(redirect, UriKind.Absolute, out var uri) || uri.Scheme != Uri.UriSchemeHttp || !IPAddress.TryParse(uri.Host.Trim('[', ']'), out var address) || !IPAddress.IsLoopback(address))
        {
            throw ProviderException.Of("unsupported", null, "the login callback must be a loopback address, got " + redirect);
        }
        var timeoutMs = effect["timeout_ms"]?.GetValue<long>() ?? 300000;
        var listener = new TcpListener(address, uri.Port);
        try
        {
            try
            {
                listener.Start();
            }
            catch (SocketException)
            {
                throw ProviderException.Of("connection", "callback_port_busy", uri.Host + ":" + uri.Port + " is in use; close the other login or use callback \"paste\"");
            }
            var shown = await UserActionAsync(effect["action"]).ConfigureAwait(false);
            if (ct.IsCancellationRequested)
            {
                return shown;
            }
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
            deadline.CancelAfter(TimeSpan.FromMilliseconds(timeoutMs));
            var origin = uri.GetLeftPart(UriPartial.Authority);
            try
            {
                while (true)
                {
                    using var client = await listener.AcceptTcpClientAsync(deadline.Token).ConfigureAwait(false);
                    var target = await ServeCallbackAsync(client, uri.AbsolutePath, deadline.Token).ConfigureAwait(false);
                    if (target is not null)
                    {
                        var input = Input("callback");
                        input["url"] = origin + target;
                        return input;
                    }
                }
            }
            catch (OperationCanceledException)
            {
                if (ct.IsCancellationRequested)
                {
                    return Input("cancelled");
                }
                throw ProviderException.Of("timeout", "login_timeout", "no sign-in callback arrived within " + (timeoutMs / 1000) + " seconds");
            }
        }
        finally
        {
            listener.Stop();
        }
    }

    private static async Task<string?> ServeCallbackAsync(TcpClient client, string path, CancellationToken token)
    {
        try
        {
            using var limit = CancellationTokenSource.CreateLinkedTokenSource(token);
            limit.CancelAfter(TimeSpan.FromSeconds(5));
            var stream = client.GetStream();
            var head = new byte[8192];
            var length = 0;
            while (length < head.Length)
            {
                var n = await stream.ReadAsync(head.AsMemory(length), limit.Token).ConfigureAwait(false);
                if (n == 0)
                {
                    break;
                }
                length += n;
                if (Encoding.ASCII.GetString(head, 0, length).Contains("\r\n\r\n", StringComparison.Ordinal))
                {
                    break;
                }
            }
            var line = Encoding.ASCII.GetString(head, 0, length).Split("\r\n")[0].Split(' ');
            var target = line.Length >= 2 && line[0] == "GET" ? line[1] : null;
            var hit = target is not null && (target.Split('?')[0] == path);
            var body = hit ? "<!doctype html><html><head><meta charset=\"utf-8\"><title>Signed in</title></head><body><p>Sign-in finished. You can close this window.</p></body></html>" : "<!doctype html><html><body>Not found</body></html>";
            var bytes = Encoding.UTF8.GetBytes(body);
            var header = Encoding.ASCII.GetBytes((hit ? "HTTP/1.1 200 OK" : "HTTP/1.1 404 Not Found") + "\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: " + bytes.Length + "\r\nConnection: close\r\n\r\n");
            await stream.WriteAsync(header, limit.Token).ConfigureAwait(false);
            await stream.WriteAsync(bytes, limit.Token).ConfigureAwait(false);
            await stream.FlushAsync(limit.Token).ConfigureAwait(false);
            return hit ? target : null;
        }
        catch (Exception e) when (e is IOException or SocketException or OperationCanceledException && !token.IsCancellationRequested)
        {
            return null;
        }
    }
}
