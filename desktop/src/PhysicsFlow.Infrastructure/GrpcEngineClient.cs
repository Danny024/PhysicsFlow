using System;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Grpc.Net.Client;
using Microsoft.Extensions.Logging;

namespace PhysicsFlow.Infrastructure;

/// <summary>
/// Manages the gRPC channel to the PhysicsFlow Python engine (port 50051).
///
/// Provides typed client accessors for SimulationService, TrainingService,
/// HistoryMatchingService, and AgentService. The channel is created lazily
/// on first use and re-created automatically if the engine restarts.
/// </summary>
public sealed class GrpcEngineClient : IDisposable
{
    private readonly ILogger<GrpcEngineClient> _log;
    private readonly SemaphoreSlim             _channelLock = new(1, 1);

    private GrpcChannel? _channel;
    private string        _endpoint = "http://localhost:50051";
    private bool          _disposed;

    // ── Proto-generated client types (populated once proto stubs are compiled) ──

    // Uncomment when proto stubs are generated:
    // private Simulation.SimulationClient?      _simulationClient;
    // private Training.TrainingClient?          _trainingClient;
    // private HistoryMatching.HMClient?         _hmClient;

    public GrpcEngineClient(ILogger<GrpcEngineClient> log)
    {
        _log = log;
    }

    // ── Channel management ────────────────────────────────────────────────────

    /// <summary>Configure the engine endpoint before first use.</summary>
    public void SetEndpoint(string host, int port)
    {
        _endpoint = $"http://{host}:{port}";
        _ = ResetChannelAsync();
    }

    /// <summary>Returns true if the gRPC channel is connected and healthy.</summary>
    public async Task<bool> IsConnectedAsync(CancellationToken ct = default)
    {
        try
        {
            var ch = await GetChannelAsync(ct);
            var state = ch.State;
            return state == Grpc.Core.ConnectivityState.Ready
                || state == Grpc.Core.ConnectivityState.Idle;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>Force a channel reconnect (e.g., after engine restart).</summary>
    public async Task ResetChannelAsync()
    {
        await _channelLock.WaitAsync();
        try
        {
            if (_channel != null)
            {
                await _channel.ShutdownAsync();
                _channel.Dispose();
                _channel = null;
            }
        }
        finally
        {
            _channelLock.Release();
        }
    }

    // ── Client accessors ──────────────────────────────────────────────────────

    // These return strongly-typed proto clients once stubs are generated.
    // Until then they return the raw channel for manual client construction.

    /// <summary>
    /// Returns the raw <see cref="GrpcChannel"/> for building proto clients.
    /// In production code, replace with typed client accessors per service.
    /// </summary>
    public async Task<GrpcChannel> GetChannelAsync(CancellationToken ct = default)
    {
        if (_channel != null
            && _channel.State != Grpc.Core.ConnectivityState.Shutdown
            && _channel.State != Grpc.Core.ConnectivityState.TransientFailure)
        {
            return _channel;
        }

        await _channelLock.WaitAsync(ct);
        try
        {
            // Double-check inside lock
            if (_channel != null
                && _channel.State != Grpc.Core.ConnectivityState.Shutdown)
            {
                return _channel;
            }

            _log.LogInformation("Opening gRPC channel → {Endpoint}", _endpoint);
            _channel = GrpcChannel.ForAddress(_endpoint, new GrpcChannelOptions
            {
                // Allow large messages for ensemble arrays (up to 256 MB)
                MaxReceiveMessageSize = 256 * 1024 * 1024,
                MaxSendMessageSize    = 64  * 1024 * 1024,
            });

            return _channel;
        }
        finally
        {
            _channelLock.Release();
        }
    }

    // ── Health check ──────────────────────────────────────────────────────────

    /// <summary>
    /// Sends a lightweight health-check ping to the Python engine.
    /// Returns latency in milliseconds, or -1 if unreachable.
    /// </summary>
    public async Task<long> PingAsync(CancellationToken ct = default)
    {
        try
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            var ch = await GetChannelAsync(ct);
            await ch.ConnectAsync(ct);
            sw.Stop();
            return sw.ElapsedMilliseconds;
        }
        catch (Exception ex)
        {
            _log.LogWarning("gRPC ping failed: {Message}", ex.Message);
            return -1;
        }
    }

    // ── IDisposable ───────────────────────────────────────────────────────────

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _channel?.Dispose();
        _channelLock.Dispose();
    }
}
