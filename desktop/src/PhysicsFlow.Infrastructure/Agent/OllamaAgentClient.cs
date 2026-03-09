using System.Runtime.CompilerServices;
using Grpc.Core;
using Grpc.Net.Client;
using Microsoft.Extensions.Logging;

namespace PhysicsFlow.Infrastructure.Agent;

/// <summary>
/// gRPC client for the Ollama-powered Reservoir AI Assistant.
///
/// Streams chat tokens from the Python AgentService over gRPC,
/// providing real-time typing effect in the UI.
/// </summary>
public class OllamaAgentClient
{
    private readonly ILogger<OllamaAgentClient> _logger;
    private GrpcChannel? _channel;
    private Proto.AgentService.AgentServiceClient? _client;

    public OllamaAgentClient(ILogger<OllamaAgentClient> logger)
    {
        _logger = logger;
    }

    public void Connect(string address = "http://localhost:50051")
    {
        _channel = GrpcChannel.ForAddress(address, new GrpcChannelOptions
        {
            MaxReceiveMessageSize = 64 * 1024 * 1024,   // 64 MB
            MaxSendMessageSize    = 64 * 1024 * 1024,
        });
        _client = new Proto.AgentService.AgentServiceClient(_channel);
        _logger.LogInformation("OllamaAgentClient connected to {Address}", address);
    }

    // ── Streaming chat ────────────────────────────────────────────────────────

    /// <summary>
    /// Send a message and stream back token-by-token response.
    /// Each yielded item is a ChatTokenResult with token text and metadata.
    /// </summary>
    public async IAsyncEnumerable<ChatTokenResult> ChatStreamAsync(
        string sessionId,
        string message,
        string model,
        string? projectPath = null,
        [EnumeratorCancellation] CancellationToken ct = default)
    {
        EnsureConnected();

        var request = new Proto.ChatRequest
        {
            SessionId      = sessionId,
            Message        = message,
            ContextProject = projectPath ?? string.Empty,
        };

        Proto.AgentService.AgentServiceClient client = _client!;

        AsyncServerStreamingCall<Proto.ChatToken> call;
        try
        {
            call = client.Chat(request, cancellationToken: ct);
        }
        catch (RpcException ex)
        {
            _logger.LogError(ex, "gRPC chat call failed");
            yield return new ChatTokenResult(
                Token: $"\n⚠ Engine error: {ex.Status.Detail}",
                IsDone: true
            );
            yield break;
        }

        await foreach (var token in call.ResponseStream.ReadAllAsync(ct))
        {
            yield return new ChatTokenResult(
                Token:       token.Token,
                IsToolCall:  token.IsToolCall,
                ToolName:    token.ToolName,
                ToolResult:  token.ToolResult,
                IsDone:      token.IsDone,
                FullResponse:token.FullResponse,
                ChartData:   token.Chart is { } c ? MapChartData(c) : null
            );
        }
    }

    // ── Model management ──────────────────────────────────────────────────────

    public async Task<List<string>> ListModelsAsync()
    {
        EnsureConnected();
        try
        {
            var response = await _client!.ListModelsAsync(new Proto.ListModelsRequest());
            return response.Models.ToList();
        }
        catch (RpcException ex)
        {
            _logger.LogWarning(ex, "Could not list Ollama models");
            return new List<string>();
        }
    }

    public async Task<bool> SetModelAsync(string modelName)
    {
        EnsureConnected();
        try
        {
            var response = await _client!.SetModelAsync(new Proto.SetModelRequest
            { ModelName = modelName });
            return response.Success;
        }
        catch (RpcException ex)
        {
            _logger.LogError(ex, "Failed to set model");
            return false;
        }
    }

    public async Task ClearHistoryAsync(string sessionId)
    {
        EnsureConnected();
        try
        {
            await _client!.ClearHistoryAsync(new Proto.ClearHistoryRequest
            { SessionId = sessionId });
        }
        catch (RpcException ex)
        {
            _logger.LogWarning(ex, "Failed to clear history");
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private void EnsureConnected()
    {
        if (_client is null)
            Connect();
    }

    private static ChartDataResult? MapChartData(Proto.ChartData proto)
    {
        if (proto is null) return null;
        return new ChartDataResult(
            ChartType: proto.ChartType,
            Title:     proto.Title,
            XLabel:    proto.XLabel,
            YLabel:    proto.YLabel,
            Series: proto.Series.Select(s => new ChartSeriesResult(
                Name:    s.Name,
                X:       s.X.ToList(),
                Y:       s.Y.ToList(),
                YLower:  s.YLower.ToList(),
                YUpper:  s.YUpper.ToList(),
                Color:   s.Color
            )).ToList()
        );
    }
}

// ── Result types ──────────────────────────────────────────────────────────────

public record ChatTokenResult(
    string Token,
    bool IsToolCall = false,
    string ToolName = "",
    string ToolResult = "",
    bool IsDone = false,
    string FullResponse = "",
    ChartDataResult? ChartData = null
);

public record ChartDataResult(
    string ChartType,
    string Title,
    string XLabel,
    string YLabel,
    List<ChartSeriesResult> Series
);

public record ChartSeriesResult(
    string Name,
    List<float> X,
    List<float> Y,
    List<float> YLower,
    List<float> YUpper,
    string Color
);
