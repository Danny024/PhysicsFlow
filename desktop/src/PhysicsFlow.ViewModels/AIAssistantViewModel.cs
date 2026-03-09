using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PhysicsFlow.Infrastructure.Agent;

namespace PhysicsFlow.ViewModels;

/// <summary>
/// ViewModel for the AI Reservoir Assistant chat panel.
///
/// Streams tokens from the Ollama agent via gRPC, displaying them
/// one-by-one for a natural typing effect. Tool call notifications
/// are shown inline as "thinking" steps.
/// </summary>
public partial class AIAssistantViewModel : ObservableObject
{
    private readonly OllamaAgentClient _agent;
    private readonly string _sessionId = Guid.NewGuid().ToString();

    [ObservableProperty] private string _inputText = string.Empty;
    [ObservableProperty] private bool _isTyping;
    [ObservableProperty] private string _selectedModel = "phi3:mini";
    [ObservableProperty] private ObservableCollection<string> _availableModels = new();
    [ObservableProperty] private string _ollamaStatusColor = "#FF6B6B";
    [ObservableProperty] private string _ollamaStatusText = "Checking Ollama...";
    [ObservableProperty] private bool _canSend = true;

    public ObservableCollection<ChatMessage> Messages { get; } = new();

    public ObservableCollection<QuickAction> QuickActions { get; } = new()
    {
        new("📊 Summarise HM results",  "Summarise the current history matching results and convergence status."),
        new("💧 Well performance",       "Which wells are performing above and below expectations? Show me the production profiles."),
        new("⚡ Mismatch analysis",      "Break down the data mismatch per well. Which wells are matching poorly?"),
        new("📈 P10/P50/P90 forecast",  "Show me the P10, P50, and P90 production forecast for all wells."),
        new("🔍 Explain αREKI",         "Explain how the αREKI history matching algorithm works in simple terms."),
        new("⚙  Training status",       "What is the current PINO surrogate training status? Show me the loss curves."),
        new("🗺  Reservoir overview",    "Give me a full summary of the current reservoir project."),
    };

    public AIAssistantViewModel(OllamaAgentClient agent)
    {
        _agent = agent;
        _ = InitialiseAsync();
    }

    private async Task InitialiseAsync()
    {
        try
        {
            var models = await _agent.ListModelsAsync();
            foreach (var m in models)
                AvailableModels.Add(m);

            if (AvailableModels.Count > 0)
            {
                SelectedModel = AvailableModels.Contains("phi3:mini")
                    ? "phi3:mini"
                    : AvailableModels[0];
                OllamaStatusColor = "#2ECC71";
                OllamaStatusText = "Ollama connected";
            }
            else
            {
                OllamaStatusColor = "#E67E22";
                OllamaStatusText = "No models found — run: ollama pull phi3:mini";
            }

            // Welcome message
            AddAssistantMessage(
                "Hello! I'm your AI Reservoir Assistant. I have access to your live " +
                "simulation data and can answer questions about your reservoir, " +
                "history matching results, and production forecasts.\n\n" +
                "Try one of the quick actions above, or ask me anything!"
            );
        }
        catch (Exception ex)
        {
            OllamaStatusColor = "#FF6B6B";
            OllamaStatusText = "Ollama not running — start Ollama first";
            AddAssistantMessage(
                $"⚠ Could not connect to Ollama: {ex.Message}\n\n" +
                "Please install Ollama from https://ollama.com and run:\n" +
                "  ollama pull phi3:mini\n" +
                "Then restart PhysicsFlow."
            );
        }
    }

    // ── Commands ──────────────────────────────────────────────────────────────

    [RelayCommand]
    private async Task SendMessageAsync()
    {
        var text = InputText.Trim();
        if (string.IsNullOrEmpty(text) || !CanSend) return;

        InputText = string.Empty;
        CanSend = false;
        IsTyping = true;

        // Add user bubble
        Messages.Add(new ChatMessage(ChatRole.User, text));

        // Start streaming assistant response
        var assistantMsg = new ChatMessage(ChatRole.Assistant, string.Empty);
        Messages.Add(assistantMsg);

        try
        {
            await foreach (var token in _agent.ChatStreamAsync(_sessionId, text, SelectedModel))
            {
                if (token.IsToolCall)
                {
                    // Show tool call as a system notification inline
                    assistantMsg.AppendToolCallNotification(token.ToolName, token.ToolResult);
                }
                else if (token.IsDone)
                {
                    IsTyping = false;
                    if (token.ChartData is { } chart)
                        assistantMsg.SetChartData(chart);
                }
                else
                {
                    assistantMsg.AppendToken(token.Token);
                }
            }
        }
        catch (Exception ex)
        {
            assistantMsg.AppendToken($"\n\n⚠ Error: {ex.Message}");
        }
        finally
        {
            IsTyping = false;
            CanSend = true;
        }
    }

    [RelayCommand]
    private async Task SendQuickActionAsync(string prompt)
    {
        InputText = prompt;
        await SendMessageAsync();
    }

    [RelayCommand]
    private async Task ClearHistoryAsync()
    {
        Messages.Clear();
        await _agent.ClearHistoryAsync(_sessionId);
        AddAssistantMessage("Conversation cleared. How can I help you?");
    }

    partial void OnSelectedModelChanged(string value)
    {
        _ = _agent.SetModelAsync(value);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private void AddAssistantMessage(string text)
    {
        Messages.Add(new ChatMessage(ChatRole.Assistant, text));
    }
}

// ── Data models ───────────────────────────────────────────────────────────────

public enum ChatRole { User, Assistant, System }

public partial class ChatMessage : ObservableObject
{
    public ChatRole Role { get; }
    [ObservableProperty] private string _content;
    [ObservableProperty] private bool _hasChart;
    [ObservableProperty] private object? _chartData;
    [ObservableProperty] private string _toolCallLog = string.Empty;
    [ObservableProperty] private bool _hasToolCalls;

    public bool IsUser      => Role == ChatRole.User;
    public bool IsAssistant => Role == ChatRole.Assistant;

    public ChatMessage(ChatRole role, string content)
    {
        Role = role;
        _content = content;
    }

    public void AppendToken(string token)
    {
        Content += token;
    }

    public void AppendToolCallNotification(string toolName, string result)
    {
        HasToolCalls = true;
        ToolCallLog += $"🔧 Called: {toolName}\n";
    }

    public void SetChartData(object chart)
    {
        ChartData = chart;
        HasChart = true;
    }
}

public record QuickAction(string Label, string Prompt);
