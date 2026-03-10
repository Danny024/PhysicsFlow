using System.Collections.ObjectModel;
using System.Linq;
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
    [ObservableProperty] private string _selectedModel = "deepseek-r1:1.5b";
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

    // Curated list of recommended Ollama models (shown even if not yet installed)
    private static readonly string[] _curatedModels =
    [
        "phi3:mini",          // ~2.3 GB  — fast, good reasoning
        "phi3:medium",        // ~7.9 GB  — higher quality
        "llama3.1:8b",        // ~4.7 GB  — Meta Llama 3.1
        "llama3.1:70b",       // ~40 GB   — large, high quality
        "llama3.2:3b",        // ~2.0 GB  — very fast
        "mistral:7b",         // ~4.1 GB  — Mistral 7B
        "mistral-nemo",       // ~7.1 GB  — Mistral Nemo
        "gemma2:9b",          // ~5.4 GB  — Google Gemma 2
        "gemma2:27b",         // ~16 GB   — larger Gemma 2
        "qwen2.5:7b",         // ~4.4 GB  — Alibaba Qwen 2.5
        "qwen2.5:14b",        // ~8.9 GB  — larger Qwen
        "deepseek-r1:8b",     // ~4.9 GB  — DeepSeek R1 reasoning
        "deepseek-r1:14b",    // ~9.0 GB  — larger DeepSeek R1
        "codellama:7b",       // ~3.8 GB  — code-focused
        "nomic-embed-text",   // ~274 MB  — embeddings
    ];

    public AIAssistantViewModel(OllamaAgentClient agent)
    {
        _agent = agent;
        _ = InitialiseAsync();
    }

    private async Task InitialiseAsync()
    {
        try
        {
            await RefreshModelsInternalAsync();

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

    private async Task RefreshModelsInternalAsync()
    {
        // Fetch installed models from Ollama
        var installed = (await _agent.ListModelsAsync()).ToHashSet(StringComparer.OrdinalIgnoreCase);

        AvailableModels.Clear();

        // Installed models first (sorted), then curated suggestions not yet installed
        foreach (var m in installed.OrderBy(x => x))
            AvailableModels.Add(m);

        foreach (var m in _curatedModels)
            if (!installed.Contains(m))
                AvailableModels.Add(m);

        if (installed.Count > 0)
        {
            // Preference order: deepseek-r1:1.5b → deepseek-r1:14b → phi3:mini → first installed
            string[] preferred = ["deepseek-r1:1.5b", "deepseek-r1:14b", "phi3:mini"];
            SelectedModel = preferred.FirstOrDefault(m => installed.Contains(m))
                            ?? installed.OrderBy(x => x).First();
            OllamaStatusColor = "#2ECC71";
            OllamaStatusText = $"Ollama connected — {installed.Count} model(s) installed";
        }
        else
        {
            SelectedModel = "deepseek-r1:1.5b";
            OllamaStatusColor = "#E67E22";
            OllamaStatusText = "No models installed — run: ollama pull deepseek-r1:1.5b";
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
    private void NewLine() => InputText += "\n";

    [RelayCommand]
    private async Task ClearHistoryAsync()
    {
        Messages.Clear();
        await _agent.ClearHistoryAsync(_sessionId);
        AddAssistantMessage("Conversation cleared. How can I help you?");
    }

    [RelayCommand]
    private async Task RefreshModelsAsync()
    {
        OllamaStatusColor = "#F39C12";
        OllamaStatusText = "Refreshing models...";
        try
        {
            await RefreshModelsInternalAsync();
        }
        catch (Exception ex)
        {
            OllamaStatusColor = "#FF6B6B";
            OllamaStatusText = $"Refresh failed: {ex.Message}";
        }
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
