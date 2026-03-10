using System.IO;
using System.Text.Json;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace PhysicsFlow.ViewModels;

public partial class SettingsViewModel : ObservableObject
{
    private static readonly string SettingsPath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
        "PhysicsFlow", "settings.json");

    // ── Engine ────────────────────────────────────────────────────────────
    [ObservableProperty] private string _engineGrpcAddress  = "http://localhost:50051";
    [ObservableProperty] private string _engineRestAddress  = "http://localhost:8000";

    // ── AI Assistant ──────────────────────────────────────────────────────
    [ObservableProperty] private string _ollamaModel       = "deepseek-r1:1.5b";
    [ObservableProperty] private string _ollamaBaseUrl     = "http://localhost:11434";

    // ── Projects ──────────────────────────────────────────────────────────
    [ObservableProperty] private string _defaultProjectDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "PhysicsFlow");

    // ── Status ────────────────────────────────────────────────────────────
    [ObservableProperty] private string _saveStatus = string.Empty;

    public event EventHandler? SettingsSaved;

    public SettingsViewModel() => Load();

    // ── Commands ──────────────────────────────────────────────────────────

    [RelayCommand]
    private void Save()
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(SettingsPath)!);
            var data = new SettingsData
            {
                EngineGrpcAddress  = EngineGrpcAddress,
                EngineRestAddress  = EngineRestAddress,
                OllamaModel        = OllamaModel,
                OllamaBaseUrl      = OllamaBaseUrl,
                DefaultProjectDir  = DefaultProjectDir,
            };
            File.WriteAllText(SettingsPath,
                JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true }));
            SaveStatus = "✓ Settings saved";
            SettingsSaved?.Invoke(this, EventArgs.Empty);
        }
        catch (Exception ex)
        {
            SaveStatus = $"Error: {ex.Message}";
        }
    }

    [RelayCommand]
    private void BrowseDefaultProjectDir()
    {
        var dlg = new System.Windows.Forms.FolderBrowserDialog
        {
            Description         = "Default folder for new projects",
            SelectedPath        = DefaultProjectDir,
            ShowNewFolderButton = true,
        };
        if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            DefaultProjectDir = dlg.SelectedPath;
    }

    // ── Load ──────────────────────────────────────────────────────────────

    private void Load()
    {
        if (!File.Exists(SettingsPath)) return;
        try
        {
            var data = JsonSerializer.Deserialize<SettingsData>(File.ReadAllText(SettingsPath));
            if (data is null) return;
            EngineGrpcAddress = data.EngineGrpcAddress ?? EngineGrpcAddress;
            EngineRestAddress = data.EngineRestAddress ?? EngineRestAddress;
            OllamaModel       = data.OllamaModel       ?? OllamaModel;
            OllamaBaseUrl     = data.OllamaBaseUrl     ?? OllamaBaseUrl;
            DefaultProjectDir = data.DefaultProjectDir ?? DefaultProjectDir;
        }
        catch { /* ignore corrupt settings */ }
    }

    private class SettingsData
    {
        public string? EngineGrpcAddress { get; set; }
        public string? EngineRestAddress { get; set; }
        public string? OllamaModel       { get; set; }
        public string? OllamaBaseUrl     { get; set; }
        public string? DefaultProjectDir { get; set; }
    }
}
