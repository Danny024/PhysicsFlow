using System.Collections.ObjectModel;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using PhysicsFlow.Infrastructure.Data;

namespace PhysicsFlow.ViewModels;

/// <summary>
/// ViewModel for the Dashboard — shows project summary stats, recent projects,
/// and well performance indicators.
/// </summary>
public partial class DashboardViewModel : ObservableObject
{
    private readonly AppDbService?          _db;
    private readonly ILogger<DashboardViewModel>? _log;

    // ── Summary statistics ────────────────────────────────────────────────────

    [ObservableProperty] private int    _totalProjects;
    [ObservableProperty] private int    _totalRuns;
    [ObservableProperty] private int    _totalWells;
    [ObservableProperty] private string _lastActivity = "Never";
    [ObservableProperty] private bool   _isLoading;
    [ObservableProperty] private string _statusMessage = "Ready";

    // ── Engine / training status ──────────────────────────────────────────────

    [ObservableProperty] private string  _currentProjectName  = "No project loaded";
    [ObservableProperty] private string  _engineStatusText    = "Starting...";
    [ObservableProperty] private string  _engineStatusColor   = "#E67E22";
    [ObservableProperty] private bool    _isTrainingActive;
    [ObservableProperty] private double  _trainingProgress;
    [ObservableProperty] private string  _trainingStatusText  = "No training run";
    [ObservableProperty] private bool    _isHmActive;
    [ObservableProperty] private double  _hmMismatch;
    [ObservableProperty] private int     _hmIteration;

    // ── Collections ───────────────────────────────────────────────────────────

    public ObservableCollection<ProjectSummaryItem> RecentProjects { get; } = new();
    public ObservableCollection<WellStatusItem>     WellStatus     { get; } = new();
    public ObservableCollection<QuickActionItem>    QuickActions   { get; } = new()
    {
        new("🆕 New Project",          "Create a new reservoir simulation project"),
        new("📂 Open Project",         "Open an existing .pfproj file"),
        new("🧠 Start Training",       "Start PINO surrogate training"),
        new("⚡ Run History Matching", "Launch αREKI history matching"),
        new("📊 View Forecast",        "Open the production forecast view"),
        new("🗺️ 3D Viewer",           "Open the 3D reservoir viewer"),
    };

    public DashboardViewModel(
        AppDbService?                 db  = null,
        ILogger<DashboardViewModel>?  log = null)
    {
        _db  = db;
        _log = log;
        _ = LoadAsync();
    }

    // ── Commands ──────────────────────────────────────────────────────────────

    [RelayCommand]
    private async Task RefreshAsync()
    {
        await LoadAsync();
    }

    // ── Data loading ──────────────────────────────────────────────────────────

    private async Task LoadAsync()
    {
        if (_db is null) return;
        IsLoading = true;
        StatusMessage = "Loading...";

        try
        {
            var summary = await _db.GetSummaryAsync();
            TotalProjects = summary.TotalProjects;
            TotalRuns     = summary.TotalRuns;
            TotalWells    = summary.TotalHMIter;   // proxy until well count query added
            LastActivity  = summary.LastActivity.HasValue
                ? summary.LastActivity.Value.ToLocalTime().ToString("dd MMM yyyy HH:mm")
                : "Never";

            var projects = await _db.GetRecentProjectsAsync(limit: 10);
            RecentProjects.Clear();
            foreach (var p in projects)
                RecentProjects.Add(new ProjectSummaryItem(
                    p.Name, p.CreatedAt,
                    p.HmCompleted ? "HM complete" : p.PinoTrained ? "Trained" : "Not started"));

            StatusMessage = "Ready";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Load error: {ex.Message}";
            _log?.LogWarning(ex, "Dashboard load failed");
        }
        finally
        {
            IsLoading = false;
        }
    }
}

// ── Supporting data records ────────────────────────────────────────────────────

public record ProjectSummaryItem(string Name, DateTime? CreatedAt, string HmStatus);
public record WellStatusItem(string WellName, string Type, double OilRate, double WaterRate);
public record QuickActionItem(string Label, string Description);
