using System.Collections.ObjectModel;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Microsoft.Extensions.Logging;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using PhysicsFlow.Infrastructure.Data;

namespace PhysicsFlow.ViewModels;

/// <summary>
/// ViewModel for the Dashboard — shows project summary stats, recent projects,
/// and well performance indicators.
/// </summary>
public partial class DashboardViewModel : ObservableObject
{
    private readonly AppDbService?           _db;
    private readonly ILogger<DashboardViewModel>? _log;

    // ── Summary statistics ────────────────────────────────────────────────────

    [ObservableProperty] private int    _totalProjects;
    [ObservableProperty] private int    _totalRuns;
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(WellCount), nameof(WellBreakdown), nameof(ProjectSummaryLine))]
    private int _totalWells;

    [ObservableProperty] private string _lastActivity = "Never";
    [ObservableProperty] private bool   _isLoading;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ProjectSummaryLine))]
    private string _statusMessage = "Ready";

    // ── Engine / training status ──────────────────────────────────────────────

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ProjectSummaryLine))]
    private string  _currentProjectName  = "No project loaded";

    [ObservableProperty] private string  _engineStatusText    = "Starting...";
    [ObservableProperty] private string  _engineStatusColor   = "#E67E22";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SurrogateStatus), nameof(SurrogateStatusColor), nameof(SurrogateDetail))]
    private bool    _isTrainingActive;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SurrogateDetail))]
    private double  _trainingProgress;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SurrogateStatus), nameof(SurrogateStatusColor), nameof(SurrogateDetail))]
    private string  _trainingStatusText  = "No training run";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HMStatus), nameof(HMStatusColor), nameof(HMDetail))]
    private bool    _isHmActive;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HMDetail))]
    private double  _hmMismatch;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HMStatus), nameof(HMStatusColor), nameof(HMDetail))]
    private int     _hmIteration;

    // ── Properties consumed by DashboardView.xaml ─────────────────────────────

    public string ProjectSummaryLine =>
        CurrentProjectName == "No project loaded"
            ? "No project loaded — open or create one to get started"
            : $"{CurrentProjectName} — {TotalWells} wells  •  {StatusMessage}";

    // Wells card
    public int    WellCount    => TotalWells;
    public string WellBreakdown => TotalWells > 0
        ? $"{TotalWells} wells configured"
        : "No wells loaded";

    // Grid card (populated from project or defaults)
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(GridDimensions), nameof(ActiveCells))]
    private int _gridNx = 46;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(GridDimensions), nameof(ActiveCells))]
    private int _gridNy = 112;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(GridDimensions), nameof(ActiveCells))]
    private int _gridNz = 22;

    public string GridDimensions => $"{GridNx} × {GridNy} × {GridNz}";
    public string ActiveCells    => $"{GridNx * GridNy * GridNz:N0} total cells";

    // Surrogate status card
    private bool HasBeenTrained => TrainingStatusText != "No training run";
    public string SurrogateStatus => IsTrainingActive ? "Training..."
        : (HasBeenTrained ? "Trained ✓" : "Not trained");
    public string SurrogateStatusColor => IsTrainingActive ? "#E67E22"
        : (HasBeenTrained ? "#2ECC71" : "#95A5A6");
    public string SurrogateDetail => IsTrainingActive
        ? $"Epoch progress: {TrainingProgress:P0}"
        : (HasBeenTrained ? TrainingStatusText : "Run 'Train PINO' to build the surrogate model");

    // HM status card
    public string HMStatus => IsHmActive ? "Running..." : HmIteration > 0 ? "Completed" : "Not run";
    public string HMStatusColor => IsHmActive ? "#E67E22" : HmIteration > 0 ? "#2ECC71" : "#95A5A6";
    public string HMDetail => IsHmActive
        ? $"Iteration {HmIteration}  |  mismatch {HmMismatch:F4}"
        : HmIteration > 0
            ? $"Converged in {HmIteration} iterations  |  mismatch {HmMismatch:F4}"
            : "No history matching results yet";

    // ── Commands ──────────────────────────────────────────────────────────────

    public event EventHandler? NewProjectRequested;
    public event EventHandler? OpenProjectRequested;
    public event EventHandler? StartTrainingRequested;
    public event EventHandler? StartHMRequested;

    [RelayCommand]
    private void NewProject()  => NewProjectRequested?.Invoke(this, EventArgs.Empty);

    [RelayCommand]
    private void OpenProject() => OpenProjectRequested?.Invoke(this, EventArgs.Empty);

    [RelayCommand(CanExecute = nameof(CanStartTraining))]
    private void StartTraining() => StartTrainingRequested?.Invoke(this, EventArgs.Empty);
    public bool CanStartTraining => !IsTrainingActive;

    [RelayCommand(CanExecute = nameof(CanStartHM))]
    private void StartHM() => StartHMRequested?.Invoke(this, EventArgs.Empty);
    public bool CanStartHM => !IsHmActive && !IsTrainingActive;

    [RelayCommand]
    private async Task RefreshAsync() => await LoadAsync();

    // ── Recent projects ───────────────────────────────────────────────────────

    public ObservableCollection<ProjectSummaryItem> RecentProjects { get; } = new();
    public ObservableCollection<WellStatusItem>     WellStatus     { get; } = new();

    [ObservableProperty] private ProjectSummaryItem? _selectedRecentProject;

    // ── Well map OxyPlot model ────────────────────────────────────────────────

    public PlotModel WellMapModel { get; } = BuildDefaultWellMap();

    // ── Constructor ──────────────────────────────────────────────────────────

    public DashboardViewModel(
        AppDbService?                 db  = null,
        ILogger<DashboardViewModel>?  log = null)
    {
        _db  = db;
        _log = log;
        _ = LoadAsync();
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
            TotalWells    = summary.TotalHMIter;
            LastActivity  = summary.LastActivity.HasValue
                ? summary.LastActivity.Value.ToLocalTime().ToString("dd MMM yyyy HH:mm")
                : "Never";

            var projects = await _db.GetRecentProjectsAsync(limit: 10);
            RecentProjects.Clear();
            foreach (var p in projects)
                RecentProjects.Add(new ProjectSummaryItem(
                    p.Name,
                    p.CreatedAt.ToString("dd MMM yyyy"),
                    p.HmCompleted ? "HM complete" : p.PinoTrained ? "Trained" : "Not started"));

            OnPropertyChanged(nameof(ProjectSummaryLine));
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

    // ── Well map placeholder ──────────────────────────────────────────────────

    private static PlotModel BuildDefaultWellMap()
    {
        var model = new PlotModel
        {
            Background = OxyColor.FromRgb(0x16, 0x20, 0x30),
            PlotAreaBorderColor = OxyColor.FromRgb(0x2A, 0x3A, 0x4A),
        };

        model.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Bottom, Title = "I (grid cells)",
            TextColor = OxyColor.FromRgb(0x88, 0x99, 0xAA),
            AxislineColor = OxyColor.FromRgb(0x2A, 0x3A, 0x4A),
            Minimum = 0, Maximum = 46,
        });
        model.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Left, Title = "J (grid cells)",
            TextColor = OxyColor.FromRgb(0x88, 0x99, 0xAA),
            AxislineColor = OxyColor.FromRgb(0x2A, 0x3A, 0x4A),
            Minimum = 0, Maximum = 112,
        });

        // Norne producers (approximate I,J)
        var producers = new ScatterSeries
        {
            MarkerType = MarkerType.Circle, MarkerSize = 6,
            MarkerFill = OxyColor.FromRgb(0x00, 0xCC, 0x66),
            Title = "Producers",
        };
        foreach (var (i, j) in new[] {
            (9,29),(11,35),(11,35),(13,43),(13,43),(20,54),(20,54),(20,58),
            (22,63),(24,63),(24,89),(25,85),(25,85),(27,100),(27,100),(31,68),
            (34,71),(35,78),(36,84),(36,84),(37,88),(38,92)})
            producers.Points.Add(new ScatterPoint(i, j));
        model.Series.Add(producers);

        // Norne injectors
        var injectors = new ScatterSeries
        {
            MarkerType = MarkerType.Triangle, MarkerSize = 6,
            MarkerFill = OxyColor.FromRgb(0x44, 0x88, 0xFF),
            Title = "Injectors",
        };
        foreach (var (i, j) in new[] { (36,84),(36,84),(38,92),(13,43) })
            injectors.Points.Add(new ScatterPoint(i, j));
        model.Series.Add(injectors);

        return model;
    }
}

// ── Supporting data records ────────────────────────────────────────────────────

public record ProjectSummaryItem(string FieldName, string LastModified, string StatusLine);
public record WellStatusItem(string WellName, string Type, double OilRate, double WaterRate);
public record QuickActionItem(string Label, string Description);
