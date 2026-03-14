using System.Collections.ObjectModel;
using System.Text.Json;
using System.Windows;
using System.Windows.Media;
using Application = System.Windows.Application;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using PhysicsFlow.Infrastructure.Engine;
using Serilog;

namespace PhysicsFlow.ViewModels;

/// <summary>
/// Main window view model: manages navigation, engine status, and AI panel toggle.
/// </summary>
public partial class MainWindowViewModel : ObservableObject
{
    private readonly EngineManager _engineManager;
    private readonly DashboardViewModel _dashboard;
    private readonly ProjectSetupViewModel _projectSetup;
    private readonly Action _openSettingsDialog;
    private readonly TrainingViewModel _training;
    private readonly HistoryMatchingViewModel _historyMatching;
    private readonly ForecastViewModel _forecast;
    private readonly ReservoirView3DViewModel _reservoirView3D;
    private readonly CrossSectionViewModel _crossSection;
    public  AIAssistantViewModel AIAssistant { get; }

    // ── Navigation ────────────────────────────────────────────────────────────

    [ObservableProperty] private ObservableObject? _currentView;
    [ObservableProperty] private string _currentProjectName = "No project loaded";

    // ── Engine status ─────────────────────────────────────────────────────────

    [ObservableProperty] private string _engineStatusText = "Starting engine...";
    [ObservableProperty] private string _engineStatusColor = "#E67E22";

    // ── AI Assistant panel ────────────────────────────────────────────────────

    [ObservableProperty] private bool _assistantPanelVisible = true;
    [ObservableProperty] private GridLength _assistantPanelWidth = new(360, GridUnitType.Pixel);

    public MainWindowViewModel(
        EngineManager engineManager,
        DashboardViewModel dashboard,
        ProjectSetupViewModel projectSetup,
        TrainingViewModel training,
        HistoryMatchingViewModel historyMatching,
        ForecastViewModel forecast,
        ReservoirView3DViewModel reservoirView3D,
        CrossSectionViewModel crossSection,
        AIAssistantViewModel aiAssistant,
        Action openSettingsDialog)
    {
        _engineManager = engineManager;
        _dashboard = dashboard;
        _projectSetup = projectSetup;
        _training = training;
        _historyMatching = historyMatching;
        _forecast = forecast;
        _reservoirView3D = reservoirView3D;
        _crossSection = crossSection;
        AIAssistant = aiAssistant;
        _openSettingsDialog = openSettingsDialog;

        // Wire up project wizard events
        _projectSetup.ProjectSaved += (_, path) =>
        {
            var name = System.IO.Path.GetFileNameWithoutExtension(path);
            CurrentProjectName             = name;
            AIAssistant.CurrentProjectPath = path;
            // Push project metadata to dashboard cards
            _dashboard.CurrentProjectName  = name;
            _dashboard.TotalWells          = _projectSetup.WellCount;
            _dashboard.GridNx              = _projectSetup.GridNx;
            _dashboard.GridNy              = _projectSetup.GridNy;
            _dashboard.GridNz              = _projectSetup.GridNz;
            _dashboard.StatusMessage       = $"Project '{name}' saved successfully";
            CurrentView = _dashboard;
        };
        _projectSetup.WizardCancelled += (_, _) => CurrentView = _dashboard;

        // Wire up dashboard quick-action buttons → navigation
        _dashboard.NewProjectRequested   += (_, _) => CurrentView = _projectSetup;
        _dashboard.OpenProjectRequested  += (_, _) => OpenProjectFile();
        _dashboard.StartTrainingRequested += (_, _) => CurrentView = _training;
        _dashboard.StartHMRequested      += (_, _) => CurrentView = _historyMatching;

        // Feed training status to dashboard status cards
        _training.PropertyChanged += (_, e) =>
        {
            switch (e.PropertyName)
            {
                case nameof(TrainingViewModel.IsTraining):
                    _dashboard.IsTrainingActive = _training.IsTraining;
                    // Training just completed with a real loss value → mark PINO as trained
                    if (!_training.IsTraining && _training.BestLoss < double.MaxValue)
                    {
                        _dashboard.IsPinoTrained      = true;
                        _dashboard.TrainingStatusText = $"Best loss: {_training.BestLoss:F6}";
                    }
                    break;
                case nameof(TrainingViewModel.ProgressPct):
                    _dashboard.TrainingProgress = _training.ProgressPct / 100.0;
                    break;
                case nameof(TrainingViewModel.BestLoss):
                    if (_training.BestLoss < double.MaxValue)
                        _dashboard.TrainingStatusText = $"Best loss: {_training.BestLoss:F6}";
                    break;
            }
        };

        // Feed HM status to dashboard status cards
        _historyMatching.PropertyChanged += (_, e) =>
        {
            switch (e.PropertyName)
            {
                case nameof(HistoryMatchingViewModel.IsRunning):
                    _dashboard.IsHmActive = _historyMatching.IsRunning;
                    break;
                case nameof(HistoryMatchingViewModel.CurrentIteration):
                    _dashboard.HmIteration = _historyMatching.CurrentIteration;
                    break;
                case nameof(HistoryMatchingViewModel.CurrentMismatch):
                    _dashboard.HmMismatch = _historyMatching.CurrentMismatch;
                    break;
            }
        };

        // Default view
        CurrentView = _dashboard;

        // Subscribe to engine status
        _engineManager.StatusChanged += OnEngineStatusChanged;

        // Start engine
        _ = StartEngineAsync();
    }

    // ── Navigation commands ────────────────────────────────────────────────────

    [RelayCommand]
    private void NavigateToDashboard()
    {
        Log.Information("[NAV] Navigate → Dashboard");
        CurrentView = _dashboard;
    }

    [RelayCommand]
    private void NavigateToProjectSetup()
    {
        Log.Information("[NAV] Navigate → ProjectSetup (vm type={Type})", _projectSetup?.GetType().Name ?? "null");
        CurrentView = _projectSetup;
        Log.Information("[NAV] CurrentView is now {Type}", CurrentView?.GetType().Name ?? "null");
    }

    [RelayCommand]
    private void NavigateToTraining()
    {
        Log.Information("[NAV] Navigate → Training");
        CurrentView = _training;
    }

    [RelayCommand]
    private void NavigateToHistoryMatching()
    {
        Log.Information("[NAV] Navigate → HistoryMatching");
        CurrentView = _historyMatching;
    }

    [RelayCommand]
    private void NavigateToForecast()
    {
        Log.Information("[NAV] Navigate → Forecast");
        CurrentView = _forecast;
    }

    [RelayCommand]
    private void NavigateTo3DViewer()
    {
        Log.Information("[NAV] Navigate → 3DViewer");
        CurrentView = _reservoirView3D;
    }

    [RelayCommand]
    private void NavigateToCrossSection()
    {
        Log.Information("[NAV] Navigate → CrossSection");
        CurrentView = _crossSection;
    }

    // ── AI panel toggle ───────────────────────────────────────────────────────

    [RelayCommand]
    private void ToggleAssistant()
    {
        AssistantPanelVisible = !AssistantPanelVisible;
        AssistantPanelWidth = AssistantPanelVisible
            ? new GridLength(360, GridUnitType.Pixel)
            : new GridLength(0, GridUnitType.Pixel);
    }

    // ── Open project file dialog ──────────────────────────────────────────────

    private void OpenProjectFile()
    {
        var dlg = new Microsoft.Win32.OpenFileDialog
        {
            Title           = "Open PhysicsFlow Project",
            Filter          = "PhysicsFlow Project (*.pfproj)|*.pfproj|All files (*.*)|*.*",
            DefaultExt      = ".pfproj",
            CheckFileExists = true,
        };

        if (dlg.ShowDialog() != true) return;

        var path = dlg.FileName;
        try
        {
            var json    = System.IO.File.ReadAllText(path);
            var doc     = JsonDocument.Parse(json);
            var root    = doc.RootElement;

            // Project name
            var name = root.TryGetProperty("name", out var np) ? np.GetString() : null;
            name ??= System.IO.Path.GetFileNameWithoutExtension(path);

            // Grid dimensions
            int nx = 46, ny = 112, nz = 22;
            if (root.TryGetProperty("grid", out var grid))
            {
                if (grid.TryGetProperty("nx", out var v)) nx = v.GetInt32();
                if (grid.TryGetProperty("ny", out v))     ny = v.GetInt32();
                if (grid.TryGetProperty("nz", out v))     nz = v.GetInt32();
            }

            // Well count
            int wellCount = 0;
            if (root.TryGetProperty("wells", out var wells))
                wellCount = wells.GetArrayLength();

            // HM results
            bool hmComplete  = false;
            int  hmIter      = 0;
            double hmMismatch = 0;
            if (root.TryGetProperty("hm_results", out var hm))
            {
                if (hm.TryGetProperty("completed", out var hmc))   hmComplete  = hmc.GetBoolean();
                if (hm.TryGetProperty("iterations", out var hmi))  hmIter      = hmi.GetInt32();
                if (hm.TryGetProperty("mismatch",   out var hmm))  hmMismatch  = hmm.GetDouble();
            }

            // Apply to dashboard
            CurrentProjectName              = name;
            AIAssistant.CurrentProjectPath  = path;
            _dashboard.CurrentProjectName   = name;
            _dashboard.GridNx               = nx;
            _dashboard.GridNy               = ny;
            _dashboard.GridNz               = nz;
            _dashboard.TotalWells           = wellCount;
            _dashboard.HmIteration          = hmIter;
            _dashboard.HmMismatch           = hmMismatch;
            _dashboard.StatusMessage        = $"Project '{name}' loaded successfully";

            Log.Information("[PROJECT] Loaded '{Name}' ({Nx}x{Ny}x{Nz}, {Wells} wells)", name, nx, ny, nz, wellCount);
        }
        catch (Exception ex)
        {
            _dashboard.StatusMessage = $"Failed to open project: {ex.Message}";
            Log.Warning(ex, "[PROJECT] Failed to open {Path}", path);
        }

        // Always navigate to dashboard so user sees the updated state
        CurrentView = _dashboard;
    }

    // ── Settings ──────────────────────────────────────────────────────────────

    [RelayCommand]
    private void OpenSettings() => _openSettingsDialog();

    // ── Engine management ─────────────────────────────────────────────────────

    private async Task StartEngineAsync()
    {
        try
        {
            EngineStatusText = "Starting engine...";
            EngineStatusColor = "#E67E22";
            await _engineManager.StartAsync();
        }
        catch (Exception ex)
        {
            EngineStatusText = "Engine failed";
            EngineStatusColor = "#E74C3C";
            System.Diagnostics.Debug.WriteLine($"Engine start error: {ex.Message}");
        }
    }

    private void OnEngineStatusChanged(object? sender, EngineStatusEventArgs e)
    {
        Application.Current.Dispatcher.Invoke(() =>
        {
            (EngineStatusText, EngineStatusColor) = e.Status switch
            {
                EngineStatus.Running  => ("Engine ready", "#2ECC71"),
                EngineStatus.Starting => ("Starting...",  "#E67E22"),
                EngineStatus.Stopped  => ("Engine stopped","#95A5A6"),
                EngineStatus.Error    => ("Engine error",  "#E74C3C"),
                _                     => ("Unknown",       "#95A5A6"),
            };
        });
    }
}
