using System.Collections.ObjectModel;
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
        _projectSetup.ProjectSaved    += (_, path) => { CurrentProjectName = System.IO.Path.GetFileNameWithoutExtension(path); CurrentView = _dashboard; };
        _projectSetup.WizardCancelled += (_, _)    => CurrentView = _dashboard;

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
