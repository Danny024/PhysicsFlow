using System.Collections.ObjectModel;
using System.Threading;
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
/// ViewModel for the αREKI History Matching workspace.
/// Shows the mismatch convergence plot, per-well fan charts,
/// and per-well RMSE heatmap. Drives the history matching via
/// the Python gRPC engine.
/// </summary>
public partial class HistoryMatchingViewModel : ObservableObject
{
    private readonly AppDbService?                    _db;
    private readonly ILogger<HistoryMatchingViewModel>? _log;
    private CancellationTokenSource?                  _cts;
    private double _initialMismatch = 1.0;

    // ── HM state ──────────────────────────────────────────────────────────────

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanStart))]
    private bool _isRunning;

    [ObservableProperty] private bool   _isConverged;
    [ObservableProperty] private int    _currentIteration;
    [ObservableProperty] private int    _maxIterations     = 20;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DataMismatch))]
    private double _currentMismatch;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Alpha))]
    private double _currentAlpha;

    [ObservableProperty] private double _sCumulative;
    [ObservableProperty] private double _progressPct;
    [ObservableProperty] private double _improvementPct;
    [ObservableProperty] private string _statusMessage     = "Ready";
    [ObservableProperty] private string _convergenceLabel  = "Not started";
    [ObservableProperty] private string _hmRunId          = string.Empty;

    // ── Configuration ─────────────────────────────────────────────────────────

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(EnsembleSize))]
    private int _configEnsembleSize = 200;

    [ObservableProperty] private int    _configMaxIterations   = 20;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(LocalisationRadius))]
    private double _configLocRadius = 12.0;   // cells (view slider: 2–40)

    [ObservableProperty] private double _configNoiseInflation  = 0.05;
    [ObservableProperty] private string _configDevice          = "cuda";

    // ── Optional feature flags (bound to CheckBoxes in view) ─────────────────

    [ObservableProperty] private bool _useGenerativePriors;
    [ObservableProperty] private bool _useCCR;
    [ObservableProperty] private bool _autoLocalisation;

    // ── View-compatible property aliases ──────────────────────────────────────

    /// <summary>View binds to EnsembleSize (two-way via NumericUpDown).</summary>
    public int EnsembleSize
    {
        get => ConfigEnsembleSize;
        set => ConfigEnsembleSize = value;
    }

    /// <summary>View binds to LocalisationRadius (two-way via Slider).</summary>
    public double LocalisationRadius
    {
        get => ConfigLocRadius;
        set => ConfigLocRadius = value;
    }

    /// <summary>View binds DataMismatch stat card to CurrentMismatch.</summary>
    public double DataMismatch => CurrentMismatch;

    /// <summary>View binds Alpha stat card to CurrentAlpha.</summary>
    public double Alpha => CurrentAlpha;

    /// <summary>View binds CanStart to IsEnabled on the Start button.</summary>
    public bool CanStart => !IsRunning;

    /// <summary>View enables Export button only when results exist.</summary>
    public bool HasResults => WellMismatches.Count > 0;

    /// <summary>View binds ConvergencePlotModel to the mismatch convergence chart.</summary>
    public PlotModel ConvergencePlotModel => MismatchPlotModel;

    /// <summary>View binds ProducerNames to the fan-chart well selector.</summary>
    public ObservableCollection<string> ProducerNames => WellNames;

    /// <summary>View binds Quantities to the quantity selector ComboBox.</summary>
    public ObservableCollection<string> Quantities => QuantityOptions;

    // ── Well selection ────────────────────────────────────────────────────────

    [ObservableProperty] private string?  _selectedWell;
    [ObservableProperty] private string?  _selectedFanWell;
    [ObservableProperty] private string   _selectedQuantity = "Oil Rate (stb/day)";

    public ObservableCollection<string> WellNames { get; } = new();
    public ObservableCollection<string> QuantityOptions { get; } = new()
        { "Oil Rate (stb/day)", "Water Rate (stb/day)", "Gas Rate (Mscf/day)", "BHP (bar)" };

    // ── OxyPlot models ────────────────────────────────────────────────────────

    [ObservableProperty] private PlotModel _mismatchPlotModel;
    [ObservableProperty] private PlotModel _fanChartModel;
    [ObservableProperty] private PlotModel _rmseHeatmapModel;

    private readonly LineSeries _mismatchSeries = new()
    {
        Title           = "Normalised Mismatch",
        Color           = OxyColors.DodgerBlue,
        StrokeThickness = 2,
        MarkerType      = MarkerType.Circle,
        MarkerSize      = 4,
    };

    // ── Per-well collections ──────────────────────────────────────────────────

    public ObservableCollection<WellRmseItem>      WellRmseTable  { get; } = new();
    public ObservableCollection<WellMismatchItem>  WellMismatches { get; } = new();

    // ── Iteration log ─────────────────────────────────────────────────────────

    public ObservableCollection<HmIterationItem> IterationLog { get; } = new();

    public HistoryMatchingViewModel(
        AppDbService?                       db  = null,
        ILogger<HistoryMatchingViewModel>?  log = null)
    {
        _db  = db;
        _log = log;

        _mismatchPlotModel = BuildMismatchPlot();
        _fanChartModel     = BuildFanChart();
        _rmseHeatmapModel  = BuildRmseHeatmap();

        // Populate representative Norne wells
        foreach (var w in new[]
        {
            "B-1H","B-2H","B-4BH","B-4DH","D-1CH","D-2H","D-3BH","D-4H",
            "E-1H","E-2H","E-3AH","E-3CH","E-4AH","K-3H"
        })
            WellNames.Add(w);

        if (WellNames.Count > 0)
        {
            SelectedWell    = WellNames[0];
            SelectedFanWell = WellNames[0];
        }
    }

    // ── Commands ──────────────────────────────────────────────────────────────

    /// <summary>View binds StartHMCommand (RelayCommand name matches method prefix).</summary>
    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task StartHMAsync()
    {
        _cts          = new CancellationTokenSource();
        IsRunning     = true;
        IsConverged   = false;
        StatusMessage = "Running αREKI...";
        MaxIterations = ConfigMaxIterations;
        IterationLog.Clear();
        WellRmseTable.Clear();
        WellMismatches.Clear();
        _mismatchSeries.Points.Clear();
        _initialMismatch = 1.0;
        ImprovementPct   = 0.0;
        OnPropertyChanged(nameof(HasResults));

        try
        {
            await SimulateHmAsync(_cts.Token);
        }
        catch (OperationCanceledException)
        {
            StatusMessage = "Stopped by user";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
            _log?.LogError(ex, "HM run failed");
        }
        finally
        {
            IsRunning = false;
        }
    }

    /// <summary>View binds StopHMCommand.</summary>
    [RelayCommand]
    private void StopHM()
    {
        _cts?.Cancel();
        StatusMessage = "Stopping...";
    }

    [RelayCommand]
    private void ExportResults()
    {
        StatusMessage = "Export — not yet implemented.";
    }

    [RelayCommand]
    private async Task LoadResultsFromDbAsync()
    {
        if (_db is null || string.IsNullOrEmpty(HmRunId)) return;
        try
        {
            var history = await _db.GetHMHistoryAsync(
                projectId: string.Empty, hmRunId: HmRunId);
            IterationLog.Clear();
            _mismatchSeries.Points.Clear();
            foreach (var it in history)
            {
                _mismatchSeries.Points.Add(new DataPoint(it.Iteration, it.Mismatch));
                IterationLog.Add(new HmIterationItem(it.Iteration, it.Mismatch,
                    it.Alpha, it.SCumulative, it.Converged));
            }
            MismatchPlotModel.InvalidatePlot(true);
        }
        catch (Exception ex)
        {
            _log?.LogWarning(ex, "Failed to load HM history");
        }
    }

    // ── Plot builders ─────────────────────────────────────────────────────────

    private PlotModel BuildMismatchPlot()
    {
        var m = new PlotModel { Title = "Mismatch Convergence (αREKI)" };
        m.Axes.Add(new LinearAxis
            { Position = AxisPosition.Bottom, Title = "Iteration", Minimum = 0 });
        m.Axes.Add(new LinearAxis
            { Position = AxisPosition.Left, Title = "Normalised Mismatch", Minimum = 0 });
        m.Series.Add(_mismatchSeries);
        return m;
    }

    private static PlotModel BuildFanChart()
    {
        var m = new PlotModel { Title = "Ensemble Fan Chart — P10/P50/P90" };
        m.Axes.Add(new DateTimeAxis { Position = AxisPosition.Bottom, Title = "Date" });
        m.Axes.Add(new LinearAxis  { Position = AxisPosition.Left,   Title = "Rate (stb/day)" });
        return m;
    }

    private static PlotModel BuildRmseHeatmap()
    {
        var m = new PlotModel { Title = "Per-Well RMSE" };
        m.Axes.Add(new CategoryAxis { Position = AxisPosition.Left,   Title = "Well" });
        m.Axes.Add(new LinearAxis   { Position = AxisPosition.Bottom, Title = "RMSE" });
        return m;
    }

    // ── Simulation stub (replaced by gRPC streaming in production) ────────────

    private async Task SimulateHmAsync(CancellationToken ct)
    {
        var rng      = new Random(7);
        double mm    = 1.0;
        double sCum  = 0.0;
        double alpha = 1.0;
        _initialMismatch = mm;

        for (int it = 1; it <= MaxIterations && !ct.IsCancellationRequested; it++)
        {
            await Task.Delay(600, ct);

            mm    *= (0.7 + rng.NextDouble() * 0.15);
            alpha  = Math.Max(0.05, alpha * 0.75);
            sCum  += alpha;

            CurrentIteration = it;
            CurrentMismatch  = mm;
            CurrentAlpha     = alpha;
            SCumulative      = sCum;
            ProgressPct      = (double)it / MaxIterations * 100.0;
            ImprovementPct   = (1.0 - mm / _initialMismatch) * 100.0;

            _mismatchSeries.Points.Add(new DataPoint(it, mm));
            MismatchPlotModel.InvalidatePlot(true);

            var converged = sCum >= 1.0 || mm < 0.05;
            IterationLog.Insert(0,
                new HmIterationItem(it, mm, alpha, sCum, converged));

            if (converged)
            {
                IsConverged      = true;
                StatusMessage    = $"Converged at iteration {it} — mismatch: {mm:F4}";
                ConvergenceLabel = $"✅ Converged  (s_cumulative={sCum:F3})";
                BuildWellRmseTable(rng);
                break;
            }
        }

        if (!IsConverged && !ct.IsCancellationRequested)
        {
            StatusMessage    = $"Max iterations reached — mismatch: {CurrentMismatch:F4}";
            ConvergenceLabel = $"⚠ Not converged (s_cumulative={SCumulative:F3})";
            BuildWellRmseTable(rng);
        }
    }

    private void BuildWellRmseTable(Random rng)
    {
        WellRmseTable.Clear();
        WellMismatches.Clear();

        foreach (var w in WellNames)
        {
            var rmse  = 0.02 + rng.NextDouble() * 0.18;
            var rmseR = Math.Round(rmse, 4);
            var qual  = rmse < 0.08 ? "Good" : rmse < 0.14 ? "Fair" : "Poor";
            var color = rmse < 0.08 ? "#2ECC71" : rmse < 0.14 ? "#E67E22" : "#E74C3C";

            WellRmseTable.Add(new WellRmseItem(w, rmseR, qual));
            WellMismatches.Add(new WellMismatchItem(w, rmseR, color));
        }

        OnPropertyChanged(nameof(HasResults));
    }
}

// ── Supporting records ────────────────────────────────────────────────────────

public record HmIterationItem(
    int    Iteration,
    double Mismatch,
    double Alpha,
    double SCumulative,
    bool   Converged);

public record WellRmseItem(string WellName, double Rmse, string Quality);

public record WellMismatchItem(string WellName, double Rmse, string MismatchColor);
