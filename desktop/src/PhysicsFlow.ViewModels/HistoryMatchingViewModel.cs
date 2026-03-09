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

    // ── HM state ──────────────────────────────────────────────────────────────

    [ObservableProperty] private bool   _isRunning;
    [ObservableProperty] private bool   _isConverged;
    [ObservableProperty] private int    _currentIteration;
    [ObservableProperty] private int    _maxIterations     = 20;
    [ObservableProperty] private double _currentMismatch;
    [ObservableProperty] private double _currentAlpha;
    [ObservableProperty] private double _sCumulative;
    [ObservableProperty] private double _progressPct;
    [ObservableProperty] private string _statusMessage     = "Ready";
    [ObservableProperty] private string _convergenceLabel  = "Not started";
    [ObservableProperty] private string _hmRunId          = string.Empty;

    // ── Configuration ─────────────────────────────────────────────────────────

    [ObservableProperty] private int    _configEnsembleSize    = 200;
    [ObservableProperty] private int    _configMaxIterations   = 20;
    [ObservableProperty] private double _configLocRadius       = 0.4;
    [ObservableProperty] private double _configNoiseInflation  = 0.05;
    [ObservableProperty] private string _configDevice          = "cuda";

    // ── Well selection ────────────────────────────────────────────────────────

    [ObservableProperty] private string?  _selectedWell;
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

    // ── Per-well RMSE table ───────────────────────────────────────────────────

    public ObservableCollection<WellRmseItem> WellRmseTable { get; } = new();

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
            SelectedWell = WellNames[0];
    }

    // ── Commands ──────────────────────────────────────────────────────────────

    [RelayCommand(CanExecute = nameof(CanStart))]
    private async Task StartHmAsync()
    {
        _cts          = new CancellationTokenSource();
        IsRunning     = true;
        IsConverged   = false;
        StatusMessage = "Running αREKI...";
        MaxIterations = ConfigMaxIterations;
        IterationLog.Clear();
        WellRmseTable.Clear();
        _mismatchSeries.Points.Clear();

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

    private bool CanStart() => !IsRunning;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void StopHm()
    {
        _cts?.Cancel();
        StatusMessage = "Stopping...";
    }

    private bool CanStop() => IsRunning;

    [RelayCommand]
    private async Task LoadResultsFromDbAsync()
    {
        if (_db is null || string.IsNullOrEmpty(HmRunId)) return;
        try
        {
            // GetHMHistoryAsync is the correct method name in AppDbService
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

        for (int it = 1; it <= MaxIterations && !ct.IsCancellationRequested; it++)
        {
            await Task.Delay(600, ct);

            // Simulated mismatch reduction
            mm    *= (0.7 + rng.NextDouble() * 0.15);
            alpha  = Math.Max(0.05, alpha * 0.75);
            sCum  += alpha;

            CurrentIteration = it;
            CurrentMismatch  = mm;
            CurrentAlpha     = alpha;
            SCumulative      = sCum;
            ProgressPct      = (double)it / MaxIterations * 100.0;

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
        foreach (var w in WellNames)
        {
            var rmse = 0.02 + rng.NextDouble() * 0.18;
            WellRmseTable.Add(new WellRmseItem(w, Math.Round(rmse, 4),
                rmse < 0.08 ? "Good" : rmse < 0.14 ? "Fair" : "Poor"));
        }
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
