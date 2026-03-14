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
        var m = new PlotModel
        {
            Title               = "Ensemble Fan Chart — P10/P50/P90",
            Background          = OxyColor.FromRgb(0x0D, 0x1B, 0x2E),
            PlotAreaBorderColor = OxyColor.FromRgb(0x2A, 0x3A, 0x4A),
            TextColor           = OxyColor.FromRgb(0x88, 0x99, 0xAA),
        };
        m.Axes.Add(new DateTimeAxis
        {
            Position      = AxisPosition.Bottom,
            Title         = "Date",
            StringFormat  = "MMM-yy",
            TextColor     = OxyColor.FromRgb(0x88, 0x99, 0xAA),
            AxislineColor = OxyColor.FromRgb(0x2A, 0x3A, 0x4A),
        });
        m.Axes.Add(new LinearAxis
        {
            Position      = AxisPosition.Left,
            Title         = "Rate (stb/day)",
            Minimum       = 0,
            TextColor     = OxyColor.FromRgb(0x88, 0x99, 0xAA),
            AxislineColor = OxyColor.FromRgb(0x2A, 0x3A, 0x4A),
        });
        return m;
    }

    /// <summary>Rebuild the fan chart for the selected well and quantity.</summary>
    private void RebuildFanChart()
    {
        var m   = FanChartModel;
        var rng = new Random((SelectedFanWell ?? "B-1H").GetHashCode()
                             ^ (SelectedQuantity ?? "").GetHashCode());

        m.Series.Clear();
        m.Axes.Clear();

        // Date axis: Norne production history 1997–2006
        m.Axes.Add(new DateTimeAxis
        {
            Position      = AxisPosition.Bottom,
            Title         = "Date",
            StringFormat  = "MMM-yy",
            TextColor     = OxyColor.FromRgb(0x88, 0x99, 0xAA),
            AxislineColor = OxyColor.FromRgb(0x2A, 0x3A, 0x4A),
            Minimum       = DateTimeAxis.ToDouble(new DateTime(1997, 1, 1)),
            Maximum       = DateTimeAxis.ToDouble(new DateTime(2006, 12, 1)),
        });

        // Choose y-axis scale and title based on quantity
        bool isOil   = SelectedQuantity?.StartsWith("Oil")   == true;
        bool isWater = SelectedQuantity?.StartsWith("Water") == true;
        bool isGas   = SelectedQuantity?.StartsWith("Gas")   == true;
        bool isBhp   = SelectedQuantity?.StartsWith("BHP")   == true;

        string yTitle = isGas ? "Rate (Mscf/day)" : isBhp ? "BHP (bar)" : "Rate (stb/day)";
        m.Axes.Add(new LinearAxis
        {
            Position      = AxisPosition.Left,
            Title         = yTitle,
            Minimum       = 0,
            TextColor     = OxyColor.FromRgb(0x88, 0x99, 0xAA),
            AxislineColor = OxyColor.FromRgb(0x2A, 0x3A, 0x4A),
        });

        // Generate monthly time points
        var dates = new List<DateTime>();
        var d     = new DateTime(1997, 1, 1);
        while (d <= new DateTime(2006, 12, 1)) { dates.Add(d); d = d.AddMonths(1); }
        int n = dates.Count;

        // Base rates per well/quantity (representative Norne values)
        double baseRate = isOil   ? 4000 + rng.NextDouble() * 3000
                        : isWater ? 1000 + rng.NextDouble() * 2000
                        : isGas   ? 5000 + rng.NextDouble() * 4000
                        : 240 + rng.NextDouble() * 40;   // BHP bar

        // Generate P10/P50/P90 curves with realistic production decline + water breakthrough
        var p10 = new double[n]; var p50 = new double[n]; var p90 = new double[n];
        var obs = new double[n];

        for (int i = 0; i < n; i++)
        {
            double t      = (double)i / n;
            double rampUp = Math.Min(1.0, i / 6.0);           // 6-month ramp-up
            double decline = isOil   ? Math.Exp(-1.8 * t)     // hyperbolic oil decline
                           : isWater ? Math.Min(1.0, t * 1.5) // water increasing
                           : isGas   ? Math.Exp(-1.2 * t)
                           : 1.0 - 0.1 * t;                   // BHP slightly declining

            double mid  = baseRate * rampUp * decline;
            double spread = mid * (0.15 + 0.10 * (1 - t));    // spread narrows over time

            p50[i] = Math.Max(0, mid + rng.NextDouble() * 40 - 20);
            p10[i] = Math.Max(0, p50[i] - spread * (0.8 + rng.NextDouble() * 0.4));
            p90[i] = p50[i] + spread * (0.8 + rng.NextDouble() * 0.4);
            obs[i] = Math.Max(0, p50[i] * (0.92 + rng.NextDouble() * 0.16)
                              + (rng.NextDouble() - 0.5) * spread * 0.3);
        }

        // P10–P90 uncertainty band (AreaSeries)
        var band = new AreaSeries
        {
            Title            = "P10–P90",
            Fill             = OxyColor.FromArgb(0x44, 0x88, 0xFF, 50),
            Color            = OxyColor.FromArgb(0x44, 0x88, 0xFF, 0),
            Color2           = OxyColor.FromArgb(0x44, 0x88, 0xFF, 0),
            StrokeThickness  = 0,
        };
        for (int i = 0; i < n; i++)
        {
            var x = DateTimeAxis.ToDouble(dates[i]);
            band.Points.Add(new DataPoint(x, p90[i]));
            band.Points2.Add(new DataPoint(x, p10[i]));
        }
        m.Series.Add(band);

        // P10 and P90 boundary lines (dashed)
        foreach (var (vals, label, col) in new[]
        {
            (p90, "P10",  OxyColor.FromArgb(160, 0x44, 0x88, 0xFF)),
            (p10, "P90",  OxyColor.FromArgb(160, 0x44, 0x88, 0xFF)),
        })
        {
            var ls = new LineSeries
            {
                Title = label, Color = col,
                StrokeThickness = 1, LineStyle = LineStyle.Dash,
            };
            for (int i = 0; i < n; i++)
                ls.Points.Add(new DataPoint(DateTimeAxis.ToDouble(dates[i]), vals[i]));
            m.Series.Add(ls);
        }

        // P50 median line (solid blue)
        var p50Line = new LineSeries
        {
            Title           = "P50 (median)",
            Color           = OxyColors.DodgerBlue,
            StrokeThickness = 2,
        };
        for (int i = 0; i < n; i++)
            p50Line.Points.Add(new DataPoint(DateTimeAxis.ToDouble(dates[i]), p50[i]));
        m.Series.Add(p50Line);

        // Observed data (scatter — orange dots)
        var obsSeries = new ScatterSeries
        {
            Title          = "Observed",
            MarkerType     = MarkerType.Circle,
            MarkerSize     = 3,
            MarkerFill     = OxyColor.FromRgb(0xFF, 0x8C, 0x00),
        };
        for (int i = 0; i < n; i += 2)   // every 2 months for clarity
            obsSeries.Points.Add(new ScatterPoint(DateTimeAxis.ToDouble(dates[i]), obs[i]));
        m.Series.Add(obsSeries);

        m.Title = $"{SelectedFanWell} — {SelectedQuantity}";
        m.InvalidatePlot(true);
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
        _initialMismatch = mm;

        for (int it = 1; it <= MaxIterations && !ct.IsCancellationRequested; it++)
        {
            await Task.Delay(50, ct);   // stub delay; real engine uses gRPC streaming

            // Proper αREKI schedule: alpha_t = 1/N so sum over all N iterations = 1.0
            // Add small noise to simulate adaptive step selection
            var rng2  = new Random(it * 13);
            double alpha = (1.0 / MaxIterations) * (0.9 + rng.NextDouble() * 0.2);
            sCum += alpha;

            // Mismatch decays quickly early, slows as it approaches noise floor
            double decay = it <= 5
                ? (0.72 + rng.NextDouble() * 0.10)   // fast initial improvement
                : (0.88 + rng.NextDouble() * 0.08);  // slower refinement
            mm = Math.Max(0.04, mm * decay);

            CurrentIteration = it;
            CurrentMismatch  = mm;
            CurrentAlpha     = alpha;
            SCumulative      = sCum;
            ProgressPct      = (double)it / MaxIterations * 100.0;
            ImprovementPct   = (1.0 - mm / _initialMismatch) * 100.0;
            StatusMessage    = $"Iteration {it}/{MaxIterations}  |  mismatch: {mm:F4}  |  α: {alpha:F4}  |  s_cum: {sCum:F3}";

            _mismatchSeries.Points.Add(new DataPoint(it, mm));
            MismatchPlotModel.InvalidatePlot(true);

            // Converge only when s_cumulative >= 1.0 AND mismatch is low,
            // or when mismatch drops below noise floor
            var converged = (sCum >= 1.0 && mm < 0.15) || mm < 0.05;
            IterationLog.Insert(0,
                new HmIterationItem(it, mm, alpha, sCum, converged));

            if (converged)
            {
                IsConverged      = true;
                StatusMessage    = $"Converged at iteration {it} — mismatch: {mm:F4}";
                ConvergenceLabel = $"Converged  (s_cumulative={sCum:F3})";
                BuildWellRmseTable(rng);
                OnPropertyChanged(nameof(HasResults));
                break;
            }
        }

        if (!IsConverged && !ct.IsCancellationRequested)
        {
            StatusMessage    = $"Max iterations reached — mismatch: {CurrentMismatch:F4}";
            ConvergenceLabel = $"Not converged (s_cumulative={SCumulative:F3})";
            BuildWellRmseTable(rng);
            OnPropertyChanged(nameof(HasResults));
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
        RebuildFanChart();
    }

    partial void OnSelectedFanWellChanged(string? value)  => RebuildFanChart();
    partial void OnSelectedQuantityChanged(string value)  => RebuildFanChart();
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
