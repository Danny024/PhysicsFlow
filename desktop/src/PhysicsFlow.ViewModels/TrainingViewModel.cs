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
/// ViewModel for the PINO Training Monitor view.
/// Streams live loss metrics from the Python engine and plots
/// total loss, PDE loss, and data loss curves in real time.
/// </summary>
public partial class TrainingViewModel : ObservableObject
{
    private readonly AppDbService?             _db;
    private readonly ILogger<TrainingViewModel>? _log;
    private CancellationTokenSource?           _cts;

    // ── Training state ────────────────────────────────────────────────────────

    [ObservableProperty] private bool   _isTraining;
    [ObservableProperty] private bool   _isPaused;
    [ObservableProperty] private int    _currentEpoch;
    [ObservableProperty] private int    _totalEpochs = 200;
    [ObservableProperty] private double _progressPct;
    [ObservableProperty] private double _currentLossTotal;
    [ObservableProperty] private double _currentLossPde;
    [ObservableProperty] private double _currentLossData;
    [ObservableProperty] private double _bestLoss = double.MaxValue;
    [ObservableProperty] private string _bestLossEpochLabel = "—";
    [ObservableProperty] private string _statusMessage = "Ready to train";
    [ObservableProperty] private string _elapsedTime  = "00:00:00";
    [ObservableProperty] private string _etaTime      = "—";
    [ObservableProperty] private string _deviceLabel  = "CPU";

    // ── Hyperparameter inputs ─────────────────────────────────────────────────

    [ObservableProperty] private int    _configEpochs      = 200;
    [ObservableProperty] private int    _configBatchSize   = 4;
    [ObservableProperty] private double _configLr          = 1e-3;
    [ObservableProperty] private int    _configWidth       = 32;
    [ObservableProperty] private int    _configModes       = 8;
    [ObservableProperty] private double _configWPde        = 0.1;
    [ObservableProperty] private double _configWData       = 1.0;
    [ObservableProperty] private string _configDevice      = "cuda";
    [ObservableProperty] private string _configEnsemble    = "500";
    [ObservableProperty] private string _configDeckPath    = string.Empty;

    // ── OxyPlot loss curves ───────────────────────────────────────────────────

    [ObservableProperty] private PlotModel _lossPlotModel;

    private readonly LineSeries _seriesTotal = new()
    {
        Title = "Total Loss", Color = OxyColors.DodgerBlue, StrokeThickness = 2
    };
    private readonly LineSeries _seriesPde = new()
    {
        Title = "PDE Loss",   Color = OxyColors.OrangeRed, StrokeThickness = 1.5,
        LineStyle = LineStyle.Dash
    };
    private readonly LineSeries _seriesData = new()
    {
        Title = "Data Loss",  Color = OxyColors.SeaGreen, StrokeThickness = 1.5,
        LineStyle = LineStyle.Dot
    };

    public ObservableCollection<EpochLogItem> EpochLog { get; } = new();

    public TrainingViewModel(
        AppDbService?               db  = null,
        ILogger<TrainingViewModel>? log = null)
    {
        _db  = db;
        _log = log;
        _lossPlotModel = BuildPlotModel();
    }

    // ── Commands ──────────────────────────────────────────────────────────────

    [RelayCommand(CanExecute = nameof(CanStartTraining))]
    private async Task StartTrainingAsync()
    {
        _cts = new CancellationTokenSource();
        IsTraining = true;
        IsPaused   = false;
        StatusMessage = "Training in progress...";
        EpochLog.Clear();
        _seriesTotal.Points.Clear();
        _seriesPde.Points.Clear();
        _seriesData.Points.Clear();

        var sw = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            // In production: call the Python engine via gRPC and stream epochs.
            // Here we simulate progress until the gRPC streaming client is wired up.
            await SimulateTrainingAsync(_cts.Token, sw);
        }
        catch (OperationCanceledException)
        {
            StatusMessage = IsPaused ? "Paused" : "Training stopped";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
            _log?.LogError(ex, "Training failed");
        }
        finally
        {
            IsTraining = false;
            sw.Stop();
        }
    }

    private bool CanStartTraining() => !IsTraining;

    [RelayCommand(CanExecute = nameof(CanPause))]
    private void PauseTraining()
    {
        IsPaused = true;
        StatusMessage = "Paused — click Resume to continue";
    }

    private bool CanPause() => IsTraining && !IsPaused;

    [RelayCommand(CanExecute = nameof(CanResume))]
    private void ResumeTraining()
    {
        IsPaused = false;
        StatusMessage = "Training in progress...";
    }

    private bool CanResume() => IsTraining && IsPaused;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void StopTraining()
    {
        _cts?.Cancel();
        StatusMessage = "Stopping...";
    }

    private bool CanStop() => IsTraining;

    [RelayCommand]
    private void ClearLog()
    {
        EpochLog.Clear();
    }

    // ── Plot construction ─────────────────────────────────────────────────────

    private PlotModel BuildPlotModel()
    {
        var model = new PlotModel { Title = "Training Loss" };
        model.Axes.Add(new LinearAxis
        {
            Position = AxisPosition.Bottom,
            Title    = "Epoch",
            Minimum  = 0,
        });
        model.Axes.Add(new LogarithmicAxis
        {
            Position = AxisPosition.Left,
            Title    = "Loss",
            Minimum  = 1e-6,
        });
        model.Series.Add(_seriesTotal);
        model.Series.Add(_seriesPde);
        model.Series.Add(_seriesData);
        return model;
    }

    private void AppendEpochToPlot(int epoch, double total, double pde, double data)
    {
        _seriesTotal.Points.Add(new DataPoint(epoch, total));
        _seriesPde.Points.Add(new DataPoint(epoch, pde));
        _seriesData.Points.Add(new DataPoint(epoch, data));
        LossPlotModel.InvalidatePlot(true);
    }

    // ── Simulation stub (replaced by gRPC streaming in production) ────────────

    private async Task SimulateTrainingAsync(CancellationToken ct, System.Diagnostics.Stopwatch sw)
    {
        var rng = new Random(42);
        TotalEpochs = ConfigEpochs;

        double loss = 0.8;
        for (int epoch = 1; epoch <= TotalEpochs && !ct.IsCancellationRequested; epoch++)
        {
            while (IsPaused && !ct.IsCancellationRequested)
                await Task.Delay(200, ct);

            // Simulated loss decay
            loss     *= (0.985 + rng.NextDouble() * 0.01);
            var pde   = loss * 0.15 * (1 + rng.NextDouble() * 0.2);
            var data  = loss * 0.80 * (1 + rng.NextDouble() * 0.05);

            CurrentEpoch      = epoch;
            CurrentLossTotal  = loss;
            CurrentLossPde    = pde;
            CurrentLossData   = data;
            ProgressPct       = (double)epoch / TotalEpochs * 100.0;
            ElapsedTime       = sw.Elapsed.ToString(@"hh\:mm\:ss");

            if (loss < BestLoss)
            {
                BestLoss          = loss;
                BestLossEpochLabel = $"Epoch {epoch}: {loss:F6}";
            }

            var remaining = TimeSpan.FromSeconds(
                sw.Elapsed.TotalSeconds / epoch * (TotalEpochs - epoch));
            EtaTime = remaining.ToString(@"hh\:mm\:ss");

            AppendEpochToPlot(epoch, loss, pde, data);
            EpochLog.Insert(0, new EpochLogItem(epoch, loss, pde, data,
                System.DateTime.Now.ToString("HH:mm:ss")));

            // Keep log to 200 rows
            while (EpochLog.Count > 200)
                EpochLog.RemoveAt(EpochLog.Count - 1);

            await Task.Delay(20, ct);  // ~50 fps update rate
        }

        if (!ct.IsCancellationRequested)
            StatusMessage = $"Training complete — best loss: {BestLoss:F6}";
    }
}

// ── Supporting record ──────────────────────────────────────────────────────────

public record EpochLogItem(int Epoch, double Total, double Pde, double Data, string Time);
