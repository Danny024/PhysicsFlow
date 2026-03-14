using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
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
/// All property names match the bindings in TrainingView.xaml exactly.
/// </summary>
public partial class TrainingViewModel : ObservableObject
{
    private readonly AppDbService?              _db;
    private readonly ILogger<TrainingViewModel>? _log;
    private CancellationTokenSource?            _cts;

    // ── Training state ────────────────────────────────────────────────────────

    [ObservableProperty] private bool   _isTraining;
    [ObservableProperty] private bool   _isPaused;
    [ObservableProperty] private int    _currentEpoch;
    [ObservableProperty] private int    _totalEpochs = 200;
    [ObservableProperty] private double _progressPct;
    [ObservableProperty] private string _progressText  = "0 / 0";
    [ObservableProperty] private string _statusMessage = "Ready to train";
    [ObservableProperty] private string _elapsedTime   = "00:00:00";
    [ObservableProperty] private double _bestLoss      = double.MaxValue;
    [ObservableProperty] private string _bestLossEpochLabel = "—";

    // Live metric properties — names match XAML bindings
    [ObservableProperty] private double _totalLoss;
    [ObservableProperty] private double _pdeLoss;
    [ObservableProperty] private double _dataLoss;
    [ObservableProperty] private double _gpuUtil;
    [ObservableProperty] private string _eta = "—";

    // ── Hyperparameter inputs — names match XAML bindings ─────────────────────

    [ObservableProperty] private int    _epochs          = 200;
    [ObservableProperty] private double _learningRate    = 1e-3;
    [ObservableProperty] private int    _trainingSamples = 500;
    [ObservableProperty] private double _pdeLossWeight   = 0.1;
    [ObservableProperty] private bool   _useGpu          = true;
    [ObservableProperty] private string _selectedMode    = "FNO-2D";

    // Mode dropdown options
    public static IReadOnlyList<string> Modes { get; } = new[]
    {
        "FNO-2D",
        "FNO-3D",
        "PINO (physics residual)",
        "U-Net surrogate",
        "DeepONet",
    };

    // ── OxyPlot — property name matches XAML: LossCurveModel ─────────────────

    [ObservableProperty] private PlotModel _lossCurveModel;

    private readonly LineSeries _seriesTotal = new()
    {
        Title = "Total Loss", Color = OxyColors.DodgerBlue, StrokeThickness = 2
    };
    private readonly LineSeries _seriesPde = new()
    {
        Title = "PDE Loss", Color = OxyColors.OrangeRed, StrokeThickness = 1.5,
        LineStyle = LineStyle.Dash
    };
    private readonly LineSeries _seriesData = new()
    {
        Title = "Data Loss", Color = OxyColors.SeaGreen, StrokeThickness = 1.5,
        LineStyle = LineStyle.Dot
    };

    public ObservableCollection<EpochLogItem> EpochLog { get; } = new();

    // ── CanExecute helpers used by XAML IsEnabled bindings ───────────────────

    public bool CanStart     => !IsTraining;
    public bool CanSaveModel => !IsTraining && BestLoss < double.MaxValue;

    // ── Constructor ───────────────────────────────────────────────────────────

    public TrainingViewModel(
        AppDbService?               db  = null,
        ILogger<TrainingViewModel>? log = null)
    {
        _db  = db;
        _log = log;
        _lossCurveModel = BuildPlotModel();
    }

    // ── Commands ──────────────────────────────────────────────────────────────

    [RelayCommand(CanExecute = nameof(CanStartTraining))]
    private async Task StartTrainingAsync()
    {
        _cts = new CancellationTokenSource();
        IsTraining    = true;
        IsPaused      = false;
        TotalLoss     = 0;
        PdeLoss       = 0;
        DataLoss      = 0;
        GpuUtil       = 0;
        StatusMessage = "Training in progress...";
        EpochLog.Clear();
        _seriesTotal.Points.Clear();
        _seriesPde.Points.Clear();
        _seriesData.Points.Clear();

        NotifyCanExecuteChanged();

        var sw = System.Diagnostics.Stopwatch.StartNew();
        try
        {
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
            NotifyCanExecuteChanged();
        }
    }

    private bool CanStartTraining() => !IsTraining;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void StopTraining()
    {
        _cts?.Cancel();
        StatusMessage = "Stopping...";
    }
    private bool CanStop() => IsTraining;

    private bool CanSaveModelEnabled() => !IsTraining && BestLoss < double.MaxValue;

    [RelayCommand(CanExecute = nameof(CanSaveModelEnabled))]
    private void SaveModel()
    {
        var dlg = new Microsoft.Win32.SaveFileDialog
        {
            Title      = "Save PINO Model Checkpoint",
            Filter     = "PhysicsFlow Model (*.pfmodel)|*.pfmodel|All files (*.*)|*.*",
            DefaultExt = ".pfmodel",
            FileName   = $"pino_{SelectedMode.Replace(" ", "_").Replace("(", "").Replace(")", "")}_{DateTime.Now:yyyyMMdd_HHmm}",
        };

        if (dlg.ShowDialog() != true) return;

        try
        {
            var info = System.Text.Json.JsonSerializer.Serialize(new
            {
                mode         = SelectedMode,
                epochs       = CurrentEpoch,
                best_loss    = BestLoss,
                pde_weight   = PdeLossWeight,
                learning_rate = LearningRate,
                saved_at     = DateTime.Now.ToString("O"),
            }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true });

            File.WriteAllText(dlg.FileName, info);
            StatusMessage = $"Model checkpoint saved: {System.IO.Path.GetFileName(dlg.FileName)}";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Save failed: {ex.Message}";
        }
    }

    [RelayCommand]
    private void ClearLog() => EpochLog.Clear();

    private void NotifyCanExecuteChanged()
    {
        StopTrainingCommand.NotifyCanExecuteChanged();
        SaveModelCommand.NotifyCanExecuteChanged();
        OnPropertyChanged(nameof(CanStart));
        OnPropertyChanged(nameof(CanSaveModel));
    }

    // ── Plot ──────────────────────────────────────────────────────────────────

    private PlotModel BuildPlotModel()
    {
        var model = new PlotModel
        {
            Title           = "Training Loss",
            Background      = OxyColor.FromRgb(0x0D, 0x1B, 0x2E),
            PlotAreaBorderColor = OxyColor.FromRgb(0x2A, 0x3A, 0x4A),
            TextColor       = OxyColor.FromRgb(0x88, 0x99, 0xAA),
        };
        model.Axes.Add(new LinearAxis
        {
            Position  = AxisPosition.Bottom,
            Title     = "Epoch",
            Minimum   = 0,
            TextColor = OxyColor.FromRgb(0x88, 0x99, 0xAA),
            AxislineColor = OxyColor.FromRgb(0x2A, 0x3A, 0x4A),
        });
        model.Axes.Add(new LogarithmicAxis
        {
            Position  = AxisPosition.Left,
            Title     = "Loss",
            Minimum   = 1e-6,
            TextColor = OxyColor.FromRgb(0x88, 0x99, 0xAA),
            AxislineColor = OxyColor.FromRgb(0x2A, 0x3A, 0x4A),
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
        LossCurveModel.InvalidatePlot(true);
    }

    // ── Simulation stub (replaced by gRPC streaming in production) ────────────

    private async Task SimulateTrainingAsync(CancellationToken ct, System.Diagnostics.Stopwatch sw)
    {
        var rng = new Random(42);
        TotalEpochs = Epochs;

        double loss = 0.8;
        for (int epoch = 1; epoch <= TotalEpochs && !ct.IsCancellationRequested; epoch++)
        {
            while (IsPaused && !ct.IsCancellationRequested)
                await Task.Delay(200, ct);

            loss    *= (0.985 + rng.NextDouble() * 0.01);
            var pde  = loss * PdeLossWeight * (1 + rng.NextDouble() * 0.2);
            var data = loss * 0.80          * (1 + rng.NextDouble() * 0.05);

            CurrentEpoch  = epoch;
            TotalLoss     = loss;
            PdeLoss       = pde;
            DataLoss      = data;
            GpuUtil       = UseGpu ? (55 + rng.NextDouble() * 40) : 0;
            ProgressPct   = (double)epoch / TotalEpochs * 100.0;
            ProgressText  = $"{epoch} / {TotalEpochs}";
            ElapsedTime   = sw.Elapsed.ToString(@"hh\:mm\:ss");

            if (loss < BestLoss)
            {
                BestLoss           = loss;
                BestLossEpochLabel = $"Epoch {epoch}: {loss:F6}";
            }

            var remaining = TimeSpan.FromSeconds(
                sw.Elapsed.TotalSeconds / epoch * (TotalEpochs - epoch));
            Eta = remaining.ToString(@"hh\:mm\:ss");

            AppendEpochToPlot(epoch, loss, pde, data);
            EpochLog.Insert(0, new EpochLogItem(epoch, loss, pde, data,
                DateTime.Now.ToString("HH:mm:ss")));

            while (EpochLog.Count > 200)
                EpochLog.RemoveAt(EpochLog.Count - 1);

            await Task.Delay(20, ct);
        }

        if (!ct.IsCancellationRequested)
            StatusMessage = $"Training complete — best loss: {BestLoss:F6}";
    }
}

// ── Supporting record ──────────────────────────────────────────────────────────

public record EpochLogItem(int Epoch, double Total, double Pde, double Data, string Time);
