using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace PhysicsFlow.ViewModels;

/// <summary>
/// ViewModel for the Production Forecast view.
/// Shows P10/P50/P90 fan charts for oil rate, water rate,
/// cumulative oil (EUR), and reservoir pressure.
/// </summary>
public partial class ForecastViewModel : ObservableObject
{
    // ── Observable properties ────────────────────────────────────────────

    [ObservableProperty] private double eurOilP50;
    [ObservableProperty] private double eurGasP50;
    [ObservableProperty] private double recoveryFactorP50;
    [ObservableProperty] private double peakOilRateP50;
    [ObservableProperty] private int forecastHorizonYears = 20;
    [ObservableProperty] private bool isCalculating;
    [ObservableProperty] private string statusMessage = "Ready";
    [ObservableProperty] private bool showHistoricalData = true;

    [ObservableProperty] private string? selectedWell;
    [ObservableProperty] private string? selectedQuantity;
    [ObservableProperty] private int selectedHorizon = 20;

    // OxyPlot models
    [ObservableProperty] private PlotModel oilRatePlotModel = new();
    [ObservableProperty] private PlotModel waterRatePlotModel = new();
    [ObservableProperty] private PlotModel cumOilPlotModel = new();
    [ObservableProperty] private PlotModel pressurePlotModel = new();

    // Collections
    public ObservableCollection<string> WellNames { get; } = new();
    public ObservableCollection<string> QuantityOptions { get; } = new()
        { "Oil Rate (stb/day)", "Water Rate (stb/day)", "Gas Rate (Mscf/day)",
          "Cum Oil (MMstb)", "Pressure (bar)" };
    public ObservableCollection<int> HorizonOptions { get; } = new()
        { 5, 10, 15, 20, 30, 40 };

    // Forecast data storage
    private ForecastData? _forecastData;

    // ── Constructor ──────────────────────────────────────────────────────

    public ForecastViewModel()
    {
        SelectedQuantity = QuantityOptions.First();
        SelectedHorizon  = 20;
        InitialisePlots();
        LoadDemoForecast();
    }

    // ── Commands ─────────────────────────────────────────────────────────

    [RelayCommand]
    private async Task RunForecast()
    {
        IsCalculating  = true;
        StatusMessage  = "Running ensemble forecast...";

        try
        {
            await Task.Delay(800);   // Replace with gRPC call
            LoadDemoForecast();
            StatusMessage = $"Forecast complete — {DateTime.Now:HH:mm:ss}";
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
        }
        finally
        {
            IsCalculating = false;
        }
    }

    [RelayCommand]
    private async Task ExportExcel()
    {
        IsCalculating = true;
        StatusMessage = "Exporting to Excel...";
        await Task.Delay(500);
        // TODO: ClosedXML export
        StatusMessage = "Excel exported";
        IsCalculating = false;
    }

    [RelayCommand]
    private async Task ExportPdf()
    {
        IsCalculating = true;
        StatusMessage = "Exporting to PDF...";
        await Task.Delay(500);
        // TODO: QuestPDF export
        StatusMessage = "PDF exported";
        IsCalculating = false;
    }

    // ── Property changed reactions ────────────────────────────────────────

    partial void OnSelectedWellChanged(string? value) => RefreshCharts();
    partial void OnSelectedQuantityChanged(string? value) => RefreshCharts();
    partial void OnShowHistoricalDataChanged(bool value) => RefreshCharts();

    // ── Plot initialisation ───────────────────────────────────────────────

    private void InitialisePlots()
    {
        OilRatePlotModel   = CreateFanChartModel("", OxyColor.FromRgb(0, 204, 102));
        WaterRatePlotModel = CreateFanChartModel("", OxyColor.FromRgb(68, 136, 255));
        CumOilPlotModel    = CreateFanChartModel("", OxyColor.FromRgb(0, 204, 102));
        PressurePlotModel  = CreateFanChartModel("", OxyColor.FromRgb(255, 107, 107));
    }

    private static PlotModel CreateFanChartModel(string title, OxyColor color)
    {
        var model = new PlotModel
        {
            Title            = title,
            Background       = OxyColors.Transparent,
            PlotAreaBackground = OxyColor.FromRgb(10, 22, 40),
            TextColor        = OxyColor.FromRgb(200, 210, 220),
            TitleColor       = OxyColor.FromRgb(200, 210, 220),
            SubtitleColor    = OxyColor.FromRgb(136, 153, 170),
        };

        model.Axes.Add(new DateTimeAxis
        {
            Position  = AxisPosition.Bottom,
            Title     = "Date",
            TextColor = OxyColor.FromRgb(136, 153, 170),
            AxislineColor = OxyColor.FromRgb(30, 50, 80),
            MajorGridlineStyle = LineStyle.Dot,
            MajorGridlineColor = OxyColor.FromRgb(20, 40, 70),
        });
        model.Axes.Add(new LinearAxis
        {
            Position  = AxisPosition.Left,
            TextColor = OxyColor.FromRgb(136, 153, 170),
            AxislineColor = OxyColor.FromRgb(30, 50, 80),
            MajorGridlineStyle = LineStyle.Dot,
            MajorGridlineColor = OxyColor.FromRgb(20, 40, 70),
            Minimum   = 0,
        });

        return model;
    }

    // ── Forecast data loading ─────────────────────────────────────────────

    private void LoadDemoForecast()
    {
        // Generate synthetic P10/P50/P90 fan data
        var start = DateTime.Today;
        var t     = Enumerable.Range(0, ForecastHorizonYears * 12)
                              .Select(i => start.AddMonths(i))
                              .ToArray();
        int n = t.Length;

        // Oil rate: exponential decline
        var oilP50  = Enumerable.Range(0, n).Select(i => 5000 * Math.Exp(-0.003 * i)).ToArray();
        var oilP10  = oilP50.Select(v => v * 1.3).ToArray();
        var oilP90  = oilP50.Select(v => v * 0.7).ToArray();

        var waterP50 = Enumerable.Range(0, n).Select(i => 1000 + 200 * i * 0.01).ToArray();
        var waterP10 = waterP50.Select(v => v * 0.7).ToArray();
        var waterP90 = waterP50.Select(v => v * 1.4).ToArray();

        var cumOilP50 = CumulativeSum(oilP50.Select(v => v * 30 / 1e6).ToArray());
        var cumOilP10 = CumulativeSum(oilP10.Select(v => v * 30 / 1e6).ToArray());
        var cumOilP90 = CumulativeSum(oilP90.Select(v => v * 30 / 1e6).ToArray());

        var pressP50 = Enumerable.Range(0, n).Select(i => 250 - 0.05 * i).ToArray();
        var pressP10 = pressP50.Select(v => v + 10).ToArray();
        var pressP90 = pressP50.Select(v => v - 10).ToArray();

        _forecastData = new ForecastData(t, oilP10, oilP50, oilP90,
                                          waterP10, waterP50, waterP90,
                                          cumOilP10, cumOilP50, cumOilP90,
                                          pressP10, pressP50, pressP90);

        // Summary stats
        EurOilP50        = cumOilP50.Last();
        EurGasP50        = EurOilP50 * 1.8;
        RecoveryFactorP50 = 0.32;
        PeakOilRateP50   = oilP50.Max();
        ForecastHorizonYears = 20;

        RefreshCharts();

        WellNames.Clear();
        WellNames.Add("FIELD");
        foreach (var name in new[] { "B-1H", "B-2H", "B-3H", "B-4H",
                                      "C-1H", "C-2H", "D-1H", "E-1H" })
            WellNames.Add(name);

        SelectedWell ??= "FIELD";
    }

    private void RefreshCharts()
    {
        if (_forecastData is null) return;
        var d = _forecastData;

        UpdateFanChart(OilRatePlotModel,   d.Time, d.OilP10, d.OilP50, d.OilP90,
                       OxyColor.FromRgb(0, 204, 102));
        UpdateFanChart(WaterRatePlotModel, d.Time, d.WaterP10, d.WaterP50, d.WaterP90,
                       OxyColor.FromRgb(68, 136, 255));
        UpdateFanChart(CumOilPlotModel,    d.Time, d.CumOilP10, d.CumOilP50, d.CumOilP90,
                       OxyColor.FromRgb(0, 204, 102));
        UpdateFanChart(PressurePlotModel,  d.Time, d.PressP10, d.PressP50, d.PressP90,
                       OxyColor.FromRgb(255, 107, 107));
    }

    private static void UpdateFanChart(
        PlotModel model,
        DateTime[] time, double[] p10, double[] p50, double[] p90,
        OxyColor color)
    {
        model.Series.Clear();

        // Shaded P10-P90 band
        var band = new AreaSeries
        {
            Color   = color.ChangeSaturation(0.4),
            Fill    = OxyColor.FromAColor(40, color),
            StrokeThickness = 0,
        };
        for (int i = 0; i < time.Length; i++)
        {
            band.Points.Add(new DataPoint(DateTimeAxis.ToDouble(time[i]), p10[i]));
            band.Points2.Add(new DataPoint(DateTimeAxis.ToDouble(time[i]), p90[i]));
        }
        model.Series.Add(band);

        // P10 dashed line
        var lineP10 = new LineSeries
        {
            Color           = OxyColor.FromAColor(160, color),
            LineStyle       = LineStyle.Dash,
            StrokeThickness = 1.5,
            Title           = "P10",
        };
        // P90 dashed line
        var lineP90 = new LineSeries
        {
            Color           = OxyColor.FromAColor(160, color),
            LineStyle       = LineStyle.Dash,
            StrokeThickness = 1.5,
            Title           = "P90",
        };
        // P50 solid line
        var lineP50 = new LineSeries
        {
            Color           = color,
            LineStyle       = LineStyle.Solid,
            StrokeThickness = 2.5,
            Title           = "P50",
        };
        for (int i = 0; i < time.Length; i++)
        {
            double dt = DateTimeAxis.ToDouble(time[i]);
            lineP10.Points.Add(new DataPoint(dt, p10[i]));
            lineP50.Points.Add(new DataPoint(dt, p50[i]));
            lineP90.Points.Add(new DataPoint(dt, p90[i]));
        }
        model.Series.Add(lineP90);
        model.Series.Add(lineP10);
        model.Series.Add(lineP50);

        model.InvalidatePlot(true);
    }

    private static double[] CumulativeSum(double[] arr)
    {
        var result = new double[arr.Length];
        double sum = 0;
        for (int i = 0; i < arr.Length; i++)
        {
            sum += arr[i];
            result[i] = sum;
        }
        return result;
    }

    // ── Nested data record ────────────────────────────────────────────────

    private sealed record ForecastData(
        DateTime[] Time,
        double[] OilP10, double[] OilP50, double[] OilP90,
        double[] WaterP10, double[] WaterP50, double[] WaterP90,
        double[] CumOilP10, double[] CumOilP50, double[] CumOilP90,
        double[] PressP10, double[] PressP50, double[] PressP90
    );
}
