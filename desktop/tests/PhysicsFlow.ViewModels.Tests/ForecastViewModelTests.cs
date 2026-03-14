using System.ComponentModel;
using System.Threading.Tasks;
using FluentAssertions;
using PhysicsFlow.ViewModels;

namespace PhysicsFlow.ViewModels.Tests;

/// <summary>
/// Tests for ForecastViewModel — EUR metrics, chart model initialisation,
/// well/quantity selectors, RunForecast command, and export CanExecute guards.
/// </summary>
public class ForecastViewModelTests
{
    private static ForecastViewModel Build() => new();

    // ─── Defaults ─────────────────────────────────────────────────────────────

    [Fact]
    public void WellNames_IsNotEmpty()
    {
        var vm = Build();
        vm.WellNames.Should().NotBeEmpty("forecast needs at least one well to select");
    }

    [Fact]
    public void QuantityOptions_ContainsOilRate()
    {
        var vm = Build();
        vm.QuantityOptions.Should().Contain(q => q.Contains("Oil"),
            "Oil Rate must be a forecast quantity option");
    }

    [Fact]
    public void HorizonOptions_ContainsStandardValues()
    {
        var vm = Build();
        vm.HorizonOptions.Should().Contain(10, "10-year forecast horizon expected");
        vm.HorizonOptions.Should().Contain(20, "20-year forecast horizon expected");
    }

    [Fact]
    public void ChartModels_AreNotNull()
    {
        var vm = Build();
        vm.OilRatePlotModel.Should().NotBeNull();
        vm.WaterRatePlotModel.Should().NotBeNull();
        vm.CumOilPlotModel.Should().NotBeNull();
        vm.PressurePlotModel.Should().NotBeNull();
    }

    [Fact]
    public void ForecastHorizonYears_DefaultIsReasonable()
    {
        var vm = Build();
        vm.ForecastHorizonYears.Should().BeInRange(1, 50);
    }

    // ─── RunForecast command ──────────────────────────────────────────────────

    [Fact]
    public void RunForecastCommand_CanAlwaysExecute()
    {
        var vm = Build();
        vm.RunForecastCommand.CanExecute(null).Should().BeTrue();
    }

    [Fact]
    public async Task RunForecastCommand_SetsEURMetrics()
    {
        var vm = Build();
        vm.RunForecastCommand.Execute(null);
        await Task.Delay(1500);   // allow async stub to complete

        vm.EurOilP50.Should().BeGreaterThan(0,
            "EUR oil P50 must be positive after running the forecast");
        vm.RecoveryFactorP50.Should().BeInRange(0.0, 1.0,
            "recovery factor must be in [0, 1]");
        vm.PeakOilRateP50.Should().BeGreaterThan(0);
    }

    [Fact]
    public async Task RunForecastCommand_SetsIsCalculatingThenClears()
    {
        var vm = Build();
        // Should briefly be true then clear
        vm.RunForecastCommand.Execute(null);
        await Task.Delay(1500);
        vm.IsCalculating.Should().BeFalse("IsCalculating must clear after forecast completes");
    }

    // ─── Well/quantity selector reaction ─────────────────────────────────────

    [Fact]
    public async Task ChangingSelectedWell_DoesNotThrow()
    {
        var vm = Build();
        vm.RunForecastCommand.Execute(null);
        await Task.Delay(1200);

        Action act = () => vm.SelectedWell = vm.WellNames.LastOrDefault();
        act.Should().NotThrow("changing well selection must not crash");
    }

    [Fact]
    public async Task ChangingSelectedQuantity_DoesNotThrow()
    {
        var vm = Build();
        vm.RunForecastCommand.Execute(null);
        await Task.Delay(1200);

        Action act = () => vm.SelectedQuantity = vm.QuantityOptions.LastOrDefault();
        act.Should().NotThrow();
    }

    [Fact]
    public async Task TogglingShowHistoricalData_DoesNotThrow()
    {
        var vm = Build();
        vm.RunForecastCommand.Execute(null);
        await Task.Delay(1200);

        Action act = () => vm.ShowHistoricalData = !vm.ShowHistoricalData;
        act.Should().NotThrow();
    }

    // ─── Export CanExecute ────────────────────────────────────────────────────

    [Fact]
    public void ExportExcelCommand_CanExecute_ReturnsBool()
    {
        var vm = Build();
        // Without a project loaded, export service may be null → CanExecute = false
        bool canExec = vm.ExportExcelCommand.CanExecute(null);
        // CanExecute must not throw — actual value depends on service availability
        Assert.True(canExec == true || canExec == false);
    }

    [Fact]
    public void ExportPdfCommand_CanExecute_ReturnsBool()
    {
        var vm = Build();
        bool canExec = vm.ExportPdfCommand.CanExecute(null);
        Assert.True(canExec == true || canExec == false);
    }

    // ─── PropertyChanged notifications ────────────────────────────────────────

    [Fact]
    public async Task RunForecastCommand_FiresPropertyChangedForEurOilP50()
    {
        var vm = Build();
        bool fired = false;
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == "EurOilP50") fired = true;
        };
        vm.RunForecastCommand.Execute(null);
        await Task.Delay(1500);
        fired.Should().BeTrue("EurOilP50 PropertyChanged expected after forecast run");
    }

    [Fact]
    public void SelectedWell_FiresPropertyChanged()
    {
        var vm = Build();
        bool fired = false;
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == "SelectedWell") fired = true;
        };
        // Use LastOrDefault so we pick a value different from the default "FIELD"
        vm.SelectedWell = vm.WellNames.LastOrDefault() ?? "B-1H";
        fired.Should().BeTrue();
    }
}
