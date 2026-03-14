using System.ComponentModel;
using System.Threading.Tasks;
using FluentAssertions;
using PhysicsFlow.ViewModels;

namespace PhysicsFlow.ViewModels.Tests;

/// <summary>
/// Tests for HistoryMatchingViewModel — αREKI configuration, CanExecute logic,
/// run/stop transitions, convergence tracking, and plot model initialisation.
/// </summary>
public class HistoryMatchingViewModelTests
{
    private static HistoryMatchingViewModel Build() => new();

    // ─── Defaults ─────────────────────────────────────────────────────────────

    [Fact]
    public void ConfigDefaults_ArePhysicallyReasonable()
    {
        var vm = Build();
        vm.ConfigEnsembleSize.Should().BeInRange(10, 2000,
            "ensemble size must be within slider bounds");
        vm.ConfigMaxIterations.Should().BeInRange(1, 100);
        vm.ConfigLocRadius.Should().BeInRange(2.0, 40.0,
            "localisation radius must be within slider bounds");
    }

    [Fact]
    public void EnsembleSizeAlias_MatchesConfigEnsembleSize()
    {
        var vm = Build();
        vm.EnsembleSize.Should().Be(vm.ConfigEnsembleSize,
            "EnsembleSize is an alias and must match ConfigEnsembleSize");
    }

    [Fact]
    public void LocRadiusAlias_MatchesConfigLocRadius()
    {
        var vm = Build();
        vm.LocalisationRadius.Should().BeApproximately(vm.ConfigLocRadius, 1e-9);
    }

    [Fact]
    public void DataMismatchAlias_MatchesCurrentMismatch()
    {
        var vm = Build();
        vm.DataMismatch.Should().BeApproximately(vm.CurrentMismatch, 1e-9);
    }

    [Fact]
    public void AlphaAlias_MatchesCurrentAlpha()
    {
        var vm = Build();
        vm.Alpha.Should().BeApproximately(vm.CurrentAlpha, 1e-9);
    }

    [Fact]
    public void WellNames_NotEmpty()
    {
        var vm = Build();
        vm.WellNames.Should().NotBeEmpty("HM needs well names for fan chart selector");
    }

    [Fact]
    public void QuantityOptions_ContainsOilRate()
    {
        var vm = Build();
        vm.QuantityOptions.Should().Contain(q => q.Contains("Oil"),
            "Oil rate is a mandatory production quantity");
    }

    [Fact]
    public void ConvergencePlotModel_InitialisedNotNull()
    {
        var vm = Build();
        vm.ConvergencePlotModel.Should().NotBeNull();
    }

    [Fact]
    public void FanChartModel_InitialisedNotNull()
    {
        var vm = Build();
        vm.FanChartModel.Should().NotBeNull();
    }

    // ─── CanExecute ───────────────────────────────────────────────────────────

    [Fact]
    public void StartHMCommand_CanExecute_WhenNotRunning()
    {
        var vm = Build();
        vm.IsRunning.Should().BeFalse();
        vm.StartHMCommand.CanExecute(null).Should().BeTrue();
    }

    [Fact]
    public async Task StartHMCommand_DisablesItself_WhenRunning()
    {
        var vm = Build();
        vm.ConfigMaxIterations = 50;
        vm.StartHMCommand.Execute(null);
        await Task.Delay(50);
        vm.StartHMCommand.CanExecute(null).Should().BeFalse(
            "cannot start a second HM run while one is active");
    }

    [Fact]
    public void StopHMCommand_CanAlwaysExecute()
    {
        var vm = Build();
        // StopCommand is always executable (stops if running, no-op if not)
        vm.StopHMCommand.CanExecute(null).Should().BeTrue();
    }

    // ─── Run / Stop transitions ───────────────────────────────────────────────

    [Fact]
    public async Task StartHM_SetsIsRunning()
    {
        var vm = Build();
        vm.ConfigMaxIterations = 50;
        vm.StartHMCommand.Execute(null);
        await Task.Delay(50);
        vm.IsRunning.Should().BeTrue();
        vm.StopHMCommand.Execute(null);
    }

    [Fact]
    public async Task StopHM_ClearsIsRunning()
    {
        var vm = Build();
        vm.ConfigMaxIterations = 50;
        vm.StartHMCommand.Execute(null);
        await Task.Delay(100);
        vm.StopHMCommand.Execute(null);
        await Task.Delay(200);
        vm.IsRunning.Should().BeFalse("stopping must clear IsRunning");
    }

    // ─── Convergence metrics ──────────────────────────────────────────────────

    [Fact]
    public async Task CurrentIteration_IncreasesDuringRun()
    {
        var vm = Build();
        vm.ConfigMaxIterations = 20;
        vm.StartHMCommand.Execute(null);
        await Task.Delay(300);
        vm.CurrentIteration.Should().BeGreaterThan(0,
            "iteration counter must advance during a run");
        vm.StopHMCommand.Execute(null);
    }

    [Fact]
    public async Task CurrentMismatch_IsPositive_DuringRun()
    {
        var vm = Build();
        vm.ConfigMaxIterations = 20;
        vm.StartHMCommand.Execute(null);
        await Task.Delay(300);
        vm.CurrentMismatch.Should().BeGreaterThan(0);
        vm.StopHMCommand.Execute(null);
    }

    [Fact]
    public async Task ProgressPct_IsBetween0And100_DuringRun()
    {
        var vm = Build();
        vm.ConfigMaxIterations = 20;
        vm.StartHMCommand.Execute(null);
        await Task.Delay(300);
        vm.ProgressPct.Should().BeInRange(0.0, 100.0);
        vm.StopHMCommand.Execute(null);
    }

    // ─── Configuration write-through ─────────────────────────────────────────

    [Fact]
    public void SetEnsembleSizeAlias_UpdatesConfigEnsembleSize()
    {
        var vm = Build();
        vm.EnsembleSize = 100;
        vm.ConfigEnsembleSize.Should().Be(100);
    }

    [Fact]
    public void SetLocRadiusAlias_UpdatesConfigLocRadius()
    {
        var vm = Build();
        vm.LocalisationRadius = 15.0;
        vm.ConfigLocRadius.Should().BeApproximately(15.0, 1e-9);
    }

    // ─── Feature flag toggles ────────────────────────────────────────────────

    [Fact]
    public void UseGenerativePriors_DefaultsFalse()
    {
        var vm = Build();
        vm.UseGenerativePriors.Should().BeFalse();
    }

    [Fact]
    public void UseCCR_DefaultsFalse()
    {
        var vm = Build();
        vm.UseCCR.Should().BeFalse();
    }

    [Fact]
    public void AutoLocalisation_DefaultsFalse()
    {
        var vm = Build();
        vm.AutoLocalisation.Should().BeFalse();
    }

    [Fact]
    public void FeatureFlags_FirePropertyChanged_WhenToggled()
    {
        var vm = Build();
        var changed = new List<string?>();
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) => changed.Add(e.PropertyName);

        vm.UseGenerativePriors = true;
        vm.UseCCR              = true;
        vm.AutoLocalisation    = true;

        changed.Should().Contain("UseGenerativePriors");
        changed.Should().Contain("UseCCR");
        changed.Should().Contain("AutoLocalisation");
    }
}
