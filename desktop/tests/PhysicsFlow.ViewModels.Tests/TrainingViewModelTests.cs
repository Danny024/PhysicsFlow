using System.ComponentModel;
using System.Threading.Tasks;
using FluentAssertions;
using PhysicsFlow.ViewModels;

namespace PhysicsFlow.ViewModels.Tests;

/// <summary>
/// Tests for TrainingViewModel — covers CanExecute logic, hyperparameter
/// defaults, command state transitions, progress tracking, and mode options.
/// </summary>
public class TrainingViewModelTests
{
    private static TrainingViewModel Build() => new();

    // ─── Defaults ─────────────────────────────────────────────────────────────

    [Fact]
    public void Defaults_ReasonableHyperparameters()
    {
        var vm = Build();
        vm.Epochs.Should().BeGreaterThan(0);
        vm.LearningRate.Should().BeInRange(1e-6, 0.1);
        vm.TrainingSamples.Should().BeGreaterThan(0);
        vm.PdeLossWeight.Should().BeInRange(0.0, 10.0);
    }

    [Fact]
    public void Modes_ContainsFnoAndPino()
    {
        TrainingViewModel.Modes.Should().Contain(m => m.Contains("FNO"),
            "at least one FNO mode must be present");
        TrainingViewModel.Modes.Should().Contain(m => m.Contains("PINO"),
            "PINO mode must be present");
    }

    [Fact]
    public void Modes_NotEmpty()
    {
        TrainingViewModel.Modes.Should().NotBeEmpty();
        TrainingViewModel.Modes.Should().HaveCountGreaterThan(1,
            "multiple architectures should be selectable");
    }

    [Fact]
    public void DefaultSelectedMode_IsInModesList()
    {
        var vm = Build();
        TrainingViewModel.Modes.Should().Contain(vm.SelectedMode,
            "the default selected mode must be in the available modes list");
    }

    [Fact]
    public void StatusMessage_InitiallyReady()
    {
        var vm = Build();
        vm.StatusMessage.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void LossCurveModel_InitiallyNotNull()
    {
        var vm = Build();
        vm.LossCurveModel.Should().NotBeNull("chart model must be initialised");
    }

    // ─── Command: StartTraining ───────────────────────────────────────────────

    [Fact]
    public void StartTrainingCommand_CanExecute_WhenNotTraining()
    {
        var vm = Build();
        vm.IsTraining.Should().BeFalse();
        vm.StartTrainingCommand.CanExecute(null).Should().BeTrue();
    }

    [Fact]
    public async Task StartTrainingCommand_SetsIsTraining()
    {
        var vm = Build();
        vm.Epochs = 100;  // 100 * 20ms = 2s; still running after 50ms check
        vm.StartTrainingCommand.Execute(null);
        await Task.Delay(50);
        vm.IsTraining.Should().BeTrue("training should be active immediately after start");
        vm.StopTrainingCommand.Execute(null);
    }

    [Fact]
    public async Task StartTrainingCommand_DisablesItself_WhileRunning()
    {
        var vm = Build();
        vm.Epochs = 100;
        vm.StartTrainingCommand.Execute(null);
        await Task.Delay(50);
        vm.StartTrainingCommand.CanExecute(null).Should().BeFalse(
            "cannot start a second training run while one is in progress");
        vm.StopTrainingCommand.Execute(null);
    }

    // ─── Command: StopTraining ────────────────────────────────────────────────

    [Fact]
    public void StopTrainingCommand_CannotExecute_WhenNotTraining()
    {
        var vm = Build();
        vm.IsTraining.Should().BeFalse();
        vm.StopTrainingCommand.CanExecute(null).Should().BeFalse();
    }

    [Fact]
    public async Task StopTrainingCommand_CanExecute_WhenTraining()
    {
        var vm = Build();
        vm.Epochs = 200;
        vm.StartTrainingCommand.Execute(null);
        await Task.Delay(50);
        vm.StopTrainingCommand.CanExecute(null).Should().BeTrue();
    }

    [Fact]
    public async Task StopTraining_SetsIsTrainingFalse()
    {
        var vm = Build();
        vm.Epochs = 200;
        vm.StartTrainingCommand.Execute(null);
        await Task.Delay(100);
        vm.StopTrainingCommand.Execute(null);
        await Task.Delay(200);
        vm.IsTraining.Should().BeFalse("stopping training should clear the IsTraining flag");
    }

    // ─── Command: SaveModel ───────────────────────────────────────────────────

    [Fact]
    public void SaveModelCommand_CannotExecute_BeforeTraining()
    {
        var vm = Build();
        // BestLoss starts at double.MaxValue → cannot save
        vm.SaveModelCommand.CanExecute(null).Should().BeFalse(
            "cannot save before any training has occurred");
    }

    // ─── Progress tracking ────────────────────────────────────────────────────

    [Fact]
    public async Task ProgressPct_IncreasesDuringTraining()
    {
        var vm = Build();
        vm.Epochs = 20;
        vm.StartTrainingCommand.Execute(null);
        await Task.Delay(200);
        vm.ProgressPct.Should().BeGreaterThan(0,
            "progress must advance while training is running");
        vm.StopTrainingCommand.Execute(null);
    }

    [Fact]
    public async Task CurrentEpoch_IncreasesDuringTraining()
    {
        var vm = Build();
        vm.Epochs = 20;
        vm.StartTrainingCommand.Execute(null);
        await Task.Delay(200);
        vm.CurrentEpoch.Should().BeGreaterThan(0);
        vm.StopTrainingCommand.Execute(null);
    }

    [Fact]
    public async Task TotalLoss_IsFiniteAndPositive_DuringTraining()
    {
        var vm = Build();
        vm.Epochs = 20;
        vm.StartTrainingCommand.Execute(null);
        await Task.Delay(200);
        double.IsFinite(vm.TotalLoss).Should().BeTrue("loss must be a finite number");
        vm.TotalLoss.Should().BeGreaterThan(0, "loss must be positive during training");
        vm.StopTrainingCommand.Execute(null);
    }

    [Fact]
    public async Task ProgressText_ReflectsCurrentEpoch()
    {
        var vm = Build();
        vm.Epochs = 50;
        vm.StartTrainingCommand.Execute(null);
        await Task.Delay(200);
        vm.ProgressText.Should().Contain("/", "format should be 'N / Total'");
        vm.StopTrainingCommand.Execute(null);
    }

    // ─── PropertyChanged notifications ────────────────────────────────────────

    [Theory]
    [InlineData("Epochs")]
    [InlineData("LearningRate")]
    [InlineData("TrainingSamples")]
    [InlineData("PdeLossWeight")]
    [InlineData("SelectedMode")]
    public void HyperparameterSetter_FiresPropertyChanged(string propName)
    {
        var vm = Build();
        bool fired = false;
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == propName) fired = true;
        };

        var prop = typeof(TrainingViewModel).GetProperty(propName)!;
        var cur = prop.GetValue(vm);
        object next = cur switch
        {
            int i    => i + 10,
            double d => d + 0.001,
            string s => TrainingViewModel.Modes.FirstOrDefault(m => m != s) ?? s,
            bool b   => !b,
            _        => cur!
        };
        prop.SetValue(vm, next);

        fired.Should().BeTrue($"{propName} setter must raise PropertyChanged");
    }
}
