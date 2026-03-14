using System.ComponentModel;
using System.Threading.Tasks;
using FluentAssertions;
using PhysicsFlow.ViewModels;

namespace PhysicsFlow.ViewModels.Tests;

/// <summary>
/// Tests for DashboardViewModel — computed stat cards, command routing events,
/// CanExecute guards, and property change propagation.
/// </summary>
public class DashboardViewModelTests
{
    private static DashboardViewModel Build() => new();

    // ─── Computed card strings ─────────────────────────────────────────────────

    [Fact]
    public void ProjectSummaryLine_NoProject_ContainsGuidanceText()
    {
        var vm = Build();
        vm.CurrentProjectName = "No project loaded";
        vm.ProjectSummaryLine.Should().Contain("open or create",
            "guidance text expected when no project is loaded");
    }

    [Fact]
    public void ProjectSummaryLine_WithProject_ContainsProjectName()
    {
        var vm = Build();
        vm.CurrentProjectName = "Norne Field";
        vm.TotalWells = 35;
        vm.ProjectSummaryLine.Should().Contain("Norne Field");
    }

    [Fact]
    public void WellCount_MatchesTotalWells()
    {
        var vm = Build();
        vm.TotalWells = 22;
        vm.WellCount.Should().Be(22);
    }

    [Fact]
    public void WellBreakdown_IsNonEmpty_WhenWellsPresent()
    {
        var vm = Build();
        vm.TotalWells = 35;
        vm.WellBreakdown.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void SurrogateStatus_IsNotTrained_WhenNoTraining()
    {
        var vm = Build();
        vm.IsTrainingActive = false;
        vm.SurrogateStatus.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void SurrogateStatus_ShowsActiveText_WhenTraining()
    {
        var vm = Build();
        vm.IsTrainingActive = true;
        vm.SurrogateStatus.ToLower().Should().ContainAny("training", "active", "running");
    }

    [Fact]
    public void HMStatus_ShowsIdleText_WhenNotRunning()
    {
        var vm = Build();
        vm.IsHmActive = false;
        vm.HMStatus.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void HMStatus_ShowsActiveText_WhenRunning()
    {
        var vm = Build();
        vm.IsHmActive = true;
        vm.HMStatus.ToLower().Should().ContainAny("running", "active", "matching");
    }

    [Fact]
    public void SurrogateStatusColor_ChangesWhenTrainingToggled()
    {
        var vm = Build();
        string colorBefore = vm.SurrogateStatusColor;
        vm.IsTrainingActive = true;
        vm.SurrogateStatusColor.Should().NotBe(colorBefore,
            "colour should change to indicate active training");
    }

    [Fact]
    public void HMStatusColor_ChangesWhenHmToggled()
    {
        var vm = Build();
        string colorBefore = vm.HMStatusColor;
        vm.IsHmActive = true;
        vm.HMStatusColor.Should().NotBe(colorBefore,
            "colour should change to indicate active HM");
    }

    // ─── Surrogate status reflects training completion ─────────────────────────

    [Fact]
    public void SurrogateStatus_ReflectsTrainingText()
    {
        var vm = Build();
        vm.IsPinoTrained      = true;
        vm.TrainingStatusText = "Trained — loss 0.00234";
        vm.SurrogateDetail.Should().Contain("0.00234",
            "SurrogateDetail must echo back the training status text");
    }

    // ─── CanExecute guards ────────────────────────────────────────────────────

    [Fact]
    public void StartTrainingCommand_IsDisabled_WhenTrainingActive()
    {
        var vm = Build();
        vm.IsTrainingActive = true;
        vm.CanStartTraining.Should().BeFalse();
    }

    [Fact]
    public void StartTrainingCommand_IsEnabled_WhenIdle()
    {
        var vm = Build();
        vm.IsTrainingActive = false;
        vm.IsHmActive = false;
        vm.CanStartTraining.Should().BeTrue();
    }

    [Fact]
    public void StartHMCommand_IsDisabled_WhenHmActive()
    {
        var vm = Build();
        vm.IsHmActive = true;
        vm.CanStartHM.Should().BeFalse();
    }

    [Fact]
    public void StartHMCommand_IsDisabled_WhenTrainingActive()
    {
        var vm = Build();
        vm.IsTrainingActive = true;
        vm.CanStartHM.Should().BeFalse("cannot start HM while PINO is training");
    }

    [Fact]
    public void StartHMCommand_IsEnabled_WhenBothIdle()
    {
        var vm = Build();
        vm.IsTrainingActive = false;
        vm.IsHmActive = false;
        vm.CanStartHM.Should().BeTrue();
    }

    // ─── Event routing ────────────────────────────────────────────────────────

    [Fact]
    public void NewProjectCommand_RaisesNewProjectRequestedEvent()
    {
        var vm = Build();
        bool fired = false;
        vm.NewProjectRequested += (_, _) => fired = true;
        vm.NewProjectCommand.Execute(null);
        fired.Should().BeTrue("NewProjectCommand must raise NewProjectRequested");
    }

    [Fact]
    public void OpenProjectCommand_RaisesOpenProjectRequestedEvent()
    {
        var vm = Build();
        bool fired = false;
        vm.OpenProjectRequested += (_, _) => fired = true;
        vm.OpenProjectCommand.Execute(null);
        fired.Should().BeTrue("OpenProjectCommand must raise OpenProjectRequested");
    }

    [Fact]
    public void StartTrainingCommand_RaisesStartTrainingRequestedEvent_WhenIdle()
    {
        var vm = Build();
        bool fired = false;
        vm.StartTrainingRequested += (_, _) => fired = true;
        vm.IsTrainingActive = false;
        vm.IsHmActive = false;
        vm.StartTrainingCommand.Execute(null);
        fired.Should().BeTrue("StartTrainingCommand must raise StartTrainingRequested");
    }

    [Fact]
    public void StartHMCommand_RaisesStartHMRequestedEvent_WhenIdle()
    {
        var vm = Build();
        bool fired = false;
        vm.StartHMRequested += (_, _) => fired = true;
        vm.IsTrainingActive = false;
        vm.IsHmActive = false;
        vm.StartHMCommand.Execute(null);
        fired.Should().BeTrue("StartHMCommand must raise StartHMRequested");
    }

    // ─── PropertyChanged propagation ─────────────────────────────────────────

    [Fact]
    public void SettingTotalWells_NotifiesWellCountAndBreakdown()
    {
        var vm = Build();
        var changed = new List<string?>();
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) => changed.Add(e.PropertyName);
        vm.TotalWells = 35;
        changed.Should().Contain("WellCount");
        changed.Should().Contain("WellBreakdown");
        changed.Should().Contain("ProjectSummaryLine");
    }

    [Fact]
    public void SettingIsTrainingActive_NotifiesSurrogateProperties()
    {
        var vm = Build();
        var changed = new List<string?>();
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) => changed.Add(e.PropertyName);
        vm.IsTrainingActive = true;
        changed.Should().Contain("SurrogateStatus");
        changed.Should().Contain("SurrogateStatusColor");
    }

    [Fact]
    public void SettingIsHmActive_NotifiesHMProperties()
    {
        var vm = Build();
        var changed = new List<string?>();
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) => changed.Add(e.PropertyName);
        vm.IsHmActive = true;
        changed.Should().Contain("HMStatus");
        changed.Should().Contain("HMStatusColor");
    }

    // ─── Dashboard refreshes well count when project is loaded ───────────────

    [Fact]
    public void TotalWells_CanBeUpdatedExternally()
    {
        var vm = Build();
        vm.TotalWells = 35;
        vm.WellCount.Should().Be(35, "dashboard must reflect updated well count");
    }

    [Fact]
    public void CurrentProjectName_CanBeUpdatedExternally()
    {
        var vm = Build();
        vm.CurrentProjectName = "My Test Project";
        vm.ProjectSummaryLine.Should().Contain("My Test Project");
    }
}
