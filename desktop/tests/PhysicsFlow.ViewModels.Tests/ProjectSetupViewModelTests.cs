using System.ComponentModel;
using FluentAssertions;
using PhysicsFlow.ViewModels;

namespace PhysicsFlow.ViewModels.Tests;

/// <summary>
/// Tests for the 5-step Project Setup Wizard ViewModel.
/// Covers: step navigation, validation, CanExecute logic, computed properties,
/// well loading, PVT defaults, schedule editing, and the save event.
/// </summary>
public class ProjectSetupViewModelTests
{
    // ─── Helpers ─────────────────────────────────────────────────────────────

    private static ProjectSetupViewModel Build() => new();

    /// <summary>Pre-loads Norne wells so step-2 validation passes when navigating beyond step 2.</summary>
    private static ProjectSetupViewModel BuildWithWells()
    {
        var vm = new ProjectSetupViewModel();
        vm.LoadNorneDefaultsCommand.Execute(null);
        return vm;
    }

    // ─── Step Navigation ─────────────────────────────────────────────────────

    [Fact]
    public void InitialStep_IsOne()
    {
        var vm = Build();
        vm.CurrentStep.Should().Be(1);
    }

    [Fact]
    public void NextCommand_AdvancesStep()
    {
        var vm = Build();
        vm.NextCommand.Execute(null);
        vm.CurrentStep.Should().Be(2);
    }

    [Fact]
    public void NextCommand_CannotExceedStep5()
    {
        var vm = Build();
        for (int i = 0; i < 10; i++) vm.NextCommand.Execute(null);
        vm.CurrentStep.Should().BeLessThanOrEqualTo(5);
    }

    [Fact]
    public void BackCommand_IsDisabled_OnStep1()
    {
        var vm = Build();
        vm.BackCommand.CanExecute(null).Should().BeFalse();
        vm.CanGoBack.Should().BeFalse();
    }

    [Fact]
    public void BackCommand_IsEnabled_AfterStep1()
    {
        var vm = Build();
        vm.NextCommand.Execute(null);   // → step 2
        vm.BackCommand.CanExecute(null).Should().BeTrue();
        vm.CanGoBack.Should().BeTrue();
    }

    [Fact]
    public void BackCommand_DecrementsStep()
    {
        var vm = Build();
        vm.NextCommand.Execute(null);  // step 2
        vm.BackCommand.Execute(null);  // back to 1
        vm.CurrentStep.Should().Be(1);
    }

    [Theory]
    [InlineData(1, true,  false, false, false, false)]
    [InlineData(2, false, true,  false, false, false)]
    [InlineData(3, false, false, true,  false, false)]
    [InlineData(4, false, false, false, true,  false)]
    [InlineData(5, false, false, false, false, true )]
    public void StepActiveFlags_CorrectForCurrentStep(
        int step,
        bool s1, bool s2, bool s3, bool s4, bool s5)
    {
        // Wells are required to pass step-2 validation when advancing beyond step 2
        var vm = BuildWithWells();
        for (int i = 1; i < step; i++) vm.NextCommand.Execute(null);

        vm.Step1Active.Should().Be(s1);
        vm.Step2Active.Should().Be(s2);
        vm.Step3Active.Should().Be(s3);
        vm.Step4Active.Should().Be(s4);
        vm.Step5Active.Should().Be(s5);
    }

    [Fact]
    public void Step1Done_IsTrueAfterAdvancing()
    {
        var vm = Build();
        vm.Step1Done.Should().BeFalse();
        vm.NextCommand.Execute(null);
        vm.Step1Done.Should().BeTrue();
    }

    [Fact]
    public void NextButtonLabel_IsNextOnStep1Through4()
    {
        var vm = BuildWithWells();
        for (int step = 1; step <= 4; step++)
        {
            vm.NextButtonLabel.Should().Contain("Next");
            vm.NextCommand.Execute(null);
        }
    }

    [Fact]
    public void NextButtonLabel_IsSaveOnStep5()
    {
        var vm = BuildWithWells();
        for (int i = 0; i < 4; i++) vm.NextCommand.Execute(null);
        vm.NextButtonLabel.Should().Contain("Save");
    }

    // ─── Wizard Cancel ────────────────────────────────────────────────────────

    [Fact]
    public void CancelCommand_RaisesWizardCancelledEvent()
    {
        var vm = Build();
        bool fired = false;
        vm.WizardCancelled += (_, _) => fired = true;
        vm.CancelCommand.Execute(null);
        fired.Should().BeTrue("CancelCommand must raise WizardCancelled");
    }

    // ─── Grid Configuration (Step 1) ─────────────────────────────────────────

    [Fact]
    public void GridDefaults_MatchNorne()
    {
        var vm = Build();
        vm.GridNx.Should().Be(46);
        vm.GridNy.Should().Be(112);
        vm.GridNz.Should().Be(22);
        vm.GridDx.Should().BeApproximately(50.0, 0.01);
        vm.GridDy.Should().BeApproximately(50.0, 0.01);
        vm.GridDz.Should().BeApproximately(20.0, 0.01);
    }

    [Fact]
    public void TotalCells_IsProductOfDimensions()
    {
        var vm = Build();
        vm.GridNx = 10;
        vm.GridNy = 20;
        vm.GridNz = 5;
        vm.TotalCells.Should().Be(10 * 20 * 5);
    }

    [Fact]
    public void TotalCells_NotifiesPropertyChanged()
    {
        var vm = Build();
        var changed = new List<string?>();
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) => changed.Add(e.PropertyName);

        vm.GridNx = 10;
        changed.Should().Contain("TotalCells");
    }

    [Fact]
    public void TotalCells_Updates_WhenNxNyNzChange()
    {
        var vm = Build();
        vm.GridNx = 5; vm.GridNy = 5; vm.GridNz = 5;
        vm.TotalCells.Should().Be(125);
    }

    // ─── Wells (Step 2) ───────────────────────────────────────────────────────

    [Fact]
    public void LoadNorneDefaultsCommand_PopulatesWells()
    {
        var vm = Build();
        vm.LoadNorneDefaultsCommand.Execute(null);
        vm.Wells.Should().NotBeEmpty("Norne has 35 wells");
    }

    [Fact]
    public void LoadNorneDefaults_HasProducersAndInjectors()
    {
        var vm = Build();
        vm.LoadNorneDefaultsCommand.Execute(null);
        vm.ProducerCount.Should().BeGreaterThan(0, "Norne has 22 producers");
        vm.InjectorCount.Should().BeGreaterThan(0, "Norne has 13 injectors");
    }

    [Fact]
    public void LoadNorneDefaults_WellCountMatchesTotal()
    {
        var vm = Build();
        vm.LoadNorneDefaultsCommand.Execute(null);
        vm.WellCount.Should().Be(vm.Wells.Count);
    }

    [Fact]
    public void AddWellCommand_AddsOneWell()
    {
        var vm = Build();
        int before = vm.Wells.Count;
        vm.AddWellCommand.Execute(null);
        vm.Wells.Count.Should().Be(before + 1);
    }

    [Fact]
    public void AddWellCommand_CanAlwaysExecute()
    {
        var vm = Build();
        vm.AddWellCommand.CanExecute(null).Should().BeTrue();
    }

    // ─── PVT Configuration (Step 3) ──────────────────────────────────────────

    [Fact]
    public void LoadNornePvtCommand_SetsReasonablePressure()
    {
        var vm = Build();
        vm.LoadNornePvtCommand.Execute(null);
        vm.PvtInitialPressure.Should().BeInRange(100.0, 1000.0,
            "Initial pressure should be a realistic reservoir value in bar");
    }

    [Fact]
    public void LoadNornePvtCommand_SetsReasonableApiGravity()
    {
        var vm = Build();
        vm.LoadNornePvtCommand.Execute(null);
        vm.PvtApiGravity.Should().BeInRange(20.0, 60.0,
            "Norne crude is ~36 API gravity");
    }

    [Fact]
    public void LoadNornePvtCommand_SetsReasonableGasGravity()
    {
        var vm = Build();
        vm.LoadNornePvtCommand.Execute(null);
        vm.PvtGasGravity.Should().BeInRange(0.5, 1.5,
            "Gas gravity relative to air must be in typical range");
    }

    [Fact]
    public void LoadNornePvtCommand_SetsReasonableSwi()
    {
        var vm = Build();
        vm.LoadNornePvtCommand.Execute(null);
        vm.PvtSwi.Should().BeInRange(0.0, 0.5,
            "Connate water saturation must be in [0, 0.5]");
    }

    // ─── Schedule (Step 4) ────────────────────────────────────────────────────

    [Fact]
    public void AddScheduleEntryCommand_AddsOneEntry()
    {
        var vm = Build();
        int before = vm.ScheduleEntries.Count;
        vm.AddScheduleEntryCommand.Execute(null);
        vm.ScheduleEntries.Count.Should().Be(before + 1);
    }

    [Fact]
    public void AddScheduleEntryCommand_CanAlwaysExecute()
    {
        var vm = Build();
        vm.AddScheduleEntryCommand.CanExecute(null).Should().BeTrue();
    }

    // ─── Review & Save (Step 5) ───────────────────────────────────────────────

    [Fact]
    public void ProjectName_DefaultIsNonEmpty()
    {
        var vm = Build();
        vm.ProjectName.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void ReviewSummary_ContainsGridInfo()
    {
        var vm = BuildWithWells();
        // Navigate to step 5
        for (int i = 0; i < 4; i++) vm.NextCommand.Execute(null);
        vm.ReviewSummary.Should().Contain("46", "should mention Nx");
    }

    // ─── PropertyChanged notifications ────────────────────────────────────────

    [Theory]
    [InlineData("GridNx")]
    [InlineData("GridNy")]
    [InlineData("GridNz")]
    [InlineData("PvtInitialPressure")]
    [InlineData("ProjectName")]
    public void SettingProperty_FiresPropertyChanged(string propName)
    {
        var vm = Build();
        bool fired = false;
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == propName) fired = true;
        };

        // Set each property via reflection
        var prop = typeof(ProjectSetupViewModel).GetProperty(propName)!;
        var current = prop.GetValue(vm);
        object newVal = current switch
        {
            int i    => i + 1,
            double d => d + 1.0,
            string s => s + "_changed",
            _        => current!
        };
        prop.SetValue(vm, newVal);

        fired.Should().BeTrue($"{propName} must raise PropertyChanged");
    }
}
