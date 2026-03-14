using System.ComponentModel;
using System.Threading.Tasks;
using FluentAssertions;
using PhysicsFlow.ViewModels;

namespace PhysicsFlow.ViewModels.Tests;

/// <summary>
/// Tests for ReservoirView3DViewModel — property selector, timestep range,
/// legend updates, animation commands, and opacity control.
/// </summary>
public class ReservoirView3DViewModelTests
{
    private static ReservoirView3DViewModel Build() => new();

    // ─── Defaults ─────────────────────────────────────────────────────────────

    [Fact]
    public void PropertyOptions_HasAtLeastThreeOptions()
    {
        var vm = Build();
        vm.PropertyOptions.Should().HaveCountGreaterThanOrEqualTo(3,
            "Pressure, Water Saturation and Permeability are mandatory");
    }

    [Fact]
    public void PropertyOptions_ContainsPressure()
    {
        var vm = Build();
        vm.PropertyOptions.Should().Contain(p => p.Contains("Pressure"));
    }

    [Fact]
    public void PropertyOptions_ContainsWaterSaturation()
    {
        var vm = Build();
        vm.PropertyOptions.Should().Contain(p => p.Contains("Saturation") || p.Contains("Water"));
    }

    [Fact]
    public void PropertyOptions_ContainsPermeability()
    {
        var vm = Build();
        vm.PropertyOptions.Should().Contain(p => p.Contains("Perm") || p.Contains("K"));
    }

    [Fact]
    public void DefaultSelectedProperty_IsInPropertyOptions()
    {
        var vm = Build();
        vm.PropertyOptions.Should().Contain(vm.SelectedProperty,
            "default selected property must be in the options list");
    }

    [Fact]
    public void TotalTimesteps_IsPositive()
    {
        var vm = Build();
        vm.TotalTimesteps.Should().BeGreaterThan(0);
    }

    [Fact]
    public void VoxelOpacity_DefaultIsInValidRange()
    {
        var vm = Build();
        vm.VoxelOpacity.Should().BeInRange(0.1, 1.0);
    }

    [Fact]
    public void ShowWells_DefaultIsTrue()
    {
        var vm = Build();
        vm.ShowWells.Should().BeTrue("wells should be visible by default");
    }

    // ─── Legend ────────────────────────────────────────────────────────────────

    [Fact]
    public void LegendTitle_IsNotEmpty()
    {
        var vm = Build();
        vm.LegendTitle.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void LegendUnit_IsNotEmpty()
    {
        var vm = Build();
        vm.LegendUnit.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void LegendMinLabel_IsNotEmpty()
    {
        var vm = Build();
        vm.LegendMinLabel.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void LegendMaxLabel_IsNotEmpty()
    {
        var vm = Build();
        vm.LegendMaxLabel.Should().NotBeNullOrWhiteSpace();
    }

    [Fact]
    public void ChangingProperty_UpdatesLegendTitle()
    {
        var vm = Build();
        string before = vm.LegendTitle;
        // Switch to a different property
        var other = vm.PropertyOptions.FirstOrDefault(p => p != vm.SelectedProperty);
        if (other is null) return; // only one option — skip
        vm.SelectedProperty = other;
        vm.LegendTitle.Should().NotBe(before,
            "LegendTitle must update when the selected property changes");
    }

    // ─── Timestep slider ──────────────────────────────────────────────────────

    [Fact]
    public void CurrentTimestepLabel_FormatIsCorrect()
    {
        var vm = Build();
        vm.CurrentTimestep = 5;
        vm.CurrentTimestepLabel.Should().Contain("5",
            "label must reflect the current timestep value");
    }

    [Fact]
    public void ChangingTimestep_FiresPropertyChanged()
    {
        var vm = Build();
        bool fired = false;
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == "CurrentTimestepLabel") fired = true;
        };
        vm.CurrentTimestep = vm.TotalTimesteps > 0 ? 1 : 0;
        fired.Should().BeTrue("CurrentTimestepLabel must notify when timestep changes");
    }

    // ─── Animation commands ───────────────────────────────────────────────────

    [Fact]
    public void PlayAnimationCommand_CanExecute()
    {
        var vm = Build();
        vm.PlayAnimationCommand.CanExecute(null).Should().BeTrue();
    }

    [Fact]
    public void PauseAnimationCommand_CanExecute()
    {
        var vm = Build();
        vm.PauseAnimationCommand.CanExecute(null).Should().BeTrue();
    }

    [Fact]
    public async Task PlayAnimation_AdvancesTimestep()
    {
        var vm = Build();
        vm.CurrentTimestep = 0;
        vm.PlayAnimationCommand.Execute(null);
        await Task.Delay(600);   // 3 frames at 200ms each
        vm.CurrentTimestep.Should().BeGreaterThan(0,
            "animation play must advance the timestep counter");
        vm.PauseAnimationCommand.Execute(null);
    }

    [Fact]
    public async Task PauseAnimation_FreezesTimestep()
    {
        var vm = Build();
        vm.PlayAnimationCommand.Execute(null);
        await Task.Delay(300);
        vm.PauseAnimationCommand.Execute(null);
        int frozen = vm.CurrentTimestep;
        await Task.Delay(400);
        vm.CurrentTimestep.Should().Be(frozen,
            "timestep must not advance after pausing");
    }

    // ─── Export command ───────────────────────────────────────────────────────

    [Fact]
    public void ExportVtkCommand_CanExecute()
    {
        var vm = Build();
        vm.ExportVtkCommand.CanExecute(null).Should().BeTrue();
    }

    [Fact]
    public void ExportVtkCommand_SetsStatusMessage()
    {
        var vm = Build();
        vm.StatusMessage = string.Empty;
        vm.ExportVtkCommand.Execute(null);
        vm.StatusMessage.Should().NotBeNullOrWhiteSpace(
            "ExportVtkCommand must report back a status message");
    }

    // ─── Opacity control ──────────────────────────────────────────────────────

    [Fact]
    public void ChangingOpacity_FiresPropertyChanged()
    {
        var vm = Build();
        bool fired = false;
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == "VoxelOpacity") fired = true;
        };
        vm.VoxelOpacity = 0.5;
        fired.Should().BeTrue();
    }

    [Fact]
    public void ReservoirModel3D_NotNull()
    {
        var vm = Build();
        vm.ReservoirModel3D.Should().NotBeNull(
            "3D model must be built on construction");
    }
}
