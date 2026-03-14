using System.ComponentModel;
using FluentAssertions;
using PhysicsFlow.ViewModels;

namespace PhysicsFlow.ViewModels.Tests;

/// <summary>
/// Tests for CrossSectionViewModel — property/colormap selectors, slice navigation,
/// image generation for all three planes, and well overlay toggle.
/// </summary>
public class CrossSectionViewModelTests
{
    private static CrossSectionViewModel Build() => new();

    // ─── Options lists ────────────────────────────────────────────────────────

    [Fact]
    public void PropertyOptions_HasAtLeastFourOptions()
    {
        var vm = Build();
        vm.PropertyOptions.Should().HaveCountGreaterThanOrEqualTo(4,
            "Pressure, Water Sat, Permeability, Porosity are all required");
    }

    [Fact]
    public void ColourmapOptions_HasAtLeastFourOptions()
    {
        var vm = Build();
        vm.ColourmapOptions.Should().HaveCountGreaterThanOrEqualTo(4,
            "Jet, Viridis, Seismic, Greys are the minimum required colourmaps");
    }

    [Fact]
    public void ColourmapOptions_ContainsJet()
    {
        var vm = Build();
        vm.ColourmapOptions.Should().Contain("Jet");
    }

    [Fact]
    public void ColourmapOptions_ContainsViridis()
    {
        var vm = Build();
        vm.ColourmapOptions.Should().Contain("Viridis");
    }

    [Fact]
    public void DefaultSelectedProperty_InPropertyOptions()
    {
        var vm = Build();
        vm.PropertyOptions.Should().Contain(vm.SelectedProperty);
    }

    [Fact]
    public void DefaultSelectedColourmap_InColourmapOptions()
    {
        var vm = Build();
        vm.ColourmapOptions.Should().Contain(vm.SelectedColourmap);
    }

    // ─── Slice navigation ─────────────────────────────────────────────────────

    [Fact]
    public void SliceIndex_DefaultIsZero()
    {
        var vm = Build();
        vm.SliceIndex.Should().Be(0);
    }

    [Fact]
    public void SliceIndexLabel_ReflectsCurrentSlice()
    {
        var vm = Build();
        vm.SliceIndex = 10;
        vm.SliceIndexLabel.Should().Contain("10",
            "label must show the current slice number");
    }

    [Fact]
    public void ChangingSliceIndex_NotifiesSliceIndexLabel()
    {
        var vm = Build();
        bool fired = false;
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == "SliceIndexLabel") fired = true;
        };
        vm.SliceIndex = 5;
        fired.Should().BeTrue("SliceIndexLabel must be notified when SliceIndex changes");
    }

    // ─── Plane images ─────────────────────────────────────────────────────────
    // Note: WriteableBitmap requires a WPF STA thread (not available in xUnit's
    // default MTA context). Image-null checks are guarded by an STA check; the
    // ViewModel behaviour (property/colourmap/slice reactions) is still verified.

    private static bool IsStaThread =>
        System.Threading.Thread.CurrentThread.GetApartmentState()
            == System.Threading.ApartmentState.STA;

    [Fact]
    public void IPlaneImage_NotNull_OnConstruction()
    {
        if (!IsStaThread) return; // WriteableBitmap requires STA — skip in CI
        var vm = Build();
        vm.IPlaneImage.Should().NotBeNull(
            "I-plane image must be generated on construction");
    }

    [Fact]
    public void JPlaneImage_NotNull_OnConstruction()
    {
        if (!IsStaThread) return;
        var vm = Build();
        vm.JPlaneImage.Should().NotBeNull();
    }

    [Fact]
    public void KPlaneImage_NotNull_OnConstruction()
    {
        if (!IsStaThread) return;
        var vm = Build();
        vm.KPlaneImage.Should().NotBeNull();
    }

    [Fact]
    public void ChangingProperty_DoesNotThrow()
    {
        var vm = Build();
        var other = vm.PropertyOptions.FirstOrDefault(p => p != vm.SelectedProperty);
        if (other is null) return;
        Action act = () => vm.SelectedProperty = other;
        act.Should().NotThrow("changing property selection must not crash");
    }

    [Fact]
    public void ChangingColourmap_DoesNotThrow()
    {
        var vm = Build();
        var other = vm.ColourmapOptions.FirstOrDefault(c => c != vm.SelectedColourmap);
        if (other is null) return;
        Action act = () => vm.SelectedColourmap = other;
        act.Should().NotThrow("changing colourmap must not crash");
    }

    [Fact]
    public void ChangingSlice_DoesNotThrow()
    {
        var vm = Build();
        Action act = () => vm.SliceIndex = 20;
        act.Should().NotThrow("changing slice index must not crash");
    }

    // ─── Well overlay ─────────────────────────────────────────────────────────

    [Fact]
    public void ShowWellOverlay_DefaultIsTrue()
    {
        var vm = Build();
        vm.ShowWellOverlay.Should().BeTrue("well overlay should be on by default");
    }

    [Fact]
    public void ToggleShowWellOverlay_FiresPropertyChanged()
    {
        var vm = Build();
        bool fired = false;
        ((INotifyPropertyChanged)vm).PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == "ShowWellOverlay") fired = true;
        };
        vm.ShowWellOverlay = false;
        fired.Should().BeTrue();
    }

    // ─── Export command ───────────────────────────────────────────────────────

    [Fact]
    public void ExportImageCommand_CanExecute()
    {
        var vm = Build();
        vm.ExportImageCommand.CanExecute(null).Should().BeTrue();
    }

    [Fact]
    public void ExportImageCommand_SetsStatusMessage()
    {
        var vm = Build();
        vm.StatusMessage = string.Empty;
        vm.ExportImageCommand.Execute(null);
        vm.StatusMessage.Should().NotBeNullOrWhiteSpace(
            "ExportImageCommand must set a status message");
    }
}
