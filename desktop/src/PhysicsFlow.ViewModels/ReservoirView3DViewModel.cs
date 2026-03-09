using System;
using System.Collections.ObjectModel;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Media3D;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace PhysicsFlow.ViewModels;

/// <summary>
/// ViewModel for the 3D reservoir property viewer.
/// Renders voxel boxes coloured by Jet colourmap for P / Sw / K
/// with animated timestep playback using HelixToolkit.Wpf.
/// </summary>
public partial class ReservoirView3DViewModel : ObservableObject
{
    // ── Observable properties ────────────────────────────────────────────

    [ObservableProperty] private string selectedProperty      = "Pressure";
    [ObservableProperty] private int    currentTimestep       = 0;
    [ObservableProperty] private int    totalTimesteps        = 20;
    [ObservableProperty] private string currentTimestepLabel  = "t = 0";
    [ObservableProperty] private double voxelOpacity          = 0.85;
    [ObservableProperty] private bool   showWells             = true;
    [ObservableProperty] private bool   isLoading             = false;
    [ObservableProperty] private bool   isAnimating           = false;
    [ObservableProperty] private string statusMessage         = "Ready — load a project to begin";

    [ObservableProperty] private Model3DGroup reservoirModel3D = new();
    [ObservableProperty] private Model3DGroup wellModel3D      = new();

    public ObservableCollection<string> PropertyOptions { get; } = new()
        { "Pressure", "Water Saturation", "Permeability" };

    // Downsampled display resolution (full 46×112×22 is too slow for real-time)
    private const int Nx = 46, Ny = 112, Nz = 22;
    private const int DsX = 4, DsY = 4;    // display every 4th cell in X and Y

    // Synthetic data cache [timestep, nz, nyDs, nxDs]
    private float[,,,]? _data;
    private CancellationTokenSource? _animCts;

    // ── Constructor ──────────────────────────────────────────────────────

    public ReservoirView3DViewModel()
    {
        _ = LoadDemoDataAsync();
    }

    // ── Property change reactions ─────────────────────────────────────────

    partial void OnSelectedPropertyChanged(string value)  => RebuildScene();
    partial void OnVoxelOpacityChanged(double value)      => RebuildScene();
    partial void OnCurrentTimestepChanged(int value)
    {
        CurrentTimestepLabel = $"t = {value}";
        RebuildScene();
    }

    // ── Commands ─────────────────────────────────────────────────────────

    [RelayCommand]
    private async Task PlayAnimation()
    {
        if (IsAnimating) return;
        IsAnimating = true;
        _animCts = new CancellationTokenSource();
        try
        {
            while (!_animCts.Token.IsCancellationRequested)
            {
                CurrentTimestep = (CurrentTimestep + 1) % TotalTimesteps;
                await Task.Delay(200, _animCts.Token);
            }
        }
        catch (OperationCanceledException) { }
        finally { IsAnimating = false; }
    }

    [RelayCommand]
    private void PauseAnimation()
    {
        _animCts?.Cancel();
        IsAnimating = false;
    }

    [RelayCommand]
    private async Task ExportVtk()
    {
        StatusMessage = "Exporting VTK...";
        await Task.Delay(300);
        StatusMessage = "VTK export complete (stub — wire to Python engine gRPC in v1.3)";
    }

    // ── Demo data loading ─────────────────────────────────────────────────

    private async Task LoadDemoDataAsync()
    {
        IsLoading = true;
        StatusMessage = "Generating synthetic demo data...";

        await Task.Run(() =>
        {
            int nxDs = Nx / DsX, nyDs = Ny / DsY;
            _data = new float[TotalTimesteps, Nz, nyDs, nxDs];
            var rng = new Random(42);

            for (int t = 0; t < TotalTimesteps; t++)
            for (int k = 0; k < Nz; k++)
            for (int j = 0; j < nyDs; j++)
            for (int i = 0; i < nxDs; i++)
            {
                double decay = 1.0 - 0.03 * t;
                _data[t, k, j, i] = (float)(
                    0.3 + 0.7 * Math.Exp(-0.05 * (i + j + k)) * decay
                    + 0.05 * rng.NextDouble());
            }
        });

        IsLoading = false;
        StatusMessage = $"Demo data loaded — {Nx}×{Ny}×{Nz} grid, {TotalTimesteps} timesteps";
        RebuildScene();
        BuildWellModel();
    }

    // ── Scene building ────────────────────────────────────────────────────

    private void RebuildScene()
    {
        if (_data is null) return;

        int t    = Math.Clamp(CurrentTimestep, 0, TotalTimesteps - 1);
        int nxDs = Nx / DsX, nyDs = Ny / DsY;
        var group = new Model3DGroup();

        for (int k = 0; k < Nz; k++)
        for (int j = 0; j < nyDs; j++)
        for (int i = 0; i < nxDs; i++)
        {
            float v = _data[t, k, j, i];
            var   c = JetColor(v);
            var   brush = new SolidColorBrush(Color.FromArgb(
                (byte)(VoxelOpacity * 255), c.R, c.G, c.B));
            brush.Freeze();

            var mesh = BuildBox(i * DsX, j * DsY, k, DsX - 0.2, DsY - 0.2, 0.8);
            var mat  = new DiffuseMaterial(brush);
            group.Children.Add(new GeometryModel3D(mesh, mat));
        }

        group.Freeze();
        ReservoirModel3D = group;
    }

    private static MeshGeometry3D BuildBox(
        double x, double y, double z, double sx, double sy, double sz)
    {
        var mesh = new MeshGeometry3D();
        double x0 = x, x1 = x + sx, y0 = y, y1 = y + sy, z0 = z, z1 = z + sz;

        var pts = new[]
        {
            new Point3D(x0,y0,z0), new Point3D(x1,y0,z0),
            new Point3D(x1,y1,z0), new Point3D(x0,y1,z0),
            new Point3D(x0,y0,z1), new Point3D(x1,y0,z1),
            new Point3D(x1,y1,z1), new Point3D(x0,y1,z1),
        };
        foreach (var p in pts) mesh.Positions.Add(p);

        // 6 faces × 2 triangles
        int[] faces = { 0,1,2, 0,2,3,  4,6,5, 4,7,6,
                        0,4,5, 0,5,1,  2,6,7, 2,7,3,
                        0,3,7, 0,7,4,  1,5,6, 1,6,2 };
        foreach (int idx in faces) mesh.TriangleIndices.Add(idx);

        mesh.Freeze();
        return mesh;
    }

    private void BuildWellModel()
    {
        var group = new Model3DGroup();
        // Representative Norne well locations
        (int i, int j)[] wells = { (10,28), (15,35), (20,42), (25,55), (30,70) };

        foreach (var (wi, wj) in wells)
        {
            var mesh = new MeshGeometry3D();
            double cx = wi * DsX / (double)DsX + 0.5;
            double cy = wj * DsY / (double)DsY + 0.5;

            for (int k = 0; k <= Nz; k++)
            {
                mesh.Positions.Add(new Point3D(cx - 0.4, cy - 0.4, k));
                mesh.Positions.Add(new Point3D(cx + 0.4, cy + 0.4, k));
            }
            for (int k = 0; k < Nz; k++)
            {
                int b = k * 2;
                mesh.TriangleIndices.Add(b);   mesh.TriangleIndices.Add(b+1); mesh.TriangleIndices.Add(b+2);
                mesh.TriangleIndices.Add(b+1); mesh.TriangleIndices.Add(b+3); mesh.TriangleIndices.Add(b+2);
            }
            mesh.Freeze();
            var mat = new DiffuseMaterial(new SolidColorBrush(Colors.Yellow));
            mat.Brush.Freeze();
            group.Children.Add(new GeometryModel3D(mesh, mat));
        }

        group.Freeze();
        WellModel3D = group;
    }

    // ── Jet colourmap ─────────────────────────────────────────────────────

    private static Color JetColor(float v)
    {
        v = Math.Clamp(v, 0f, 1f);
        byte r, g, b;
        if      (v < 0.25f) { r = 0;   g = (byte)(v * 4 * 255); b = 255; }
        else if (v < 0.5f)  { r = 0;   g = 255; b = (byte)((1 - (v - 0.25f) * 4) * 255); }
        else if (v < 0.75f) { r = (byte)((v - 0.5f) * 4 * 255); g = 255; b = 0; }
        else                { r = 255; g = (byte)((1 - (v - 0.75f) * 4) * 255); b = 0; }
        return Color.FromRgb(r, g, b);
    }
}
