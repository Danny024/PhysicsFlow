using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using Color = System.Windows.Media.Color;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace PhysicsFlow.ViewModels;

/// <summary>
/// ViewModel for the 2D cross-section viewer.
/// Renders I / J / K plane slices of reservoir properties
/// as colour-mapped bitmaps using WriteableBitmap.
/// </summary>
public partial class CrossSectionViewModel : ObservableObject
{
    // ── Observable properties ─────────────────────────────────────────────

    [ObservableProperty] private string selectedProperty  = "Pressure";
    [ObservableProperty] private string selectedColourmap = "Jet";
    [ObservableProperty] private int    sliceIndex        = 0;
    [ObservableProperty] private string sliceIndexLabel   = "Slice 0";
    [ObservableProperty] private bool   showWellOverlay   = true;
    [ObservableProperty] private string statusMessage     = "Ready";

    [ObservableProperty] private BitmapSource? iPlaneImage;
    [ObservableProperty] private BitmapSource? jPlaneImage;
    [ObservableProperty] private BitmapSource? kPlaneImage;

    public ObservableCollection<string> PropertyOptions { get; } = new()
        { "Pressure", "Water Saturation", "Permeability", "Porosity" };
    public ObservableCollection<string> ColourmapOptions { get; } = new()
        { "Jet", "Viridis", "Seismic", "Greys" };

    // Grid dimensions (Norne)
    private const int Nx = 46, Ny = 112, Nz = 22;

    // Synthetic 3D property field [nz, ny, nx]
    private float[,,]? _field;

    // ── Constructor ───────────────────────────────────────────────────────

    public CrossSectionViewModel()
    {
        _ = LoadDemoFieldAsync();
    }

    // ── Property change reactions ─────────────────────────────────────────

    partial void OnSelectedPropertyChanged(string value)  => RefreshAll();
    partial void OnSelectedColourmapChanged(string value) => RefreshAll();
    partial void OnSliceIndexChanged(int value)
    {
        SliceIndexLabel = $"Slice {value}";
        RefreshAll();
    }

    // ── Commands ──────────────────────────────────────────────────────────

    [RelayCommand]
    private void ExportImage() =>
        StatusMessage = "Image export stub — implement save dialog in v1.3";

    // ── Demo data ─────────────────────────────────────────────────────────

    private async Task LoadDemoFieldAsync()
    {
        StatusMessage = "Generating synthetic field...";
        await Task.Run(() =>
        {
            _field = new float[Nz, Ny, Nx];
            var rng = new Random(1234);
            for (int k = 0; k < Nz; k++)
            for (int j = 0; j < Ny; j++)
            for (int i = 0; i < Nx; i++)
                _field[k, j, i] = (float)(
                    Math.Exp(-0.02 * (i + j * 0.5 + k * 2))
                    + 0.1 * rng.NextDouble());
        });
        SliceIndex    = Ny / 2;
        StatusMessage = $"Demo field loaded — {Nx}×{Ny}×{Nz}";
        RefreshAll();
    }

    // ── Slice rendering ───────────────────────────────────────────────────

    private void RefreshAll()
    {
        if (_field is null) return;
        IPlaneImage = RenderIPlane(Math.Clamp(SliceIndex, 0, Nx - 1));
        JPlaneImage = RenderJPlane(Math.Clamp(SliceIndex, 0, Ny - 1));
        KPlaneImage = RenderKPlane(Math.Clamp(SliceIndex, 0, Nz - 1));
    }

    // I-plane: fix I → bitmap width=Ny, height=Nz
    private BitmapSource RenderIPlane(int iIdx)
    {
        const int w = Ny, h = Nz;
        var px = new byte[h * w * 4];
        for (int k = 0; k < h; k++)
        for (int j = 0; j < w; j++)
        {
            var c = MapColour(_field![k, j, iIdx]);
            int o = (k * w + j) * 4;
            px[o] = c.B; px[o+1] = c.G; px[o+2] = c.R; px[o+3] = 255;
        }
        return ToBitmap(px, w, h);
    }

    // J-plane: fix J → bitmap width=Nx, height=Nz
    private BitmapSource RenderJPlane(int jIdx)
    {
        const int w = Nx, h = Nz;
        var px = new byte[h * w * 4];
        for (int k = 0; k < h; k++)
        for (int i = 0; i < w; i++)
        {
            var c = MapColour(_field![k, jIdx, i]);
            int o = (k * w + i) * 4;
            px[o] = c.B; px[o+1] = c.G; px[o+2] = c.R; px[o+3] = 255;
        }
        return ToBitmap(px, w, h);
    }

    // K-plane: fix K → bitmap width=Nx, height=Ny
    private BitmapSource RenderKPlane(int kIdx)
    {
        const int w = Nx, h = Ny;
        var px = new byte[h * w * 4];
        for (int j = 0; j < h; j++)
        for (int i = 0; i < w; i++)
        {
            var c = MapColour(_field![kIdx, j, i]);
            int o = (j * w + i) * 4;
            px[o] = c.B; px[o+1] = c.G; px[o+2] = c.R; px[o+3] = 255;
        }
        return ToBitmap(px, w, h);
    }

    private static BitmapSource ToBitmap(byte[] pixels, int w, int h)
    {
        var bmp = new WriteableBitmap(w, h, 96, 96, PixelFormats.Bgr32, null);
        bmp.WritePixels(new Int32Rect(0, 0, w, h), pixels, w * 4, 0);
        bmp.Freeze();
        return bmp;
    }

    // ── Colourmaps ────────────────────────────────────────────────────────

    private Color MapColour(float v)
    {
        v = Math.Clamp(v, 0f, 1f);
        return SelectedColourmap switch
        {
            "Viridis" => ViridisColor(v),
            "Seismic" => SeismicColor(v),
            "Greys"   => Color.FromRgb((byte)(v*255), (byte)(v*255), (byte)(v*255)),
            _         => JetColor(v),
        };
    }

    private static Color JetColor(float v)
    {
        byte r, g, b;
        if      (v < 0.25f) { r = 0;   g = (byte)(v*4*255); b = 255; }
        else if (v < 0.5f)  { r = 0;   g = 255; b = (byte)((1-(v-0.25f)*4)*255); }
        else if (v < 0.75f) { r = (byte)((v-0.5f)*4*255); g = 255; b = 0; }
        else                { r = 255; g = (byte)((1-(v-0.75f)*4)*255); b = 0; }
        return Color.FromRgb(r, g, b);
    }

    private static Color ViridisColor(float v)
    {
        byte r = (byte)(Math.Clamp(-0.5 + v * 2.5, 0, 1) * 255);
        byte g = (byte)(Math.Sin(v * Math.PI) * 220);
        byte b = (byte)(Math.Clamp(1 - v * 2, 0, 1) * 255);
        return Color.FromRgb(r, g, b);
    }

    private static Color SeismicColor(float v)
    {
        if (v < 0.5f)
        {
            byte c = (byte)((0.5f - v) * 2 * 255);
            return Color.FromRgb(c, c, 255);
        }
        else
        {
            byte c = (byte)((v - 0.5f) * 2 * 255);
            return Color.FromRgb(255, (byte)(255-c), (byte)(255-c));
        }
    }
}
