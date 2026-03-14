using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace PhysicsFlow.ViewModels;

/// <summary>
/// ViewModel for the 5-step New Project Setup wizard.
///
/// Step 1 — Grid: Nx/Ny/Nz dimensions or Eclipse deck import
/// Step 2 — Wells: COMPDAT import or manual entry
/// Step 3 — PVT: fluid properties
/// Step 4 — Schedule: production/injection targets
/// Step 5 — Review: summary + save as .pfproj
/// </summary>
public partial class ProjectSetupViewModel : ObservableObject
{
    // ── Step navigation ──────────────────────────────────────────────────

    [ObservableProperty] [NotifyPropertyChangedFor(nameof(Step1Active), nameof(Step1Done), nameof(Step1Pending),
                                                    nameof(Step2Active), nameof(Step2Pending),
                                                    nameof(Step3Active), nameof(Step3Pending),
                                                    nameof(Step4Active), nameof(Step4Pending),
                                                    nameof(Step5Active), nameof(Step5Pending),
                                                    nameof(CanGoBack),   nameof(NextButtonLabel))]
    private int currentStep = 1;

    public bool Step1Active   => CurrentStep == 1;
    public bool Step1Done     => CurrentStep > 1;
    public bool Step1Pending  => CurrentStep < 1;
    public bool Step2Active   => CurrentStep == 2;
    public bool Step2Pending  => CurrentStep < 2;
    public bool Step3Active   => CurrentStep == 3;
    public bool Step3Pending  => CurrentStep < 3;
    public bool Step4Active   => CurrentStep == 4;
    public bool Step4Pending  => CurrentStep < 4;
    public bool Step5Active   => CurrentStep == 5;
    public bool Step5Pending  => CurrentStep < 5;

    public bool   CanGoBack      => CurrentStep > 1;
    public string NextButtonLabel => CurrentStep == 5 ? "Save Project ✓" : "Next →";

    // ── Step 1: Grid ─────────────────────────────────────────────────────

    [ObservableProperty] [NotifyPropertyChangedFor(nameof(TotalCells))] private int gridNx = 46;
    [ObservableProperty] [NotifyPropertyChangedFor(nameof(TotalCells))] private int gridNy = 112;
    [ObservableProperty] [NotifyPropertyChangedFor(nameof(TotalCells))] private int gridNz = 22;
    [ObservableProperty] private double gridDx = 50.0;
    [ObservableProperty] private double gridDy = 50.0;
    [ObservableProperty] private double gridDz = 20.0;
    [ObservableProperty] private double gridDepth = 2000.0;
    [ObservableProperty] private string? eclipseDeckPath;

    public int TotalCells  => GridNx * GridNy * GridNz;
    public int ActiveCells => TotalCells;   // Updated after ACTNUM parse

    // ── Step 2: Wells ─────────────────────────────────────────────────────

    public ObservableCollection<WellRow> Wells { get; } = new();

    public int WellCount     => Wells.Count;
    public int ProducerCount => Wells.Count(w => w.WellType == "PRODUCER");
    public int InjectorCount => Wells.Count(w => w.WellType == "INJECTOR");

    // ── Step 3: PVT ───────────────────────────────────────────────────────

    [ObservableProperty] private double pvtInitialPressure = 277.0;
    [ObservableProperty] private double pvtTemperature = 90.0;
    [ObservableProperty] private double pvtApiGravity = 40.0;
    [ObservableProperty] private double pvtGasGravity = 0.7;
    [ObservableProperty] private double pvtSwi = 0.20;

    // ── Step 4: Schedule ──────────────────────────────────────────────────

    public ObservableCollection<ScheduleRow> ScheduleEntries { get; } = new();

    // ── Step 5: Review ────────────────────────────────────────────────────

    [ObservableProperty] private string projectName = $"NewProject_{DateTime.Today:yyyy-MM-dd}";
    [ObservableProperty] private string savePath = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments),
        "PhysicsFlow");

    [ObservableProperty] private string validationMessage = string.Empty;
    [ObservableProperty] private string reviewSummary = string.Empty;

    // Events
    public event EventHandler<string>? ProjectSaved;
    public event EventHandler? WizardCancelled;

    // ── Commands ─────────────────────────────────────────────────────────

    [RelayCommand]
    private void Next()
    {
        ValidationMessage = string.Empty;
        if (!ValidateCurrentStep()) return;

        if (CurrentStep < 5)
        {
            CurrentStep++;
            if (CurrentStep == 5)
                RefreshReviewSummary();
        }
        else
        {
            _ = SaveProjectAsync();
        }
    }

    [RelayCommand(CanExecute = nameof(CanGoBack))]
    private void Back() => CurrentStep--;

    [RelayCommand]
    private void Cancel() => WizardCancelled?.Invoke(this, EventArgs.Empty);

    [RelayCommand]
    private void BrowseEclipseDeck()
    {
        // Microsoft.Win32.OpenFileDialog
        var dlg = new Microsoft.Win32.OpenFileDialog
        {
            Title  = "Select Eclipse DATA file",
            Filter = "Eclipse Deck (*.DATA)|*.DATA|All files (*.*)|*.*",
        };
        if (dlg.ShowDialog() == true)
        {
            EclipseDeckPath = dlg.FileName;
            ParseEclipseDeck(dlg.FileName);
        }
    }

    [RelayCommand]
    private void ImportCompdat()
    {
        if (string.IsNullOrEmpty(EclipseDeckPath)) return;
        ParseEclipseDeck(EclipseDeckPath);
    }

    [RelayCommand]
    private void LoadNorneDefaults()
    {
        Wells.Clear();
        // 22 producers
        var producers = new[]
        {
            ("B-1H", 9, 29), ("B-2H", 11, 35), ("B-3H", 11, 35), ("B-4H", 13, 43),
            ("B-5H", 13, 43), ("C-1H", 20, 54), ("C-2H", 20, 54), ("C-3H", 20, 58),
            ("C-4H", 22, 63), ("C-4AH", 24, 63), ("D-1H", 24, 89), ("D-2H", 25, 85),
            ("D-3AH", 25, 85), ("D-4H", 27, 100), ("E-1H", 27, 100), ("E-2H", 31, 68),
            ("E-3H", 34, 71), ("E-4H", 35, 78), ("E-4AH", 36, 84), ("F-1H", 36, 84),
            ("F-2H", 37, 88), ("F-3H", 38, 92)
        };
        foreach (var (n, i, j) in producers)
            Wells.Add(new WellRow { Name = n, WellType = "PRODUCER",
                                    PerfCount = 3, BhpLimitBar = 150.0 });

        // 9 water injectors + 4 gas injectors (abbreviated)
        foreach (var n in new[] { "E-3AH", "E-4BH", "F-4H", "B-4BH" })
            Wells.Add(new WellRow { Name = n, WellType = "INJECTOR",
                                    PerfCount = 2, BhpLimitBar = 400.0 });

        OnPropertyChanged(nameof(WellCount));
        OnPropertyChanged(nameof(ProducerCount));
        OnPropertyChanged(nameof(InjectorCount));
    }

    [RelayCommand]
    private void AddWell()
    {
        Wells.Add(new WellRow
        {
            Name = $"WELL-{Wells.Count + 1}",
            WellType = "PRODUCER",
            PerfCount = 1,
            BhpLimitBar = 150.0,
        });
        OnPropertyChanged(nameof(WellCount));
        OnPropertyChanged(nameof(ProducerCount));
        OnPropertyChanged(nameof(InjectorCount));
    }

    [RelayCommand]
    private void LoadNornePvt()
    {
        PvtInitialPressure = 277.0;
        PvtTemperature     = 90.0;
        PvtApiGravity      = 40.0;
        PvtGasGravity      = 0.7;
        PvtSwi             = 0.20;
    }

    [RelayCommand]
    private void AddScheduleEntry()
    {
        ScheduleEntries.Add(new ScheduleRow
        {
            WellName     = Wells.FirstOrDefault()?.Name ?? "FIELD",
            StartDate    = "2001-01-01",
            EndDate      = "2001-12-31",
            ControlMode  = "ORAT",
            TargetValue  = 5000,
            Unit         = "stb/day",
        });
    }

    [RelayCommand]
    private void BrowseSavePath()
    {
        // FolderBrowserDialog equivalent
        var dlg = new System.Windows.Forms.FolderBrowserDialog
        {
            Description         = "Select folder to save project",
            SelectedPath        = SavePath,
            ShowNewFolderButton = true,
        };
        if (dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            SavePath = dlg.SelectedPath;
    }

    // ── Private helpers ──────────────────────────────────────────────────

    private bool ValidateCurrentStep()
    {
        switch (CurrentStep)
        {
            case 1:
                if (GridNx <= 0 || GridNy <= 0 || GridNz <= 0)
                { ValidationMessage = "Grid dimensions must be > 0"; return false; }
                if (TotalCells > 5_000_000)
                { ValidationMessage = "Warning: >5M cells may be slow. Continue?"; }
                break;
            case 2:
                if (Wells.Count == 0)
                { ValidationMessage = "Add at least one well (or load Norne defaults)"; return false; }
                break;
            case 3:
                if (PvtInitialPressure <= 0)
                { ValidationMessage = "Initial pressure must be positive"; return false; }
                break;
            case 5:
                if (string.IsNullOrWhiteSpace(ProjectName))
                { ValidationMessage = "Enter a project name"; return false; }
                break;
        }
        return true;
    }

    private void ParseEclipseDeck(string deckPath)
    {
        // Stub — will call Python engine via gRPC to parse deck
        // For now, just update grid dims from filename hint
        if (deckPath.ToUpperInvariant().Contains("NORNE"))
        {
            GridNx = 46; GridNy = 112; GridNz = 22;
            LoadNorneDefaults();
        }
    }

    private void RefreshReviewSummary()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Project   : {ProjectName}");
        sb.AppendLine($"Grid      : {GridNx} × {GridNy} × {GridNz}  ({TotalCells:N0} cells)");
        sb.AppendLine($"Cell size : dx={GridDx}m  dy={GridDy}m  dz={GridDz}m");
        sb.AppendLine($"Depth     : {GridDepth}m TVDSS");
        sb.AppendLine($"Wells     : {WellCount} ({ProducerCount} producers, {InjectorCount} injectors)");
        sb.AppendLine($"PVT       : Pi={PvtInitialPressure} bar  T={PvtTemperature}°C  API={PvtApiGravity}°");
        sb.AppendLine($"Schedule  : {ScheduleEntries.Count} control periods");
        sb.AppendLine();
        sb.AppendLine("Eclipse deck : " + (EclipseDeckPath ?? "(none)"));
        ReviewSummary = sb.ToString();
    }

    private async Task SaveProjectAsync()
    {
        ValidationMessage = string.Empty;
        if (!ValidateCurrentStep()) return;

        try
        {
            Directory.CreateDirectory(SavePath);
            var filePath = Path.Combine(SavePath, ProjectName + ".pfproj");

            // Build JSON project file
            var projectJson = BuildProjectJson();
            await File.WriteAllTextAsync(filePath, projectJson);

            ProjectSaved?.Invoke(this, filePath);
        }
        catch (Exception ex)
        {
            ValidationMessage = $"Failed to save: {ex.Message}";
        }
    }

    private string BuildProjectJson()
    {
        var sb = new StringBuilder();
        sb.AppendLine("{");
        sb.AppendLine($"  \"version\": \"1.1.0\",");
        sb.AppendLine($"  \"name\": \"{ProjectName}\",");
        sb.AppendLine($"  \"created\": \"{DateTime.Now:O}\",");
        sb.AppendLine($"  \"modified\": \"{DateTime.Now:O}\",");
        sb.AppendLine($"  \"grid\": {{");
        sb.AppendLine($"    \"nx\": {GridNx}, \"ny\": {GridNy}, \"nz\": {GridNz},");
        sb.AppendLine($"    \"dx\": {GridDx}, \"dy\": {GridDy}, \"dz\": {GridDz},");
        sb.AppendLine($"    \"depth\": {GridDepth}");
        sb.AppendLine($"  }},");
        sb.AppendLine($"  \"pvt\": {{");
        sb.AppendLine($"    \"initial_pressure_bar\": {PvtInitialPressure},");
        sb.AppendLine($"    \"temperature_c\": {PvtTemperature},");
        sb.AppendLine($"    \"api_gravity\": {PvtApiGravity},");
        sb.AppendLine($"    \"gas_gravity\": {PvtGasGravity},");
        sb.AppendLine($"    \"swi\": {PvtSwi}");
        sb.AppendLine($"  }},");

        // Serialize actual wells collection
        var wellList = Wells.ToList();
        sb.AppendLine($"  \"wells\": [");
        for (int wi = 0; wi < wellList.Count; wi++)
        {
            var w     = wellList[wi];
            var comma = wi < wellList.Count - 1 ? "," : "";
            sb.AppendLine($"    {{\"name\": \"{w.Name}\", \"type\": \"{w.WellType}\", " +
                          $"\"perf_count\": {w.PerfCount}, \"bhp_limit_bar\": {w.BhpLimitBar}}}{comma}");
        }
        sb.AppendLine($"  ],");

        sb.AppendLine($"  \"schedule\": [],");
        sb.AppendLine($"  \"eclipse_deck_path\": {(EclipseDeckPath == null ? "null" : $"\"{EclipseDeckPath}\"")},");
        sb.AppendLine($"  \"model_paths\": {{}},");
        sb.AppendLine($"  \"hm_results\": {{}},");
        sb.AppendLine($"  \"forecast\": {{}},");
        sb.AppendLine($"  \"notes\": \"\"");
        sb.AppendLine("}");
        return sb.ToString();
    }
}

// ── Row models for DataGrids ─────────────────────────────────────────────

public class WellRow : ObservableObject
{
    public string Name { get; set; } = string.Empty;
    public string WellType { get; set; } = "PRODUCER";
    public int PerfCount { get; set; }
    public double BhpLimitBar { get; set; }
}

public class ScheduleRow : ObservableObject
{
    public string WellName { get; set; } = string.Empty;
    public string StartDate { get; set; } = string.Empty;
    public string EndDate { get; set; } = string.Empty;
    public string ControlMode { get; set; } = "ORAT";
    public double TargetValue { get; set; }
    public string Unit { get; set; } = "stb/day";
}
