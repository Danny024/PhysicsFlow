using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ClosedXML.Excel;
using Microsoft.Extensions.Logging;
using PhysicsFlow.Infrastructure.Data.Entities;
using PhysicsFlow.Infrastructure.Reports;

namespace PhysicsFlow.Infrastructure.Export;

public class ExcelExportService : IExcelExportService
{
    private readonly ILogger<ExcelExportService> _log;

    public ExcelExportService(ILogger<ExcelExportService> log) => _log = log;

    // ── Well data export ──────────────────────────────────────────────────

    public Task<string> ExportWellDataAsync(
        List<WellObservationEntity> wellObs,
        string projectName,
        string outputPath)
    {
        return Task.Run(() =>
        {
            outputPath = EnsureXlsx(outputPath);
            using var wb = new XLWorkbook();

            // Summary sheet
            var ws = wb.Worksheets.Add("Summary");
            ws.Cell("A1").Value = $"PhysicsFlow — Well Production Data Export";
            ws.Cell("A1").Style.Font.Bold = true; ws.Cell("A1").Style.Font.FontSize = 14;
            ws.Cell("A2").Value = $"Project: {projectName}";
            ws.Cell("A3").Value = $"Exported: {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC";
            ws.Cell("A4").Value = $"Total records: {wellObs.Count}";

            ws.Cell("A6").Value = "Well";
            ws.Cell("B6").Value = "Records";
            ws.Cell("C6").Value = "Date Range";
            StyleHeader(ws.Row(6));

            var wellNames = wellObs.Select(w => w.WellName).Distinct().OrderBy(n => n).ToList();
            int r = 7;
            foreach (var wn in wellNames)
            {
                var rows = wellObs.Where(w => w.WellName == wn).OrderBy(w => w.Date).ToList();
                ws.Cell(r, 1).Value = wn;
                ws.Cell(r, 2).Value = rows.Count;
                ws.Cell(r, 3).Value = $"{rows.First().Date:yyyy-MM-dd} — {rows.Last().Date:yyyy-MM-dd}";
                r++;
            }
            ws.Columns().AdjustToContents();

            // One sheet per well
            foreach (var wn in wellNames)
            {
                var sheet = wb.Worksheets.Add(wn.Length > 31 ? wn[..31] : wn);
                sheet.Cell("A1").Value = $"Well: {wn}"; sheet.Cell("A1").Style.Font.Bold = true;

                var hdrs = new[]
                {
                    "Date","Timestep",
                    "Obs WOPR","Obs WWPR","Obs WGPR","Obs WBHP","Obs WWCT",
                    "Sim WOPR","Sim WWPR","Sim WGPR","Sim WBHP",
                    "P10 WOPR","P90 WOPR","Source"
                };
                for (int c = 0; c < hdrs.Length; c++) sheet.Cell(3, c+1).Value = hdrs[c];
                StyleHeader(sheet.Row(3));

                var rows = wellObs.Where(w => w.WellName == wn).OrderBy(w => w.Date).ToList();
                for (int i = 0; i < rows.Count; i++)
                {
                    var row = rows[i]; int dr = i + 4;
                    sheet.Cell(dr, 1).Value = row.Date.ToString("yyyy-MM-dd");
                    sheet.Cell(dr, 2).Value = row.Timestep;
                    SetOpt(sheet, dr, 3,  row.ObsWopr);
                    SetOpt(sheet, dr, 4,  row.ObsWwpr);
                    SetOpt(sheet, dr, 5,  row.ObsWgpr);
                    SetOpt(sheet, dr, 6,  row.ObsWbhp);
                    SetOpt(sheet, dr, 7,  row.ObsWwct);
                    SetOpt(sheet, dr, 8,  row.SimWopr);
                    SetOpt(sheet, dr, 9,  row.SimWwpr);
                    SetOpt(sheet, dr, 10, row.SimWgpr);
                    SetOpt(sheet, dr, 11, row.SimWbhp);
                    SetOpt(sheet, dr, 12, row.P10Wopr);
                    SetOpt(sheet, dr, 13, row.P90Wopr);
                    sheet.Cell(dr, 14).Value = row.DataSource;
                }
                sheet.Columns().AdjustToContents();
            }

            wb.SaveAs(outputPath);
            _log.LogInformation("Well data exported: {Path}", outputPath);
            return outputPath;
        });
    }

    // ── Ensemble statistics export ────────────────────────────────────────

    public Task<string> ExportEnsembleStatisticsAsync(
        EURReportData eur,
        string projectName,
        string outputPath)
    {
        return Task.Run(() =>
        {
            outputPath = EnsureXlsx(outputPath);
            using var wb = new XLWorkbook();

            var ws = wb.Worksheets.Add("EUR Summary");
            ws.Cell("A1").Value = $"PhysicsFlow — Ensemble Statistics — {projectName}";
            ws.Cell("A1").Style.Font.Bold = true; ws.Cell("A1").Style.Font.FontSize = 14;
            ws.Cell("A2").Value = $"Exported: {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC";

            ws.Cell("A4").Value = "EUR Summary"; ws.Cell("A4").Style.Font.Bold = true;

            var labels = new[]
            {
                "EUR Oil P10 (MMstb)", "EUR Oil P50 (MMstb)", "EUR Oil P90 (MMstb)",
                "EUR Gas P50 (Bscf)", "Recovery Factor P50",
                "Peak Oil Rate P50 (stb/d)", "Forecast Horizon (years)"
            };
            var values = new object[]
            {
                eur.EurOilP10, eur.EurOilP50, eur.EurOilP90,
                eur.EurGasP50, eur.RecoveryFactorP50,
                eur.PeakOilRateP50, eur.ForecastHorizonYears
            };
            for (int i = 0; i < labels.Length; i++)
            {
                ws.Cell(5+i, 1).Value = labels[i];
                ws.Cell(5+i, 2).Value = (ClosedXML.Excel.XLCellValue)(dynamic)values[i];
            }
            ws.Columns().AdjustToContents();

            // Per-well P50 rates
            if (eur.WellEurP50.Count > 0)
            {
                var ws2 = wb.Worksheets.Add("Well P50 Rates");
                ws2.Cell("A1").Value = "Monthly P50 Oil Rates by Well (stb/d)";
                ws2.Cell("A1").Style.Font.Bold = true;
                ws2.Cell(3, 1).Value = "Month";
                int col = 1;
                foreach (var wn in eur.WellNames) { col++; ws2.Cell(3, col).Value = wn; }
                StyleHeader(ws2.Row(3));

                int maxLen = eur.WellEurP50.Values.Max(rr => rr.Length);
                for (int month = 0; month < maxLen; month++)
                {
                    int row = month + 4;
                    ws2.Cell(row, 1).Value = month + 1;
                    col = 1;
                    foreach (var wn in eur.WellNames)
                    {
                        col++;
                        if (eur.WellEurP50.TryGetValue(wn, out var rates) && month < rates.Length)
                            ws2.Cell(row, col).Value = rates[month];
                    }
                }
                ws2.Columns().AdjustToContents();
            }

            wb.SaveAs(outputPath);
            _log.LogInformation("Ensemble statistics exported: {Path}", outputPath);
            return outputPath;
        });
    }

    // ── Training history export ───────────────────────────────────────────

    public Task<string> ExportTrainingHistoryAsync(
        List<TrainingEpochEntity> epochs,
        string runId,
        string outputPath)
    {
        return Task.Run(() =>
        {
            outputPath = EnsureXlsx(outputPath);
            using var wb = new XLWorkbook();
            var ws = wb.Worksheets.Add("Training History");

            ws.Cell("A1").Value = $"PhysicsFlow — Training Loss History — Run {runId}";
            ws.Cell("A1").Style.Font.Bold = true;
            ws.Cell("A2").Value = $"Total epochs: {epochs.Count}";

            var hdrs = new[]
            {
                "Epoch","Loss Total","Loss PDE","Loss Data",
                "Loss Well","Loss IC","Loss BC","GPU Util (%)","Recorded At"
            };
            for (int i = 0; i < hdrs.Length; i++) ws.Cell(4, i+1).Value = hdrs[i];
            StyleHeader(ws.Row(4));

            for (int i = 0; i < epochs.Count; i++)
            {
                var ep = epochs[i]; int r = i + 5;
                ws.Cell(r, 1).Value = ep.Epoch;
                ws.Cell(r, 2).Value = ep.LossTotal;
                ws.Cell(r, 3).Value = ep.LossPde;
                ws.Cell(r, 4).Value = ep.LossData;
                ws.Cell(r, 5).Value = ep.LossWell;
                ws.Cell(r, 6).Value = ep.LossIc;
                ws.Cell(r, 7).Value = ep.LossBc;
                if (ep.GpuUtil.HasValue) ws.Cell(r, 8).Value = ep.GpuUtil.Value;
                ws.Cell(r, 9).Value = ep.RecordedAt.ToString("yyyy-MM-dd HH:mm:ss");
            }
            for (int c = 2; c <= 8; c++)
                ws.Column(c).Style.NumberFormat.Format = "0.000000";
            ws.Columns().AdjustToContents();

            wb.SaveAs(outputPath);
            _log.LogInformation("Training history exported: {Path}", outputPath);
            return outputPath;
        });
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private static void StyleHeader(IXLRow row)
    {
        row.Style.Font.Bold = true;
        row.Style.Fill.BackgroundColor = XLColor.FromArgb(0xD4, 0xED, 0xDA);
        row.Style.Border.BottomBorder = XLBorderStyleValues.Medium;
    }

    private static void SetOpt(IXLWorksheet ws, int r, int c, double? v)
    {
        if (v.HasValue) ws.Cell(r, c).Value = v.Value;
    }

    private static string EnsureXlsx(string p) =>
        p.EndsWith(".xlsx", StringComparison.OrdinalIgnoreCase) ? p : p + ".xlsx";
}
