using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using PhysicsFlow.Infrastructure.Data.Entities;
using QuestPDF.Fluent;
using QuestPDF.Helpers;
using QuestPDF.Infrastructure;

namespace PhysicsFlow.Infrastructure.Reports;

public class ReportService : IReportService
{
    private readonly ILogger<ReportService> _log;

    public ReportService(ILogger<ReportService> log)
    {
        _log = log;
        QuestPDF.Settings.License = LicenseType.Community;
    }

    // ── HM Summary Report ─────────────────────────────────────────────────

    public Task<string> GenerateHMSummaryReportAsync(
        ProjectEntity project,
        List<HMIterationEntity> iterations,
        List<WellObservationEntity> wellObs,
        string outputPath)
    {
        return Task.Run(() =>
        {
            outputPath = EnsurePdf(outputPath);
            Document.Create(c => c.Page(p =>
            {
                p.Size(PageSizes.A4);
                p.Margin(2, Unit.Centimetre);
                p.DefaultTextStyle(t => t.FontSize(10).FontColor(Colors.Grey.Darken3));
                p.Header().Element(HMHeader(project));
                p.Content().Column(col => HMContent(col, project, iterations, wellObs));
                p.Footer().Element(Footer);
            })).GeneratePdf(outputPath);
            _log.LogInformation("HM summary report: {Path}", outputPath);
            return outputPath;
        });
    }

    private static Action<IContainer> HMHeader(ProjectEntity project) =>
        c => c.Column(col =>
        {
            col.Item().Row(row => row.RelativeItem().Column(inner =>
            {
                inner.Item().Text("PhysicsFlow — History Matching Summary")
                     .Bold().FontSize(16).FontColor(Colors.Green.Darken2);
                inner.Item().Text($"Project: {project.Name}  |  Grid: {project.Nx}×{project.Ny}×{project.Nz}")
                     .FontSize(11).FontColor(Colors.Grey.Darken2);
                inner.Item().Text($"Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC")
                     .FontSize(9).FontColor(Colors.Grey.Medium);
            }));
            col.Item().PaddingTop(8).LineHorizontal(1).LineColor(Colors.Green.Darken2);
        });

    private static void HMContent(
        ColumnDescriptor col,
        ProjectEntity project,
        List<HMIterationEntity> iterations,
        List<WellObservationEntity> wellObs)
    {
        // Project summary
        col.Item().PaddingTop(16).Column(inner =>
        {
            inner.Item().Text("Project Summary").Bold().FontSize(12);
            inner.Item().PaddingTop(4).Table(t =>
            {
                t.ColumnsDefinition(c =>
                {
                    c.RelativeColumn(); c.RelativeColumn();
                    c.RelativeColumn(); c.RelativeColumn();
                });
                t.Header(h =>
                {
                    foreach (var lbl in new[] { "Grid", "Wells", "Ensemble", "Converged" })
                        h.Cell().Background(Colors.Green.Lighten4).Padding(4).Text(lbl).Bold();
                });
                t.Cell().Padding(4).Text($"{project.Nx}×{project.Ny}×{project.Nz}");
                t.Cell().Padding(4).Text($"{project.NWells}");
                t.Cell().Padding(4).Text(iterations.Count > 0 ? "200" : "—");
                t.Cell().Padding(4).Text(project.HmConverged == true ? "Yes ✓" : "No");
            });
        });

        // Convergence table
        if (iterations.Count > 0)
        {
            col.Item().PaddingTop(20).Column(inner =>
            {
                inner.Item().Text("Convergence History").Bold().FontSize(12);
                inner.Item().PaddingTop(4).Table(t =>
                {
                    t.ColumnsDefinition(c =>
                    {
                        c.ConstantColumn(60); c.RelativeColumn();
                        c.RelativeColumn();   c.RelativeColumn();
                    });
                    t.Header(h =>
                    {
                        foreach (var lbl in new[] { "Iter", "Mismatch", "α", "s_cum" })
                            h.Cell().Background(Colors.Green.Lighten4).Padding(4).Text(lbl).Bold();
                    });
                    foreach (var it in iterations.Take(25))
                    {
                        t.Cell().Padding(3).Text(it.Iteration.ToString());
                        t.Cell().Padding(3).Text($"{it.Mismatch:F4}");
                        t.Cell().Padding(3).Text($"{it.Alpha:F4}");
                        t.Cell().Padding(3).Text($"{it.SCumulative:F4}");
                    }
                });
            });
        }

        // Per-well RMSE
        var wellNames = wellObs.Select(w => w.WellName).Distinct().OrderBy(n => n).ToList();
        if (wellNames.Count > 0)
        {
            col.Item().PaddingTop(20).Column(inner =>
            {
                inner.Item().Text("Per-Well Data Mismatch (RMSE)").Bold().FontSize(12);
                inner.Item().PaddingTop(4).Table(t =>
                {
                    t.ColumnsDefinition(c =>
                    {
                        c.RelativeColumn(2); c.RelativeColumn();
                        c.RelativeColumn();  c.RelativeColumn();
                    });
                    t.Header(h =>
                    {
                        foreach (var lbl in new[] { "Well", "WOPR RMSE", "WWPR RMSE", "WBHP RMSE" })
                            h.Cell().Background(Colors.Green.Lighten4).Padding(4).Text(lbl).Bold();
                    });
                    foreach (var wn in wellNames.Take(30))
                    {
                        var rows  = wellObs.Where(w => w.WellName == wn).ToList();
                        double wo = Rmse(rows, w => w.ObsWopr, w => w.SimWopr);
                        double ww = Rmse(rows, w => w.ObsWwpr, w => w.SimWwpr);
                        double wb = Rmse(rows, w => w.ObsWbhp, w => w.SimWbhp);
                        t.Cell().Padding(3).Text(wn);
                        t.Cell().Padding(3).Text(double.IsNaN(wo) ? "—" : $"{wo:F4}");
                        t.Cell().Padding(3).Text(double.IsNaN(ww) ? "—" : $"{ww:F4}");
                        t.Cell().Padding(3).Text(double.IsNaN(wb) ? "—" : $"{wb:F4}");
                    }
                });
            });
        }

        // Best mismatch highlight
        if (project.BestMismatch.HasValue)
        {
            col.Item().PaddingTop(20).Background(Colors.Green.Lighten5).Padding(12).Column(inner =>
            {
                inner.Item().Text($"Best Overall Mismatch: {project.BestMismatch:F4}")
                     .Bold().FontSize(14).FontColor(Colors.Green.Darken2);
                inner.Item().Text("Target < 0.10 for reservoir certification screening")
                     .FontSize(9).FontColor(Colors.Grey.Medium);
            });
        }
    }

    // ── EUR / Forecast Report ─────────────────────────────────────────────

    public Task<string> GenerateEURReportAsync(
        ProjectEntity project,
        EURReportData eur,
        string outputPath)
    {
        return Task.Run(() =>
        {
            outputPath = EnsurePdf(outputPath);
            Document.Create(c => c.Page(p =>
            {
                p.Size(PageSizes.A4);
                p.Margin(2, Unit.Centimetre);
                p.DefaultTextStyle(t => t.FontSize(10).FontColor(Colors.Grey.Darken3));
                p.Header().Element(EURHeader(project));
                p.Content().Column(col => EURContent(col, eur));
                p.Footer().Element(Footer);
            })).GeneratePdf(outputPath);
            _log.LogInformation("EUR report: {Path}", outputPath);
            return outputPath;
        });
    }

    private static Action<IContainer> EURHeader(ProjectEntity project) =>
        c => c.Column(col =>
        {
            col.Item().Row(row => row.RelativeItem().Column(inner =>
            {
                inner.Item().Text("PhysicsFlow — EUR & Production Forecast Report")
                     .Bold().FontSize(16).FontColor(Colors.Blue.Darken2);
                inner.Item().Text($"Project: {project.Name}")
                     .FontSize(11).FontColor(Colors.Grey.Darken2);
                inner.Item().Text($"Generated: {DateTime.UtcNow:yyyy-MM-dd HH:mm} UTC")
                     .FontSize(9).FontColor(Colors.Grey.Medium);
            }));
            col.Item().PaddingTop(8).LineHorizontal(1).LineColor(Colors.Blue.Darken2);
        });

    private static void EURContent(ColumnDescriptor col, EURReportData eur)
    {
        // EUR summary table
        col.Item().PaddingTop(16).Column(inner =>
        {
            inner.Item().Text("Estimated Ultimate Recovery (EUR)").Bold().FontSize(12);
            inner.Item().PaddingTop(4).Table(t =>
            {
                t.ColumnsDefinition(c =>
                {
                    c.RelativeColumn(2); c.RelativeColumn();
                    c.RelativeColumn();  c.RelativeColumn();
                });
                t.Header(h =>
                {
                    foreach (var lbl in new[] { "Metric", "P10 (Optimistic)", "P50 (Base)", "P90 (Conservative)" })
                        h.Cell().Background(Colors.Blue.Lighten4).Padding(4).Text(lbl).Bold();
                });

                void Row(string metric, string p10, string p50, string p90)
                {
                    t.Cell().Padding(4).Text(metric);
                    t.Cell().Padding(4).Text(p10);
                    t.Cell().Padding(4).Text(p50);
                    t.Cell().Padding(4).Text(p90);
                }

                Row("EUR Oil (MMstb)",       $"{eur.EurOilP10:F2}", $"{eur.EurOilP50:F2}", $"{eur.EurOilP90:F2}");
                Row("EUR Gas (Bscf)",        "—", $"{eur.EurGasP50:F2}", "—");
                Row("Recovery Factor",       "—", $"{eur.RecoveryFactorP50:P1}", "—");
                Row("Peak Oil Rate (stb/d)", "—", $"{eur.PeakOilRateP50:N0}", "—");
                Row("Forecast Horizon",      "—", $"{eur.ForecastHorizonYears} years", "—");
            });
        });

        // Per-well EUR
        if (eur.WellEurP50.Count > 0)
        {
            col.Item().PaddingTop(20).Column(inner =>
            {
                inner.Item().Text("P50 EUR by Well").Bold().FontSize(12);
                inner.Item().PaddingTop(4).Table(t =>
                {
                    t.ColumnsDefinition(c => { c.RelativeColumn(2); c.RelativeColumn(); });
                    t.Header(h =>
                    {
                        h.Cell().Background(Colors.Blue.Lighten4).Padding(4).Text("Well").Bold();
                        h.Cell().Background(Colors.Blue.Lighten4).Padding(4).Text("EUR P50 (MMstb)").Bold();
                    });
                    foreach (var (wn, rates) in eur.WellEurP50)
                    {
                        double cum = rates.Sum() * 30 / 1e6;
                        t.Cell().Padding(3).Text(wn);
                        t.Cell().Padding(3).Text($"{cum:F3}");
                    }
                });
            });
        }

        // Disclaimer
        col.Item().PaddingTop(30).Background(Colors.Yellow.Lighten4).Padding(10).Column(inner =>
        {
            inner.Item().Text("Disclaimer").Bold().FontSize(9);
            inner.Item().Text(
                "EUR estimates are generated by the PhysicsFlow PINO surrogate and are for " +
                "screening purposes only. Not to be used for reserve certification without " +
                "independent validation by a qualified reservoir engineer using OPM FLOW or Eclipse.")
                .FontSize(8).FontColor(Colors.Grey.Darken2);
        });
    }

    // ── Shared footer ─────────────────────────────────────────────────────

    private static void Footer(IContainer c) =>
        c.Row(row =>
        {
            row.RelativeItem().Text(t =>
            {
                t.Span("PhysicsFlow v1.2.0 — ").FontSize(8).FontColor(Colors.Grey.Medium);
                t.Span("Confidential").FontSize(8).FontColor(Colors.Grey.Medium);
            });
            row.ConstantItem(60).AlignRight().Text(t =>
            {
                t.Span("Page ").FontSize(8);
                t.CurrentPageNumber().FontSize(8);
                t.Span(" / ").FontSize(8);
                t.TotalPages().FontSize(8);
            });
        });

    // ── Helpers ───────────────────────────────────────────────────────────

    private static string EnsurePdf(string p) =>
        p.EndsWith(".pdf", StringComparison.OrdinalIgnoreCase) ? p : p + ".pdf";

    private static double Rmse(
        List<WellObservationEntity> rows,
        Func<WellObservationEntity, double?> obs,
        Func<WellObservationEntity, double?> sim)
    {
        var pairs = rows
            .Select(r => (o: obs(r), s: sim(r)))
            .Where(p => p.o.HasValue && p.s.HasValue)
            .ToList();
        return pairs.Count == 0
            ? double.NaN
            : Math.Sqrt(pairs.Average(p => Math.Pow(p.o!.Value - p.s!.Value, 2)));
    }
}
