using System.Collections.Generic;
using System.Threading.Tasks;
using PhysicsFlow.Infrastructure.Data.Entities;

namespace PhysicsFlow.Infrastructure.Reports;

public interface IReportService
{
    /// <summary>Generates a History Matching summary PDF report.</summary>
    Task<string> GenerateHMSummaryReportAsync(
        ProjectEntity project,
        List<HMIterationEntity> iterations,
        List<WellObservationEntity> wellObs,
        string outputPath);

    /// <summary>Generates a EUR / Production Forecast PDF report.</summary>
    Task<string> GenerateEURReportAsync(
        ProjectEntity project,
        EURReportData eurData,
        string outputPath);
}

/// <summary>Data transfer object carrying EUR values for report generation.</summary>
public record EURReportData(
    double EurOilP10,
    double EurOilP50,
    double EurOilP90,
    double EurGasP50,
    double RecoveryFactorP50,
    double PeakOilRateP50,
    int    ForecastHorizonYears,
    Dictionary<string, double[]> WellEurP50,
    string[] WellNames
);
