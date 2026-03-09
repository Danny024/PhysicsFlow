using System.Collections.Generic;
using System.Threading.Tasks;
using PhysicsFlow.Infrastructure.Data.Entities;
using PhysicsFlow.Infrastructure.Reports;

namespace PhysicsFlow.Infrastructure.Export;

public interface IExcelExportService
{
    /// <summary>Exports observed + simulated well production time series to Excel.</summary>
    Task<string> ExportWellDataAsync(
        List<WellObservationEntity> wellObs,
        string projectName,
        string outputPath);

    /// <summary>Exports P10/P50/P90 ensemble statistics to Excel.</summary>
    Task<string> ExportEnsembleStatisticsAsync(
        EURReportData eurData,
        string projectName,
        string outputPath);

    /// <summary>Exports epoch-by-epoch training loss history to Excel.</summary>
    Task<string> ExportTrainingHistoryAsync(
        List<TrainingEpochEntity> epochs,
        string runId,
        string outputPath);
}
