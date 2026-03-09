using System;
using System.ComponentModel.DataAnnotations.Schema;

namespace PhysicsFlow.Infrastructure.Data.Entities;

[Table("hm_iterations")]
public class HMIterationEntity
{
    [Column("id")]            public int     Id          { get; set; }
    [Column("project_id")]    public string  ProjectId   { get; set; } = string.Empty;
    [Column("hm_run_id")]     public string  HmRunId     { get; set; } = string.Empty;
    [Column("iteration")]     public int     Iteration   { get; set; }
    [Column("mismatch")]      public double  Mismatch    { get; set; }
    [Column("alpha")]         public double  Alpha       { get; set; }
    [Column("s_cumulative")]  public double  SCumulative { get; set; }
    [Column("improvement_pct")] public double? ImprovementPct { get; set; }
    [Column("converged")]     public bool    Converged   { get; set; }
    [Column("recorded_at")]   public DateTime RecordedAt { get; set; }
    // P10/P50/P90 stored as JSON strings in SQLite
    [Column("p50_snapshot")]  public string? P50Snapshot { get; set; }
    [Column("per_well_rmse")] public string? PerWellRmse { get; set; }
}
