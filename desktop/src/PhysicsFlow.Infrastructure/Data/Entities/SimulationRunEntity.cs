using System;
using System.ComponentModel.DataAnnotations.Schema;

namespace PhysicsFlow.Infrastructure.Data.Entities;

[Table("simulation_runs")]
public class SimulationRunEntity
{
    [Column("id")]          public string   Id        { get; set; } = Guid.NewGuid().ToString();
    [Column("project_id")]  public string   ProjectId { get; set; } = string.Empty;
    [Column("run_type")]    public string   RunType   { get; set; } = string.Empty;
    [Column("status")]      public string   Status    { get; set; } = "pending";

    [Column("started_at")]      public DateTime  StartedAt    { get; set; } = DateTime.UtcNow;
    [Column("completed_at")]    public DateTime? CompletedAt  { get; set; }
    [Column("duration_seconds")] public double?  DurationSeconds { get; set; }

    [Column("input_hash")]   public string?  InputHash  { get; set; }
    [Column("random_seed")]  public int?     RandomSeed { get; set; }
    [Column("n_timesteps")]  public int?     NTimesteps { get; set; }
    [Column("n_ensemble")]   public int?     NEnsemble  { get; set; }

    [Column("rmse_pressure")]     public double? RmsePressure    { get; set; }
    [Column("rmse_sw")]           public double? RmseSw          { get; set; }
    [Column("loss_total")]        public double? LossTotal        { get; set; }
    [Column("loss_pde")]          public double? LossPde          { get; set; }
    [Column("loss_data")]         public double? LossData         { get; set; }
    [Column("epochs_completed")]  public int?    EpochsCompleted  { get; set; }
    [Column("best_epoch")]        public int?    BestEpoch        { get; set; }
    [Column("model_path")]        public string? ModelPath        { get; set; }
    [Column("error_message")]     public string? ErrorMessage     { get; set; }
}
