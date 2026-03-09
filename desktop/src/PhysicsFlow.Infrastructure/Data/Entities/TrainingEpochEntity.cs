using System;
using System.ComponentModel.DataAnnotations.Schema;

namespace PhysicsFlow.Infrastructure.Data.Entities;

[Table("training_epochs")]
public class TrainingEpochEntity
{
    [Column("id")]           public int    Id         { get; set; }
    [Column("run_id")]       public string RunId      { get; set; } = string.Empty;
    [Column("epoch")]        public int    Epoch      { get; set; }
    [Column("loss_total")]   public double LossTotal  { get; set; }
    [Column("loss_pde")]     public double LossPde    { get; set; }
    [Column("loss_data")]    public double LossData   { get; set; }
    [Column("loss_well")]    public double LossWell   { get; set; }
    [Column("loss_ic")]      public double LossIc     { get; set; }
    [Column("loss_bc")]      public double LossBc     { get; set; }
    [Column("gpu_util")]     public double? GpuUtil   { get; set; }
    [Column("recorded_at")]  public DateTime RecordedAt { get; set; }
}
