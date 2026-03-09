using System;
using System.ComponentModel.DataAnnotations.Schema;

namespace PhysicsFlow.Infrastructure.Data.Entities;

[Table("model_versions")]
public class ModelVersionEntity
{
    [Column("id")]               public string  Id             { get; set; } = Guid.NewGuid().ToString();
    [Column("project_id")]       public string  ProjectId      { get; set; } = string.Empty;
    [Column("model_type")]       public string  ModelType      { get; set; } = string.Empty;
    [Column("version_tag")]      public string  VersionTag     { get; set; } = "latest";
    [Column("file_path")]        public string  FilePath       { get; set; } = string.Empty;
    [Column("file_size_bytes")]  public long?   FileSizeBytes  { get; set; }
    [Column("file_sha256")]      public string? FileSha256     { get; set; }
    [Column("epochs_trained")]   public int?    EpochsTrained  { get; set; }
    [Column("loss_total")]       public double? LossTotal      { get; set; }
    [Column("rmse_pressure")]    public double? RmsePressure   { get; set; }
    [Column("rmse_sw")]          public double? RmseSw         { get; set; }
    [Column("is_active")]        public bool    IsActive       { get; set; } = true;
    [Column("created_at")]       public DateTime CreatedAt     { get; set; } = DateTime.UtcNow;
    [Column("notes")]            public string? Notes          { get; set; }
}
