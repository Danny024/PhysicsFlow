using System;
using System.ComponentModel.DataAnnotations.Schema;

namespace PhysicsFlow.Infrastructure.Data.Entities;

[Table("projects")]
public class ProjectEntity
{
    [Column("id")]           public string    Id          { get; set; } = Guid.NewGuid().ToString();
    [Column("name")]         public string    Name        { get; set; } = string.Empty;
    [Column("pfproj_path")]  public string    PfprojPath  { get; set; } = string.Empty;
    [Column("created_at")]   public DateTime  CreatedAt   { get; set; } = DateTime.UtcNow;
    [Column("modified_at")]  public DateTime  ModifiedAt  { get; set; } = DateTime.UtcNow;
    [Column("last_opened_at")] public DateTime? LastOpenedAt { get; set; }

    [Column("nx")] public int? Nx { get; set; }
    [Column("ny")] public int? Ny { get; set; }
    [Column("nz")] public int? Nz { get; set; }
    [Column("n_wells")] public int? NWells { get; set; }

    [Column("pino_trained")]  public bool  PinoTrained  { get; set; }
    [Column("hm_completed")]  public bool  HmCompleted  { get; set; }
    [Column("hm_converged")]  public bool  HmConverged  { get; set; }
    [Column("best_mismatch")] public double? BestMismatch { get; set; }
    [Column("notes")]         public string? Notes        { get; set; }
}
