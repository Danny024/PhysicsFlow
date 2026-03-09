using System;
using System.IO;
using Microsoft.EntityFrameworkCore;
using PhysicsFlow.Infrastructure.Data.Entities;

namespace PhysicsFlow.Infrastructure.Data;

/// <summary>
/// Entity Framework Core DbContext for the PhysicsFlow SQLite database.
///
/// Database file location (resolved in order):
///   1. PHYSICSFLOW_DB_PATH environment variable
///   2. %APPDATA%\PhysicsFlow\physicsflow.db  (Windows)
///   3. ~/.physicsflow/physicsflow.db          (Linux/Mac)
///
/// The same SQLite file is shared with the Python engine — the schema
/// is created by Python (SQLAlchemy) and this .NET context provides
/// read/write access from the UI layer (project registry, audit log display,
/// run history, well observation charts).
/// </summary>
public class PhysicsFlowDbContext : DbContext
{
    public PhysicsFlowDbContext(DbContextOptions<PhysicsFlowDbContext> options)
        : base(options) { }

    // ── DbSets ───────────────────────────────────────────────────────────

    public DbSet<ProjectEntity>         Projects         { get; set; } = null!;
    public DbSet<SimulationRunEntity>   SimulationRuns   { get; set; } = null!;
    public DbSet<TrainingEpochEntity>   TrainingEpochs   { get; set; } = null!;
    public DbSet<HMIterationEntity>     HMIterations     { get; set; } = null!;
    public DbSet<WellObservationEntity> WellObservations { get; set; } = null!;
    public DbSet<ModelVersionEntity>    ModelVersions    { get; set; } = null!;
    public DbSet<AuditLogEntity>        AuditLog         { get; set; } = null!;

    // ── Model configuration ───────────────────────────────────────────────

    protected override void OnModelCreating(ModelBuilder mb)
    {
        base.OnModelCreating(mb);

        // projects
        mb.Entity<ProjectEntity>(e =>
        {
            e.ToTable("projects");
            e.HasKey(x => x.Id);
            e.HasIndex(x => x.Name);
            e.HasIndex(x => x.PfprojPath).IsUnique();
            e.Property(x => x.CreatedAt).HasColumnType("TEXT");
            e.Property(x => x.ModifiedAt).HasColumnType("TEXT");
            e.Property(x => x.LastOpenedAt).HasColumnType("TEXT");
        });

        // simulation_runs
        mb.Entity<SimulationRunEntity>(e =>
        {
            e.ToTable("simulation_runs");
            e.HasKey(x => x.Id);
            e.HasIndex(x => new { x.ProjectId, x.RunType });
            e.HasIndex(x => x.StartedAt);
            e.HasOne<ProjectEntity>()
             .WithMany()
             .HasForeignKey(x => x.ProjectId)
             .OnDelete(DeleteBehavior.Cascade);
        });

        // training_epochs
        mb.Entity<TrainingEpochEntity>(e =>
        {
            e.ToTable("training_epochs");
            e.HasKey(x => x.Id);
            e.HasIndex(x => x.RunId);
            e.HasOne<SimulationRunEntity>()
             .WithMany()
             .HasForeignKey(x => x.RunId)
             .OnDelete(DeleteBehavior.Cascade);
        });

        // hm_iterations
        mb.Entity<HMIterationEntity>(e =>
        {
            e.ToTable("hm_iterations");
            e.HasKey(x => x.Id);
            e.HasIndex(x => new { x.ProjectId, x.HmRunId });
            e.HasOne<ProjectEntity>()
             .WithMany()
             .HasForeignKey(x => x.ProjectId)
             .OnDelete(DeleteBehavior.Cascade);
        });

        // well_observations
        mb.Entity<WellObservationEntity>(e =>
        {
            e.ToTable("well_observations");
            e.HasKey(x => x.Id);
            e.HasIndex(x => new { x.ProjectId, x.WellName, x.Date });
            e.HasOne<ProjectEntity>()
             .WithMany()
             .HasForeignKey(x => x.ProjectId)
             .OnDelete(DeleteBehavior.Cascade);
        });

        // model_versions
        mb.Entity<ModelVersionEntity>(e =>
        {
            e.ToTable("model_versions");
            e.HasKey(x => x.Id);
            e.HasIndex(x => new { x.ProjectId, x.ModelType });
            e.HasOne<ProjectEntity>()
             .WithMany()
             .HasForeignKey(x => x.ProjectId)
             .OnDelete(DeleteBehavior.Cascade);
        });

        // audit_log — read-only from .NET (Python owns writes)
        mb.Entity<AuditLogEntity>(e =>
        {
            e.ToTable("audit_log");
            e.HasKey(x => x.Id);
            e.HasIndex(x => x.Timestamp);
            e.HasIndex(x => new { x.EventType, x.ProjectId });
        });
    }

    // ── Static factory ────────────────────────────────────────────────────

    public static string ResolveDbPath()
    {
        var env = Environment.GetEnvironmentVariable("PHYSICSFLOW_DB_PATH");
        if (!string.IsNullOrWhiteSpace(env)) return env;

        var appData = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
        var dir     = Path.Combine(appData, "PhysicsFlow");
        Directory.CreateDirectory(dir);
        return Path.Combine(dir, "physicsflow.db");
    }

    public static DbContextOptions<PhysicsFlowDbContext> BuildOptions(string? dbPath = null)
    {
        var path = dbPath ?? ResolveDbPath();
        return new DbContextOptionsBuilder<PhysicsFlowDbContext>()
            .UseSqlite($"Data Source={path};Cache=Shared")
            .Options;
    }
}
