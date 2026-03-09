using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;
using PhysicsFlow.Infrastructure.Data.Entities;

namespace PhysicsFlow.Infrastructure.Data;

/// <summary>
/// Application-level database service used by ViewModels.
/// Wraps EF Core operations for the PhysicsFlow SQLite database.
/// The database schema is owned by the Python engine (SQLAlchemy).
/// This service provides read-optimised queries for UI display.
/// </summary>
public class AppDbService
{
    private readonly ILogger<AppDbService> _log;
    private readonly DbContextOptions<PhysicsFlowDbContext> _options;

    public AppDbService(ILogger<AppDbService> log)
    {
        _log     = log;
        _options = PhysicsFlowDbContext.BuildOptions();
        EnsureCreated();
    }

    // ── Schema initialisation ─────────────────────────────────────────────

    private void EnsureCreated()
    {
        try
        {
            using var db = new PhysicsFlowDbContext(_options);
            db.Database.EnsureCreated();
            _log.LogInformation("Database ready: {Path}", PhysicsFlowDbContext.ResolveDbPath());
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "Failed to initialise database");
        }
    }

    private PhysicsFlowDbContext Ctx() => new(_options);

    // ── Projects ──────────────────────────────────────────────────────────

    public async Task<List<ProjectEntity>> GetRecentProjectsAsync(int limit = 20)
    {
        await using var db = Ctx();
        return await db.Projects
            .OrderByDescending(p => p.LastOpenedAt ?? p.ModifiedAt)
            .Take(limit)
            .ToListAsync();
    }

    public async Task<ProjectEntity?> GetProjectByPathAsync(string pfprojPath)
    {
        await using var db = Ctx();
        return await db.Projects
            .FirstOrDefaultAsync(p => p.PfprojPath == pfprojPath);
    }

    public async Task UpsertProjectAsync(ProjectEntity project)
    {
        await using var db = Ctx();
        var existing = await db.Projects.FindAsync(project.Id);
        if (existing is null)
            db.Projects.Add(project);
        else
        {
            existing.Name         = project.Name;
            existing.ModifiedAt   = DateTime.UtcNow;
            existing.Nx           = project.Nx;
            existing.Ny           = project.Ny;
            existing.Nz           = project.Nz;
            existing.NWells       = project.NWells;
            existing.PinoTrained  = project.PinoTrained;
            existing.HmCompleted  = project.HmCompleted;
            existing.HmConverged  = project.HmConverged;
            existing.BestMismatch = project.BestMismatch;
        }
        await db.SaveChangesAsync();
    }

    public async Task MarkProjectOpenedAsync(string projectId)
    {
        await using var db = Ctx();
        var proj = await db.Projects.FindAsync(projectId);
        if (proj is not null)
        {
            proj.LastOpenedAt = DateTime.UtcNow;
            await db.SaveChangesAsync();
        }
    }

    // ── Simulation runs ───────────────────────────────────────────────────

    public async Task<List<SimulationRunEntity>> GetRecentRunsAsync(
        string projectId, int limit = 50)
    {
        await using var db = Ctx();
        return await db.SimulationRuns
            .Where(r => r.ProjectId == projectId)
            .OrderByDescending(r => r.StartedAt)
            .Take(limit)
            .ToListAsync();
    }

    public async Task<SimulationRunEntity?> GetLastTrainingRunAsync(string projectId)
    {
        await using var db = Ctx();
        return await db.SimulationRuns
            .Where(r => r.ProjectId == projectId
                     && r.RunType   == "training"
                     && r.Status    == "completed")
            .OrderByDescending(r => r.CompletedAt)
            .FirstOrDefaultAsync();
    }

    // ── Training epoch history ────────────────────────────────────────────

    public async Task<List<TrainingEpochEntity>> GetEpochHistoryAsync(string runId)
    {
        await using var db = Ctx();
        return await db.TrainingEpochs
            .Where(e => e.RunId == runId)
            .OrderBy(e => e.Epoch)
            .ToListAsync();
    }

    // ── HM iterations ─────────────────────────────────────────────────────

    public async Task<List<HMIterationEntity>> GetHMHistoryAsync(
        string projectId, string? hmRunId = null)
    {
        await using var db = Ctx();
        var q = db.HMIterations.Where(h => h.ProjectId == projectId);
        if (hmRunId is not null)
            q = q.Where(h => h.HmRunId == hmRunId);
        return await q.OrderBy(h => h.Iteration).ToListAsync();
    }

    public async Task<List<string>> GetHMRunIdsAsync(string projectId)
    {
        await using var db = Ctx();
        return await db.HMIterations
            .Where(h => h.ProjectId == projectId)
            .Select(h => h.HmRunId)
            .Distinct()
            .ToListAsync();
    }

    // ── Well observations ─────────────────────────────────────────────────

    public async Task<List<WellObservationEntity>> GetWellTimeSeriesAsync(
        string projectId, string wellName)
    {
        await using var db = Ctx();
        return await db.WellObservations
            .Where(w => w.ProjectId == projectId && w.WellName == wellName)
            .OrderBy(w => w.Date)
            .ToListAsync();
    }

    public async Task<List<string>> GetWellNamesAsync(string projectId)
    {
        await using var db = Ctx();
        return await db.WellObservations
            .Where(w => w.ProjectId == projectId)
            .Select(w => w.WellName)
            .Distinct()
            .OrderBy(n => n)
            .ToListAsync();
    }

    // ── Model versions ────────────────────────────────────────────────────

    public async Task<ModelVersionEntity?> GetActiveModelAsync(
        string projectId, string modelType)
    {
        await using var db = Ctx();
        return await db.ModelVersions
            .Where(m => m.ProjectId == projectId
                     && m.ModelType  == modelType
                     && m.IsActive   == true)
            .OrderByDescending(m => m.CreatedAt)
            .FirstOrDefaultAsync();
    }

    public async Task<List<ModelVersionEntity>> GetModelHistoryAsync(
        string projectId, string modelType)
    {
        await using var db = Ctx();
        return await db.ModelVersions
            .Where(m => m.ProjectId == projectId && m.ModelType == modelType)
            .OrderByDescending(m => m.CreatedAt)
            .ToListAsync();
    }

    // ── Audit log ─────────────────────────────────────────────────────────

    public async Task<List<AuditLogEntity>> GetAuditLogAsync(
        string? projectId = null, int limit = 200)
    {
        await using var db = Ctx();
        var q = db.AuditLog.AsQueryable();
        if (projectId is not null)
            q = q.Where(a => a.ProjectId == projectId);
        return await q
            .OrderByDescending(a => a.Timestamp)
            .Take(limit)
            .ToListAsync();
    }

    // ── Dashboard summary ─────────────────────────────────────────────────

    public async Task<DbSummary> GetSummaryAsync()
    {
        await using var db = Ctx();
        return new DbSummary
        {
            TotalProjects    = await db.Projects.CountAsync(),
            TotalRuns        = await db.SimulationRuns.CountAsync(),
            TotalHMIter      = await db.HMIterations.CountAsync(),
            TotalAuditEvents = await db.AuditLog.CountAsync(),
            LastActivity     = await db.AuditLog
                                        .OrderByDescending(a => a.Timestamp)
                                        .Select(a => (DateTime?)a.Timestamp)
                                        .FirstOrDefaultAsync(),
        };
    }
}

public record DbSummary(
    int TotalProjects    = 0,
    int TotalRuns        = 0,
    int TotalHMIter      = 0,
    int TotalAuditEvents = 0,
    DateTime? LastActivity = null
);
