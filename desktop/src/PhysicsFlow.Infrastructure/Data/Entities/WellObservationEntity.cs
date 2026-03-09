using System;
using System.ComponentModel.DataAnnotations.Schema;

namespace PhysicsFlow.Infrastructure.Data.Entities;

[Table("well_observations")]
public class WellObservationEntity
{
    [Column("id")]          public int      Id        { get; set; }
    [Column("project_id")]  public string   ProjectId { get; set; } = string.Empty;
    [Column("well_name")]   public string   WellName  { get; set; } = string.Empty;
    [Column("date")]        public DateTime Date      { get; set; }
    [Column("timestep")]    public int      Timestep  { get; set; }

    [Column("obs_wopr")]    public double?  ObsWopr   { get; set; }
    [Column("obs_wwpr")]    public double?  ObsWwpr   { get; set; }
    [Column("obs_wgpr")]    public double?  ObsWgpr   { get; set; }
    [Column("obs_wbhp")]    public double?  ObsWbhp   { get; set; }
    [Column("obs_wwct")]    public double?  ObsWwct   { get; set; }

    [Column("sim_wopr")]    public double?  SimWopr   { get; set; }
    [Column("sim_wwpr")]    public double?  SimWwpr   { get; set; }
    [Column("sim_wgpr")]    public double?  SimWgpr   { get; set; }
    [Column("sim_wbhp")]    public double?  SimWbhp   { get; set; }

    [Column("p10_wopr")]    public double?  P10Wopr   { get; set; }
    [Column("p90_wopr")]    public double?  P90Wopr   { get; set; }
    [Column("data_source")] public string   DataSource { get; set; } = "eclipse";
}
