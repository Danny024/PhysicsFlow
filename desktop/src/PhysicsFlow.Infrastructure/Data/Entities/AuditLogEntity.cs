using System;
using System.ComponentModel.DataAnnotations.Schema;

namespace PhysicsFlow.Infrastructure.Data.Entities;

[Table("audit_log")]
public class AuditLogEntity
{
    [Column("id")]             public int      Id           { get; set; }
    [Column("timestamp")]      public DateTime Timestamp    { get; set; }
    [Column("event_type")]     public string   EventType    { get; set; } = string.Empty;
    [Column("project_id")]     public string?  ProjectId    { get; set; }
    [Column("project_name")]   public string?  ProjectName  { get; set; }
    [Column("description")]    public string   Description  { get; set; } = string.Empty;
    [Column("entity_type")]    public string?  EntityType   { get; set; }
    [Column("entity_id")]      public string?  EntityId     { get; set; }
    [Column("username")]       public string?  Username     { get; set; }
    [Column("hostname")]       public string?  Hostname     { get; set; }
    [Column("success")]        public bool     Success      { get; set; } = true;
    [Column("error_message")]  public string?  ErrorMessage { get; set; }
    [Column("metadata")]       public string?  Metadata     { get; set; }
}
