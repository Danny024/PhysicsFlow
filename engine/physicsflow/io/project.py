"""
PhysicsFlow — .pfproj Project File Format.

A .pfproj file is a JSON document that captures all configuration and
state needed to reproduce a PhysicsFlow study:

    {
      "version": "1.1.0",
      "name": "Norne History Match Q4-2024",
      "created": "2024-11-01T12:00:00",
      "grid": { ... GridConfig ... },
      "pvt":  { ... PVTConfig ...  },
      "wells": [ { ... WellConfig ... }, ... ],
      "schedule": { ... },
      "model_paths": { "pino": "models/pino_latest.pt", "ccr": "models/ccr.pkl" },
      "hm_results": { "best_mismatch": 0.12, "n_iterations": 18, ... },
      "notes": "..."
    }

Usage:
    proj = PhysicsFlowProject.new("Norne Q4")
    proj.save("my_study.pfproj")
    proj = PhysicsFlowProject.load("my_study.pfproj")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


PFPROJ_VERSION = "1.2.0"
PFPROJ_EXTENSION = ".pfproj"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-structures stored in the project file
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScheduleEntry:
    """One production/injection target period."""
    start_date: str          # ISO 8601 date string
    end_date: str
    well_name: str
    control_mode: str        # 'ORAT', 'WRAT', 'GRAT', 'BHP', 'RATE'
    target_value: float
    unit: str


@dataclass
class HMResults:
    """Summary of the last history matching run."""
    n_ensemble: int = 0
    n_iterations: int = 0
    best_mismatch: float = float('inf')
    final_alpha: float = 0.0
    s_cumulative: float = 0.0
    converged: bool = False
    completed_at: Optional[str] = None
    per_well_rmse: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelPaths:
    """Paths to trained model checkpoints (relative to project file)."""
    pino: Optional[str] = None    # FNO/PINO surrogate
    ccr:  Optional[str] = None    # CCR well surrogate
    vcae: Optional[str] = None    # VCAE encoder
    ddim: Optional[str] = None    # DDIM prior


# ─────────────────────────────────────────────────────────────────────────────
# Main project dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PhysicsFlowProject:
    """
    Complete PhysicsFlow study project.

    Serialises/deserialises to/from .pfproj JSON format.
    """
    name: str
    version: str = PFPROJ_VERSION
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    modified: str = field(default_factory=lambda: datetime.now().isoformat())

    # Reservoir model configuration
    grid: Dict[str, Any] = field(default_factory=dict)
    pvt: Dict[str, Any] = field(default_factory=dict)
    wells: List[Dict[str, Any]] = field(default_factory=list)
    schedule: List[Dict[str, Any]] = field(default_factory=list)

    # Eclipse deck reference (optional)
    eclipse_deck_path: Optional[str] = None
    las_files: List[str] = field(default_factory=list)

    # Trained models
    model_paths: ModelPaths = field(default_factory=ModelPaths)

    # History matching results
    hm_results: HMResults = field(default_factory=HMResults)

    # Forecast results (P10/P50/P90 arrays per well)
    forecast: Dict[str, Any] = field(default_factory=dict)

    # Free-text notes
    notes: str = ''

    # ── Factory methods ────────────────────────────────────────────────────

    @classmethod
    def new(cls, name: str) -> "PhysicsFlowProject":
        """Create a blank project with Norne defaults."""
        from ..config import EngineConfig
        cfg = EngineConfig()
        return cls(
            name=name,
            grid={
                'nx': cfg.default_nx, 'ny': cfg.default_ny, 'nz': cfg.default_nz,
                'dx': cfg.default_dx, 'dy': cfg.default_dy, 'dz': cfg.default_dz,
                'depth': 2000.0,
            },
            pvt={
                'initial_pressure_bar': 277.0,
                'temperature_c': 90.0,
                'api_gravity': 40.0,
                'gas_gravity': 0.7,
                'swi': 0.2,
            },
        )

    @classmethod
    def from_eclipse(cls, name: str, deck_path: str) -> "PhysicsFlowProject":
        """
        Create a project by reading an Eclipse deck.
        Grid, wells, and PVT are populated from the deck.
        """
        from ..io.eclipse_reader import EclipseReader
        reader = EclipseReader(deck_path)
        Nx, Ny, Nz = reader._dimensions()
        proj = cls.new(name)
        proj.eclipse_deck_path = str(Path(deck_path).resolve())
        proj.grid.update({'nx': Nx, 'ny': Ny, 'nz': Nz})

        try:
            wells = reader.wells()
            proj.wells = [_well_to_dict(w) for w in wells]
        except Exception:
            pass

        return proj

    # ── Serialisation ──────────────────────────────────────────────────────

    def save(self, path, password: Optional[str] = None) -> Path:
        """
        Save project to .pfproj file.

        Parameters
        ----------
        path     : destination path (extension forced to .pfproj)
        password : if provided, the saved file is additionally encrypted to
                   <path>.enc using AES-256-GCM (requires `cryptography` package)

        Returns the path of the written file (.pfproj or .pfproj.enc).
        """
        path = Path(path)
        if path.suffix.lower() != PFPROJ_EXTENSION:
            path = path.with_suffix(PFPROJ_EXTENSION)

        self.modified = datetime.now().isoformat()

        data = {
            'version':           self.version,
            'name':              self.name,
            'created':           self.created,
            'modified':          self.modified,
            'grid':              self.grid,
            'pvt':               self.pvt,
            'wells':             self.wells,
            'schedule':          self.schedule,
            'eclipse_deck_path': self.eclipse_deck_path,
            'las_files':         self.las_files,
            'model_paths':       asdict(self.model_paths),
            'hm_results':        asdict(self.hm_results),
            'forecast':          self.forecast,
            'notes':             self.notes,
        }

        path.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')

        if password is not None:
            from .crypto import encrypt_pfproj, _HAS_CRYPTOGRAPHY
            if not _HAS_CRYPTOGRAPHY:
                raise ImportError(
                    "The 'cryptography' package is required for encryption. "
                    "Install with: pip install cryptography"
                )
            enc_path = encrypt_pfproj(path, password=password, remove_original=True)
            return enc_path

        return path

    @classmethod
    def load(cls, path, password: Optional[str] = None) -> "PhysicsFlowProject":
        """
        Load a project from a .pfproj (or .pfproj.enc) file.

        Parameters
        ----------
        path     : path to the .pfproj or .pfproj.enc file
        password : required if the file is AES-256-GCM encrypted (.pfproj.enc)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Project file not found: {path}")

        # Handle encrypted files
        if str(path).endswith('.pfproj.enc') or path.suffix == '.enc':
            if password is None:
                raise ValueError(
                    "This project file is encrypted. Provide `password` to decrypt it."
                )
            from .crypto import decrypt_pfproj
            import tempfile, os
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_plain = Path(tmpdir) / "project.pfproj"
                decrypt_pfproj(path, password=password, output_path=tmp_plain)
                data = json.loads(tmp_plain.read_text(encoding='utf-8'))
        else:
            data = json.loads(path.read_text(encoding='utf-8'))

        model_paths = ModelPaths(**data.get('model_paths', {}))
        hm_raw = data.get('hm_results', {})
        hm_results = HMResults(
            n_ensemble=hm_raw.get('n_ensemble', 0),
            n_iterations=hm_raw.get('n_iterations', 0),
            best_mismatch=hm_raw.get('best_mismatch', float('inf')),
            final_alpha=hm_raw.get('final_alpha', 0.0),
            s_cumulative=hm_raw.get('s_cumulative', 0.0),
            converged=hm_raw.get('converged', False),
            completed_at=hm_raw.get('completed_at'),
            per_well_rmse=hm_raw.get('per_well_rmse', {}),
        )

        return cls(
            name=data.get('name', path.stem),
            version=data.get('version', PFPROJ_VERSION),
            created=data.get('created', ''),
            modified=data.get('modified', ''),
            grid=data.get('grid', {}),
            pvt=data.get('pvt', {}),
            wells=data.get('wells', []),
            schedule=data.get('schedule', []),
            eclipse_deck_path=data.get('eclipse_deck_path'),
            las_files=data.get('las_files', []),
            model_paths=model_paths,
            hm_results=hm_results,
            forecast=data.get('forecast', {}),
            notes=data.get('notes', ''),
        )

    # ── Convenience ────────────────────────────────────────────────────────

    def summary(self) -> str:
        n_prod = sum(1 for w in self.wells if w.get('well_type') == 'PRODUCER')
        n_inj  = len(self.wells) - n_prod
        return (
            f"Project  : {self.name}\n"
            f"Version  : {self.version}\n"
            f"Grid     : {self.grid.get('nx')}×{self.grid.get('ny')}×{self.grid.get('nz')}\n"
            f"Wells    : {n_prod} producers, {n_inj} injectors\n"
            f"PINO     : {self.model_paths.pino or 'not trained'}\n"
            f"HM best  : {self.hm_results.best_mismatch:.4f} "
            f"({'converged' if self.hm_results.converged else 'not run'})\n"
        )

    def add_las_file(self, path: str) -> None:
        p = str(Path(path).resolve())
        if p not in self.las_files:
            self.las_files.append(p)

    def update_hm_results(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if hasattr(self.hm_results, k):
                setattr(self.hm_results, k, v)
        self.hm_results.completed_at = datetime.now().isoformat()


def _well_to_dict(well) -> dict:
    """Convert a WellConfig to a JSON-serialisable dict."""
    try:
        perfs = [
            {'i': p.i, 'j': p.j, 'k': p.k,
             'skin': p.skin, 'wellbore_radius': p.wellbore_radius}
            for p in well.perforations
        ]
        return {
            'name': well.name,
            'well_type': well.well_type.name if hasattr(well.well_type, 'name') else str(well.well_type),
            'perforations': perfs,
            'bhp_limit': well.bhp_limit,
        }
    except Exception:
        return {'name': str(well)}
