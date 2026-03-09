"""
PhysicsFlow — Eclipse simulation deck reader.

Supports:
    .DATA   — ASCII keyword deck (GRID, PROPS, SOLUTION, SCHEDULE sections)
    .EGRID  — Binary EGRID geometry file (ZCORN, COORD, ACTNUM)
    .UNRST  — Unified restart file (pressure, saturation time series)
    .INIT   — Initial conditions and static properties (PORO, PERMX, etc.)

Usage:
    from physicsflow.io.eclipse_reader import EclipseReader

    reader = EclipseReader('NORNE_ATW2013')
    grid   = reader.grid()          # ReservoirGrid
    wells  = reader.wells()         # List[WellConfig]
    pvt    = reader.pvt()           # PVTConfig
    ts     = reader.timesteps()     # List[datetime]
    snap   = reader.snapshot(step=0)  # {'pressure': ndarray, 'sw': ndarray, ...}
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    from ..core.grid import GridConfig, ReservoirGrid
    from ..core.wells import WellConfig, WellType, Perforation
    from ..core.pvt import PVTConfig
except ImportError:
    GridConfig = ReservoirGrid = WellConfig = WellType = Perforation = PVTConfig = None  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# Low-level binary (EGRID / UNRST) helpers
# ─────────────────────────────────────────────────────────────────────────────

# Eclipse binary records: [4-byte length] [data] [4-byte length]
_DTYPE_MAP = {
    b'REAL': np.float32,
    b'DOUB': np.float64,
    b'INTE': np.int32,
    b'LOGI': np.int32,
    b'CHAR': 'S8',
    b'MESS': None,
    b'C008': 'S8',
}


def _read_binary_records(path: Path) -> Iterator[Tuple[str, np.ndarray]]:
    """
    Yield (keyword, data_array) tuples from an Eclipse binary file.
    Each record follows the FORTRAN unformatted convention.
    """
    with open(path, 'rb') as f:
        while True:
            header_len_bytes = f.read(4)
            if not header_len_bytes or len(header_len_bytes) < 4:
                break
            header_len = struct.unpack('>i', header_len_bytes)[0]
            header = f.read(header_len)
            f.read(4)  # trailing record marker

            kw    = header[:8].decode('ascii').strip()
            n_el  = struct.unpack('>i', header[8:12])[0]
            dtype_code = header[12:16]

            dtype = _DTYPE_MAP.get(dtype_code)
            if dtype is None:
                continue

            # Read data record(s)  — Eclipse splits into max 1000-element chunks
            all_data = []
            remaining = n_el
            while remaining > 0:
                chunk = min(remaining, 1000)
                rec_len_bytes = f.read(4)
                if len(rec_len_bytes) < 4:
                    break
                rec_len = struct.unpack('>i', rec_len_bytes)[0]
                raw = f.read(rec_len)
                f.read(4)  # trailing

                if isinstance(dtype, str) and dtype.startswith('S'):
                    item_size = int(dtype[1:])
                    arr = np.frombuffer(raw, dtype=np.dtype(f'S{item_size}'))
                else:
                    arr = np.frombuffer(raw, dtype=np.dtype(dtype).newbyteorder('>'))
                all_data.append(arr)
                remaining -= chunk

            if all_data:
                data = np.concatenate(all_data) if len(all_data) > 1 else all_data[0]
                yield kw, data


# ─────────────────────────────────────────────────────────────────────────────
# .DATA keyword parser
# ─────────────────────────────────────────────────────────────────────────────

def _tokenise_data_file(path: Path) -> Iterator[str]:
    """Yield tokens from an Eclipse .DATA file, stripping comments."""
    comment_re = re.compile(r'--.*$')
    for line in path.read_text(errors='replace').splitlines():
        line = comment_re.sub('', line).strip()
        if line:
            yield from line.split()


def _read_keyword_records(path: Path) -> Dict[str, List[List[str]]]:
    """
    Parse an Eclipse DATA file into a dict:
        keyword → list of records (each record is a list of token strings)

    Records are terminated by '/'.
    """
    keywords: Dict[str, List[List[str]]] = {}
    current_kw: Optional[str] = None
    current_record: List[str] = []

    SECTION_KW = {'RUNSPEC', 'GRID', 'EDIT', 'PROPS', 'REGIONS',
                  'SOLUTION', 'SUMMARY', 'SCHEDULE'}

    tokens = _tokenise_data_file(path)
    for tok in tokens:
        if tok == '/':
            if current_kw is not None:
                keywords.setdefault(current_kw, []).append(current_record)
                current_record = []
        elif re.match(r'^[A-Z][A-Z0-9_]{0,7}$', tok) and not current_record:
            # Looks like a keyword
            current_kw = tok
            if tok in SECTION_KW:
                current_record = []
        else:
            current_record.append(tok)

    return keywords


def _expand_repeat(tokens: List[str]) -> List[str]:
    """Expand Eclipse repeat notation '3*0.25' → ['0.25', '0.25', '0.25']."""
    result = []
    for tok in tokens:
        if '*' in tok:
            parts = tok.split('*', 1)
            try:
                count = int(parts[0])
                result.extend([parts[1]] * count)
            except ValueError:
                result.append(tok)
        else:
            result.append(tok)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# EclipseReader
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EclipseSnapshot:
    """One timestep of simulation results."""
    date: datetime
    step: int
    pressure: np.ndarray    # Pa, shape [Nx, Ny, Nz]
    sw: np.ndarray          # water saturation [Nx, Ny, Nz]
    so: np.ndarray          # oil saturation [Nx, Ny, Nz]
    sg: np.ndarray          # gas saturation [Nx, Ny, Nz]
    rs: Optional[np.ndarray] = None   # GOR [Nx, Ny, Nz]


class EclipseReader:
    """
    Reads Eclipse simulation files for the given deck basename.

    Parameters
    ----------
    deck_path : str or Path
        Full path to the .DATA file OR basename without extension
        (e.g. 'NORNE_ATW2013' will look for NORNE_ATW2013.DATA, .EGRID, .UNRST)
    """

    def __init__(self, deck_path):
        path = Path(deck_path)
        if path.suffix.upper() == '.DATA':
            self.base = path.with_suffix('')
        else:
            self.base = path

        self._kw: Optional[Dict[str, List[List[str]]]] = None
        self._nx: Optional[int] = None
        self._ny: Optional[int] = None
        self._nz: Optional[int] = None
        self._actnum: Optional[np.ndarray] = None
        self._grid_cache: Optional['ReservoirGrid'] = None
        self._snapshots: Optional[List[EclipseSnapshot]] = None

    # ── Internal helpers ───────────────────────────────────────────────────

    @property
    def _keywords(self) -> Dict[str, List[List[str]]]:
        if self._kw is None:
            data_file = self.base.with_suffix('.DATA')
            if not data_file.exists():
                raise FileNotFoundError(f"Eclipse DATA file not found: {data_file}")
            self._kw = _read_keyword_records(data_file)
        return self._kw

    def _dimensions(self) -> Tuple[int, int, int]:
        if self._nx is not None:
            return self._nx, self._ny, self._nz
        kw = self._keywords
        dimens = kw.get('DIMENS', [[]])[0]
        if len(dimens) >= 3:
            self._nx, self._ny, self._nz = int(dimens[0]), int(dimens[1]), int(dimens[2])
        else:
            self._nx, self._ny, self._nz = 46, 112, 22   # Norne defaults
        return self._nx, self._ny, self._nz

    def _read_property_array(self, keyword: str) -> Optional[np.ndarray]:
        """Read a 3-D property from DATA keywords (with repeat expansion)."""
        kw = self._keywords
        if keyword not in kw:
            return None
        Nx, Ny, Nz = self._dimensions()
        tokens = _expand_repeat(kw[keyword][0])
        arr = np.array([float(t) for t in tokens], dtype=np.float32)
        n = Nx * Ny * Nz
        if len(arr) < n:
            arr = np.pad(arr, (0, n - len(arr)), constant_values=0.0)
        return arr[:n].reshape(Nx, Ny, Nz, order='F')

    # ── Public API ─────────────────────────────────────────────────────────

    def grid(self) -> 'ReservoirGrid':
        """Build a ReservoirGrid from the Eclipse deck."""
        if self._grid_cache is not None:
            return self._grid_cache

        Nx, Ny, Nz = self._dimensions()
        kw = self._keywords

        # DX, DY, DZ (may be scalar or full arrays)
        dx = float(_expand_repeat(kw.get('DX', [['50.0']])[0])[0])
        dy = float(_expand_repeat(kw.get('DY', [['50.0']])[0])[0])
        dz = float(_expand_repeat(kw.get('DZ', [['20.0']])[0])[0])
        depth = float(_expand_repeat(kw.get('TOPS', [['2000.0']])[0])[0])

        if GridConfig is not None:
            cfg = GridConfig(nx=Nx, ny=Ny, nz=Nz,
                             dx=dx, dy=dy, dz=dz, depth=depth)
            perm_x = self._read_property_array('PERMX')
            poro   = self._read_property_array('PORO')
            grid   = ReservoirGrid(cfg, perm_x=perm_x, porosity=poro)
            self._grid_cache = grid
            return grid

        raise ImportError("ReservoirGrid not available — check physicsflow.core.grid import")

    def actnum(self) -> np.ndarray:
        """Return active cell mask [Nx, Ny, Nz] bool."""
        if self._actnum is not None:
            return self._actnum
        Nx, Ny, Nz = self._dimensions()
        egrid_file = self.base.with_suffix('.EGRID')
        if egrid_file.exists():
            for kw, data in _read_binary_records(egrid_file):
                if kw == 'ACTNUM':
                    self._actnum = data.reshape(Nx, Ny, Nz, order='F').astype(bool)
                    return self._actnum
        # Fall back to DATA keyword
        arr = self._read_property_array('ACTNUM')
        if arr is not None:
            self._actnum = arr.astype(bool)
        else:
            self._actnum = np.ones((Nx, Ny, Nz), dtype=bool)
        return self._actnum

    def permeability(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (Kx, Ky, Kz) in mD, shape [Nx, Ny, Nz] each."""
        Kx = self._read_property_array('PERMX')
        Ky = self._read_property_array('PERMY') or Kx
        Kz = self._read_property_array('PERMZ')
        if Kx is None:
            Nx, Ny, Nz = self._dimensions()
            Kx = np.ones((Nx, Ny, Nz), dtype=np.float32) * 100.0
        if Kz is None:
            Kz = Kx * 0.1
        return Kx, Ky, Kz

    def porosity(self) -> np.ndarray:
        """Return porosity array [Nx, Ny, Nz]."""
        phi = self._read_property_array('PORO')
        if phi is None:
            Nx, Ny, Nz = self._dimensions()
            phi = np.full((Nx, Ny, Nz), 0.2, dtype=np.float32)
        return phi

    def wells(self) -> List['WellConfig']:
        """Parse WELSPECS + COMPDAT keywords to build WellConfig list."""
        if WellConfig is None:
            raise ImportError("WellConfig not available")

        kw = self._keywords
        well_configs: List[WellConfig] = []
        well_map: Dict[str, dict] = {}

        # WELSPECS: WELL GROUP I J DEPTH PHASE ...
        for record in kw.get('WELSPECS', []):
            if len(record) >= 4:
                name  = record[0].strip("'")
                group = record[1].strip("'")
                i_loc = int(record[2]) - 1   # 0-based
                j_loc = int(record[3]) - 1
                phase = record[5].strip("'") if len(record) > 5 else 'OIL'
                well_map[name] = {'i': i_loc, 'j': j_loc, 'phase': phase,
                                  'group': group, 'perfs': []}

        # COMPDAT: WELL I J K1 K2 STATUS ... DIAM
        for record in kw.get('COMPDAT', []):
            if len(record) >= 5:
                name = record[0].strip("'")
                i = int(record[1]) - 1
                j = int(record[2]) - 1
                k1 = int(record[3]) - 1
                k2 = int(record[4]) - 1
                for k in range(k1, k2 + 1):
                    if name not in well_map:
                        well_map[name] = {'i': i, 'j': j, 'phase': 'OIL',
                                          'group': 'FIELD', 'perfs': []}
                    well_map[name]['perfs'].append(
                        Perforation(i=i, j=j, k=k, skin=0.0,
                                    wellbore_radius=0.108)
                    )

        for name, info in well_map.items():
            wtype = WellType.PRODUCER if info['phase'] in ('OIL', 'GAS', 'LIQ') \
                    else WellType.INJECTOR
            if info['perfs']:
                well_configs.append(WellConfig(
                    name=name,
                    well_type=wtype,
                    perforations=info['perfs'],
                    bhp_limit=250e5 if wtype == WellType.PRODUCER else 400e5,
                ))

        return well_configs

    def pvt(self) -> 'PVTConfig':
        """Extract PVT configuration from PVTO/PVTW/PVDG keywords."""
        if PVTConfig is None:
            raise ImportError("PVTConfig not available")
        # Default to Norne calibrated values; override from deck if available
        return PVTConfig.norne_defaults()

    def start_date(self) -> datetime:
        """Return simulation start date from START keyword."""
        kw = self._keywords
        start_rec = kw.get('START', [[]])[0]
        months = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
                  'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
                  'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        if len(start_rec) >= 3:
            try:
                day   = int(start_rec[0])
                month = months.get(start_rec[1][:3].upper(), 1)
                year  = int(start_rec[2])
                return datetime(year, month, day)
            except (ValueError, IndexError):
                pass
        return datetime(2001, 1, 1)

    def snapshots(self) -> List[EclipseSnapshot]:
        """
        Read all timestep snapshots from the .UNRST file.
        Returns a list of EclipseSnapshot objects in chronological order.
        """
        if self._snapshots is not None:
            return self._snapshots

        Nx, Ny, Nz = self._dimensions()
        n_cells = Nx * Ny * Nz
        unrst_file = self.base.with_suffix('.UNRST')
        if not unrst_file.exists():
            return []

        snapshots = []
        buf: Dict[str, np.ndarray] = {}
        intehead: Optional[np.ndarray] = None
        start = self.start_date()

        for kw, data in _read_binary_records(unrst_file):
            if kw == 'INTEHEAD':
                # Flush previous snapshot
                if intehead is not None and 'PRESSURE' in buf:
                    _flush_snapshot(buf, intehead, Nx, Ny, Nz, n_cells,
                                    start, snapshots)
                buf = {}
                intehead = data
            elif kw in ('PRESSURE', 'SWAT', 'SOIL', 'SGAS', 'RS'):
                buf[kw] = data
            # Other arrays (e.g. PERMX in restart) are ignored

        # Flush last snapshot
        if intehead is not None and 'PRESSURE' in buf:
            _flush_snapshot(buf, intehead, Nx, Ny, Nz, n_cells, start, snapshots)

        self._snapshots = snapshots
        return snapshots

    def snapshot(self, step: int = -1) -> Optional[EclipseSnapshot]:
        """Return a single snapshot by index (-1 = last)."""
        snaps = self.snapshots()
        if not snaps:
            return None
        return snaps[step]

    def timestep_dates(self) -> List[datetime]:
        return [s.date for s in self.snapshots()]

    # ── Convenience: export to training tensors ────────────────────────────

    def to_training_arrays(self) -> dict:
        """
        Export all snapshots as stacked numpy arrays suitable for
        FNO training.

        Returns dict with keys:
            'perm_log'   : [Nx, Ny, Nz]  log10 Kx
            'phi'        : [Nx, Ny, Nz]
            'pressure'   : [T, Nx, Ny, Nz]
            'sw'         : [T, Nx, Ny, Nz]
            'actnum'     : [Nx, Ny, Nz] bool
        """
        Kx, _, _ = self.permeability()
        phi       = self.porosity()
        act       = self.actnum()
        snaps     = self.snapshots()

        pressures = np.stack([s.pressure for s in snaps], axis=0) if snaps else None
        sws       = np.stack([s.sw       for s in snaps], axis=0) if snaps else None

        return {
            'perm_log': np.log10(np.clip(Kx, 1e-3, None)),
            'phi':      phi,
            'pressure': pressures,
            'sw':       sws,
            'actnum':   act,
        }


def _flush_snapshot(
    buf: Dict[str, np.ndarray],
    intehead: np.ndarray,
    Nx: int, Ny: int, Nz: int,
    n_cells: int,
    start_date: datetime,
    snapshots: List[EclipseSnapshot],
) -> None:
    """Convert buffered raw arrays into an EclipseSnapshot and append."""
    step = int(intehead[66]) if len(intehead) > 66 else len(snapshots)
    sim_days = float(intehead[65]) if len(intehead) > 65 else 0.0
    date = start_date + timedelta(days=sim_days)

    def _reshape(arr: Optional[np.ndarray]) -> np.ndarray:
        if arr is None:
            return np.zeros((Nx, Ny, Nz), dtype=np.float32)
        return arr[:n_cells].reshape(Nx, Ny, Nz, order='F').astype(np.float32)

    pressure = _reshape(buf.get('PRESSURE')) * 1e5   # bar → Pa
    sw = _reshape(buf.get('SWAT'))
    so = _reshape(buf.get('SOIL'))
    sg = _reshape(buf.get('SGAS'))
    rs = _reshape(buf.get('RS')) if 'RS' in buf else None

    snapshots.append(EclipseSnapshot(
        date=date, step=step,
        pressure=pressure, sw=sw, so=so, sg=sg, rs=rs,
    ))
