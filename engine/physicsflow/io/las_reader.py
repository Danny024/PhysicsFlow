"""
PhysicsFlow — LAS 2.0 Well Log Reader.

Parses Log ASCII Standard (LAS) version 2.0 files.
Specification: CWLS LAS 2.0 (1992).

Sections:
    ~VERSION   — LAS version, wrap mode
    ~WELL      — Well identification information
    ~CURVE     — Curve names, units, descriptions
    ~PARAMETER — Miscellaneous parameters
    ~OTHER     — Free-form comments
    ~ASCII     — Curve data

Usage:
    from physicsflow.io.las_reader import LASReader

    log = LASReader.read('NORNE_B2H.las')
    depth  = log.curve('DEPTH')       # numpy array
    gamma  = log.curve('GR')          # numpy array
    print(log.well_info)              # dict of header values
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LASCurve:
    """One curve (column) from a LAS file."""
    mnemonic: str
    unit: str
    api_code: str
    description: str
    data: np.ndarray

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (f"LASCurve(mnemonic={self.mnemonic!r}, unit={self.unit!r}, "
                f"n={len(self.data)})")


@dataclass
class LASParameter:
    """One line from the ~PARAMETER section."""
    mnemonic: str
    unit: str
    value: str
    description: str


@dataclass
class WellLog:
    """
    Complete parsed LAS file.

    Attributes
    ----------
    path         : source file path
    version      : LAS version string (e.g. '2.0')
    wrap         : True if wrap-mode
    null_value   : NULL substitution value (default -9999.25)
    well_info    : dict of well header fields (UWI, COMP, WELL, FLD, ...)
    curves       : ordered list of LASCurve objects
    parameters   : dict of parameter mnemonics → LASParameter
    other_text   : free-form ~OTHER section text
    """
    path: Optional[Path]
    version: str
    wrap: bool
    null_value: float
    well_info: Dict[str, str]
    curves: List[LASCurve]
    parameters: Dict[str, LASParameter] = field(default_factory=dict)
    other_text: str = ''

    # ── Convenience methods ────────────────────────────────────────────────

    def curve(self, mnemonic: str) -> np.ndarray:
        """Return data array for the named curve. Raises KeyError if not found."""
        mnemonic = mnemonic.upper()
        for c in self.curves:
            if c.mnemonic.upper() == mnemonic:
                return c.data
        available = [c.mnemonic for c in self.curves]
        raise KeyError(f"Curve {mnemonic!r} not found. Available: {available}")

    def has_curve(self, mnemonic: str) -> bool:
        return any(c.mnemonic.upper() == mnemonic.upper() for c in self.curves)

    @property
    def depth(self) -> np.ndarray:
        """Return the depth index curve (first curve by LAS convention)."""
        if self.curves:
            return self.curves[0].data
        return np.array([], dtype=np.float64)

    @property
    def depth_unit(self) -> str:
        if self.curves:
            return self.curves[0].unit
        return ''

    @property
    def n_samples(self) -> int:
        return len(self.curves[0].data) if self.curves else 0

    @property
    def curve_names(self) -> List[str]:
        return [c.mnemonic for c in self.curves]

    def to_dict(self) -> Dict[str, np.ndarray]:
        """Export all curves as {mnemonic: array}."""
        return {c.mnemonic: c.data for c in self.curves}

    def depth_range(self) -> tuple:
        """Return (min_depth, max_depth) in depth_unit."""
        d = self.depth
        if len(d) == 0:
            return (0.0, 0.0)
        valid = d[~np.isnan(d)]
        return (float(valid.min()), float(valid.max())) if len(valid) else (0.0, 0.0)

    def resample(self, new_depth: np.ndarray) -> 'WellLog':
        """
        Linearly interpolate all curves to a new depth array.
        Returns a new WellLog with interpolated data.
        """
        old_depth = self.depth
        new_curves = []
        for i, c in enumerate(self.curves):
            if i == 0:
                new_data = new_depth
            else:
                new_data = np.interp(new_depth, old_depth, c.data,
                                     left=np.nan, right=np.nan)
            new_curves.append(LASCurve(c.mnemonic, c.unit, c.api_code,
                                       c.description, new_data))
        return WellLog(
            path=self.path, version=self.version, wrap=self.wrap,
            null_value=self.null_value, well_info=self.well_info,
            curves=new_curves, parameters=self.parameters,
            other_text=self.other_text,
        )

    def __repr__(self) -> str:
        name = self.well_info.get('WELL', str(self.path))
        return (f"WellLog(well={name!r}, n_curves={len(self.curves)}, "
                f"n_samples={self.n_samples})")


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

# LAS 2.0 header line pattern:
#   MNEMONIC.UNIT   VALUE  : DESCRIPTION
_HEADER_RE = re.compile(
    r'^\s*([A-Za-z0-9_\-\.]+)'     # mnemonic
    r'\s*\.\s*([^\s:]*)'            # .unit (optional)
    r'\s+(.*?)\s*:\s*(.*?)\s*$'     # value : description
)


class LASReader:
    """
    LAS 2.0 file parser.

    Usage
    -----
    log = LASReader.read('well.las')
    log = LASReader.read_string(las_text)
    """

    @classmethod
    def read(cls, path) -> WellLog:
        """Parse a LAS file from disk."""
        path = Path(path)
        text = path.read_text(errors='replace')
        return cls._parse(text, path)

    @classmethod
    def read_string(cls, text: str) -> WellLog:
        """Parse a LAS file from a string."""
        return cls._parse(text, path=None)

    @classmethod
    def _parse(cls, text: str, path: Optional[Path]) -> WellLog:
        sections = cls._split_sections(text)

        # ── ~VERSION ──────────────────────────────────────────────────────
        ver_items = cls._parse_header_section(sections.get('VERSION', ''))
        version   = ver_items.get('VERS', {}).get('value', '2.0')
        wrap_str  = ver_items.get('WRAP', {}).get('value', 'NO').upper()
        wrap      = wrap_str == 'YES'

        # ── ~WELL ─────────────────────────────────────────────────────────
        well_items = cls._parse_header_section(sections.get('WELL', ''))
        well_info  = {k: v['value'] for k, v in well_items.items()}
        null_val   = float(well_items.get('NULL', {}).get('value', '-9999.25') or '-9999.25')

        # ── ~CURVE ────────────────────────────────────────────────────────
        curve_defs = cls._parse_header_section(sections.get('CURVE', ''))
        curve_meta = [
            (mnem, info.get('unit', ''), info.get('api_code', ''),
             info.get('description', ''))
            for mnem, info in curve_defs.items()
        ]

        # ── ~PARAMETER ────────────────────────────────────────────────────
        param_items = cls._parse_header_section(sections.get('PARAMETER', ''))
        parameters  = {
            k: LASParameter(mnemonic=k, unit=v.get('unit', ''),
                            value=v.get('value', ''),
                            description=v.get('description', ''))
            for k, v in param_items.items()
        }

        # ── ~OTHER ────────────────────────────────────────────────────────
        other_text = sections.get('OTHER', '').strip()

        # ── ~ASCII data ───────────────────────────────────────────────────
        ascii_text = sections.get('ASCII', '') or sections.get('A', '')
        data_matrix = cls._parse_ascii(ascii_text, null_val, wrap,
                                       n_curves=len(curve_meta))

        # Build LASCurve objects
        curves = []
        for i, (mnem, unit, api, desc) in enumerate(curve_meta):
            col = data_matrix[:, i] if i < data_matrix.shape[1] else np.array([])
            curves.append(LASCurve(mnemonic=mnem, unit=unit,
                                   api_code=api, description=desc, data=col))

        return WellLog(
            path=path, version=version, wrap=wrap, null_value=null_val,
            well_info=well_info, curves=curves,
            parameters=parameters, other_text=other_text,
        )

    # ── Section splitting ─────────────────────────────────────────────────

    @staticmethod
    def _split_sections(text: str) -> Dict[str, str]:
        """Split a LAS text into sections keyed by section name."""
        sections: Dict[str, str] = {}
        current_name: Optional[str] = None
        lines: List[str] = []

        for raw_line in text.splitlines():
            # Strip inline comments
            line = raw_line.split('#')[0]
            stripped = line.strip()

            if stripped.startswith('~'):
                if current_name is not None:
                    sections[current_name] = '\n'.join(lines)
                # Extract section name (first word after ~, uppercase)
                section_token = stripped[1:].split()[0].upper() if len(stripped) > 1 else ''
                current_name = section_token
                lines = []
            else:
                lines.append(line)

        if current_name is not None:
            sections[current_name] = '\n'.join(lines)

        return sections

    @staticmethod
    def _parse_header_section(text: str) -> Dict[str, dict]:
        """
        Parse header lines from a single section.
        Returns {mnemonic: {unit, value, api_code, description}}.
        """
        result: Dict[str, dict] = {}
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            m = _HEADER_RE.match(stripped)
            if m:
                mnem = m.group(1).strip().upper()
                unit = m.group(2).strip()
                val  = m.group(3).strip()
                desc = m.group(4).strip()
                result[mnem] = {'unit': unit, 'value': val,
                                'api_code': '', 'description': desc}
        return result

    @staticmethod
    def _parse_ascii(text: str, null_val: float, wrap: bool,
                     n_curves: int) -> np.ndarray:
        """
        Parse the ~ASCII data section.
        Returns shape [n_rows, n_curves] float64 array.
        Null values are replaced with np.nan.
        """
        tokens: List[float] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            for tok in stripped.split():
                try:
                    tokens.append(float(tok))
                except ValueError:
                    tokens.append(np.nan)

        if n_curves == 0 or not tokens:
            return np.empty((0, max(n_curves, 1)), dtype=np.float64)

        arr = np.array(tokens, dtype=np.float64)
        # Trim to multiple of n_curves
        rem = len(arr) % n_curves
        if rem:
            arr = arr[:-rem]
        matrix = arr.reshape(-1, n_curves)
        # Replace null values
        matrix[np.isclose(matrix, null_val, rtol=1e-4, atol=1e-2)] = np.nan
        return matrix


# ─────────────────────────────────────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────────────────────────────────────

def read_las(path) -> WellLog:
    """Read a LAS 2.0 file from disk. Alias for LASReader.read()."""
    return LASReader.read(path)


def read_las_directory(directory) -> Dict[str, WellLog]:
    """
    Read all .las / .LAS files in a directory.
    Returns {stem: WellLog} mapping.
    """
    directory = Path(directory)
    logs = {}
    for las_path in sorted(directory.glob('*.las')) + sorted(directory.glob('*.LAS')):
        try:
            logs[las_path.stem] = LASReader.read(las_path)
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to read {las_path}: {e}")
    return logs
