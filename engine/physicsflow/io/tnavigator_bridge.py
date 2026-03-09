"""
PhysicsFlow — tNavigator Bridge (v2.0)

Bidirectional conversion between:
  - tNavigator ASCII .sim decks (Eclipse-compatible keyword format)
  - PhysicsFlow .pfproj JSON project files

The parser is keyword-based and handles the most common reservoir simulation
sections: RUNSPEC, GRID, PROPS, SOLUTION, SCHEDULE, WELSPECS, COMPDAT, WCONPROD.

No external tNavigator libraries are required — the bridge reads/writes plain
ASCII text, matching tNavigator's Eclipse-compatible input format.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class WellSpec:
    name:  str
    group: str = "FIELD"
    i:     int = 0
    j:     int = 0
    depth: float = 0.0
    type:  str = "PROD"     # PROD | INJ


@dataclass
class SimDeck:
    """Parsed representation of a tNavigator / Eclipse .sim deck."""
    title:    str = ""
    nx:       int = 0
    ny:       int = 0
    nz:       int = 0
    n_active: int = 0
    days:     list[float]    = field(default_factory=list)
    wells:    list[WellSpec] = field(default_factory=list)
    keywords: dict[str, list[str]] = field(default_factory=dict)


# ── Tokeniser helpers ─────────────────────────────────────────────────────────

_COMMENT_RE = re.compile(r"--.*")


def _strip(line: str) -> str:
    return _COMMENT_RE.sub("", line).strip()


def _tokens(line: str) -> list[str]:
    return _strip(line).split()


# ── Bridge ────────────────────────────────────────────────────────────────────

class TNavigatorBridge:
    """Parse a .sim deck and expose conversion helpers."""

    def __init__(self, sim_path: str | Path):
        self.path = Path(sim_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Sim file not found: {self.path}")
        self.deck = self._parse(self.path.read_text(encoding="utf-8", errors="replace"))

    # ── Parsing ───────────────────────────────────────────────────────────────

    def _parse(self, text: str) -> SimDeck:
        deck   = SimDeck()
        lines  = text.splitlines()
        i      = 0
        n      = len(lines)

        while i < n:
            tok = _tokens(lines[i])
            if not tok:
                i += 1
                continue

            kw = tok[0].upper()

            if kw == "TITLE":
                i += 1
                if i < n:
                    deck.title = _strip(lines[i])

            elif kw == "DIMENS":
                i += 1
                while i < n:
                    t = _tokens(lines[i])
                    if t and t[0] != "/":
                        try:
                            deck.nx, deck.ny, deck.nz = int(t[0]), int(t[1]), int(t[2])
                        except (ValueError, IndexError):
                            pass
                        i += 1
                        break
                    i += 1

            elif kw in ("ACTNUM", "PORO", "PERMX", "PERMY", "PERMZ",
                        "NTG", "TOPS", "DX", "DY", "DZ"):
                # Collect raw data block until '/'
                raw: list[str] = []
                i += 1
                while i < n:
                    stripped = _strip(lines[i])
                    if stripped.endswith("/"):
                        raw.append(stripped[:-1])
                        break
                    raw.append(stripped)
                    i += 1
                deck.keywords[kw] = raw

            elif kw == "WELSPECS":
                i += 1
                while i < n:
                    t = _tokens(lines[i])
                    if not t or t[0] == "/":
                        break
                    if len(t) >= 4:
                        try:
                            ws = WellSpec(
                                name=t[0], group=t[1],
                                i=int(t[2]), j=int(t[3]),
                                depth=float(t[4]) if len(t) > 4 and t[4] != "1*" else 0.0,
                            )
                            deck.wells.append(ws)
                        except (ValueError, IndexError):
                            pass
                    i += 1

            elif kw == "WCONPROD":
                i += 1
                while i < n:
                    t = _tokens(lines[i])
                    if not t or t[0] == "/":
                        break
                    # Mark well type from status flag
                    if len(t) >= 2 and t[1].upper() == "OPEN":
                        for ws in deck.wells:
                            if ws.name == t[0]:
                                ws.type = "PROD"
                    i += 1

            elif kw == "TSTEP":
                i += 1
                while i < n:
                    stripped = _strip(lines[i])
                    if stripped.endswith("/"):
                        stripped = stripped[:-1]
                        for v in stripped.split():
                            try:
                                deck.days.append(float(v))
                            except ValueError:
                                pass
                        break
                    for v in stripped.split():
                        try:
                            deck.days.append(float(v))
                        except ValueError:
                            pass
                    i += 1

            i += 1

        # Estimate n_active from ACTNUM if present
        if "ACTNUM" in deck.keywords:
            raw_vals: list[str] = []
            for chunk in deck.keywords["ACTNUM"]:
                raw_vals.extend(chunk.split())
            total_active = 0
            for v in raw_vals:
                if "*" in v:
                    parts = v.split("*")
                    try:
                        count, val = int(parts[0]), int(parts[1])
                        total_active += count * val
                    except (ValueError, IndexError):
                        pass
                else:
                    try:
                        total_active += int(v)
                    except ValueError:
                        pass
            deck.n_active = total_active
        else:
            deck.n_active = deck.nx * deck.ny * deck.nz

        return deck

    # ── Summary dict ─────────────────────────────────────────────────────────

    def to_summary(self) -> dict[str, Any]:
        d = self.deck
        return {
            "title":         d.title,
            "nx":            d.nx,
            "ny":            d.ny,
            "nz":            d.nz,
            "n_active":      d.n_active,
            "n_wells":       len(d.wells),
            "producers":     [w.name for w in d.wells if w.type == "PROD"],
            "injectors":     [w.name for w in d.wells if w.type == "INJ"],
            "n_timesteps":   len(d.days),
            "total_days":    sum(d.days),
            "keywords_found": sorted(d.keywords.keys()),
        }

    # ── Convert to pfproj JSON ────────────────────────────────────────────────

    def to_pfproj(self, project_name: str = "") -> dict[str, Any]:
        d   = self.deck
        sum = self.to_summary()
        return {
            "format":  "pfproj",
            "version": "2.0",
            "name":    project_name or d.title or self.path.stem,
            "source":  {"type": "tnavigator", "sim_path": str(self.path)},
            "grid": {
                "nx": d.nx, "ny": d.ny, "nz": d.nz,
                "n_active": d.n_active,
            },
            "wells": [
                {
                    "name":  w.name,
                    "group": w.group,
                    "i": w.i, "j": w.j,
                    "depth": w.depth,
                    "type":  w.type,
                }
                for w in d.wells
            ],
            "schedule": {
                "timesteps_days": d.days,
                "total_days":     sum["total_days"],
            },
            "keywords": {k: True for k in d.keywords},
        }

    # ── Convert back to .sim ──────────────────────────────────────────────────

    @staticmethod
    def from_pfproj(pfproj_path: str | Path) -> "TNavigatorBridge":
        """
        Load a .pfproj JSON and return a TNavigatorBridge wrapping a synthetic
        deck (populated enough for to_sim() to produce valid output).
        """
        data = json.loads(Path(pfproj_path).read_text(encoding="utf-8"))

        # Create a minimal .sim in memory so we can reuse to_sim()
        grid = data.get("grid", {})
        wells_data = data.get("wells", [])
        schedule = data.get("schedule", {})

        lines = [
            f"TITLE",
            f"  {data.get('name', 'PhysicsFlow Export')}",
            "",
            "RUNSPEC",
            "",
            "DIMENS",
            f"  {grid.get('nx', 1)}  {grid.get('ny', 1)}  {grid.get('nz', 1)}  /",
            "",
            "OIL",
            "WATER",
            "GAS",
            "",
            "GRID",
            "",
        ]

        # Write WELSPECS
        if wells_data:
            lines += ["", "WELSPECS"]
            for w in wells_data:
                lines.append(
                    f"  {w['name']:<10} {w.get('group', 'FIELD'):<10} "
                    f"{w.get('i', 1):4d} {w.get('j', 1):4d} "
                    f"{w.get('depth', 0.0):8.1f}  OIL  /"
                )
            lines.append("/")

        # Write TSTEP
        timesteps = schedule.get("timesteps_days", [30.0])
        if timesteps:
            lines += ["", "TSTEP"]
            chunks = [timesteps[k:k+10] for k in range(0, len(timesteps), 10)]
            for chunk in chunks:
                lines.append("  " + "  ".join(f"{v:.1f}" for v in chunk))
            lines.append("/")

        lines += ["", "END", ""]

        # Write to a temp path and parse it so the normal bridge path works
        import tempfile, os
        fd, tmp = tempfile.mkstemp(suffix=".sim")
        os.close(fd)
        Path(tmp).write_text("\n".join(lines), encoding="utf-8")
        bridge = TNavigatorBridge(tmp)
        os.unlink(tmp)
        return bridge

    def to_sim(self) -> str:
        """Render the deck as a tNavigator-compatible ASCII .sim string."""
        d = self.deck
        out: list[str] = []

        out.append("-- Generated by PhysicsFlow v2.0 tNavigator Bridge")
        out.append(f"-- Source: {self.path.name}")
        out.append("")

        if d.title:
            out += ["TITLE", f"  {d.title}", ""]

        out += [
            "RUNSPEC",
            "",
            "DIMENS",
            f"  {d.nx}  {d.ny}  {d.nz}  /",
            "",
            "OIL",
            "WATER",
            "GAS",
            "",
            "GRID",
            "",
        ]

        for kw, raw in d.keywords.items():
            out.append(kw)
            out.extend(raw)
            out.append("/")
            out.append("")

        if d.wells:
            out += ["WELSPECS"]
            for w in d.wells:
                out.append(
                    f"  {w.name:<10} {w.group:<10} {w.i:4d} {w.j:4d} "
                    f"{w.depth:8.1f}  OIL  /"
                )
            out += ["/", ""]

        if d.days:
            out.append("TSTEP")
            chunks = [d.days[k:k+10] for k in range(0, len(d.days), 10)]
            for chunk in chunks:
                out.append("  " + "  ".join(f"{v:.1f}" for v in chunk))
            out += ["/", ""]

        out += ["END", ""]
        return "\n".join(out)
