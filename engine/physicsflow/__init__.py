"""
PhysicsFlow Engine — AI-Accelerated Reservoir Simulation Platform.

Core package exposing the main engine components.
"""

__version__ = "1.0.0-dev"
__author__ = "PhysicsFlow Technologies"

from .config import EngineConfig
from .core.pvt import BlackOilPVT, PVTConfig
from .core.grid import ReservoirGrid, GridConfig
from .core.wells import WellConfig, WellType, PeacemannWellModel, norne_default_wells

__all__ = [
    "EngineConfig",
    "BlackOilPVT",
    "PVTConfig",
    "ReservoirGrid",
    "GridConfig",
    "WellConfig",
    "WellType",
    "PeacemannWellModel",
    "norne_default_wells",
]
