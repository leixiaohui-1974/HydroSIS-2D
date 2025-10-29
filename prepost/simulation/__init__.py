"""
Simulation setup module for HydroSIS-2D

Provides complete simulation configuration integrating all preprocessing modules.
"""

from .simulation_config import (
    SimulationConfig,
    create_dam_break_simulation,
    create_channel_flow_simulation
)

__all__ = [
    'SimulationConfig',
    'create_dam_break_simulation',
    'create_channel_flow_simulation'
]
