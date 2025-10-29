"""
Boundary conditions module for HydroSIS-2D preprocessing

Provides tools for defining and managing boundary conditions.
"""

from .bc_types import (
    BCType,
    BCLocation,
    BoundaryCondition,
    WallBC,
    InflowBC,
    OutflowBC,
    PeriodicBC,
    TransmissiveBC,
    TimeSeriesBC,
    FunctionBC,
    create_bc_from_dict
)

from .bc_manager import BoundaryConditionManager

__all__ = [
    # Enums
    'BCType',
    'BCLocation',

    # Base class
    'BoundaryCondition',

    # BC types
    'WallBC',
    'InflowBC',
    'OutflowBC',
    'PeriodicBC',
    'TransmissiveBC',
    'TimeSeriesBC',
    'FunctionBC',

    # Factory
    'create_bc_from_dict',

    # Manager
    'BoundaryConditionManager'
]
