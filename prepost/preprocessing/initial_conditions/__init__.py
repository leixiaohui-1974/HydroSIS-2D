# -*- coding: utf-8 -*-
"""
Initial conditions module for HydroSIS-2D preprocessing

Provides tools for defining and managing initial conditions for simulations.
"""

from .ic_types import (
    ICType,
    InitialCondition,
    UniformIC,
    DamBreakIC,
    DryBedIC,
    GaussianHumpIC,
    ParabolicBowlIC,
    CustomFieldIC,
    create_ic_from_dict
)

from .ic_manager import InitialConditionManager

__all__ = [
    # Enum
    'ICType',

    # Base class
    'InitialCondition',

    # IC types
    'UniformIC',
    'DamBreakIC',
    'DryBedIC',
    'GaussianHumpIC',
    'ParabolicBowlIC',
    'CustomFieldIC',

    # Factory
    'create_ic_from_dict',

    # Manager
    'InitialConditionManager'
]
