# -*- coding: utf-8 -*-
"""
Boundary condition types for HydroSIS-2D

Define various boundary condition types for shallow water simulations.
"""

import numpy as np
from typing import Union, Tuple, Optional, Callable, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class BCType(Enum):
    """Enumeration of boundary condition types"""
    WALL = "wall"                    # Solid wall (reflective)
    INFLOW = "inflow"                # Inflow boundary
    OUTFLOW = "outflow"              # Outflow boundary
    PERIODIC = "periodic"            # Periodic boundary
    TRANSMISSIVE = "transmissive"    # Transmissive/open boundary
    TIME_SERIES = "time_series"      # Time-dependent boundary


class BCLocation(Enum):
    """Enumeration of boundary locations"""
    WEST = "west"      # Left boundary (x_min)
    EAST = "east"      # Right boundary (x_max)
    SOUTH = "south"    # Bottom boundary (y_min)
    NORTH = "north"    # Top boundary (y_max)


class BoundaryCondition:
    """
    Base class for boundary conditions

    Attributes:
        bc_type: Type of boundary condition
        location: Boundary location
        description: Human-readable description
    """

    def __init__(self, bc_type: BCType, location: BCLocation, description: str = ""):
        """
        Initialize boundary condition

        Args:
            bc_type: Boundary condition type
            location: Boundary location
            description: Optional description
        """
        self.bc_type = bc_type
        self.location = location
        self.description = description or f"{bc_type.value} at {location.value}"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(type={self.bc_type.value}, location={self.location.value})>"

    def to_dict(self) -> dict:
        """Export boundary condition to dictionary"""
        return {
            'type': self.bc_type.value,
            'location': self.location.value,
            'description': self.description
        }


class WallBC(BoundaryCondition):
    """
    Wall (reflective) boundary condition

    Enforces zero normal velocity at boundary.
    Used for solid walls, impermeable boundaries.
    """

    def __init__(self, location: BCLocation, roughness: float = 0.0):
        """
        Initialize wall boundary condition

        Args:
            location: Boundary location
            roughness: Wall roughness coefficient (Manning's n)
        """
        super().__init__(BCType.WALL, location)
        self.roughness = roughness

    def to_dict(self) -> dict:
        data = super().to_dict()
        data['roughness'] = self.roughness
        return data


class InflowBC(BoundaryCondition):
    """
    Inflow boundary condition

    Prescribes water depth and velocity at inflow boundary.
    Can be constant or time-varying.
    """

    def __init__(self,
                 location: BCLocation,
                 depth: float,
                 velocity_x: float = 0.0,
                 velocity_y: float = 0.0):
        """
        Initialize inflow boundary condition

        Args:
            location: Boundary location
            depth: Water depth [m]
            velocity_x: X-velocity component [m/s]
            velocity_y: Y-velocity component [m/s]
        """
        super().__init__(BCType.INFLOW, location)
        self.depth = depth
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            'depth': self.depth,
            'velocity_x': self.velocity_x,
            'velocity_y': self.velocity_y
        })
        return data


class OutflowBC(BoundaryCondition):
    """
    Outflow boundary condition

    Allows water to exit the domain.
    Can use zero-gradient or specified depth.
    """

    def __init__(self,
                 location: BCLocation,
                 outflow_type: str = "zero_gradient",
                 depth: Optional[float] = None):
        """
        Initialize outflow boundary condition

        Args:
            location: Boundary location
            outflow_type: Type of outflow ('zero_gradient', 'fixed_depth')
            depth: Fixed depth for 'fixed_depth' type [m]
        """
        super().__init__(BCType.OUTFLOW, location)
        self.outflow_type = outflow_type
        self.depth = depth

        if outflow_type == "fixed_depth" and depth is None:
            raise ValueError("Fixed depth outflow requires depth specification")

    def to_dict(self) -> dict:
        data = super().to_dict()
        data['outflow_type'] = self.outflow_type
        if self.depth is not None:
            data['depth'] = self.depth
        return data


class PeriodicBC(BoundaryCondition):
    """
    Periodic boundary condition

    Creates periodic connection between opposite boundaries.
    """

    def __init__(self, location: BCLocation, paired_location: BCLocation):
        """
        Initialize periodic boundary condition

        Args:
            location: Primary boundary location
            paired_location: Paired boundary location
        """
        super().__init__(BCType.PERIODIC, location)
        self.paired_location = paired_location

        # Validate pairing
        valid_pairs = {
            (BCLocation.WEST, BCLocation.EAST),
            (BCLocation.EAST, BCLocation.WEST),
            (BCLocation.SOUTH, BCLocation.NORTH),
            (BCLocation.NORTH, BCLocation.SOUTH)
        }

        if (location, paired_location) not in valid_pairs:
            raise ValueError(f"Invalid periodic boundary pair: {location.value} - {paired_location.value}")

    def to_dict(self) -> dict:
        data = super().to_dict()
        data['paired_location'] = self.paired_location.value
        return data


class TransmissiveBC(BoundaryCondition):
    """
    Transmissive (open) boundary condition

    Allows waves to pass through with minimal reflection.
    Uses characteristics-based approach.
    """

    def __init__(self, location: BCLocation):
        """
        Initialize transmissive boundary condition

        Args:
            location: Boundary location
        """
        super().__init__(BCType.TRANSMISSIVE, location)


class TimeSeriesBC(BoundaryCondition):
    """
    Time-series boundary condition

    Prescribes time-varying values at boundary.
    Supports linear interpolation between time points.
    """

    def __init__(self,
                 location: BCLocation,
                 times: np.ndarray,
                 depths: np.ndarray,
                 velocity_x: Optional[np.ndarray] = None,
                 velocity_y: Optional[np.ndarray] = None):
        """
        Initialize time-series boundary condition

        Args:
            location: Boundary location
            times: Time values [s]
            depths: Water depth time series [m]
            velocity_x: Optional X-velocity time series [m/s]
            velocity_y: Optional Y-velocity time series [m/s]
        """
        super().__init__(BCType.TIME_SERIES, location)

        # Validate inputs
        if len(times) != len(depths):
            raise ValueError("Times and depths must have same length")

        if velocity_x is not None and len(velocity_x) != len(times):
            raise ValueError("Times and velocity_x must have same length")

        if velocity_y is not None and len(velocity_y) != len(times):
            raise ValueError("Times and velocity_y must have same length")

        # Sort by time
        sort_idx = np.argsort(times)
        self.times = times[sort_idx]
        self.depths = depths[sort_idx]
        self.velocity_x = velocity_x[sort_idx] if velocity_x is not None else np.zeros_like(times)
        self.velocity_y = velocity_y[sort_idx] if velocity_y is not None else np.zeros_like(times)

    def interpolate(self, time: float) -> Tuple[float, float, float]:
        """
        Interpolate boundary values at given time

        Args:
            time: Time value [s]

        Returns:
            Tuple of (depth, velocity_x, velocity_y)
        """
        # Handle out-of-bounds
        if time <= self.times[0]:
            return self.depths[0], self.velocity_x[0], self.velocity_y[0]
        if time >= self.times[-1]:
            return self.depths[-1], self.velocity_x[-1], self.velocity_y[-1]

        # Linear interpolation
        depth = np.interp(time, self.times, self.depths)
        vel_x = np.interp(time, self.times, self.velocity_x)
        vel_y = np.interp(time, self.times, self.velocity_y)

        return depth, vel_x, vel_y

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            'times': self.times.tolist(),
            'depths': self.depths.tolist(),
            'velocity_x': self.velocity_x.tolist(),
            'velocity_y': self.velocity_y.tolist()
        })
        return data


class FunctionBC(BoundaryCondition):
    """
    Function-based boundary condition

    Prescribes values using user-defined functions.
    Most flexible boundary condition type.
    """

    def __init__(self,
                 location: BCLocation,
                 depth_func: Callable[[float], float],
                 velocity_x_func: Optional[Callable[[float], float]] = None,
                 velocity_y_func: Optional[Callable[[float], float]] = None):
        """
        Initialize function-based boundary condition

        Args:
            location: Boundary location
            depth_func: Function for depth: f(t) -> h
            velocity_x_func: Optional function for x-velocity: f(t) -> u
            velocity_y_func: Optional function for y-velocity: f(t) -> v
        """
        super().__init__(BCType.TIME_SERIES, location)  # Use TIME_SERIES type
        self.depth_func = depth_func
        self.velocity_x_func = velocity_x_func or (lambda t: 0.0)
        self.velocity_y_func = velocity_y_func or (lambda t: 0.0)

    def evaluate(self, time: float) -> Tuple[float, float, float]:
        """
        Evaluate boundary condition at given time

        Args:
            time: Time value [s]

        Returns:
            Tuple of (depth, velocity_x, velocity_y)
        """
        depth = self.depth_func(time)
        vel_x = self.velocity_x_func(time)
        vel_y = self.velocity_y_func(time)

        return depth, vel_x, vel_y

    def to_dict(self) -> dict:
        data = super().to_dict()
        data['note'] = "Function-based BC (functions not serializable)"
        return data


# Factory function to create boundary conditions from dictionary
def create_bc_from_dict(bc_data: dict) -> BoundaryCondition:
    """
    Create boundary condition from dictionary

    Args:
        bc_data: Dictionary with BC specification

    Returns:
        BoundaryCondition instance
    """
    bc_type = BCType(bc_data['type'])
    location = BCLocation(bc_data['location'])

    if bc_type == BCType.WALL:
        return WallBC(location, roughness=bc_data.get('roughness', 0.0))

    elif bc_type == BCType.INFLOW:
        return InflowBC(
            location,
            depth=bc_data['depth'],
            velocity_x=bc_data.get('velocity_x', 0.0),
            velocity_y=bc_data.get('velocity_y', 0.0)
        )

    elif bc_type == BCType.OUTFLOW:
        return OutflowBC(
            location,
            outflow_type=bc_data.get('outflow_type', 'zero_gradient'),
            depth=bc_data.get('depth')
        )

    elif bc_type == BCType.PERIODIC:
        paired_loc = BCLocation(bc_data['paired_location'])
        return PeriodicBC(location, paired_loc)

    elif bc_type == BCType.TRANSMISSIVE:
        return TransmissiveBC(location)

    elif bc_type == BCType.TIME_SERIES:
        return TimeSeriesBC(
            location,
            times=np.array(bc_data['times']),
            depths=np.array(bc_data['depths']),
            velocity_x=np.array(bc_data.get('velocity_x', [])) if 'velocity_x' in bc_data else None,
            velocity_y=np.array(bc_data.get('velocity_y', [])) if 'velocity_y' in bc_data else None
        )

    else:
        raise ValueError(f"Unknown boundary condition type: {bc_type}")
