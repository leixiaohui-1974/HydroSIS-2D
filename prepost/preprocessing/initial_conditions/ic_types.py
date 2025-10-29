"""
Initial condition types for HydroSIS-2D

Define various initial condition configurations for shallow water simulations.
"""

import numpy as np
from typing import Union, Tuple, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ICType(Enum):
    """Enumeration of initial condition types"""
    UNIFORM = "uniform"                  # Uniform initial state
    DAM_BREAK = "dam_break"             # Dam break scenario
    RAIN = "rain"                       # Rainfall initial condition
    DRY_BED = "dry_bed"                 # Dry bed (zero depth)
    FROM_FILE = "from_file"             # Load from file
    CUSTOM = "custom"                   # Custom field


class InitialCondition:
    """
    Base class for initial conditions

    Represents the initial state of water depth, velocity, and elevation
    for a 2D shallow water simulation.

    Attributes:
        ic_type: Type of initial condition
        description: Human-readable description
    """

    def __init__(self, ic_type: ICType, description: str = ""):
        """
        Initialize initial condition

        Args:
            ic_type: Initial condition type
            description: Optional description
        """
        self.ic_type = ic_type
        self.description = description or f"{ic_type.value} initial condition"

    def generate(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate initial condition fields

        Args:
            nx: Number of cells in x-direction
            ny: Number of cells in y-direction

        Returns:
            Tuple of (depth, velocity_x, velocity_y) arrays
        """
        raise NotImplementedError("Subclasses must implement generate()")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(type={self.ic_type.value})>"

    def to_dict(self) -> dict:
        """Export initial condition to dictionary"""
        return {
            'type': self.ic_type.value,
            'description': self.description
        }


class UniformIC(InitialCondition):
    """
    Uniform initial condition

    Constant depth and velocity across entire domain.
    """

    def __init__(self,
                 depth: float = 0.0,
                 velocity_x: float = 0.0,
                 velocity_y: float = 0.0):
        """
        Initialize uniform initial condition

        Args:
            depth: Uniform water depth [m]
            velocity_x: Uniform x-velocity [m/s]
            velocity_y: Uniform y-velocity [m/s]
        """
        super().__init__(ICType.UNIFORM)
        self.depth = depth
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y

    def generate(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate uniform fields"""
        h = np.full((nx, ny), self.depth, dtype=float)
        u = np.full((nx, ny), self.velocity_x, dtype=float)
        v = np.full((nx, ny), self.velocity_y, dtype=float)

        logger.info(f"Generated uniform IC: h={self.depth}m, u={self.velocity_x}m/s, v={self.velocity_y}m/s")
        return h, u, v

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            'depth': self.depth,
            'velocity_x': self.velocity_x,
            'velocity_y': self.velocity_y
        })
        return data


class DamBreakIC(InitialCondition):
    """
    Dam break initial condition

    Step function in water depth at dam location.
    Classic benchmark problem for shallow water codes.
    """

    def __init__(self,
                 dam_position: float,
                 upstream_depth: float,
                 downstream_depth: float = 0.0,
                 orientation: str = 'x'):
        """
        Initialize dam break initial condition

        Args:
            dam_position: Dam location (0-1, fraction of domain length)
            upstream_depth: Water depth upstream of dam [m]
            downstream_depth: Water depth downstream of dam [m]
            orientation: Dam orientation ('x' or 'y')
        """
        super().__init__(ICType.DAM_BREAK)
        self.dam_position = dam_position
        self.upstream_depth = upstream_depth
        self.downstream_depth = downstream_depth
        self.orientation = orientation

        if not 0 <= dam_position <= 1:
            raise ValueError("Dam position must be between 0 and 1")

    def generate(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate dam break fields"""
        h = np.zeros((nx, ny), dtype=float)
        u = np.zeros((nx, ny), dtype=float)
        v = np.zeros((nx, ny), dtype=float)

        if self.orientation == 'x':
            # Dam perpendicular to x-axis
            dam_index = int(self.dam_position * nx)
            h[:dam_index, :] = self.upstream_depth
            h[dam_index:, :] = self.downstream_depth
        elif self.orientation == 'y':
            # Dam perpendicular to y-axis
            dam_index = int(self.dam_position * ny)
            h[:, :dam_index] = self.upstream_depth
            h[:, dam_index:] = self.downstream_depth
        else:
            raise ValueError(f"Unknown orientation: {self.orientation}")

        logger.info(f"Generated dam break IC at position {self.dam_position} (orientation: {self.orientation})")
        logger.info(f"  Upstream depth: {self.upstream_depth}m, Downstream depth: {self.downstream_depth}m")

        return h, u, v

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            'dam_position': self.dam_position,
            'upstream_depth': self.upstream_depth,
            'downstream_depth': self.downstream_depth,
            'orientation': self.orientation
        })
        return data


class DryBedIC(InitialCondition):
    """
    Dry bed initial condition

    Zero water depth everywhere. Used for rainfall or inflow scenarios.
    """

    def __init__(self):
        """Initialize dry bed initial condition"""
        super().__init__(ICType.DRY_BED)

    def generate(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate dry bed fields"""
        h = np.zeros((nx, ny), dtype=float)
        u = np.zeros((nx, ny), dtype=float)
        v = np.zeros((nx, ny), dtype=float)

        logger.info("Generated dry bed IC (h=0 everywhere)")
        return h, u, v


class GaussianHumpIC(InitialCondition):
    """
    Gaussian hump initial condition

    Localized water surface elevation. Used for testing wave propagation.
    """

    def __init__(self,
                 center_x: float = 0.5,
                 center_y: float = 0.5,
                 amplitude: float = 1.0,
                 width: float = 0.1,
                 base_depth: float = 0.0):
        """
        Initialize Gaussian hump initial condition

        Args:
            center_x: X-coordinate of hump center (0-1, fraction of domain)
            center_y: Y-coordinate of hump center (0-1, fraction of domain)
            amplitude: Hump amplitude [m]
            width: Hump width (0-1, fraction of domain)
            base_depth: Base water depth [m]
        """
        super().__init__(ICType.CUSTOM, "Gaussian hump")
        self.center_x = center_x
        self.center_y = center_y
        self.amplitude = amplitude
        self.width = width
        self.base_depth = base_depth

    def generate(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate Gaussian hump fields"""
        # Create normalized coordinates
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Gaussian function
        r_squared = ((X - self.center_x)**2 + (Y - self.center_y)**2) / (self.width**2)
        h = self.base_depth + self.amplitude * np.exp(-r_squared)

        # Zero velocity
        u = np.zeros((nx, ny), dtype=float)
        v = np.zeros((nx, ny), dtype=float)

        logger.info(f"Generated Gaussian hump IC at ({self.center_x}, {self.center_y})")
        logger.info(f"  Amplitude: {self.amplitude}m, Width: {self.width}, Base depth: {self.base_depth}m")

        return h, u, v

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            'center_x': self.center_x,
            'center_y': self.center_y,
            'amplitude': self.amplitude,
            'width': self.width,
            'base_depth': self.base_depth
        })
        return data


class ParabolicBowlIC(InitialCondition):
    """
    Parabolic bowl initial condition

    Water at rest in a parabolic basin. Analytical solution exists.
    Used for testing well-balanced schemes.
    """

    def __init__(self,
                 bowl_depth: float = 1.0,
                 water_depth: float = 0.5):
        """
        Initialize parabolic bowl initial condition

        Args:
            bowl_depth: Maximum depth of parabolic bowl [m]
            water_depth: Initial water depth at center [m]
        """
        super().__init__(ICType.CUSTOM, "Parabolic bowl")
        self.bowl_depth = bowl_depth
        self.water_depth = water_depth

    def generate(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate parabolic bowl fields"""
        # Create normalized coordinates
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Parabolic bed
        r_squared = X**2 + Y**2
        bed = self.bowl_depth * r_squared

        # Water surface at constant elevation
        water_surface = self.water_depth

        # Water depth
        h = np.maximum(water_surface - bed, 0.0)

        # Zero velocity (at rest)
        u = np.zeros((nx, ny), dtype=float)
        v = np.zeros((nx, ny), dtype=float)

        logger.info(f"Generated parabolic bowl IC")
        logger.info(f"  Bowl depth: {self.bowl_depth}m, Water depth: {self.water_depth}m")

        return h, u, v

    def to_dict(self) -> dict:
        data = super().to_dict()
        data.update({
            'bowl_depth': self.bowl_depth,
            'water_depth': self.water_depth
        })
        return data


class CustomFieldIC(InitialCondition):
    """
    Custom field initial condition

    User-provided arrays for depth and velocity.
    """

    def __init__(self,
                 depth: np.ndarray,
                 velocity_x: Optional[np.ndarray] = None,
                 velocity_y: Optional[np.ndarray] = None):
        """
        Initialize custom field initial condition

        Args:
            depth: Water depth array
            velocity_x: X-velocity array (optional, defaults to zero)
            velocity_y: Y-velocity array (optional, defaults to zero)
        """
        super().__init__(ICType.CUSTOM, "Custom field")

        self.depth_field = depth.copy()
        self.velocity_x_field = velocity_x.copy() if velocity_x is not None else np.zeros_like(depth)
        self.velocity_y_field = velocity_y.copy() if velocity_y is not None else np.zeros_like(depth)

        # Validate shapes
        if self.velocity_x_field.shape != depth.shape:
            raise ValueError("velocity_x shape must match depth shape")
        if self.velocity_y_field.shape != depth.shape:
            raise ValueError("velocity_y shape must match depth shape")

    def generate(self, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return custom fields"""
        # Check if dimensions match
        if self.depth_field.shape != (nx, ny):
            logger.warning(f"Custom field shape {self.depth_field.shape} does not match requested ({nx}, {ny})")
            logger.warning("Returning fields as-is without resizing")

        logger.info(f"Using custom IC fields ({self.depth_field.shape})")

        return self.depth_field, self.velocity_x_field, self.velocity_y_field

    def to_dict(self) -> dict:
        data = super().to_dict()
        data['note'] = "Custom arrays not serializable"
        return data


# Factory function
def create_ic_from_dict(ic_data: dict) -> InitialCondition:
    """
    Create initial condition from dictionary

    Args:
        ic_data: Dictionary with IC specification

    Returns:
        InitialCondition instance
    """
    ic_type = ICType(ic_data['type'])

    if ic_type == ICType.UNIFORM:
        return UniformIC(
            depth=ic_data.get('depth', 0.0),
            velocity_x=ic_data.get('velocity_x', 0.0),
            velocity_y=ic_data.get('velocity_y', 0.0)
        )

    elif ic_type == ICType.DAM_BREAK:
        return DamBreakIC(
            dam_position=ic_data['dam_position'],
            upstream_depth=ic_data['upstream_depth'],
            downstream_depth=ic_data.get('downstream_depth', 0.0),
            orientation=ic_data.get('orientation', 'x')
        )

    elif ic_type == ICType.DRY_BED:
        return DryBedIC()

    elif ic_type == ICType.CUSTOM:
        # Cannot recreate custom fields from dict
        logger.warning("Cannot recreate custom field IC from dictionary")
        return DryBedIC()  # Fallback

    else:
        raise ValueError(f"Unknown initial condition type: {ic_type}")
