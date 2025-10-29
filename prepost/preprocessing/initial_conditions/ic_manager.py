"""
Initial condition manager for HydroSIS-2D

Manage and validate initial conditions for simulation domains.
"""

import numpy as np
from typing import Tuple, Optional
import logging
import json

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from preprocessing.mesh_generation import StructuredMesh, DomainParams
from preprocessing.geometry import TerrainData, TerrainProcessor
from .ic_types import (
    InitialCondition, ICType,
    UniformIC, DamBreakIC, DryBedIC,
    GaussianHumpIC, CustomFieldIC,
    create_ic_from_dict
)

logger = logging.getLogger(__name__)


class InitialConditionManager:
    """
    Manage initial conditions for a simulation

    Features:
        - Store and generate initial fields
        - Validate physical consistency
        - Apply terrain elevation
        - Export/import IC configurations
    """

    def __init__(self, mesh: StructuredMesh, terrain: Optional[np.ndarray] = None):
        """
        Initialize initial condition manager

        Args:
            mesh: Structured mesh
            terrain: Optional terrain elevation array
        """
        self.mesh = mesh
        self.terrain = terrain if terrain is not None else np.zeros((mesh.nx, mesh.ny))
        self.ic: Optional[InitialCondition] = None

        # Generated fields
        self.depth: Optional[np.ndarray] = None
        self.velocity_x: Optional[np.ndarray] = None
        self.velocity_y: Optional[np.ndarray] = None

        logger.info(f"Initialized InitialConditionManager for {mesh.nx}×{mesh.ny} mesh")

    def set_initial_condition(self, ic: InitialCondition) -> None:
        """
        Set initial condition

        Args:
            ic: InitialCondition object
        """
        self.ic = ic
        logger.info(f"Set initial condition: {ic.ic_type.value}")

        # Generate fields
        self.generate()

    def generate(self) -> None:
        """Generate initial condition fields"""
        if self.ic is None:
            raise ValueError("No initial condition set")

        self.depth, self.velocity_x, self.velocity_y = self.ic.generate(
            self.mesh.nx,
            self.mesh.ny
        )

        logger.info("Generated initial condition fields")
        logger.info(f"  Depth range: [{np.min(self.depth):.3f}, {np.max(self.depth):.3f}] m")
        logger.info(f"  Velocity range: u=[{np.min(self.velocity_x):.3f}, {np.max(self.velocity_x):.3f}] m/s")
        logger.info(f"  Velocity range: v=[{np.min(self.velocity_y):.3f}, {np.max(self.velocity_y):.3f}] m/s")

    def validate(self) -> Tuple[bool, list]:
        """
        Validate initial conditions

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        if self.ic is None:
            errors.append("No initial condition set")
            return False, errors

        if self.depth is None:
            errors.append("Initial fields not generated")
            return False, errors

        # Check for negative depths
        if np.any(self.depth < 0):
            errors.append(f"Negative depths found (min: {np.min(self.depth):.3f}m)")

        # Check for NaN or Inf
        if np.any(~np.isfinite(self.depth)):
            errors.append("Non-finite values in depth field")

        if np.any(~np.isfinite(self.velocity_x)):
            errors.append("Non-finite values in velocity_x field")

        if np.any(~np.isfinite(self.velocity_y)):
            errors.append("Non-finite values in velocity_y field")

        # Check Froude number (warn if supercritical everywhere)
        g = 9.81
        wet_cells = self.depth > 1e-6
        if np.any(wet_cells):
            velocity_mag = np.sqrt(self.velocity_x**2 + self.velocity_y**2)
            froude = velocity_mag[wet_cells] / np.sqrt(g * self.depth[wet_cells])
            max_froude = np.max(froude)

            if max_froude > 10.0:
                errors.append(f"Very high Froude number detected (max: {max_froude:.2f})")

        # Check for depth below terrain
        water_surface = self.depth + self.terrain
        if np.any(water_surface < self.terrain):
            errors.append("Water surface below terrain")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("Initial condition validation passed")
        else:
            logger.warning(f"Initial condition validation failed with {len(errors)} error(s)")

        return is_valid, errors

    def apply_terrain(self, terrain: np.ndarray) -> None:
        """
        Apply terrain elevation

        Updates the terrain array and adjusts water surface accordingly.

        Args:
            terrain: Terrain elevation array
        """
        if terrain.shape != (self.mesh.nx, self.mesh.ny):
            raise ValueError(f"Terrain shape {terrain.shape} does not match mesh {(self.mesh.nx, self.mesh.ny)}")

        self.terrain = terrain.copy()
        logger.info("Applied terrain elevation to initial conditions")

    def get_water_surface_elevation(self) -> np.ndarray:
        """
        Compute water surface elevation

        Returns:
            Water surface elevation array (depth + terrain)
        """
        if self.depth is None:
            raise ValueError("Initial fields not generated")

        return self.depth + self.terrain

    def get_total_volume(self) -> float:
        """
        Compute total water volume

        Returns:
            Total volume [m³]
        """
        if self.depth is None:
            raise ValueError("Initial fields not generated")

        # Cell areas
        dx = self.mesh.dx
        dy = self.mesh.dy
        cell_area = dx * dy

        # Total volume
        volume = np.sum(self.depth) * cell_area

        logger.info(f"Total water volume: {volume:.2f} m³")

        return volume

    def get_statistics(self) -> dict:
        """
        Compute statistics of initial conditions

        Returns:
            Dictionary with statistics
        """
        if self.depth is None:
            raise ValueError("Initial fields not generated")

        # Water depth stats
        wet_cells = self.depth > 1e-6
        num_wet = np.sum(wet_cells)

        stats = {
            'num_cells': self.mesh.ncells,
            'num_wet_cells': int(num_wet),
            'wet_fraction': num_wet / self.mesh.ncells,
            'depth': {
                'min': float(np.min(self.depth)),
                'max': float(np.max(self.depth)),
                'mean': float(np.mean(self.depth)),
                'std': float(np.std(self.depth))
            },
            'velocity_x': {
                'min': float(np.min(self.velocity_x)),
                'max': float(np.max(self.velocity_x)),
                'mean': float(np.mean(self.velocity_x)),
                'std': float(np.std(self.velocity_x))
            },
            'velocity_y': {
                'min': float(np.min(self.velocity_y)),
                'max': float(np.max(self.velocity_y)),
                'mean': float(np.mean(self.velocity_y)),
                'std': float(np.std(self.velocity_y))
            },
            'total_volume': float(self.get_total_volume())
        }

        # Froude number (for wet cells)
        if num_wet > 0:
            g = 9.81
            velocity_mag = np.sqrt(self.velocity_x[wet_cells]**2 + self.velocity_y[wet_cells]**2)
            froude = velocity_mag / np.sqrt(g * self.depth[wet_cells])
            stats['froude_number'] = {
                'min': float(np.min(froude)),
                'max': float(np.max(froude)),
                'mean': float(np.mean(froude))
            }

        return stats

    def export_to_vtk(self, filepath: str) -> None:
        """
        Export initial conditions to VTK file

        Args:
            filepath: Output file path
        """
        if self.depth is None:
            raise ValueError("Initial fields not generated")

        try:
            import pyvista as pv
        except ImportError:
            logger.error("PyVista not available for VTK export")
            raise

        # Create structured grid
        x = np.linspace(self.mesh.domain.xmin, self.mesh.domain.xmax, self.mesh.nx)
        y = np.linspace(self.mesh.domain.ymin, self.mesh.domain.ymax, self.mesh.ny)
        z = np.zeros((self.mesh.nx, self.mesh.ny))  # 2D grid

        grid = pv.StructuredGrid()
        X, Y = np.meshgrid(x, y, indexing='ij')
        grid.points = np.c_[X.ravel(), Y.ravel(), z.ravel()]
        grid.dimensions = [self.mesh.nx, self.mesh.ny, 1]

        # Add fields
        grid.point_data['depth'] = self.depth.ravel()
        grid.point_data['velocity_x'] = self.velocity_x.ravel()
        grid.point_data['velocity_y'] = self.velocity_y.ravel()
        grid.point_data['terrain'] = self.terrain.ravel()
        grid.point_data['water_surface'] = self.get_water_surface_elevation().ravel()

        # Velocity magnitude
        vel_mag = np.sqrt(self.velocity_x**2 + self.velocity_y**2)
        grid.point_data['velocity_magnitude'] = vel_mag.ravel()

        # Save
        grid.save(filepath)
        logger.info(f"Exported initial conditions to {filepath}")

    def export_to_json(self, filepath: str) -> None:
        """
        Export initial condition configuration to JSON

        Args:
            filepath: Output file path
        """
        if self.ic is None:
            raise ValueError("No initial condition set")

        data = {
            'mesh': {
                'nx': self.mesh.nx,
                'ny': self.mesh.ny,
                'domain': {
                    'xmin': self.mesh.domain.xmin,
                    'xmax': self.mesh.domain.xmax,
                    'ymin': self.mesh.domain.ymin,
                    'ymax': self.mesh.domain.ymax
                }
            },
            'initial_condition': self.ic.to_dict()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported IC configuration to {filepath}")

    def export_arrays_to_numpy(self, filepath_prefix: str) -> None:
        """
        Export initial condition arrays to NumPy .npz file

        Args:
            filepath_prefix: Output file prefix (without extension)
        """
        if self.depth is None:
            raise ValueError("Initial fields not generated")

        filepath = f"{filepath_prefix}.npz"

        np.savez(
            filepath,
            depth=self.depth,
            velocity_x=self.velocity_x,
            velocity_y=self.velocity_y,
            terrain=self.terrain,
            x=self.mesh.x,
            y=self.mesh.y
        )

        logger.info(f"Exported IC arrays to {filepath}")

    @staticmethod
    def import_arrays_from_numpy(filepath: str, mesh: StructuredMesh) -> 'InitialConditionManager':
        """
        Import initial condition arrays from NumPy .npz file

        Args:
            filepath: Input file path
            mesh: Structured mesh

        Returns:
            InitialConditionManager instance
        """
        data = np.load(filepath)

        manager = InitialConditionManager(mesh, terrain=data['terrain'])

        # Create custom field IC
        ic = CustomFieldIC(
            depth=data['depth'],
            velocity_x=data['velocity_x'],
            velocity_y=data['velocity_y']
        )

        manager.set_initial_condition(ic)

        logger.info(f"Imported IC arrays from {filepath}")

        return manager

    def summary(self) -> str:
        """Generate summary report"""
        lines = []
        lines.append("=" * 60)
        lines.append("INITIAL CONDITIONS SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Mesh: {self.mesh.nx} × {self.mesh.ny} = {self.mesh.ncells} cells")
        lines.append(f"Domain: [{self.mesh.domain.xmin}, {self.mesh.domain.xmax}] × [{self.mesh.domain.ymin}, {self.mesh.domain.ymax}]")
        lines.append("")

        if self.ic:
            lines.append(f"Initial Condition Type: {self.ic.ic_type.value}")
            lines.append(f"Description: {self.ic.description}")
            lines.append("")

            if self.depth is not None:
                stats = self.get_statistics()

                lines.append("Field Statistics:")
                lines.append(f"  Wet cells: {stats['num_wet_cells']} ({100*stats['wet_fraction']:.1f}%)")
                lines.append(f"  Total volume: {stats['total_volume']:.2f} m³")
                lines.append("")
                lines.append(f"  Depth: [{stats['depth']['min']:.3f}, {stats['depth']['max']:.3f}] m (mean: {stats['depth']['mean']:.3f})")
                lines.append(f"  Velocity X: [{stats['velocity_x']['min']:.3f}, {stats['velocity_x']['max']:.3f}] m/s")
                lines.append(f"  Velocity Y: [{stats['velocity_y']['min']:.3f}, {stats['velocity_y']['max']:.3f}] m/s")

                if 'froude_number' in stats:
                    lines.append(f"  Froude number: [{stats['froude_number']['min']:.3f}, {stats['froude_number']['max']:.3f}] (mean: {stats['froude_number']['mean']:.3f})")

                lines.append("")

                # Validation
                is_valid, errors = self.validate()
                lines.append("Validation Status: " + ("✓ VALID" if is_valid else "✗ INVALID"))
                if errors:
                    lines.append("Errors:")
                    for error in errors:
                        lines.append(f"  - {error}")
            else:
                lines.append("Fields not yet generated")
        else:
            lines.append("No initial condition set")

        lines.append("=" * 60)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<InitialConditionManager(mesh={self.mesh.nx}×{self.mesh.ny}, ic={self.ic.ic_type.value if self.ic else 'None'})>"
