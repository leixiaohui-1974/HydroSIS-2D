# -*- coding: utf-8 -*-
"""
Adaptive mesh generation for HydroSIS-2D

Provides adaptive mesh refinement based on terrain features, gradients,
and user-defined refinement zones.
"""

import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging

from .mesh_generator import DomainParams, MeshParams, StructuredMesh, MeshGenerator

logger = logging.getLogger(__name__)


@dataclass
class RefinementZone:
    """
    Define a rectangular refinement zone

    Attributes:
        xmin, xmax, ymin, ymax: Zone boundaries
        refinement_level: Refinement level (1 = 2x finer, 2 = 4x finer, etc.)
        priority: Zone priority (higher = processed first)
    """
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    refinement_level: int = 1
    priority: int = 0

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) is inside zone"""
        return (self.xmin <= x <= self.xmax and
                self.ymin <= y <= self.ymax)

    def get_refinement_factor(self) -> int:
        """Get cell refinement factor (2^level)"""
        return 2 ** self.refinement_level


class AdaptiveMeshGenerator(MeshGenerator):
    """
    Advanced mesh generator with adaptive refinement capabilities

    Supports:
        - User-defined refinement zones
        - Terrain gradient-based refinement
        - Multiple refinement levels
        - Smooth transitions between refinement levels
    """

    def __init__(self, domain: DomainParams):
        """
        Initialize adaptive mesh generator

        Args:
            domain: Domain parameters
        """
        super().__init__(domain)
        self.refinement_zones: List[RefinementZone] = []
        self.refinement_map = None  # 2D array storing refinement levels

    def add_refinement_zone(self, xmin: float, xmax: float,
                           ymin: float, ymax: float,
                           refinement_level: int = 1,
                           priority: int = 0) -> None:
        """
        Add a rectangular refinement zone

        Args:
            xmin, xmax: Zone extent in x-direction
            ymin, ymax: Zone extent in y-direction
            refinement_level: How many times to refine (1 = 2x, 2 = 4x, etc.)
            priority: Zone priority for overlapping zones
        """
        zone = RefinementZone(xmin, xmax, ymin, ymax, refinement_level, priority)
        self.refinement_zones.append(zone)
        logger.info(f"Added refinement zone: level={refinement_level}, "
                   f"bounds=[{xmin:.2f}, {xmax:.2f}] x [{ymin:.2f}, {ymax:.2f}]")

    def add_circular_refinement_zone(self, center_x: float, center_y: float,
                                    radius: float, refinement_level: int = 1) -> None:
        """
        Add a circular refinement zone (approximated as square)

        Args:
            center_x, center_y: Circle center
            radius: Circle radius
            refinement_level: Refinement level
        """
        self.add_refinement_zone(
            xmin=center_x - radius,
            xmax=center_x + radius,
            ymin=center_y - radius,
            ymax=center_y + radius,
            refinement_level=refinement_level
        )

    def generate_refinement_map_from_zones(self, base_nx: int, base_ny: int) -> np.ndarray:
        """
        Generate refinement map from user-defined zones

        Args:
            base_nx, base_ny: Base mesh resolution

        Returns:
            2D array with refinement levels for each cell
        """
        # Create base mesh for mapping
        base_mesh_params = MeshParams(nx=base_nx, ny=base_ny)
        base_mesh = StructuredMesh(self.domain, base_mesh_params)

        # Initialize refinement map (0 = base resolution)
        refinement_map = np.zeros((base_nx, base_ny), dtype=int)

        # Sort zones by priority (highest first)
        sorted_zones = sorted(self.refinement_zones,
                            key=lambda z: z.priority,
                            reverse=True)

        # Apply refinement zones
        for zone in sorted_zones:
            for i in range(base_nx):
                for j in range(base_ny):
                    x, y = base_mesh.get_cell_coordinates(i, j)
                    if zone.contains_point(x, y):
                        refinement_map[i, j] = max(refinement_map[i, j],
                                                  zone.refinement_level)

        self.refinement_map = refinement_map
        logger.info(f"Generated refinement map: {np.sum(refinement_map > 0)} cells to refine")
        return refinement_map

    def generate_refinement_map_from_terrain(self, terrain_data: np.ndarray,
                                            gradient_threshold: float = 0.1,
                                            max_refinement_level: int = 2) -> np.ndarray:
        """
        Generate refinement map based on terrain gradients

        Args:
            terrain_data: 2D array of terrain elevations (same size as base mesh)
            gradient_threshold: Minimum gradient magnitude to trigger refinement
            max_refinement_level: Maximum refinement level

        Returns:
            2D array with refinement levels
        """
        ny, nx = terrain_data.shape

        # Calculate terrain gradients
        grad_x, grad_y = np.gradient(terrain_data)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize gradients to [0, 1]
        grad_max = np.max(grad_magnitude)
        if grad_max > 0:
            grad_normalized = grad_magnitude / grad_max
        else:
            grad_normalized = grad_magnitude

        # Map gradients to refinement levels
        refinement_map = np.zeros((nx, ny), dtype=int)

        for level in range(1, max_refinement_level + 1):
            threshold = gradient_threshold * (max_refinement_level - level + 1) / max_refinement_level
            mask = grad_normalized > threshold
            refinement_map[mask] = level

        self.refinement_map = refinement_map
        cells_to_refine = np.sum(refinement_map > 0)
        logger.info(f"Terrain-based refinement: {cells_to_refine} cells to refine "
                   f"(max gradient: {grad_max:.4f})")
        return refinement_map

    def generate_multiresolution_mesh(self, base_nx: int, base_ny: int,
                                     use_terrain: bool = False,
                                     terrain_data: Optional[np.ndarray] = None) -> StructuredMesh:
        """
        Generate mesh with multiple resolution levels

        Note: For structured Cartesian mesh, this creates the finest uniform mesh
        that covers all refinement requirements.

        Args:
            base_nx, base_ny: Base mesh resolution
            use_terrain: Whether to use terrain-based refinement
            terrain_data: Terrain elevation data (required if use_terrain=True)

        Returns:
            StructuredMesh at finest required resolution
        """
        # Generate refinement map
        if use_terrain:
            if terrain_data is None:
                raise ValueError("terrain_data required when use_terrain=True")
            self.generate_refinement_map_from_terrain(terrain_data)
        else:
            self.generate_refinement_map_from_zones(base_nx, base_ny)

        # For structured mesh, determine finest required resolution
        max_refinement = np.max(self.refinement_map) if self.refinement_map is not None else 0
        refinement_factor = 2 ** max_refinement

        # Generate uniform mesh at finest resolution
        fine_nx = base_nx * refinement_factor
        fine_ny = base_ny * refinement_factor

        logger.info(f"Generating uniform mesh at finest resolution: {fine_nx}x{fine_ny}")
        self.mesh = self.generate_uniform_mesh(fine_nx, fine_ny)

        return self.mesh

    def generate_adaptive_mesh_smart(self, base_cell_size: float,
                                    refinement_criterion: Optional[Callable] = None) -> StructuredMesh:
        """
        Generate adaptive mesh with smart cell sizing

        Args:
            base_cell_size: Base cell size (coarsest level)
            refinement_criterion: Optional function(x, y) -> refinement_level

        Returns:
            StructuredMesh with variable resolution
        """
        # Calculate base mesh dimensions
        base_nx = int(np.ceil(self.domain.length_x / base_cell_size))
        base_ny = int(np.ceil(self.domain.length_y / base_cell_size))

        if refinement_criterion:
            # Apply custom refinement criterion
            base_mesh_params = MeshParams(nx=base_nx, ny=base_ny)
            base_mesh = StructuredMesh(self.domain, base_mesh_params)

            refinement_map = np.zeros((base_nx, base_ny), dtype=int)
            for i in range(base_nx):
                for j in range(base_ny):
                    x, y = base_mesh.get_cell_coordinates(i, j)
                    refinement_map[i, j] = refinement_criterion(x, y)

            self.refinement_map = refinement_map

        # Generate multiresolution mesh
        return self.generate_multiresolution_mesh(base_nx, base_ny)

    def visualize_refinement_map(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Visualize the refinement map

        Args:
            figsize: Figure size in inches
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib not available for visualization")
            return

        if self.refinement_map is None:
            raise RuntimeError("No refinement map generated")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot refinement levels
        im = ax.imshow(self.refinement_map.T, origin='lower',
                      extent=[self.domain.xmin, self.domain.xmax,
                             self.domain.ymin, self.domain.ymax],
                      cmap='YlOrRd', interpolation='nearest')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Refinement Level')

        # Draw refinement zone boundaries
        for zone in self.refinement_zones:
            rect = plt.Rectangle((zone.xmin, zone.ymin),
                                zone.xmax - zone.xmin,
                                zone.ymax - zone.ymin,
                                fill=False, edgecolor='blue',
                                linewidth=2, linestyle='--')
            ax.add_patch(rect)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Adaptive Mesh Refinement Map')
        ax.set_aspect('equal')

        plt.tight_layout()
        return fig, ax

    def get_refinement_statistics(self) -> dict:
        """Get statistics about refinement"""
        if self.refinement_map is None:
            return {}

        total_cells = self.refinement_map.size
        cells_by_level = {}

        for level in range(np.max(self.refinement_map) + 1):
            count = np.sum(self.refinement_map == level)
            cells_by_level[level] = {
                'count': int(count),
                'percentage': float(count / total_cells * 100)
            }

        return {
            'total_base_cells': total_cells,
            'cells_by_level': cells_by_level,
            'max_refinement_level': int(np.max(self.refinement_map)),
            'cells_to_refine': int(np.sum(self.refinement_map > 0))
        }


# Utility functions

def create_adaptive_mesh_with_zones(xmin: float, xmax: float,
                                    ymin: float, ymax: float,
                                    base_nx: int, base_ny: int,
                                    zones: List[Tuple[float, float, float, float, int]]) -> StructuredMesh:
    """
    Convenience function to create adaptive mesh with refinement zones

    Args:
        xmin, xmax, ymin, ymax: Domain extent
        base_nx, base_ny: Base mesh resolution
        zones: List of (xmin, xmax, ymin, ymax, refinement_level) tuples

    Returns:
        StructuredMesh object
    """
    domain = DomainParams(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    generator = AdaptiveMeshGenerator(domain)

    for zone in zones:
        generator.add_refinement_zone(*zone)

    return generator.generate_multiresolution_mesh(base_nx, base_ny)


def refine_mesh_near_features(mesh: StructuredMesh,
                              feature_points: List[Tuple[float, float]],
                              refinement_radius: float,
                              refinement_level: int = 1) -> StructuredMesh:
    """
    Refine mesh near specific feature points (e.g., observation points, structures)

    Args:
        mesh: Base mesh
        feature_points: List of (x, y) coordinates
        refinement_radius: Radius around each point to refine
        refinement_level: Refinement level

    Returns:
        Refined StructuredMesh
    """
    domain = DomainParams(
        xmin=mesh.domain.xmin,
        xmax=mesh.domain.xmax,
        ymin=mesh.domain.ymin,
        ymax=mesh.domain.ymax
    )

    generator = AdaptiveMeshGenerator(domain)

    # Add circular refinement zones around each feature
    for x, y in feature_points:
        generator.add_circular_refinement_zone(x, y, refinement_radius, refinement_level)

    return generator.generate_multiresolution_mesh(mesh.nx, mesh.ny)
