"""
Basic mesh generator for HydroSIS-2D

Provides structured Cartesian mesh generation with configurable parameters.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DomainParams:
    """Parameters defining the computational domain"""
    xmin: float = 0.0
    xmax: float = 100.0
    ymin: float = 0.0
    ymax: float = 100.0

    def __post_init__(self):
        if self.xmax <= self.xmin:
            raise ValueError("xmax must be greater than xmin")
        if self.ymax <= self.ymin:
            raise ValueError("ymax must be greater than ymin")

    @property
    def length_x(self) -> float:
        """Domain length in x-direction"""
        return self.xmax - self.xmin

    @property
    def length_y(self) -> float:
        """Domain length in y-direction"""
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        """Total domain area"""
        return self.length_x * self.length_y


@dataclass
class MeshParams:
    """Parameters for mesh generation"""
    nx: int = 100
    ny: int = 100
    dx: Optional[float] = None
    dy: Optional[float] = None

    def __post_init__(self):
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("Grid dimensions must be positive")


class StructuredMesh:
    """
    Structured Cartesian mesh for HydroSIS-2D

    Attributes:
        nx, ny: Number of cells in x and y directions
        dx, dy: Cell spacing
        x, y: Cell center coordinates (2D arrays)
        xmin, xmax, ymin, ymax: Domain boundaries
    """

    def __init__(self, domain: DomainParams, mesh_params: MeshParams):
        """
        Initialize structured mesh

        Args:
            domain: Domain parameters
            mesh_params: Mesh parameters
        """
        self.domain = domain
        self.mesh_params = mesh_params

        # Calculate cell spacing
        self.dx = mesh_params.dx if mesh_params.dx is not None else domain.length_x / mesh_params.nx
        self.dy = mesh_params.dy if mesh_params.dy is not None else domain.length_y / mesh_params.ny

        # Actual number of cells (may differ if dx/dy provided)
        self.nx = int(np.ceil(domain.length_x / self.dx))
        self.ny = int(np.ceil(domain.length_y / self.dy))

        # Generate cell centers
        x_centers = domain.xmin + self.dx * (np.arange(self.nx) + 0.5)
        y_centers = domain.ymin + self.dy * (np.arange(self.ny) + 0.5)

        self.x, self.y = np.meshgrid(x_centers, y_centers, indexing='ij')

        logger.info(f"Created structured mesh: {self.nx}×{self.ny} cells, "
                   f"dx={self.dx:.4f}, dy={self.dy:.4f}")

    @property
    def ncells(self) -> int:
        """Total number of cells"""
        return self.nx * self.ny

    @property
    def aspect_ratio(self) -> float:
        """Cell aspect ratio (dy/dx)"""
        return self.dy / self.dx

    def get_cell_area(self) -> float:
        """Get cell area (uniform for structured mesh)"""
        return self.dx * self.dy

    def get_cell_coordinates(self, i: int, j: int) -> Tuple[float, float]:
        """
        Get coordinates of cell (i, j)

        Args:
            i: Cell index in x-direction (0 to nx-1)
            j: Cell index in y-direction (0 to ny-1)

        Returns:
            (x, y) coordinates of cell center
        """
        if i < 0 or i >= self.nx or j < 0 or j >= self.ny:
            raise IndexError(f"Cell index ({i}, {j}) out of bounds")
        return self.x[i, j], self.y[i, j]

    def find_cell(self, x: float, y: float) -> Tuple[int, int]:
        """
        Find cell index containing point (x, y)

        Args:
            x, y: Point coordinates

        Returns:
            (i, j) cell indices
        """
        i = int((x - self.domain.xmin) / self.dx)
        j = int((y - self.domain.ymin) / self.dy)

        # Clamp to valid range
        i = max(0, min(i, self.nx - 1))
        j = max(0, min(j, self.ny - 1))

        return i, j

    def get_info(self) -> Dict[str, Any]:
        """Get mesh information dictionary"""
        return {
            'nx': self.nx,
            'ny': self.ny,
            'ncells': self.ncells,
            'dx': self.dx,
            'dy': self.dy,
            'xmin': self.domain.xmin,
            'xmax': self.domain.xmax,
            'ymin': self.domain.ymin,
            'ymax': self.domain.ymax,
            'cell_area': self.get_cell_area(),
            'aspect_ratio': self.aspect_ratio
        }


class MeshGenerator:
    """
    Main mesh generator for HydroSIS-2D

    Generates structured Cartesian meshes with various configuration options.
    """

    def __init__(self, domain: DomainParams):
        """
        Initialize mesh generator

        Args:
            domain: Domain parameters defining computational extent
        """
        self.domain = domain
        self.mesh = None
        logger.info(f"Initialized MeshGenerator for domain: "
                   f"[{domain.xmin}, {domain.xmax}] × [{domain.ymin}, {domain.ymax}]")

    def generate_uniform_mesh(self, nx: int, ny: int) -> StructuredMesh:
        """
        Generate uniform structured mesh

        Args:
            nx: Number of cells in x-direction
            ny: Number of cells in y-direction

        Returns:
            StructuredMesh object
        """
        mesh_params = MeshParams(nx=nx, ny=ny)
        self.mesh = StructuredMesh(self.domain, mesh_params)
        return self.mesh

    def generate_mesh_with_spacing(self, dx: float, dy: float) -> StructuredMesh:
        """
        Generate mesh with specified cell spacing

        Args:
            dx: Cell spacing in x-direction
            dy: Cell spacing in y-direction

        Returns:
            StructuredMesh object
        """
        # Calculate required number of cells
        nx = int(np.ceil(self.domain.length_x / dx))
        ny = int(np.ceil(self.domain.length_y / dy))

        mesh_params = MeshParams(nx=nx, ny=ny, dx=dx, dy=dy)
        self.mesh = StructuredMesh(self.domain, mesh_params)
        return self.mesh

    def generate_mesh_from_resolution(self, target_cell_area: float) -> StructuredMesh:
        """
        Generate mesh targeting a specific cell area

        Args:
            target_cell_area: Target area per cell

        Returns:
            StructuredMesh object
        """
        # Calculate spacing to achieve target area (assuming square cells)
        dx = dy = np.sqrt(target_cell_area)
        return self.generate_mesh_with_spacing(dx, dy)

    def export_to_hydrosis_format(self, output_file: str) -> None:
        """
        Export mesh configuration to HydroSIS-2D INI format

        Args:
            output_file: Path to output INI file
        """
        if self.mesh is None:
            raise RuntimeError("No mesh generated. Call generate_*_mesh() first.")

        config_content = f"""# HydroSIS-2D Mesh Configuration
# Generated by MeshGenerator

[Grid]
nx = {self.mesh.nx}
ny = {self.mesh.ny}
dx = {self.mesh.dx}
dy = {self.mesh.dy}
xmin = {self.domain.xmin}
ymin = {self.domain.ymin}
xmax = {self.domain.xmax}
ymax = {self.domain.ymax}

# Mesh Statistics
# Total cells: {self.mesh.ncells}
# Cell area: {self.mesh.get_cell_area():.6f}
# Aspect ratio: {self.mesh.aspect_ratio:.4f}
"""

        with open(output_file, 'w') as f:
            f.write(config_content)

        logger.info(f"Exported mesh configuration to {output_file}")

    def visualize_mesh(self, show_every: int = 10, figsize: Tuple[int, int] = (10, 8)):
        """
        Visualize the mesh grid

        Args:
            show_every: Show every Nth grid line (for large meshes)
            figsize: Figure size in inches
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.error("Matplotlib not available for visualization")
            return

        if self.mesh is None:
            raise RuntimeError("No mesh generated")

        fig, ax = plt.subplots(figsize=figsize)

        # Draw vertical lines
        for i in range(0, self.mesh.nx + 1, show_every):
            x = self.domain.xmin + i * self.mesh.dx
            ax.plot([x, x], [self.domain.ymin, self.domain.ymax],
                   'k-', linewidth=0.5, alpha=0.5)

        # Draw horizontal lines
        for j in range(0, self.mesh.ny + 1, show_every):
            y = self.domain.ymin + j * self.mesh.dy
            ax.plot([self.domain.xmin, self.domain.xmax], [y, y],
                   'k-', linewidth=0.5, alpha=0.5)

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'Structured Mesh: {self.mesh.nx}×{self.mesh.ny} cells')
        ax.set_aspect('equal')
        ax.grid(False)

        # Add mesh info text
        info_text = (f"Cells: {self.mesh.ncells}\n"
                    f"dx: {self.mesh.dx:.3f} m\n"
                    f"dy: {self.mesh.dy:.3f} m\n"
                    f"Area: {self.mesh.get_cell_area():.3f} m²")
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='wheat', alpha=0.8))

        plt.tight_layout()
        return fig, ax


# Utility functions

def create_uniform_mesh(xmin: float, xmax: float, ymin: float, ymax: float,
                       nx: int, ny: int) -> StructuredMesh:
    """
    Convenience function to create uniform mesh in one call

    Args:
        xmin, xmax: Domain extent in x
        ymin, ymax: Domain extent in y
        nx, ny: Number of cells

    Returns:
        StructuredMesh object
    """
    domain = DomainParams(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    generator = MeshGenerator(domain)
    return generator.generate_uniform_mesh(nx, ny)


def create_mesh_with_target_size(xmin: float, xmax: float, ymin: float, ymax: float,
                                 target_cell_size: float) -> StructuredMesh:
    """
    Convenience function to create mesh with target cell size

    Args:
        xmin, xmax: Domain extent in x
        ymin, ymax: Domain extent in y
        target_cell_size: Target cell edge length (for square cells)

    Returns:
        StructuredMesh object
    """
    domain = DomainParams(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    generator = MeshGenerator(domain)
    return generator.generate_mesh_with_spacing(target_cell_size, target_cell_size)
