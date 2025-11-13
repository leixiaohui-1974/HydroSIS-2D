# -*- coding: utf-8 -*-
"""
3D Visualization Engine for HydroSIS-2D

Provides advanced 3D visualization capabilities using PyVista for
rendering water surfaces, terrain, velocity fields, and simulation results.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List, Union
import logging

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    pv = None

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from preprocessing.mesh_generation import StructuredMesh

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Advanced 3D visualization engine for HydroSIS-2D results

    Features:
        - 3D water surface rendering
        - Terrain visualization
        - Velocity vector fields
        - Streamlines and flow visualization
        - Multiple rendering modes
        - High-quality image and video export
    """

    def __init__(self, offscreen: bool = False, window_size: Tuple[int, int] = (1920, 1080)):
        """
        Initialize visualization engine

        Args:
            offscreen: Run in offscreen mode (for server/batch processing)
            window_size: Window size for rendering (width, height)
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista is required for visualization. Install with: pip install pyvista")

        self.offscreen = offscreen
        self.window_size = window_size
        self.plotter = None
        self.mesh_data = {}
        self.current_time = 0.0

        logger.info(f"Initialized VisualizationEngine (offscreen={offscreen})")

    def create_plotter(self, **kwargs) -> pv.Plotter:
        """
        Create a new PyVista plotter

        Args:
            **kwargs: Additional arguments for pv.Plotter

        Returns:
            PyVista Plotter object
        """
        plotter_kwargs = {
            'off_screen': self.offscreen,
            'window_size': self.window_size,
            'notebook': False
        }
        plotter_kwargs.update(kwargs)

        self.plotter = pv.Plotter(**plotter_kwargs)
        return self.plotter

    def create_structured_grid(self, mesh: StructuredMesh,
                              elevation: Optional[np.ndarray] = None) -> pv.StructuredGrid:
        """
        Create PyVista StructuredGrid from HydroSIS-2D mesh

        Args:
            mesh: StructuredMesh object
            elevation: Optional 2D array of elevations (z-coordinates)

        Returns:
            PyVista StructuredGrid
        """
        # Create coordinate arrays
        x = mesh.x  # Shape: (nx, ny)
        y = mesh.y

        if elevation is None:
            z = np.zeros_like(x)
        else:
            if elevation.shape != (mesh.nx, mesh.ny):
                raise ValueError(f"Elevation shape {elevation.shape} doesn't match mesh shape ({mesh.nx}, {mesh.ny})")
            z = elevation

        # Create structured grid
        grid = pv.StructuredGrid(x, y, z)

        return grid

    def visualize_mesh(self, mesh: StructuredMesh,
                      show_edges: bool = True,
                      edge_color: str = 'black',
                      opacity: float = 0.5,
                      color: str = 'lightgray') -> pv.Plotter:
        """
        Visualize mesh grid structure

        Args:
            mesh: StructuredMesh object
            show_edges: Show mesh edges
            edge_color: Color of mesh edges
            opacity: Mesh opacity
            color: Mesh face color

        Returns:
            PyVista Plotter with mesh visualization
        """
        if self.plotter is None:
            self.create_plotter()

        # Create grid
        grid = self.create_structured_grid(mesh)

        # Add mesh to plotter
        self.plotter.add_mesh(
            grid,
            show_edges=show_edges,
            edge_color=edge_color,
            opacity=opacity,
            color=color,
            line_width=1
        )

        # Set labels
        self.plotter.add_axes()
        self.plotter.set_background('white')

        return self.plotter

    def visualize_terrain(self, mesh: StructuredMesh,
                         terrain: np.ndarray,
                         cmap: str = 'terrain',
                         show_scalar_bar: bool = True,
                         scalar_bar_title: str = 'Elevation [m]') -> pv.Plotter:
        """
        Visualize terrain elevation

        Args:
            mesh: StructuredMesh object
            terrain: 2D array of terrain elevations
            cmap: Colormap name
            show_scalar_bar: Show colorbar
            scalar_bar_title: Title for colorbar

        Returns:
            PyVista Plotter with terrain visualization
        """
        if self.plotter is None:
            self.create_plotter()

        # Create grid with terrain elevation
        grid = self.create_structured_grid(mesh, elevation=terrain)

        # Add elevation as cell data
        grid.cell_data['elevation'] = terrain.flatten(order='F')

        # Add terrain to plotter
        self.plotter.add_mesh(
            grid,
            scalars='elevation',
            cmap=cmap,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={'title': scalar_bar_title},
            lighting=True,
            smooth_shading=True
        )

        self.plotter.add_axes()
        self.plotter.set_background('white')

        return self.plotter

    def visualize_water_surface(self, mesh: StructuredMesh,
                               water_depth: np.ndarray,
                               terrain: np.ndarray,
                               cmap: str = 'Blues',
                               show_scalar_bar: bool = True,
                               opacity: float = 0.8,
                               min_depth: float = 0.01) -> pv.Plotter:
        """
        Visualize water surface (terrain + water depth)

        Args:
            mesh: StructuredMesh object
            water_depth: 2D array of water depths
            terrain: 2D array of terrain elevations
            cmap: Colormap for water depth
            show_scalar_bar: Show colorbar
            opacity: Water surface opacity
            min_depth: Minimum depth to display (for dry areas)

        Returns:
            PyVista Plotter with water surface
        """
        if self.plotter is None:
            self.create_plotter()

        # Water surface elevation = terrain + depth
        water_surface = terrain + water_depth

        # Create water depth mask (only show wet areas)
        wet_mask = water_depth >= min_depth

        # Create grid for terrain
        terrain_grid = self.create_structured_grid(mesh, elevation=terrain)
        terrain_grid.cell_data['elevation'] = terrain.flatten(order='F')

        # Add terrain
        self.plotter.add_mesh(
            terrain_grid,
            scalars='elevation',
            cmap='terrain',
            show_scalar_bar=False,
            lighting=True
        )

        # Create grid for water surface
        water_grid = self.create_structured_grid(mesh, elevation=water_surface)
        water_grid.cell_data['depth'] = water_depth.flatten(order='F')

        # Add water surface
        self.plotter.add_mesh(
            water_grid,
            scalars='depth',
            cmap=cmap,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={'title': 'Water Depth [m]'},
            opacity=opacity,
            lighting=True,
            smooth_shading=True
        )

        self.plotter.add_axes()
        self.plotter.set_background('white')

        return self.plotter

    def add_velocity_vectors(self, mesh: StructuredMesh,
                            u_velocity: np.ndarray,
                            v_velocity: np.ndarray,
                            elevation: Optional[np.ndarray] = None,
                            scale_factor: float = 1.0,
                            color: str = 'red',
                            arrow_scale: str = 'auto') -> None:
        """
        Add velocity vector field to current plot

        Args:
            mesh: StructuredMesh object
            u_velocity: 2D array of x-velocities
            v_velocity: 2D array of y-velocities
            elevation: Optional elevation for vector placement
            scale_factor: Scale factor for vectors
            color: Vector color
            arrow_scale: Arrow scaling mode ('auto', 'manual')
        """
        if self.plotter is None:
            raise RuntimeError("Create plotter first with create_plotter()")

        # Create point cloud at cell centers
        x_flat = mesh.x.flatten()
        y_flat = mesh.y.flatten()

        if elevation is not None:
            z_flat = elevation.flatten()
        else:
            z_flat = np.zeros_like(x_flat)

        points = np.column_stack([x_flat, y_flat, z_flat])

        # Create velocity vectors (u, v, 0)
        u_flat = u_velocity.flatten()
        v_flat = v_velocity.flatten()
        w_flat = np.zeros_like(u_flat)

        vectors = np.column_stack([u_flat, v_flat, w_flat])

        # Create point cloud
        point_cloud = pv.PolyData(points)
        point_cloud['vectors'] = vectors

        # Add arrows
        arrows = point_cloud.glyph(
            orient='vectors',
            scale='vectors',
            factor=scale_factor
        )

        self.plotter.add_mesh(arrows, color=color, label='Velocity')

    def visualize_velocity_magnitude(self, mesh: StructuredMesh,
                                    u_velocity: np.ndarray,
                                    v_velocity: np.ndarray,
                                    terrain: Optional[np.ndarray] = None,
                                    cmap: str = 'jet',
                                    show_scalar_bar: bool = True) -> pv.Plotter:
        """
        Visualize velocity magnitude as colored surface

        Args:
            mesh: StructuredMesh object
            u_velocity: 2D array of x-velocities
            v_velocity: 2D array of y-velocities
            terrain: Optional terrain elevation
            cmap: Colormap name
            show_scalar_bar: Show colorbar

        Returns:
            PyVista Plotter with velocity visualization
        """
        if self.plotter is None:
            self.create_plotter()

        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(u_velocity**2 + v_velocity**2)

        # Create grid
        if terrain is None:
            terrain = np.zeros_like(u_velocity)

        grid = self.create_structured_grid(mesh, elevation=terrain)
        grid.cell_data['velocity'] = velocity_magnitude.flatten(order='F')

        # Add to plotter
        self.plotter.add_mesh(
            grid,
            scalars='velocity',
            cmap=cmap,
            show_scalar_bar=show_scalar_bar,
            scalar_bar_args={'title': 'Velocity Magnitude [m/s]'},
            lighting=True
        )

        self.plotter.add_axes()
        self.plotter.set_background('white')

        return self.plotter

    def show(self, title: str = "HydroSIS-2D Visualization") -> None:
        """
        Display the visualization window

        Args:
            title: Window title
        """
        if self.plotter is None:
            raise RuntimeError("No visualization created. Call a visualization method first.")

        self.plotter.add_title(title, font_size=14)

        if not self.offscreen:
            self.plotter.show()

    def screenshot(self, filename: str, transparent_background: bool = False) -> None:
        """
        Save screenshot of current visualization

        Args:
            filename: Output filename (PNG, JPEG, etc.)
            transparent_background: Use transparent background
        """
        if self.plotter is None:
            raise RuntimeError("No visualization created")

        self.plotter.screenshot(filename, transparent_background=transparent_background)
        logger.info(f"Screenshot saved to {filename}")

    def export_to_vtk(self, filename: str) -> None:
        """
        Export current scene to VTK file

        Args:
            filename: Output VTK filename
        """
        if self.plotter is None:
            raise RuntimeError("No visualization created")

        # Save the scene
        self.plotter.export_vtkjs(filename)
        logger.info(f"VTK scene exported to {filename}")

    def close(self) -> None:
        """Close the plotter and clean up resources"""
        if self.plotter is not None:
            self.plotter.close()
            self.plotter = None


# Utility functions

def quick_mesh_view(mesh: StructuredMesh, save_to: Optional[str] = None) -> None:
    """
    Quick visualization of mesh structure

    Args:
        mesh: StructuredMesh to visualize
        save_to: Optional filename to save screenshot
    """
    engine = VisualizationEngine(offscreen=(save_to is not None))
    engine.visualize_mesh(mesh)

    if save_to:
        engine.screenshot(save_to)
    else:
        engine.show("Mesh Structure")

    engine.close()


def quick_terrain_view(mesh: StructuredMesh, terrain: np.ndarray,
                       save_to: Optional[str] = None) -> None:
    """
    Quick visualization of terrain

    Args:
        mesh: StructuredMesh object
        terrain: 2D terrain elevation array
        save_to: Optional filename to save screenshot
    """
    engine = VisualizationEngine(offscreen=(save_to is not None))
    engine.visualize_terrain(mesh, terrain)

    if save_to:
        engine.screenshot(save_to)
    else:
        engine.show("Terrain Elevation")

    engine.close()


def quick_water_view(mesh: StructuredMesh, water_depth: np.ndarray,
                    terrain: np.ndarray, save_to: Optional[str] = None) -> None:
    """
    Quick visualization of water surface

    Args:
        mesh: StructuredMesh object
        water_depth: 2D water depth array
        terrain: 2D terrain elevation array
        save_to: Optional filename to save screenshot
    """
    engine = VisualizationEngine(offscreen=(save_to is not None))
    engine.visualize_water_surface(mesh, water_depth, terrain)

    if save_to:
        engine.screenshot(save_to)
    else:
        engine.show("Water Surface")

    engine.close()
