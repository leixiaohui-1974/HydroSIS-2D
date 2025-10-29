"""
Animation generation for HydroSIS-2D visualizations

Create high-quality animations from simulation time series.
"""

import numpy as np
from typing import List, Optional, Callable, Tuple
import logging
import os

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from preprocessing.mesh_generation import StructuredMesh
from .visualization_engine import VisualizationEngine

logger = logging.getLogger(__name__)


class AnimationGenerator:
    """
    Generate animations from HydroSIS-2D simulation results

    Supports:
        - Time series animations
        - Multiple fields (water depth, velocity, etc.)
        - Various export formats (MP4, GIF, image sequence)
        - Customizable rendering options
    """

    def __init__(self, mesh: StructuredMesh):
        """
        Initialize animation generator

        Args:
            mesh: StructuredMesh object for simulation domain
        """
        self.mesh = mesh
        self.time_series = []
        self.current_frame = 0

        logger.info(f"Initialized AnimationGenerator for {mesh.nx}×{mesh.ny} mesh")

    def add_timestep(self, time: float, data: dict) -> None:
        """
        Add a timestep to the animation

        Args:
            time: Simulation time
            data: Dictionary with field data (e.g., {'h': depth_array, 'u': u_array, ...})
        """
        self.time_series.append({
            'time': time,
            'data': data.copy()
        })

    def load_from_vtk_series(self, file_pattern: str) -> None:
        """
        Load time series from VTK files

        Args:
            file_pattern: Pattern for VTK files (e.g., "results/output_*.vtk")
        """
        import glob

        files = sorted(glob.glob(file_pattern))
        logger.info(f"Found {len(files)} VTK files matching pattern: {file_pattern}")

        for i, filename in enumerate(files):
            # Extract time from filename or use index
            time = float(i)  # Simple approach, can be improved

            # Load VTK file
            grid = pv.read(filename)

            # Extract data
            data = {}
            for key in grid.cell_data.keys():
                data[key] = grid.cell_data[key].reshape(self.mesh.nx, self.mesh.ny)

            self.add_timestep(time, data)

        logger.info(f"Loaded {len(self.time_series)} timesteps")

    def create_water_surface_animation(self,
                                      terrain: np.ndarray,
                                      output_file: str,
                                      fps: int = 20,
                                      field: str = 'h',
                                      cmap: str = 'Blues',
                                      time_label: bool = True) -> None:
        """
        Create animation of water surface evolution

        Args:
            terrain: 2D terrain elevation array
            output_file: Output filename (e.g., "animation.mp4")
            fps: Frames per second
            field: Field name for water depth in data dict
            cmap: Colormap
            time_label: Show time label
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for animation")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Initialize plot with first frame
        first_data = self.time_series[0]['data'][field]
        water_surface = terrain + first_data

        im = ax.imshow(
            water_surface.T,
            origin='lower',
            extent=[self.mesh.domain.xmin, self.mesh.domain.xmax,
                   self.mesh.domain.ymin, self.mesh.domain.ymax],
            cmap=cmap,
            vmin=np.min(terrain),
            vmax=np.max(terrain) + np.max([ts['data'][field] for ts in self.time_series])
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Water Surface Elevation [m]')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')

        if time_label:
            time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                              verticalalignment='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        def update(frame):
            """Update function for animation"""
            timestep = self.time_series[frame]
            water_depth = timestep['data'][field]
            water_surface = terrain + water_depth

            im.set_array(water_surface.T)

            if time_label:
                time_text.set_text(f"Time: {timestep['time']:.2f} s")

            return im,

        # Create animation
        anim = animation.FuncAnimation(
            fig, update,
            frames=len(self.time_series),
            interval=1000/fps,
            blit=True
        )

        # Save animation
        logger.info(f"Saving animation to {output_file}...")

        if output_file.endswith('.gif'):
            anim.save(output_file, writer='pillow', fps=fps)
        else:
            # Use ffmpeg for MP4
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='HydroSIS-2D'), bitrate=5000)
            anim.save(output_file, writer=writer)

        plt.close(fig)
        logger.info(f"Animation saved: {output_file}")

    def create_velocity_animation(self,
                                 output_file: str,
                                 fps: int = 20,
                                 u_field: str = 'u',
                                 v_field: str = 'v',
                                 cmap: str = 'jet') -> None:
        """
        Create animation of velocity magnitude

        Args:
            output_file: Output filename
            fps: Frames per second
            u_field: Field name for u-velocity
            v_field: Field name for v-velocity
            cmap: Colormap
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for animation")

        fig, ax = plt.subplots(figsize=(12, 8))

        # Calculate max velocity for consistent colorbar
        max_vel = 0
        for ts in self.time_series:
            u = ts['data'].get(u_field, np.zeros((self.mesh.nx, self.mesh.ny)))
            v = ts['data'].get(v_field, np.zeros((self.mesh.nx, self.mesh.ny)))
            vel_mag = np.sqrt(u**2 + v**2)
            max_vel = max(max_vel, np.max(vel_mag))

        # Initialize plot
        first_u = self.time_series[0]['data'].get(u_field, np.zeros((self.mesh.nx, self.mesh.ny)))
        first_v = self.time_series[0]['data'].get(v_field, np.zeros((self.mesh.nx, self.mesh.ny)))
        first_vel = np.sqrt(first_u**2 + first_v**2)

        im = ax.imshow(
            first_vel.T,
            origin='lower',
            extent=[self.mesh.domain.xmin, self.mesh.domain.xmax,
                   self.mesh.domain.ymin, self.mesh.domain.ymax],
            cmap=cmap,
            vmin=0,
            vmax=max_vel
        )

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Velocity Magnitude [m/s]')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')

        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        def update(frame):
            timestep = self.time_series[frame]
            u = timestep['data'].get(u_field, np.zeros((self.mesh.nx, self.mesh.ny)))
            v = timestep['data'].get(v_field, np.zeros((self.mesh.nx, self.mesh.ny)))
            vel_mag = np.sqrt(u**2 + v**2)

            im.set_array(vel_mag.T)
            time_text.set_text(f"Time: {timestep['time']:.2f} s")

            return im,

        anim = animation.FuncAnimation(
            fig, update,
            frames=len(self.time_series),
            interval=1000/fps,
            blit=True
        )

        logger.info(f"Saving velocity animation to {output_file}...")

        if output_file.endswith('.gif'):
            anim.save(output_file, writer='pillow', fps=fps)
        else:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='HydroSIS-2D'), bitrate=5000)
            anim.save(output_file, writer=writer)

        plt.close(fig)
        logger.info(f"Velocity animation saved: {output_file}")

    def export_image_sequence(self,
                            output_dir: str,
                            terrain: np.ndarray,
                            field: str = 'h',
                            cmap: str = 'Blues',
                            prefix: str = 'frame') -> List[str]:
        """
        Export animation frames as image sequence

        Args:
            output_dir: Output directory
            terrain: Terrain elevation
            field: Field to visualize
            cmap: Colormap
            prefix: Filename prefix

        Returns:
            List of generated filenames
        """
        os.makedirs(output_dir, exist_ok=True)

        filenames = []

        for i, timestep in enumerate(self.time_series):
            water_depth = timestep['data'][field]
            water_surface = terrain + water_depth

            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))

            im = ax.imshow(
                water_surface.T,
                origin='lower',
                extent=[self.mesh.domain.xmin, self.mesh.domain.xmax,
                       self.mesh.domain.ymin, self.mesh.domain.ymax],
                cmap=cmap
            )

            plt.colorbar(im, ax=ax, label='Water Surface Elevation [m]')

            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_title(f'Time: {timestep["time"]:.2f} s')
            ax.set_aspect('equal')

            # Save
            filename = os.path.join(output_dir, f"{prefix}_{i:04d}.png")
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close(fig)

            filenames.append(filename)

        logger.info(f"Exported {len(filenames)} frames to {output_dir}")

        return filenames

    def create_3d_animation(self,
                           terrain: np.ndarray,
                           output_file: str,
                           field: str = 'h',
                           fps: int = 20,
                           cmap: str = 'Blues') -> None:
        """
        Create 3D animation using PyVista

        Args:
            terrain: Terrain elevation
            output_file: Output filename (MP4)
            field: Field to visualize
            fps: Frames per second
            cmap: Colormap
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista required for 3D animation")

        # Create visualization engine in offscreen mode
        engine = VisualizationEngine(offscreen=True)
        plotter = engine.create_plotter()

        # Open movie file
        plotter.open_movie(output_file, framerate=fps)

        # Animate frames
        for timestep in self.time_series:
            plotter.clear()

            water_depth = timestep['data'][field]

            # Visualize water surface
            engine.visualize_water_surface(
                self.mesh, water_depth, terrain,
                cmap=cmap, opacity=0.8
            )

            # Add time label
            plotter.add_text(
                f"Time: {timestep['time']:.2f} s",
                position='upper_left',
                font_size=12
            )

            # Write frame
            plotter.write_frame()

        # Close movie
        plotter.close()
        engine.close()

        logger.info(f"3D animation saved: {output_file}")


# Utility functions

def quick_animation(mesh: StructuredMesh,
                   time_series_data: List[Tuple[float, np.ndarray]],
                   terrain: np.ndarray,
                   output_file: str,
                   fps: int = 20) -> None:
    """
    Quick animation generation from time series

    Args:
        mesh: StructuredMesh object
        time_series_data: List of (time, water_depth) tuples
        terrain: Terrain elevation
        output_file: Output filename
        fps: Frames per second
    """
    gen = AnimationGenerator(mesh)

    for time, water_depth in time_series_data:
        gen.add_timestep(time, {'h': water_depth})

    gen.create_water_surface_animation(terrain, output_file, fps=fps)
