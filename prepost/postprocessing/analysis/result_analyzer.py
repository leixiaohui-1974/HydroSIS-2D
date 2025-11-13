# -*- coding: utf-8 -*-
"""
Result analyzer for HydroSIS-2D simulation outputs

Load, process, and analyze HydroSIS-2D simulation results.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
import logging
import os
import glob

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from preprocessing.mesh_generation import StructuredMesh, DomainParams, MeshParams

logger = logging.getLogger(__name__)


class SimulationResult:
    """
    Container for a single simulation timestep result

    Attributes:
        time: Simulation time [s]
        data: Dictionary of field data (h, u, v, etc.)
        mesh: Associated mesh
    """

    def __init__(self, time: float, data: Dict[str, np.ndarray], mesh: Optional[StructuredMesh] = None):
        """
        Initialize simulation result

        Args:
            time: Simulation time
            data: Dictionary mapping field names to 2D arrays
            mesh: Optional StructuredMesh object
        """
        self.time = time
        self.data = data
        self.mesh = mesh

    def get_field(self, field_name: str) -> np.ndarray:
        """Get field data by name"""
        if field_name not in self.data:
            raise KeyError(f"Field '{field_name}' not found in result")
        return self.data[field_name]

    def available_fields(self) -> List[str]:
        """Get list of available field names"""
        return list(self.data.keys())

    def compute_velocity_magnitude(self) -> np.ndarray:
        """Compute velocity magnitude from u and v components"""
        if 'u' in self.data and 'v' in self.data:
            return np.sqrt(self.data['u']**2 + self.data['v']**2)
        else:
            raise ValueError("Velocity components 'u' and 'v' not found")

    def compute_froude_number(self, gravity: float = 9.81) -> np.ndarray:
        """
        Compute Froude number

        Args:
            gravity: Gravitational acceleration [m/s^2]

        Returns:
            Froude number array
        """
        if 'h' not in self.data:
            raise ValueError("Water depth 'h' not found")

        vel_mag = self.compute_velocity_magnitude()
        h = self.data['h']

        # Avoid division by zero
        Fr = np.zeros_like(h)
        mask = h > 1e-6
        Fr[mask] = vel_mag[mask] / np.sqrt(gravity * h[mask])

        return Fr


class ResultAnalyzer:
    """
    Main result analyzer for HydroSIS-2D outputs

    Features:
        - Load VTK result files
        - Time series management
        - Field extraction and processing
        - Statistical analysis
        - Mass conservation checking
    """

    def __init__(self, mesh: Optional[StructuredMesh] = None):
        """
        Initialize result analyzer

        Args:
            mesh: Optional StructuredMesh object
        """
        self.mesh = mesh
        self.results: List[SimulationResult] = []
        self.times: List[float] = []

        logger.info("Initialized ResultAnalyzer")

    def load_vtk_file(self, filename: str, time: Optional[float] = None) -> SimulationResult:
        """
        Load a single VTK result file

        Args:
            filename: Path to VTK file
            time: Simulation time (if None, extracted from filename or index)

        Returns:
            SimulationResult object
        """
        if not PYVISTA_AVAILABLE:
            raise ImportError("PyVista required for VTK file loading")

        if not os.path.exists(filename):
            raise FileNotFoundError(f"VTK file not found: {filename}")

        # Load VTK file
        grid = pv.read(filename)

        # Extract time from filename if not provided
        if time is None:
            # Try to extract time from filename pattern like "output_t0001.vtk"
            import re
            match = re.search(r't(\d+)', os.path.basename(filename))
            if match:
                time = float(match.group(1))
            else:
                time = 0.0

        # Extract field data
        data = {}
        for field_name in grid.cell_data.keys():
            field_array = grid.cell_data[field_name]

            # Reshape to 2D if possible
            if self.mesh is not None:
                try:
                    data[field_name] = field_array.reshape(self.mesh.nx, self.mesh.ny, order='F')
                except:
                    data[field_name] = field_array
            else:
                data[field_name] = field_array

        logger.info(f"Loaded VTK file: {filename} (t={time:.2f} s, {len(data)} fields)")

        return SimulationResult(time, data, self.mesh)

    def load_vtk_series(self, pattern: str, sort: bool = True) -> None:
        """
        Load a series of VTK files matching a pattern

        Args:
            pattern: File pattern (e.g., "results/output_*.vtk")
            sort: Sort files by name
        """
        files = glob.glob(pattern)

        if not files:
            raise FileNotFoundError(f"No files found matching pattern: {pattern}")

        if sort:
            files.sort()

        logger.info(f"Found {len(files)} VTK files matching pattern: {pattern}")

        for i, filename in enumerate(files):
            try:
                result = self.load_vtk_file(filename, time=float(i))
                self.add_result(result)
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")

        logger.info(f"Loaded {len(self.results)} timesteps")

    def add_result(self, result: SimulationResult) -> None:
        """Add a result to the time series"""
        self.results.append(result)
        self.times.append(result.time)

    def get_result(self, index: int) -> SimulationResult:
        """Get result by index"""
        return self.results[index]

    def get_result_at_time(self, time: float, tolerance: float = 1e-6) -> Optional[SimulationResult]:
        """
        Get result closest to specified time

        Args:
            time: Target time
            tolerance: Time tolerance

        Returns:
            SimulationResult or None if not found
        """
        for result in self.results:
            if abs(result.time - time) < tolerance:
                return result
        return None

    def get_field_series(self, field_name: str) -> List[np.ndarray]:
        """
        Get time series of a specific field

        Args:
            field_name: Field name (e.g., 'h', 'u', 'v')

        Returns:
            List of field arrays for each timestep
        """
        series = []
        for result in self.results:
            if field_name in result.data:
                series.append(result.data[field_name])
        return series

    def compute_max_field(self, field_name: str) -> np.ndarray:
        """
        Compute maximum value of field over all timesteps

        Args:
            field_name: Field name

        Returns:
            2D array of maximum values
        """
        series = self.get_field_series(field_name)
        if not series:
            raise ValueError(f"Field '{field_name}' not found in results")

        return np.maximum.reduce(series)

    def compute_time_of_max(self, field_name: str) -> np.ndarray:
        """
        Compute time when maximum value occurs for each cell

        Args:
            field_name: Field name

        Returns:
            2D array of times when maximum occurred
        """
        series = self.get_field_series(field_name)
        if not series:
            raise ValueError(f"Field '{field_name}' not found in results")

        # Stack series along time dimension
        field_3d = np.stack(series, axis=-1)

        # Find index of maximum along time axis
        max_indices = np.argmax(field_3d, axis=-1)

        # Convert indices to times
        times_array = np.array(self.times)
        time_of_max = times_array[max_indices]

        return time_of_max

    def compute_inundation_duration(self, depth_threshold: float = 0.01) -> np.ndarray:
        """
        Compute inundation duration for each cell

        Args:
            depth_threshold: Minimum depth to consider as inundated [m]

        Returns:
            2D array of inundation durations [s]
        """
        h_series = self.get_field_series('h')
        if not h_series:
            raise ValueError("Water depth 'h' not found in results")

        # Stack depths along time dimension
        h_3d = np.stack(h_series, axis=-1)

        # Count timesteps where depth exceeds threshold
        inundated = h_3d > depth_threshold
        num_inundated = np.sum(inundated, axis=-1)

        # Compute duration (assuming uniform time steps)
        if len(self.times) > 1:
            dt = np.mean(np.diff(self.times))
        else:
            dt = 1.0

        duration = num_inundated * dt

        return duration

    def compute_total_volume(self, result: SimulationResult) -> float:
        """
        Compute total water volume

        Args:
            result: SimulationResult object

        Returns:
            Total volume [m^3]
        """
        if 'h' not in result.data:
            raise ValueError("Water depth 'h' not found")

        if self.mesh is None:
            raise ValueError("Mesh required for volume calculation")

        h = result.data['h']
        cell_area = self.mesh.dx * self.mesh.dy

        return np.sum(h) * cell_area

    def check_mass_conservation(self, reference_index: int = 0) -> Dict[str, float]:
        """
        Check mass conservation throughout simulation

        Args:
            reference_index: Index of reference timestep

        Returns:
            Dictionary with conservation statistics
        """
        if not self.results:
            raise ValueError("No results loaded")

        ref_volume = self.compute_total_volume(self.results[reference_index])

        volumes = []
        relative_errors = []

        for result in self.results:
            vol = self.compute_total_volume(result)
            volumes.append(vol)
            rel_error = abs(vol - ref_volume) / ref_volume if ref_volume > 0 else 0
            relative_errors.append(rel_error)

        return {
            'reference_volume': ref_volume,
            'volumes': volumes,
            'relative_errors': relative_errors,
            'max_error': max(relative_errors),
            'mean_error': np.mean(relative_errors),
            'final_volume': volumes[-1],
            'volume_change': volumes[-1] - ref_volume
        }

    def get_statistics(self, field_name: str) -> Dict[str, float]:
        """
        Compute statistics for a field across all timesteps

        Args:
            field_name: Field name

        Returns:
            Dictionary of statistics
        """
        series = self.get_field_series(field_name)
        if not series:
            raise ValueError(f"Field '{field_name}' not found")

        # Combine all data
        all_data = np.concatenate([arr.flatten() for arr in series])

        return {
            'min': float(np.min(all_data)),
            'max': float(np.max(all_data)),
            'mean': float(np.mean(all_data)),
            'std': float(np.std(all_data)),
            'median': float(np.median(all_data))
        }

    def summary(self) -> str:
        """Generate summary report of loaded results"""
        if not self.results:
            return "No results loaded"

        lines = []
        lines.append("=" * 60)
        lines.append("SIMULATION RESULTS SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Number of timesteps: {len(self.results)}")
        lines.append(f"Time range: {self.times[0]:.2f} - {self.times[-1]:.2f} s")
        lines.append(f"Time step: {np.mean(np.diff(self.times)):.4f} s (average)")
        lines.append("")

        lines.append("Available fields:")
        fields = self.results[0].available_fields()
        for field in fields:
            lines.append(f"  - {field}")
        lines.append("")

        if self.mesh is not None:
            lines.append("Mesh information:")
            lines.append(f"  Grid: {self.mesh.nx} x {self.mesh.ny} = {self.mesh.ncells} cells")
            lines.append(f"  Domain: [{self.mesh.domain.xmin:.2f}, {self.mesh.domain.xmax:.2f}] x "
                       f"[{self.mesh.domain.ymin:.2f}, {self.mesh.domain.ymax:.2f}] m")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


# Utility functions

def quick_load_results(pattern: str, mesh: Optional[StructuredMesh] = None) -> ResultAnalyzer:
    """
    Quick utility to load VTK result series

    Args:
        pattern: File pattern for VTK files
        mesh: Optional StructuredMesh

    Returns:
        ResultAnalyzer with loaded results
    """
    analyzer = ResultAnalyzer(mesh=mesh)
    analyzer.load_vtk_series(pattern)
    return analyzer
