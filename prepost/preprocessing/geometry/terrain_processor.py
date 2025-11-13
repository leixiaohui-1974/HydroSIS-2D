# -*- coding: utf-8 -*-
"""
Terrain data processing tools for HydroSIS-2D

Process, interpolate, and analyze terrain elevation data.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging
from scipy.interpolate import RegularGridInterpolator, RectBivariateSpline
from scipy.ndimage import gaussian_filter, uniform_filter

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from preprocessing.mesh_generation import StructuredMesh
from .terrain_reader import TerrainData

logger = logging.getLogger(__name__)


class TerrainProcessor:
    """
    Process and manipulate terrain elevation data

    Features:
        - Interpolation to simulation mesh
        - Smoothing and filtering
        - Gradient computation
        - Statistical analysis
    """

    def __init__(self, terrain: TerrainData):
        """
        Initialize terrain processor

        Args:
            terrain: TerrainData object
        """
        self.terrain = terrain

    @property
    def valid_data(self) -> np.ndarray:
        """Get terrain data with nodata values masked"""
        mask = self.terrain.data != self.terrain.nodata_value
        return self.terrain.data[mask]

    def interpolate_to_mesh(self, mesh: StructuredMesh, method: str = 'linear') -> np.ndarray:
        """
        Interpolate terrain data to simulation mesh

        Args:
            mesh: Target StructuredMesh
            method: Interpolation method ('linear', 'cubic')

        Returns:
            2D array of elevation values on mesh
        """
        # Create coordinate arrays for terrain grid
        terrain_x = self.terrain.xllcorner + np.arange(self.terrain.ncols) * self.terrain.cellsize
        terrain_y = self.terrain.yllcorner + np.arange(self.terrain.nrows) * self.terrain.cellsize

        # Create interpolator
        if method == 'linear':
            interpolator = RegularGridInterpolator(
                (terrain_x, terrain_y),
                self.terrain.data,
                method='linear',
                bounds_error=False,
                fill_value=0.0
            )
        elif method == 'cubic':
            # Use RectBivariateSpline for cubic interpolation
            spline = RectBivariateSpline(terrain_x, terrain_y, self.terrain.data, kx=3, ky=3)

            # Evaluate on mesh
            elevation = np.zeros((mesh.nx, mesh.ny))
            for i in range(mesh.nx):
                for j in range(mesh.ny):
                    x = mesh.x[i, j]
                    y = mesh.y[i, j]
                    if (self.terrain.xllcorner <= x <= self.terrain.xllcorner + self.terrain.ncols * self.terrain.cellsize and
                        self.terrain.yllcorner <= y <= self.terrain.yllcorner + self.terrain.nrows * self.terrain.cellsize):
                        elevation[i, j] = spline(x, y)[0, 0]
                    else:
                        elevation[i, j] = 0.0

            logger.info(f"Interpolated terrain to mesh using {method} method")
            return elevation
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        # Flatten mesh coordinates
        mesh_points = np.column_stack([mesh.x.ravel(), mesh.y.ravel()])

        # Interpolate
        elevation = interpolator(mesh_points).reshape(mesh.nx, mesh.ny)

        logger.info(f"Interpolated terrain to mesh using {method} method")
        logger.info(f"  Mesh size: {mesh.nx} x {mesh.ny}")
        logger.info(f"  Elevation range: [{np.min(elevation):.2f}, {np.max(elevation):.2f}] m")

        return elevation

    def smooth_gaussian(self, sigma: float = 1.0) -> TerrainData:
        """
        Apply Gaussian smoothing filter

        Args:
            sigma: Standard deviation for Gaussian kernel

        Returns:
            New TerrainData with smoothed elevation
        """
        # Create copy of data
        smoothed_data = self.terrain.data.copy()

        # Mask nodata values
        mask = smoothed_data != self.terrain.nodata_value

        # Apply Gaussian filter only to valid data
        smoothed_data[mask] = gaussian_filter(smoothed_data[mask].reshape(np.sum(mask)), sigma)

        # Create new TerrainData
        terrain_smooth = TerrainData(
            data=smoothed_data,
            ncols=self.terrain.ncols,
            nrows=self.terrain.nrows,
            xllcorner=self.terrain.xllcorner,
            yllcorner=self.terrain.yllcorner,
            cellsize=self.terrain.cellsize,
            nodata_value=self.terrain.nodata_value
        )

        logger.info(f"Applied Gaussian smoothing with sigma={sigma}")

        return terrain_smooth

    def smooth_uniform(self, size: int = 3) -> TerrainData:
        """
        Apply uniform (box) smoothing filter

        Args:
            size: Filter window size

        Returns:
            New TerrainData with smoothed elevation
        """
        # Apply uniform filter
        smoothed_data = uniform_filter(self.terrain.data, size=size, mode='nearest')

        # Restore nodata values
        nodata_mask = self.terrain.data == self.terrain.nodata_value
        smoothed_data[nodata_mask] = self.terrain.nodata_value

        # Create new TerrainData
        terrain_smooth = TerrainData(
            data=smoothed_data,
            ncols=self.terrain.ncols,
            nrows=self.terrain.nrows,
            xllcorner=self.terrain.xllcorner,
            yllcorner=self.terrain.yllcorner,
            cellsize=self.terrain.cellsize,
            nodata_value=self.terrain.nodata_value
        )

        logger.info(f"Applied uniform smoothing with size={size}")

        return terrain_smooth

    def compute_gradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute terrain gradient (slope)

        Returns:
            Tuple of (gradient_x, gradient_y) arrays
        """
        # Compute gradients using central differences
        grad_y, grad_x = np.gradient(self.terrain.data, self.terrain.cellsize)

        # Mask nodata values
        nodata_mask = self.terrain.data == self.terrain.nodata_value
        grad_x[nodata_mask] = 0.0
        grad_y[nodata_mask] = 0.0

        logger.info("Computed terrain gradients")

        return grad_x, grad_y

    def compute_slope(self) -> np.ndarray:
        """
        Compute terrain slope magnitude

        Returns:
            2D array of slope values [dimensionless]
        """
        grad_x, grad_y = self.compute_gradient()
        slope = np.sqrt(grad_x**2 + grad_y**2)

        logger.info(f"Computed terrain slope (max: {np.max(slope):.4f})")

        return slope

    def compute_aspect(self) -> np.ndarray:
        """
        Compute terrain aspect (direction of slope)

        Returns:
            2D array of aspect angles [degrees, 0-360]
        """
        grad_x, grad_y = self.compute_gradient()

        # Compute aspect in radians, convert to degrees
        aspect = np.arctan2(-grad_y, grad_x)
        aspect_deg = np.degrees(aspect)

        # Convert to 0-360 range
        aspect_deg = (90 - aspect_deg) % 360

        # Mask nodata values
        nodata_mask = self.terrain.data == self.terrain.nodata_value
        aspect_deg[nodata_mask] = -1

        logger.info("Computed terrain aspect")

        return aspect_deg

    def compute_curvature(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute terrain curvature

        Returns:
            Tuple of (profile_curvature, planform_curvature)
        """
        # Compute first derivatives
        grad_y, grad_x = np.gradient(self.terrain.data, self.terrain.cellsize)

        # Compute second derivatives
        grad_yy, grad_xy = np.gradient(grad_y, self.terrain.cellsize)
        grad_yx, grad_xx = np.gradient(grad_x, self.terrain.cellsize)

        # Profile curvature (in direction of steepest descent)
        p = grad_x**2 + grad_y**2
        profile_curvature = np.zeros_like(self.terrain.data)
        mask = p > 1e-10
        profile_curvature[mask] = -(grad_xx[mask] * grad_x[mask]**2 +
                                      2 * grad_xy[mask] * grad_x[mask] * grad_y[mask] +
                                      grad_yy[mask] * grad_y[mask]**2) / (p[mask] * np.sqrt(p[mask] + 1))

        # Planform curvature (perpendicular to steepest descent)
        planform_curvature = np.zeros_like(self.terrain.data)
        planform_curvature[mask] = -(grad_xx[mask] * grad_y[mask]**2 -
                                       2 * grad_xy[mask] * grad_x[mask] * grad_y[mask] +
                                       grad_yy[mask] * grad_x[mask]**2) / (p[mask] * np.sqrt(p[mask] + 1))

        # Mask nodata values
        nodata_mask = self.terrain.data == self.terrain.nodata_value
        profile_curvature[nodata_mask] = 0.0
        planform_curvature[nodata_mask] = 0.0

        logger.info("Computed terrain curvatures")

        return profile_curvature, planform_curvature

    def identify_refinement_zones(self, slope_threshold: float = 0.1) -> np.ndarray:
        """
        Identify zones requiring mesh refinement based on terrain features

        Args:
            slope_threshold: Minimum slope for refinement

        Returns:
            2D boolean array indicating refinement zones
        """
        # Compute slope
        slope = self.compute_slope()

        # Identify steep areas
        steep_zones = slope > slope_threshold

        # Count steep cells
        num_steep = np.sum(steep_zones)
        total_cells = np.sum(self.terrain.data != self.terrain.nodata_value)
        pct_steep = 100.0 * num_steep / total_cells if total_cells > 0 else 0.0

        logger.info(f"Identified {num_steep} cells ({pct_steep:.1f}%) requiring refinement")

        return steep_zones

    def resample(self, new_cellsize: float) -> TerrainData:
        """
        Resample terrain to different cell size

        Args:
            new_cellsize: New cell size [m]

        Returns:
            Resampled TerrainData
        """
        # Calculate new dimensions
        new_ncols = int(self.terrain.ncols * self.terrain.cellsize / new_cellsize)
        new_nrows = int(self.terrain.nrows * self.terrain.cellsize / new_cellsize)

        # Create new coordinate arrays
        old_x = self.terrain.xllcorner + np.arange(self.terrain.ncols) * self.terrain.cellsize
        old_y = self.terrain.yllcorner + np.arange(self.terrain.nrows) * self.terrain.cellsize

        new_x = self.terrain.xllcorner + np.arange(new_ncols) * new_cellsize
        new_y = self.terrain.yllcorner + np.arange(new_nrows) * new_cellsize

        # Create interpolator
        interpolator = RegularGridInterpolator(
            (old_x, old_y),
            self.terrain.data.T,
            method='linear',
            bounds_error=False,
            fill_value=self.terrain.nodata_value
        )

        # Create new grid
        new_X, new_Y = np.meshgrid(new_x, new_y, indexing='ij')
        new_points = np.column_stack([new_X.ravel(), new_Y.ravel()])

        # Interpolate
        new_data = interpolator(new_points).reshape(new_ncols, new_nrows)

        # Create new TerrainData
        terrain_resampled = TerrainData(
            data=new_data,
            ncols=new_ncols,
            nrows=new_nrows,
            xllcorner=self.terrain.xllcorner,
            yllcorner=self.terrain.yllcorner,
            cellsize=new_cellsize,
            nodata_value=self.terrain.nodata_value
        )

        logger.info(f"Resampled terrain from {self.terrain.cellsize:.2f}m to {new_cellsize:.2f}m")
        logger.info(f"  Old size: {self.terrain.ncols} x {self.terrain.nrows}")
        logger.info(f"  New size: {new_ncols} x {new_nrows}")

        return terrain_resampled

    def clip_to_extent(self, xmin: float, xmax: float, ymin: float, ymax: float) -> TerrainData:
        """
        Clip terrain to specified extent

        Args:
            xmin, xmax, ymin, ymax: Extent bounds

        Returns:
            Clipped TerrainData
        """
        # Find indices corresponding to extent
        i_min = max(0, int((xmin - self.terrain.xllcorner) / self.terrain.cellsize))
        i_max = min(self.terrain.ncols, int((xmax - self.terrain.xllcorner) / self.terrain.cellsize))
        j_min = max(0, int((ymin - self.terrain.yllcorner) / self.terrain.cellsize))
        j_max = min(self.terrain.nrows, int((ymax - self.terrain.yllcorner) / self.terrain.cellsize))

        # Clip data
        clipped_data = self.terrain.data[i_min:i_max, j_min:j_max]

        # Calculate new corner
        new_xllcorner = self.terrain.xllcorner + i_min * self.terrain.cellsize
        new_yllcorner = self.terrain.yllcorner + j_min * self.terrain.cellsize

        # Create new TerrainData
        terrain_clipped = TerrainData(
            data=clipped_data,
            ncols=clipped_data.shape[0],
            nrows=clipped_data.shape[1],
            xllcorner=new_xllcorner,
            yllcorner=new_yllcorner,
            cellsize=self.terrain.cellsize,
            nodata_value=self.terrain.nodata_value
        )

        logger.info(f"Clipped terrain to extent [{xmin}, {xmax}] x [{ymin}, {ymax}]")
        logger.info(f"  New size: {terrain_clipped.ncols} x {terrain_clipped.nrows}")

        return terrain_clipped

    def fill_nodata(self, method: str = 'nearest') -> TerrainData:
        """
        Fill nodata values using interpolation

        Args:
            method: Interpolation method ('nearest', 'linear')

        Returns:
            TerrainData with filled values
        """
        from scipy.interpolate import griddata

        data_filled = self.terrain.data.copy()

        # Find valid and invalid points
        valid_mask = data_filled != self.terrain.nodata_value
        invalid_mask = ~valid_mask

        if not np.any(invalid_mask):
            logger.info("No nodata values to fill")
            return TerrainData(
                data=data_filled,
                ncols=self.terrain.ncols,
                nrows=self.terrain.nrows,
                xllcorner=self.terrain.xllcorner,
                yllcorner=self.terrain.yllcorner,
                cellsize=self.terrain.cellsize,
                nodata_value=self.terrain.nodata_value
            )

        # Create coordinate grids
        x = np.arange(self.terrain.ncols)
        y = np.arange(self.terrain.nrows)
        xx, yy = np.meshgrid(x, y, indexing='ij')

        # Get valid points and values
        valid_points = np.column_stack([xx[valid_mask], yy[valid_mask]])
        valid_values = data_filled[valid_mask]

        # Get invalid points
        invalid_points = np.column_stack([xx[invalid_mask], yy[invalid_mask]])

        # Interpolate
        filled_values = griddata(valid_points, valid_values, invalid_points, method=method)

        # Fill the data
        data_filled[invalid_mask] = filled_values

        terrain_filled = TerrainData(
            data=data_filled,
            ncols=self.terrain.ncols,
            nrows=self.terrain.nrows,
            xllcorner=self.terrain.xllcorner,
            yllcorner=self.terrain.yllcorner,
            cellsize=self.terrain.cellsize,
            nodata_value=self.terrain.nodata_value
        )

        num_filled = np.sum(invalid_mask)
        logger.info(f"Filled {num_filled} nodata values using {method} interpolation")

        return terrain_filled


def process_terrain_for_simulation(terrain: TerrainData,
                                   mesh: StructuredMesh,
                                   smooth: bool = True,
                                   smooth_sigma: float = 1.0) -> np.ndarray:
    """
    Convenience function to process terrain for simulation

    Args:
        terrain: Input TerrainData
        mesh: Target simulation mesh
        smooth: Apply smoothing filter
        smooth_sigma: Smoothing parameter

    Returns:
        Elevation array on simulation mesh
    """
    processor = TerrainProcessor(terrain)

    # Optionally smooth terrain
    if smooth:
        terrain_smooth = processor.smooth_gaussian(sigma=smooth_sigma)
        processor = TerrainProcessor(terrain_smooth)

    # Interpolate to mesh
    elevation = processor.interpolate_to_mesh(mesh, method='linear')

    return elevation
