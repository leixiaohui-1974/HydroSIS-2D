# -*- coding: utf-8 -*-
"""
Profile extraction tools for HydroSIS-2D results

Extract 1D profiles from 2D simulation data along specified lines.
"""

import numpy as np
from typing import Tuple, List, Dict
import logging

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from preprocessing.mesh_generation import StructuredMesh

logger = logging.getLogger(__name__)


class ProfileExtractor:
    """
    Extract 1D profiles from 2D data fields

    Supports extraction along:
        - X-axis lines (constant y)
        - Y-axis lines (constant x)
        - Arbitrary diagonal lines
    """

    def __init__(self, mesh: StructuredMesh):
        """
        Initialize profile extractor

        Args:
            mesh: StructuredMesh object
        """
        self.mesh = mesh

    def extract_x_profile(self, field: np.ndarray, y: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract profile along x-axis at constant y

        Args:
            field: 2D field array
            y: Y-coordinate

        Returns:
            Tuple of (x_coords, values)
        """
        # Find nearest j index
        j = int((y - self.mesh.domain.ymin) / self.mesh.dy)
        j = np.clip(j, 0, self.mesh.ny - 1)

        x_coords = self.mesh.x[:, j]
        values = field[:, j]

        return x_coords, values

    def extract_y_profile(self, field: np.ndarray, x: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract profile along y-axis at constant x

        Args:
            field: 2D field array
            x: X-coordinate

        Returns:
            Tuple of (y_coords, values)
        """
        # Find nearest i index
        i = int((x - self.mesh.domain.xmin) / self.mesh.dx)
        i = np.clip(i, 0, self.mesh.nx - 1)

        y_coords = self.mesh.y[i, :]
        values = field[i, :]

        return y_coords, values

    def extract_diagonal_profile(self, field: np.ndarray,
                                 start: Tuple[float, float],
                                 end: Tuple[float, float],
                                 num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract profile along arbitrary line

        Args:
            field: 2D field array
            start: Starting point (x, y)
            end: Ending point (x, y)
            num_points: Number of interpolation points

        Returns:
            Tuple of (distances, values)
        """
        from scipy.interpolate import RegularGridInterpolator

        # Create interpolator
        x_grid = self.mesh.x[:, 0]
        y_grid = self.mesh.y[0, :]
        interpolator = RegularGridInterpolator((x_grid, y_grid), field, bounds_error=False, fill_value=0)

        # Generate points along line
        x_points = np.linspace(start[0], end[0], num_points)
        y_points = np.linspace(start[1], end[1], num_points)

        # Interpolate values
        points = np.column_stack([x_points, y_points])
        values = interpolator(points)

        # Compute distances along line
        distances = np.linspace(0, np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2), num_points)

        return distances, values
