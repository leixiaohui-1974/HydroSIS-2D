# -*- coding: utf-8 -*-
"""
Synthetic geometry generator for HydroSIS-2D

Generate simple geometric shapes and terrain features for testing.
"""

import numpy as np
from typing import Tuple, Optional, List
import logging

from .terrain_reader import TerrainData

logger = logging.getLogger(__name__)


class GeometryGenerator:
    """
    Generate synthetic terrain and geometric features

    Provides methods for creating:
        - Flat surfaces
        - Inclined planes
        - Gaussian hills
        - Valleys
        - Composite terrains
    """

    @staticmethod
    def flat_surface(ncols: int, nrows: int,
                    elevation: float = 0.0,
                    xllcorner: float = 0.0,
                    yllcorner: float = 0.0,
                    cellsize: float = 1.0) -> TerrainData:
        """
        Generate flat horizontal surface

        Args:
            ncols: Number of columns
            nrows: Number of rows
            elevation: Constant elevation [m]
            xllcorner: X coordinate of lower-left corner
            yllcorner: Y coordinate of lower-left corner
            cellsize: Cell size [m]

        Returns:
            TerrainData with flat surface
        """
        data = np.full((ncols, nrows), elevation, dtype=float)

        terrain = TerrainData(
            data=data,
            ncols=ncols,
            nrows=nrows,
            xllcorner=xllcorner,
            yllcorner=yllcorner,
            cellsize=cellsize
        )

        logger.info(f"Generated flat surface at elevation {elevation} m")
        return terrain

    @staticmethod
    def inclined_plane(ncols: int, nrows: int,
                      slope_x: float = 0.01,
                      slope_y: float = 0.0,
                      base_elevation: float = 0.0,
                      xllcorner: float = 0.0,
                      yllcorner: float = 0.0,
                      cellsize: float = 1.0) -> TerrainData:
        """
        Generate inclined plane

        Args:
            ncols: Number of columns
            nrows: Number of rows
            slope_x: Slope in x-direction [dimensionless]
            slope_y: Slope in y-direction [dimensionless]
            base_elevation: Elevation at lower-left corner [m]
            xllcorner: X coordinate of lower-left corner
            yllcorner: Y coordinate of lower-left corner
            cellsize: Cell size [m]

        Returns:
            TerrainData with inclined plane
        """
        # Create coordinate arrays
        x = np.arange(ncols) * cellsize
        y = np.arange(nrows) * cellsize

        # Create meshgrid
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Compute elevation
        data = base_elevation + slope_x * X + slope_y * Y

        terrain = TerrainData(
            data=data,
            ncols=ncols,
            nrows=nrows,
            xllcorner=xllcorner,
            yllcorner=yllcorner,
            cellsize=cellsize
        )

        logger.info(f"Generated inclined plane with slopes ({slope_x}, {slope_y})")
        return terrain

    @staticmethod
    def gaussian_hill(ncols: int, nrows: int,
                     center_x: Optional[float] = None,
                     center_y: Optional[float] = None,
                     height: float = 10.0,
                     width: float = 50.0,
                     base_elevation: float = 0.0,
                     xllcorner: float = 0.0,
                     yllcorner: float = 0.0,
                     cellsize: float = 1.0) -> TerrainData:
        """
        Generate Gaussian hill

        Args:
            ncols: Number of columns
            nrows: Number of rows
            center_x: X coordinate of hill center (default: domain center)
            center_y: Y coordinate of hill center (default: domain center)
            height: Hill height [m]
            width: Hill characteristic width [m]
            base_elevation: Base elevation [m]
            xllcorner: X coordinate of lower-left corner
            yllcorner: Y coordinate of lower-left corner
            cellsize: Cell size [m]

        Returns:
            TerrainData with Gaussian hill
        """
        # Create coordinate arrays
        x = xllcorner + np.arange(ncols) * cellsize
        y = yllcorner + np.arange(nrows) * cellsize

        # Default center to domain center
        if center_x is None:
            center_x = xllcorner + (ncols * cellsize) / 2
        if center_y is None:
            center_y = yllcorner + (nrows * cellsize) / 2

        # Create meshgrid
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Compute Gaussian hill
        r_squared = (X - center_x)**2 + (Y - center_y)**2
        sigma = width / 3.0  # 3-sigma width
        data = base_elevation + height * np.exp(-r_squared / (2 * sigma**2))

        terrain = TerrainData(
            data=data,
            ncols=ncols,
            nrows=nrows,
            xllcorner=xllcorner,
            yllcorner=yllcorner,
            cellsize=cellsize
        )

        logger.info(f"Generated Gaussian hill at ({center_x}, {center_y}) with height {height} m")
        return terrain

    @staticmethod
    def valley(ncols: int, nrows: int,
              orientation: str = 'x',
              depth: float = 10.0,
              width: float = 50.0,
              base_elevation: float = 20.0,
              xllcorner: float = 0.0,
              yllcorner: float = 0.0,
              cellsize: float = 1.0) -> TerrainData:
        """
        Generate parabolic valley

        Args:
            ncols: Number of columns
            nrows: Number of rows
            orientation: Valley orientation ('x' or 'y')
            depth: Valley depth [m]
            width: Valley characteristic width [m]
            base_elevation: Elevation at valley sides [m]
            xllcorner: X coordinate of lower-left corner
            yllcorner: Y coordinate of lower-left corner
            cellsize: Cell size [m]

        Returns:
            TerrainData with valley
        """
        # Create coordinate arrays
        x = xllcorner + np.arange(ncols) * cellsize
        y = yllcorner + np.arange(nrows) * cellsize

        X, Y = np.meshgrid(x, y, indexing='ij')

        if orientation == 'x':
            # Valley runs along x-axis (varies in y)
            center_y = yllcorner + (nrows * cellsize) / 2
            dist = Y - center_y
        elif orientation == 'y':
            # Valley runs along y-axis (varies in x)
            center_x = xllcorner + (ncols * cellsize) / 2
            dist = X - center_x
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

        # Parabolic profile
        data = base_elevation - depth * np.exp(-(dist / width)**2)

        terrain = TerrainData(
            data=data,
            ncols=ncols,
            nrows=nrows,
            xllcorner=xllcorner,
            yllcorner=yllcorner,
            cellsize=cellsize
        )

        logger.info(f"Generated valley along {orientation}-axis with depth {depth} m")
        return terrain

    @staticmethod
    def random_terrain(ncols: int, nrows: int,
                      amplitude: float = 5.0,
                      wavelength: float = 20.0,
                      base_elevation: float = 0.0,
                      xllcorner: float = 0.0,
                      yllcorner: float = 0.0,
                      cellsize: float = 1.0,
                      seed: Optional[int] = None) -> TerrainData:
        """
        Generate random terrain using Perlin-like noise

        Args:
            ncols: Number of columns
            nrows: Number of rows
            amplitude: Terrain amplitude [m]
            wavelength: Characteristic wavelength [m]
            base_elevation: Base elevation [m]
            xllcorner: X coordinate of lower-left corner
            yllcorner: Y coordinate of lower-left corner
            cellsize: Cell size [m]
            seed: Random seed for reproducibility

        Returns:
            TerrainData with random terrain
        """
        if seed is not None:
            np.random.seed(seed)

        # Number of waves
        n_waves_x = int(ncols * cellsize / wavelength)
        n_waves_y = int(nrows * cellsize / wavelength)

        # Create coordinate arrays
        x = np.arange(ncols) * cellsize / (ncols * cellsize)
        y = np.arange(nrows) * cellsize / (nrows * cellsize)
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Generate random terrain by summing sinusoids
        data = np.zeros((ncols, nrows))

        for i in range(max(1, n_waves_x)):
            for j in range(max(1, n_waves_y)):
                kx = 2 * np.pi * (i + 1)
                ky = 2 * np.pi * (j + 1)
                phase_x = 2 * np.pi * np.random.rand()
                phase_y = 2 * np.pi * np.random.rand()
                amp = amplitude / ((i + 1) * (j + 1))**0.5

                data += amp * np.sin(kx * X + phase_x) * np.sin(ky * Y + phase_y)

        data += base_elevation

        terrain = TerrainData(
            data=data,
            ncols=ncols,
            nrows=nrows,
            xllcorner=xllcorner,
            yllcorner=yllcorner,
            cellsize=cellsize
        )

        logger.info(f"Generated random terrain with amplitude {amplitude} m")
        return terrain

    @staticmethod
    def dam_break_channel(length: float = 500.0,
                         width: float = 100.0,
                         dam_position: float = 250.0,
                         upstream_elevation: float = 10.0,
                         downstream_elevation: float = 0.0,
                         cellsize: float = 1.0) -> TerrainData:
        """
        Generate terrain for classic dam break problem

        Args:
            length: Channel length [m]
            width: Channel width [m]
            dam_position: Dam location along channel [m]
            upstream_elevation: Elevation upstream of dam [m]
            downstream_elevation: Elevation downstream of dam [m]
            cellsize: Cell size [m]

        Returns:
            TerrainData for dam break scenario
        """
        ncols = int(length / cellsize)
        nrows = int(width / cellsize)

        # Create coordinate arrays
        x = np.arange(ncols) * cellsize
        y = np.arange(nrows) * cellsize
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Step function at dam
        data = np.where(X < dam_position,
                       upstream_elevation,
                       downstream_elevation)

        terrain = TerrainData(
            data=data,
            ncols=ncols,
            nrows=nrows,
            xllcorner=0.0,
            yllcorner=0.0,
            cellsize=cellsize
        )

        logger.info(f"Generated dam break channel (length={length} m, dam at x={dam_position} m)")
        return terrain

    @staticmethod
    def composite_terrain(features: List[Tuple[str, dict]],
                         ncols: int,
                         nrows: int,
                         xllcorner: float = 0.0,
                         yllcorner: float = 0.0,
                         cellsize: float = 1.0) -> TerrainData:
        """
        Generate composite terrain by combining multiple features

        Args:
            features: List of (feature_type, kwargs) tuples
            ncols: Number of columns
            nrows: Number of rows
            xllcorner: X coordinate of lower-left corner
            yllcorner: Y coordinate of lower-left corner
            cellsize: Cell size [m]

        Returns:
            TerrainData with combined features

        Example:
            features = [
                ('inclined_plane', {'slope_x': 0.01, 'base_elevation': 0.0}),
                ('gaussian_hill', {'center_x': 50, 'center_y': 25, 'height': 10})
            ]
        """
        # Start with zeros
        data = np.zeros((ncols, nrows))

        gen = GeometryGenerator()

        for feature_type, kwargs in features:
            # Add common parameters
            kwargs.update({
                'ncols': ncols,
                'nrows': nrows,
                'xllcorner': xllcorner,
                'yllcorner': yllcorner,
                'cellsize': cellsize
            })

            # Generate feature
            if feature_type == 'flat_surface':
                terrain = gen.flat_surface(**kwargs)
                data += terrain.data - kwargs.get('elevation', 0.0)  # Additive

            elif feature_type == 'inclined_plane':
                terrain = gen.inclined_plane(**kwargs)
                data += terrain.data - kwargs.get('base_elevation', 0.0)

            elif feature_type == 'gaussian_hill':
                terrain = gen.gaussian_hill(**kwargs)
                data += terrain.data - kwargs.get('base_elevation', 0.0)

            elif feature_type == 'valley':
                terrain = gen.valley(**kwargs)
                data += terrain.data - kwargs.get('base_elevation', 0.0)

            elif feature_type == 'random_terrain':
                terrain = gen.random_terrain(**kwargs)
                data += terrain.data - kwargs.get('base_elevation', 0.0)

            else:
                logger.warning(f"Unknown feature type: {feature_type}")

        terrain_composite = TerrainData(
            data=data,
            ncols=ncols,
            nrows=nrows,
            xllcorner=xllcorner,
            yllcorner=yllcorner,
            cellsize=cellsize
        )

        logger.info(f"Generated composite terrain with {len(features)} features")
        return terrain_composite


# Convenience functions

def create_test_terrain(terrain_type: str = 'hill', **kwargs) -> TerrainData:
    """
    Quick function to create common test terrains

    Args:
        terrain_type: Type of terrain ('flat', 'hill', 'valley', 'incline', 'random')
        **kwargs: Additional parameters for terrain generator

    Returns:
        TerrainData object
    """
    gen = GeometryGenerator()

    # Default parameters
    defaults = {
        'ncols': 100,
        'nrows': 50,
        'cellsize': 1.0,
        'xllcorner': 0.0,
        'yllcorner': 0.0
    }
    defaults.update(kwargs)

    if terrain_type == 'flat':
        return gen.flat_surface(**defaults)
    elif terrain_type == 'hill':
        return gen.gaussian_hill(**defaults)
    elif terrain_type == 'valley':
        return gen.valley(**defaults)
    elif terrain_type == 'incline':
        return gen.inclined_plane(**defaults)
    elif terrain_type == 'random':
        return gen.random_terrain(**defaults)
    else:
        raise ValueError(f"Unknown terrain type: {terrain_type}")
