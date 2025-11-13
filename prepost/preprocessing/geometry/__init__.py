# -*- coding: utf-8 -*-
"""
Geometry processing module for HydroSIS-2D preprocessing

Provides tools for:
    - Terrain data import and processing
    - Geometric shape generation
    - Domain definition
"""

from .terrain_reader import TerrainData, TerrainReader, ASCIIGridReader, load_terrain, save_terrain
from .terrain_processor import TerrainProcessor, process_terrain_for_simulation
from .geometry_generator import GeometryGenerator, create_test_terrain

__all__ = [
    'TerrainData',
    'TerrainReader',
    'ASCIIGridReader',
    'load_terrain',
    'save_terrain',
    'TerrainProcessor',
    'process_terrain_for_simulation',
    'GeometryGenerator',
    'create_test_terrain'
]
