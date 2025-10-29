"""
Preprocessing module for HydroSIS-2D

Provides tools for:
    - Mesh generation (structured and unstructured)
    - Geometry processing and terrain handling
    - Boundary condition setup
"""

from .mesh_generation import MeshGenerator, AdaptiveMeshGenerator
from .geometry import GeometryGenerator, TerrainProcessor, TerrainReader
from .boundary_conditions import BoundaryConditionManager, WallBC, InflowBC, OutflowBC

__all__ = [
    # Mesh generation
    'MeshGenerator',
    'AdaptiveMeshGenerator',

    # Geometry processing
    'GeometryGenerator',
    'TerrainProcessor',
    'TerrainReader',

    # Boundary conditions
    'BoundaryConditionManager',
    'WallBC',
    'InflowBC',
    'OutflowBC'
]
