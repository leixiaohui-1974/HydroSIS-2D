"""
Preprocessing module for HydroSIS-2D

Provides tools for:
    - Mesh generation (structured and unstructured)
    - Geometry processing and CAD import
    - Boundary condition setup
"""

from .mesh_generation import MeshGenerator, AdaptiveMeshGenerator

__all__ = [
    'MeshGenerator',
    'AdaptiveMeshGenerator'
]

# TODO: Add when implemented
# from .geometry import GeometryProcessor
# from .boundary_conditions import BoundaryConditionEditor
