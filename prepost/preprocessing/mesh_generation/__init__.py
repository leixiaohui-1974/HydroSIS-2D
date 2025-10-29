"""
Mesh generation module for HydroSIS-2D preprocessing

Provides structured and unstructured mesh generation capabilities with
adaptive refinement features.
"""

from .mesh_generator import MeshGenerator, DomainParams, MeshParams, StructuredMesh
from .adaptive_mesh import AdaptiveMeshGenerator, RefinementZone
from .mesh_quality import MeshQualityChecker, QualityMetrics
from .mesh_io import MeshIO

__all__ = [
    'MeshGenerator',
    'AdaptiveMeshGenerator',
    'MeshQualityChecker',
    'MeshIO',
    'DomainParams',
    'MeshParams',
    'StructuredMesh',
    'RefinementZone',
    'QualityMetrics'
]
