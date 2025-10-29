"""
Postprocessing module for HydroSIS-2D

Provides tools for:
    - 3D visualization and rendering
    - Result analysis and statistics
    - Report generation
"""

from .visualization import VisualizationEngine, AnimationGenerator

__all__ = [
    'VisualizationEngine',
    'AnimationGenerator'
]

# TODO: Add when implemented
# from .analysis import ResultAnalyzer
# from .reporting import ReportGenerator
