# -*- coding: utf-8 -*-
"""
HydroSIS-2D Pre/Post-processing Suite

A comprehensive preprocessing and postprocessing toolkit for HydroSIS-2D
hydrodynamic simulation software.

Modules:
    - preprocessing: Mesh generation, geometry processing, boundary conditions
    - postprocessing: Visualization, analysis, reporting
    - gui: Graphical user interface

Author: HydroSIS-2D Development Team
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "HydroSIS-2D Development Team"

from . import preprocessing
from . import postprocessing

__all__ = ['preprocessing', 'postprocessing']
