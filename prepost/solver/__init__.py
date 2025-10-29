"""
HydroSIS-2D Solver Module

This module provides a Python reference implementation of the 2D shallow water
equations solver. It serves as a prototype for the GPU-accelerated CUDA version.

Main Classes:
    ShallowWaterSolver: Main solver class
    SolverConfig: Configuration parameters

Example:
    >>> from solver import ShallowWaterSolver, SolverConfig
    >>> from preprocessing.mesh_generation import MeshGenerator, DomainParams
    >>>
    >>> # Create mesh
    >>> domain = DomainParams(0, 100, 0, 50)
    >>> mesh_gen = MeshGenerator(domain)
    >>> mesh = mesh_gen.generate_uniform_mesh(100, 50)
    >>>
    >>> # Initialize solver
    >>> terrain = np.zeros((100, 50))
    >>> config = SolverConfig(t_end=10.0, cfl=0.5)
    >>> solver = ShallowWaterSolver(mesh, terrain, config)
    >>>
    >>> # Set initial conditions
    >>> h0 = dam_break_initial_condition(mesh)
    >>> solver.set_initial_conditions(h0, u0=0, v0=0)
    >>>
    >>> # Run solver
    >>> results = solver.solve()
"""

from .shallow_water_solver import ShallowWaterSolver, SolverConfig

__all__ = ['ShallowWaterSolver', 'SolverConfig']

__version__ = '0.1.0'
__author__ = 'HydroSIS-2D Development Team'
