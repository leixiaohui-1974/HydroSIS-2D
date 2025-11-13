# -*- coding: utf-8 -*-
"""
Complete simulation configuration for HydroSIS-2D

Integrates all preprocessing modules into a unified simulation setup.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
import json
import os
from datetime import datetime

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.mesh_generation import (
    DomainParams, MeshGenerator, AdaptiveMeshGenerator, StructuredMesh
)
from preprocessing.geometry import TerrainData, TerrainProcessor, GeometryGenerator
from preprocessing.boundary_conditions import (
    BoundaryConditionManager, BCLocation, BoundaryCondition
)
from preprocessing.initial_conditions import (
    InitialConditionManager, InitialCondition
)

logger = logging.getLogger(__name__)


class SimulationConfig:
    """
    Complete simulation configuration

    Integrates mesh, terrain, boundary conditions, and initial conditions
    into a unified simulation setup with validation and export capabilities.

    Attributes:
        name: Simulation name
        description: Simulation description
        mesh: Structured mesh
        terrain: Terrain elevation array
        bc_manager: Boundary condition manager
        ic_manager: Initial condition manager
    """

    def __init__(self, name: str = "HydroSIS-2D Simulation", description: str = ""):
        """
        Initialize simulation configuration

        Args:
            name: Simulation name
            description: Optional description
        """
        self.name = name
        self.description = description
        self.created_at = datetime.now().isoformat()

        # Components
        self.domain: Optional[DomainParams] = None
        self.mesh: Optional[StructuredMesh] = None
        self.terrain: Optional[np.ndarray] = None
        self.bc_manager: Optional[BoundaryConditionManager] = None
        self.ic_manager: Optional[InitialConditionManager] = None

        # Simulation parameters
        self.simulation_time: float = 0.0
        self.output_interval: float = 0.0
        self.cfl_number: float = 0.5

        logger.info(f"Created simulation configuration: {name}")

    def set_domain_and_mesh(self,
                           xmin: float, xmax: float,
                           ymin: float, ymax: float,
                           nx: int, ny: int) -> None:
        """
        Set domain and generate uniform mesh

        Args:
            xmin, xmax: X-domain extent
            ymin, ymax: Y-domain extent
            nx, ny: Number of cells
        """
        self.domain = DomainParams(xmin, xmax, ymin, ymax)
        generator = MeshGenerator(self.domain)
        self.mesh = generator.generate_uniform_mesh(nx, ny)

        logger.info(f"Set domain: [{xmin}, {xmax}] x [{ymin}, {ymax}]")
        logger.info(f"Generated mesh: {nx} x {ny} = {self.mesh.ncells} cells")

    def set_mesh(self, mesh: StructuredMesh) -> None:
        """
        Set pre-generated mesh

        Args:
            mesh: StructuredMesh object
        """
        self.mesh = mesh
        self.domain = mesh.domain

        logger.info(f"Set custom mesh: {mesh.nx} x {mesh.ny} cells")

    def set_terrain(self, terrain: np.ndarray) -> None:
        """
        Set terrain elevation

        Args:
            terrain: Terrain elevation array
        """
        if self.mesh is None:
            raise ValueError("Mesh must be set before terrain")

        if terrain.shape != (self.mesh.nx, self.mesh.ny):
            raise ValueError(f"Terrain shape {terrain.shape} does not match mesh {(self.mesh.nx, self.mesh.ny)}")

        self.terrain = terrain.copy()
        logger.info("Set terrain elevation")

    def set_flat_terrain(self, elevation: float = 0.0) -> None:
        """
        Set flat terrain at constant elevation

        Args:
            elevation: Terrain elevation [m]
        """
        if self.mesh is None:
            raise ValueError("Mesh must be set before terrain")

        self.terrain = np.full((self.mesh.nx, self.mesh.ny), elevation)
        logger.info(f"Set flat terrain at {elevation} m")

    def setup_boundary_conditions(self) -> BoundaryConditionManager:
        """
        Initialize boundary condition manager

        Returns:
            BoundaryConditionManager instance
        """
        if self.domain is None:
            raise ValueError("Domain must be set before boundary conditions")

        self.bc_manager = BoundaryConditionManager(self.domain)
        logger.info("Initialized boundary condition manager")

        return self.bc_manager

    def setup_initial_conditions(self) -> InitialConditionManager:
        """
        Initialize initial condition manager

        Returns:
            InitialConditionManager instance
        """
        if self.mesh is None:
            raise ValueError("Mesh must be set before initial conditions")

        terrain = self.terrain if self.terrain is not None else np.zeros((self.mesh.nx, self.mesh.ny))
        self.ic_manager = InitialConditionManager(self.mesh, terrain=terrain)
        logger.info("Initialized initial condition manager")

        return self.ic_manager

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate complete simulation configuration

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check required components
        if self.domain is None:
            errors.append("Domain not set")
        if self.mesh is None:
            errors.append("Mesh not set")
        if self.bc_manager is None:
            errors.append("Boundary conditions not set")
        if self.ic_manager is None:
            errors.append("Initial conditions not set")

        # Validate boundary conditions
        if self.bc_manager is not None:
            bc_valid, bc_errors = self.bc_manager.validate()
            if not bc_valid:
                errors.extend([f"BC: {err}" for err in bc_errors])

        # Validate initial conditions
        if self.ic_manager is not None:
            ic_valid, ic_errors = self.ic_manager.validate()
            if not ic_valid:
                errors.extend([f"IC: {err}" for err in ic_errors])

        # Check simulation parameters
        if self.simulation_time <= 0:
            errors.append("Simulation time must be positive")

        if self.output_interval <= 0:
            errors.append("Output interval must be positive")

        if self.cfl_number <= 0 or self.cfl_number > 1.0:
            errors.append("CFL number must be in (0, 1]")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("[OK] Simulation configuration validation passed")
        else:
            logger.warning(f"[ERROR] Simulation configuration validation failed with {len(errors)} error(s)")

        return is_valid, errors

    def set_simulation_parameters(self,
                                 simulation_time: float,
                                 output_interval: float,
                                 cfl_number: float = 0.5) -> None:
        """
        Set simulation parameters

        Args:
            simulation_time: Total simulation time [s]
            output_interval: Output interval [s]
            cfl_number: CFL number for stability (default: 0.5)
        """
        self.simulation_time = simulation_time
        self.output_interval = output_interval
        self.cfl_number = cfl_number

        logger.info(f"Set simulation parameters:")
        logger.info(f"  Time: {simulation_time} s")
        logger.info(f"  Output interval: {output_interval} s")
        logger.info(f"  CFL: {cfl_number}")

    def export_configuration(self, output_dir: str) -> None:
        """
        Export complete simulation configuration

        Args:
            output_dir: Output directory
        """
        os.makedirs(output_dir, exist_ok=True)

        # Export metadata
        metadata = {
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at,
            'domain': {
                'xmin': self.domain.xmin,
                'xmax': self.domain.xmax,
                'ymin': self.domain.ymin,
                'ymax': self.domain.ymax
            } if self.domain else None,
            'mesh': {
                'nx': self.mesh.nx,
                'ny': self.mesh.ny,
                'ncells': self.mesh.ncells,
                'dx': self.mesh.dx,
                'dy': self.mesh.dy
            } if self.mesh else None,
            'simulation_parameters': {
                'simulation_time': self.simulation_time,
                'output_interval': self.output_interval,
                'cfl_number': self.cfl_number
            }
        }

        with open(os.path.join(output_dir, 'simulation_config.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Exported metadata to {output_dir}/simulation_config.json")

        # Export boundary conditions
        if self.bc_manager:
            self.bc_manager.export_to_json(os.path.join(output_dir, 'boundary_conditions.json'))
            logger.info(f"Exported boundary conditions to {output_dir}/boundary_conditions.json")

        # Export initial conditions
        if self.ic_manager:
            self.ic_manager.export_to_json(os.path.join(output_dir, 'initial_conditions.json'))
            self.ic_manager.export_arrays_to_numpy(os.path.join(output_dir, 'initial_fields'))
            self.ic_manager.export_to_vtk(os.path.join(output_dir, 'initial_conditions.vtk'))
            logger.info(f"Exported initial conditions to {output_dir}/")

        # Export mesh
        if self.mesh:
            from preprocessing.mesh_generation.mesh_io import MeshIO
            MeshIO.export_to_vtk(self.mesh, os.path.join(output_dir, 'mesh.vtk'))
            MeshIO.export_to_json(self.mesh, os.path.join(output_dir, 'mesh.json'))
            logger.info(f"Exported mesh to {output_dir}/")

        # Export terrain
        if self.terrain is not None:
            np.save(os.path.join(output_dir, 'terrain.npy'), self.terrain)
            logger.info(f"Exported terrain to {output_dir}/terrain.npy")

        logger.info(f"[OK] Complete configuration exported to {output_dir}/")

    def get_statistics(self) -> Dict:
        """
        Get simulation statistics

        Returns:
            Dictionary with statistics
        """
        stats = {
            'name': self.name,
            'mesh': {
                'nx': self.mesh.nx if self.mesh else 0,
                'ny': self.mesh.ny if self.mesh else 0,
                'ncells': self.mesh.ncells if self.mesh else 0,
                'dx': self.mesh.dx if self.mesh else 0,
                'dy': self.mesh.dy if self.mesh else 0
            },
            'domain': {
                'xmin': self.domain.xmin if self.domain else 0,
                'xmax': self.domain.xmax if self.domain else 0,
                'ymin': self.domain.ymin if self.domain else 0,
                'ymax': self.domain.ymax if self.domain else 0,
                'area': (self.domain.xmax - self.domain.xmin) * (self.domain.ymax - self.domain.ymin) if self.domain else 0
            },
            'simulation': {
                'time': self.simulation_time,
                'output_interval': self.output_interval,
                'cfl': self.cfl_number,
                'num_outputs': int(self.simulation_time / self.output_interval) if self.output_interval > 0 else 0
            }
        }

        # Add initial condition statistics
        if self.ic_manager and self.ic_manager.depth is not None:
            ic_stats = self.ic_manager.get_statistics()
            stats['initial_conditions'] = ic_stats

        return stats

    def summary(self) -> str:
        """
        Generate summary report

        Returns:
            Summary string
        """
        lines = []
        lines.append("=" * 70)
        lines.append(f"SIMULATION CONFIGURATION: {self.name}")
        lines.append("=" * 70)
        lines.append("")

        if self.description:
            lines.append(f"Description: {self.description}")
            lines.append("")

        # Domain and mesh
        if self.domain and self.mesh:
            lines.append("DOMAIN AND MESH:")
            lines.append(f"  Domain: [{self.domain.xmin}, {self.domain.xmax}] x [{self.domain.ymin}, {self.domain.ymax}]")
            lines.append(f"  Mesh: {self.mesh.nx} x {self.mesh.ny} = {self.mesh.ncells} cells")
            lines.append(f"  Cell size: Δx={self.mesh.dx:.3f} m, Δy={self.mesh.dy:.3f} m")
            lines.append(f"  Domain area: {(self.domain.xmax-self.domain.xmin)*(self.domain.ymax-self.domain.ymin):.1f} m^2")
            lines.append("")

        # Terrain
        if self.terrain is not None:
            lines.append("TERRAIN:")
            lines.append(f"  Elevation range: [{np.min(self.terrain):.2f}, {np.max(self.terrain):.2f}] m")
            lines.append(f"  Mean elevation: {np.mean(self.terrain):.2f} m")
            lines.append("")

        # Boundary conditions
        if self.bc_manager:
            lines.append("BOUNDARY CONDITIONS:")
            for loc in BCLocation:
                bc = self.bc_manager.get_boundary(loc)
                if bc:
                    lines.append(f"  {loc.value.upper()}: {bc.bc_type.value}")
            lines.append("")

        # Initial conditions
        if self.ic_manager and self.ic_manager.ic:
            lines.append("INITIAL CONDITIONS:")
            lines.append(f"  Type: {self.ic_manager.ic.ic_type.value}")
            if self.ic_manager.depth is not None:
                stats = self.ic_manager.get_statistics()
                lines.append(f"  Total volume: {stats['total_volume']:.1f} m^3")
                lines.append(f"  Wet cells: {stats['num_wet_cells']} ({100*stats['wet_fraction']:.1f}%)")
                lines.append(f"  Depth range: [{stats['depth']['min']:.2f}, {stats['depth']['max']:.2f}] m")
            lines.append("")

        # Simulation parameters
        lines.append("SIMULATION PARAMETERS:")
        lines.append(f"  Simulation time: {self.simulation_time} s")
        lines.append(f"  Output interval: {self.output_interval} s")
        lines.append(f"  Number of outputs: {int(self.simulation_time/self.output_interval) if self.output_interval > 0 else 0}")
        lines.append(f"  CFL number: {self.cfl_number}")
        lines.append("")

        # Validation
        is_valid, errors = self.validate()
        lines.append("VALIDATION STATUS: " + ("[OK] VALID - Ready to run" if is_valid else "[ERROR] INVALID - Issues found"))
        if errors:
            lines.append("Issues:")
            for error in errors:
                lines.append(f"  - {error}")
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)

    def __repr__(self) -> str:
        mesh_str = f"{self.mesh.nx}x{self.mesh.ny}" if self.mesh else "None"
        return f"<SimulationConfig(name='{self.name}', mesh={mesh_str})>"


# Factory functions for common scenarios

def create_dam_break_simulation(length: float = 200.0,
                                width: float = 100.0,
                                dam_position: float = 0.5,
                                upstream_depth: float = 10.0,
                                downstream_depth: float = 0.0,
                                nx: int = 100,
                                ny: int = 50,
                                simulation_time: float = 10.0) -> SimulationConfig:
    """
    Create dam break simulation configuration

    Args:
        length: Channel length [m]
        width: Channel width [m]
        dam_position: Dam position (0-1)
        upstream_depth: Upstream water depth [m]
        downstream_depth: Downstream water depth [m]
        nx, ny: Mesh resolution
        simulation_time: Simulation time [s]

    Returns:
        SimulationConfig instance
    """
    from preprocessing.initial_conditions import DamBreakIC
    from preprocessing.boundary_conditions import WallBC

    config = SimulationConfig(name="Dam Break", description="Classic dam break test case")

    # Setup mesh
    config.set_domain_and_mesh(0.0, length, 0.0, width, nx, ny)

    # Flat terrain
    config.set_flat_terrain(0.0)

    # Wall boundaries
    bc_manager = config.setup_boundary_conditions()
    bc_manager.set_all_walls()

    # Dam break initial condition
    ic_manager = config.setup_initial_conditions()
    ic = DamBreakIC(dam_position, upstream_depth, downstream_depth, orientation='x')
    ic_manager.set_initial_condition(ic)

    # Simulation parameters
    config.set_simulation_parameters(simulation_time, output_interval=0.5)

    logger.info("Created dam break simulation configuration")

    return config


def create_channel_flow_simulation(length: float = 500.0,
                                   width: float = 100.0,
                                   inflow_depth: float = 5.0,
                                   inflow_velocity: float = 2.0,
                                   slope: float = 0.001,
                                   nx: int = 100,
                                   ny: int = 50,
                                   simulation_time: float = 100.0) -> SimulationConfig:
    """
    Create channel flow simulation configuration

    Args:
        length: Channel length [m]
        width: Channel width [m]
        inflow_depth: Inflow water depth [m]
        inflow_velocity: Inflow velocity [m/s]
        slope: Channel bed slope
        nx, ny: Mesh resolution
        simulation_time: Simulation time [s]

    Returns:
        SimulationConfig instance
    """
    from preprocessing.initial_conditions import DryBedIC
    from preprocessing.geometry import GeometryGenerator

    config = SimulationConfig(name="Channel Flow", description="Steady channel flow")

    # Setup mesh
    config.set_domain_and_mesh(0.0, length, 0.0, width, nx, ny)

    # Inclined terrain
    terrain_gen = GeometryGenerator()
    terrain = terrain_gen.inclined_plane(nx, ny, slope_x=slope, cellsize=length/nx)
    config.set_terrain(terrain.data)

    # Channel boundaries
    bc_manager = config.setup_boundary_conditions()
    bc_manager.set_channel_bcs(inflow_depth, inflow_velocity)

    # Dry bed initial condition
    ic_manager = config.setup_initial_conditions()
    ic = DryBedIC()
    ic_manager.set_initial_condition(ic)

    # Simulation parameters
    config.set_simulation_parameters(simulation_time, output_interval=5.0)

    logger.info("Created channel flow simulation configuration")

    return config
