# -*- coding: utf-8 -*-
"""
Boundary condition manager for HydroSIS-2D

Manage and validate boundary conditions for simulation domains.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import json

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from preprocessing.mesh_generation import StructuredMesh, DomainParams
from .bc_types import (
    BoundaryCondition, BCLocation, BCType,
    WallBC, InflowBC, OutflowBC, PeriodicBC,
    TransmissiveBC, TimeSeriesBC,
    create_bc_from_dict
)

logger = logging.getLogger(__name__)


class BoundaryConditionManager:
    """
    Manage boundary conditions for a simulation domain

    Features:
        - Store BCs for all domain boundaries
        - Validate BC consistency
        - Export/import BC configurations
        - Apply BCs to mesh
    """

    def __init__(self, domain: DomainParams):
        """
        Initialize boundary condition manager

        Args:
            domain: Domain parameters
        """
        self.domain = domain
        self.boundaries: Dict[BCLocation, BoundaryCondition] = {}

        logger.info(f"Initialized BoundaryConditionManager for domain {domain}")

    def set_boundary(self, bc: BoundaryCondition) -> None:
        """
        Set boundary condition for a location

        Args:
            bc: Boundary condition to set
        """
        if bc.location in self.boundaries:
            logger.warning(f"Overwriting existing BC at {bc.location.value}")

        self.boundaries[bc.location] = bc
        logger.info(f"Set {bc.bc_type.value} BC at {bc.location.value}")

    def get_boundary(self, location: BCLocation) -> Optional[BoundaryCondition]:
        """
        Get boundary condition at location

        Args:
            location: Boundary location

        Returns:
            BoundaryCondition or None if not set
        """
        return self.boundaries.get(location)

    def remove_boundary(self, location: BCLocation) -> None:
        """
        Remove boundary condition at location

        Args:
            location: Boundary location
        """
        if location in self.boundaries:
            del self.boundaries[location]
            logger.info(f"Removed BC at {location.value}")

    def has_boundary(self, location: BCLocation) -> bool:
        """Check if boundary condition is set at location"""
        return location in self.boundaries

    def get_all_boundaries(self) -> List[BoundaryCondition]:
        """Get list of all boundary conditions"""
        return list(self.boundaries.values())

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate boundary condition configuration

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check that all boundaries are specified
        required_locations = {BCLocation.WEST, BCLocation.EAST,
                            BCLocation.SOUTH, BCLocation.NORTH}

        for loc in required_locations:
            if loc not in self.boundaries:
                errors.append(f"Missing boundary condition at {loc.value}")

        # Validate periodic boundary pairs
        periodic_bcs = [bc for bc in self.boundaries.values()
                       if bc.bc_type == BCType.PERIODIC]

        for bc in periodic_bcs:
            paired_bc = self.boundaries.get(bc.paired_location)
            if paired_bc is None:
                errors.append(f"Periodic BC at {bc.location.value} has no pair at {bc.paired_location.value}")
            elif paired_bc.bc_type != BCType.PERIODIC:
                errors.append(f"Periodic BC at {bc.location.value} paired with non-periodic BC at {bc.paired_location.value}")

        # Check for conflicting inflow/outflow
        inflow_count = sum(1 for bc in self.boundaries.values() if bc.bc_type == BCType.INFLOW)
        outflow_count = sum(1 for bc in self.boundaries.values() if bc.bc_type == BCType.OUTFLOW)

        if inflow_count > 0 and outflow_count == 0:
            errors.append("Inflow BC specified but no outflow BC (mass conservation may be violated)")

        # Validate time-series BCs
        for bc in self.boundaries.values():
            if isinstance(bc, TimeSeriesBC):
                if len(bc.times) < 2:
                    errors.append(f"Time-series BC at {bc.location.value} must have at least 2 time points")
                if np.any(bc.depths < 0):
                    errors.append(f"Time-series BC at {bc.location.value} contains negative depths")

        is_valid = len(errors) == 0

        if is_valid:
            logger.info("Boundary condition validation passed")
        else:
            logger.warning(f"Boundary condition validation failed with {len(errors)} error(s)")

        return is_valid, errors

    def set_all_walls(self) -> None:
        """
        Convenience method to set all boundaries as walls

        Useful for enclosed domains.
        """
        for location in BCLocation:
            self.set_boundary(WallBC(location))

        logger.info("Set all boundaries to wall BCs")

    def set_channel_bcs(self,
                       inflow_depth: float,
                       inflow_velocity: float = 0.0,
                       outflow_type: str = "zero_gradient") -> None:
        """
        Convenience method for typical channel flow setup

        Sets west as inflow, east as outflow, north/south as walls.

        Args:
            inflow_depth: Inflow depth [m]
            inflow_velocity: Inflow velocity [m/s]
            outflow_type: Outflow boundary type
        """
        # Inflow at west
        self.set_boundary(InflowBC(BCLocation.WEST, inflow_depth, inflow_velocity, 0.0))

        # Outflow at east
        self.set_boundary(OutflowBC(BCLocation.EAST, outflow_type))

        # Walls at north and south
        self.set_boundary(WallBC(BCLocation.NORTH))
        self.set_boundary(WallBC(BCLocation.SOUTH))

        logger.info("Set channel flow boundary conditions")

    def apply_to_mesh(self, mesh: StructuredMesh) -> Dict[str, np.ndarray]:
        """
        Apply boundary conditions to mesh

        Creates boundary masks for each BC type.

        Args:
            mesh: StructuredMesh object

        Returns:
            Dictionary of boundary masks
        """
        masks = {}

        # Create masks for each boundary
        west_mask = np.zeros((mesh.nx, mesh.ny), dtype=bool)
        east_mask = np.zeros((mesh.nx, mesh.ny), dtype=bool)
        south_mask = np.zeros((mesh.nx, mesh.ny), dtype=bool)
        north_mask = np.zeros((mesh.nx, mesh.ny), dtype=bool)

        # Set boundary cells
        west_mask[0, :] = True
        east_mask[-1, :] = True
        south_mask[:, 0] = True
        north_mask[:, -1] = True

        masks['west'] = west_mask
        masks['east'] = east_mask
        masks['south'] = south_mask
        masks['north'] = north_mask

        logger.info(f"Applied boundary conditions to mesh ({mesh.nx}x{mesh.ny})")

        return masks

    def get_boundary_cells(self, mesh: StructuredMesh,
                          location: BCLocation) -> np.ndarray:
        """
        Get cell indices for specified boundary

        Args:
            mesh: StructuredMesh object
            location: Boundary location

        Returns:
            Array of cell indices (i, j) pairs
        """
        if location == BCLocation.WEST:
            indices = [(0, j) for j in range(mesh.ny)]
        elif location == BCLocation.EAST:
            indices = [(mesh.nx - 1, j) for j in range(mesh.ny)]
        elif location == BCLocation.SOUTH:
            indices = [(i, 0) for i in range(mesh.nx)]
        elif location == BCLocation.NORTH:
            indices = [(i, mesh.ny - 1) for i in range(mesh.nx)]
        else:
            raise ValueError(f"Unknown boundary location: {location}")

        return np.array(indices)

    def export_to_json(self, filepath: str) -> None:
        """
        Export boundary conditions to JSON file

        Args:
            filepath: Output file path
        """
        data = {
            'domain': {
                'xmin': self.domain.xmin,
                'xmax': self.domain.xmax,
                'ymin': self.domain.ymin,
                'ymax': self.domain.ymax
            },
            'boundaries': {}
        }

        for loc, bc in self.boundaries.items():
            data['boundaries'][loc.value] = bc.to_dict()

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported boundary conditions to {filepath}")

    @staticmethod
    def import_from_json(filepath: str) -> 'BoundaryConditionManager':
        """
        Import boundary conditions from JSON file

        Args:
            filepath: Input file path

        Returns:
            BoundaryConditionManager instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Create domain
        domain_data = data['domain']
        domain = DomainParams(
            xmin=domain_data['xmin'],
            xmax=domain_data['xmax'],
            ymin=domain_data['ymin'],
            ymax=domain_data['ymax']
        )

        # Create manager
        manager = BoundaryConditionManager(domain)

        # Load boundaries
        for loc_str, bc_data in data['boundaries'].items():
            bc = create_bc_from_dict(bc_data)
            manager.set_boundary(bc)

        logger.info(f"Imported boundary conditions from {filepath}")

        return manager

    def summary(self) -> str:
        """Generate summary report of boundary conditions"""
        lines = []
        lines.append("=" * 60)
        lines.append("BOUNDARY CONDITIONS SUMMARY")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Domain: [{self.domain.xmin}, {self.domain.xmax}] x [{self.domain.ymin}, {self.domain.ymax}]")
        lines.append("")

        for location in BCLocation:
            bc = self.boundaries.get(location)
            if bc:
                lines.append(f"{location.value.upper()}:")
                lines.append(f"  Type: {bc.bc_type.value}")
                lines.append(f"  Description: {bc.description}")

                # Type-specific info
                if isinstance(bc, WallBC):
                    lines.append(f"  Roughness: {bc.roughness}")
                elif isinstance(bc, InflowBC):
                    lines.append(f"  Depth: {bc.depth} m")
                    lines.append(f"  Velocity: ({bc.velocity_x}, {bc.velocity_y}) m/s")
                elif isinstance(bc, OutflowBC):
                    lines.append(f"  Outflow type: {bc.outflow_type}")
                    if bc.depth is not None:
                        lines.append(f"  Fixed depth: {bc.depth} m")
                elif isinstance(bc, PeriodicBC):
                    lines.append(f"  Paired with: {bc.paired_location.value}")
                elif isinstance(bc, TimeSeriesBC):
                    lines.append(f"  Time range: [{bc.times[0]:.2f}, {bc.times[-1]:.2f}] s")
                    lines.append(f"  Depth range: [{np.min(bc.depths):.2f}, {np.max(bc.depths):.2f}] m")

                lines.append("")
            else:
                lines.append(f"{location.value.upper()}: NOT SET")
                lines.append("")

        # Validation status
        is_valid, errors = self.validate()
        lines.append("Validation Status: " + ("[OK] VALID" if is_valid else "[ERROR] INVALID"))
        if errors:
            lines.append("Errors:")
            for error in errors:
                lines.append(f"  - {error}")

        lines.append("=" * 60)

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<BoundaryConditionManager({len(self.boundaries)} boundaries)>"
