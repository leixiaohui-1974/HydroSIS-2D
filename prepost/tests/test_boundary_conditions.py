# -*- coding: utf-8 -*-
"""
Unit tests for boundary conditions module
"""

import pytest
import numpy as np
import tempfile
import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from preprocessing.boundary_conditions import (
    BCType,
    BCLocation,
    BoundaryCondition,
    WallBC,
    InflowBC,
    OutflowBC,
    PeriodicBC,
    TransmissiveBC,
    TimeSeriesBC,
    FunctionBC,
    BoundaryConditionManager,
    create_bc_from_dict
)
from preprocessing.mesh_generation import DomainParams, MeshGenerator


class TestBCEnums:
    """Test boundary condition enums"""

    def test_bc_type_enum(self):
        """Test BCType enumeration"""
        assert BCType.WALL.value == "wall"
        assert BCType.INFLOW.value == "inflow"
        assert BCType.OUTFLOW.value == "outflow"
        assert BCType.PERIODIC.value == "periodic"

    def test_bc_location_enum(self):
        """Test BCLocation enumeration"""
        assert BCLocation.WEST.value == "west"
        assert BCLocation.EAST.value == "east"
        assert BCLocation.SOUTH.value == "south"
        assert BCLocation.NORTH.value == "north"


class TestWallBC:
    """Test wall boundary condition"""

    def test_creation(self):
        """Test wall BC creation"""
        bc = WallBC(BCLocation.WEST, roughness=0.03)

        assert bc.bc_type == BCType.WALL
        assert bc.location == BCLocation.WEST
        assert bc.roughness == 0.03

    def test_to_dict(self):
        """Test dictionary export"""
        bc = WallBC(BCLocation.SOUTH, roughness=0.02)
        data = bc.to_dict()

        assert data['type'] == 'wall'
        assert data['location'] == 'south'
        assert data['roughness'] == 0.02


class TestInflowBC:
    """Test inflow boundary condition"""

    def test_creation(self):
        """Test inflow BC creation"""
        bc = InflowBC(
            BCLocation.WEST,
            depth=5.0,
            velocity_x=2.0,
            velocity_y=0.0
        )

        assert bc.bc_type == BCType.INFLOW
        assert bc.depth == 5.0
        assert bc.velocity_x == 2.0
        assert bc.velocity_y == 0.0

    def test_to_dict(self):
        """Test dictionary export"""
        bc = InflowBC(BCLocation.WEST, depth=3.0, velocity_x=1.5)
        data = bc.to_dict()

        assert data['type'] == 'inflow'
        assert data['depth'] == 3.0
        assert data['velocity_x'] == 1.5
        assert data['velocity_y'] == 0.0


class TestOutflowBC:
    """Test outflow boundary condition"""

    def test_zero_gradient(self):
        """Test zero gradient outflow"""
        bc = OutflowBC(BCLocation.EAST, outflow_type='zero_gradient')

        assert bc.bc_type == BCType.OUTFLOW
        assert bc.outflow_type == 'zero_gradient'
        assert bc.depth is None

    def test_fixed_depth(self):
        """Test fixed depth outflow"""
        bc = OutflowBC(BCLocation.EAST, outflow_type='fixed_depth', depth=2.0)

        assert bc.outflow_type == 'fixed_depth'
        assert bc.depth == 2.0

    def test_fixed_depth_without_value(self):
        """Test that fixed depth requires depth value"""
        with pytest.raises(ValueError, match="Fixed depth outflow requires depth"):
            OutflowBC(BCLocation.EAST, outflow_type='fixed_depth')


class TestPeriodicBC:
    """Test periodic boundary condition"""

    def test_valid_pairing(self):
        """Test valid periodic pairing"""
        bc = PeriodicBC(BCLocation.WEST, BCLocation.EAST)

        assert bc.bc_type == BCType.PERIODIC
        assert bc.paired_location == BCLocation.EAST

    def test_invalid_pairing(self):
        """Test invalid periodic pairing"""
        with pytest.raises(ValueError, match="Invalid periodic boundary pair"):
            PeriodicBC(BCLocation.WEST, BCLocation.NORTH)


class TestTransmissiveBC:
    """Test transmissive boundary condition"""

    def test_creation(self):
        """Test transmissive BC creation"""
        bc = TransmissiveBC(BCLocation.EAST)

        assert bc.bc_type == BCType.TRANSMISSIVE
        assert bc.location == BCLocation.EAST


class TestTimeSeriesBC:
    """Test time-series boundary condition"""

    def test_creation(self):
        """Test time-series BC creation"""
        times = np.array([0.0, 10.0, 20.0])
        depths = np.array([1.0, 2.0, 1.5])
        velocity_x = np.array([0.5, 1.0, 0.8])

        bc = TimeSeriesBC(
            BCLocation.WEST,
            times=times,
            depths=depths,
            velocity_x=velocity_x
        )

        assert bc.bc_type == BCType.TIME_SERIES
        assert len(bc.times) == 3
        np.testing.assert_array_equal(bc.times, times)

    def test_sorting(self):
        """Test that times are sorted"""
        times = np.array([20.0, 0.0, 10.0])
        depths = np.array([1.5, 1.0, 2.0])

        bc = TimeSeriesBC(BCLocation.WEST, times=times, depths=depths)

        # Should be sorted
        assert bc.times[0] == 0.0
        assert bc.times[1] == 10.0
        assert bc.times[2] == 20.0
        assert bc.depths[0] == 1.0  # Corresponding to time 0
        assert bc.depths[1] == 2.0  # Corresponding to time 10
        assert bc.depths[2] == 1.5  # Corresponding to time 20

    def test_interpolation(self):
        """Test time interpolation"""
        times = np.array([0.0, 10.0, 20.0])
        depths = np.array([1.0, 2.0, 1.0])
        vel_x = np.array([0.0, 1.0, 0.0])

        bc = TimeSeriesBC(BCLocation.WEST, times=times, depths=depths, velocity_x=vel_x)

        # Interpolate at t=5 (should be between 1.0 and 2.0)
        h, u, v = bc.interpolate(5.0)
        assert h == pytest.approx(1.5)
        assert u == pytest.approx(0.5)

        # Interpolate at t=0 (boundary)
        h, u, v = bc.interpolate(0.0)
        assert h == 1.0
        assert u == 0.0

        # Interpolate at t=20 (boundary)
        h, u, v = bc.interpolate(20.0)
        assert h == 1.0
        assert u == 0.0

        # Interpolate beyond range (should clamp)
        h, u, v = bc.interpolate(30.0)
        assert h == 1.0  # Last value
        assert u == 0.0

    def test_mismatched_lengths(self):
        """Test error on mismatched array lengths"""
        times = np.array([0.0, 10.0])
        depths = np.array([1.0, 2.0, 3.0])  # Wrong length

        with pytest.raises(ValueError, match="must have same length"):
            TimeSeriesBC(BCLocation.WEST, times=times, depths=depths)


class TestFunctionBC:
    """Test function-based boundary condition"""

    def test_creation(self):
        """Test function BC creation"""
        def depth_func(t):
            return 1.0 + 0.5 * np.sin(2 * np.pi * t / 10.0)

        def vel_func(t):
            return 0.5 * t

        bc = FunctionBC(
            BCLocation.WEST,
            depth_func=depth_func,
            velocity_x_func=vel_func
        )

        assert bc.location == BCLocation.WEST

    def test_evaluation(self):
        """Test function evaluation"""
        def depth_func(t):
            return 2.0 + t

        def vel_func(t):
            return 0.5 * t

        bc = FunctionBC(
            BCLocation.WEST,
            depth_func=depth_func,
            velocity_x_func=vel_func
        )

        h, u, v = bc.evaluate(5.0)
        assert h == 7.0  # 2.0 + 5.0
        assert u == 2.5  # 0.5 * 5.0
        assert v == 0.0  # Default


class TestBoundaryConditionManager:
    """Test boundary condition manager"""

    def setup_method(self):
        """Setup for each test"""
        self.domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=50.0)
        self.manager = BoundaryConditionManager(self.domain)

    def test_creation(self):
        """Test manager creation"""
        assert self.manager.domain == self.domain
        assert len(self.manager.boundaries) == 0

    def test_set_boundary(self):
        """Test setting boundary condition"""
        bc = WallBC(BCLocation.WEST)
        self.manager.set_boundary(bc)

        assert self.manager.has_boundary(BCLocation.WEST)
        assert self.manager.get_boundary(BCLocation.WEST) == bc

    def test_remove_boundary(self):
        """Test removing boundary condition"""
        bc = WallBC(BCLocation.WEST)
        self.manager.set_boundary(bc)
        assert self.manager.has_boundary(BCLocation.WEST)

        self.manager.remove_boundary(BCLocation.WEST)
        assert not self.manager.has_boundary(BCLocation.WEST)

    def test_get_all_boundaries(self):
        """Test getting all boundaries"""
        bc1 = WallBC(BCLocation.WEST)
        bc2 = WallBC(BCLocation.EAST)
        self.manager.set_boundary(bc1)
        self.manager.set_boundary(bc2)

        boundaries = self.manager.get_all_boundaries()
        assert len(boundaries) == 2

    def test_validation_missing_boundaries(self):
        """Test validation with missing boundaries"""
        # Set only west boundary
        self.manager.set_boundary(WallBC(BCLocation.WEST))

        is_valid, errors = self.manager.validate()
        assert not is_valid
        assert len(errors) == 3  # Missing east, south, north

    def test_validation_complete(self):
        """Test validation with complete boundaries"""
        self.manager.set_all_walls()

        is_valid, errors = self.manager.validate()
        assert is_valid
        assert len(errors) == 0

    def test_validation_periodic_pair(self):
        """Test validation of periodic pairing"""
        # Set west as periodic but not east
        self.manager.set_boundary(PeriodicBC(BCLocation.WEST, BCLocation.EAST))
        self.manager.set_boundary(WallBC(BCLocation.SOUTH))
        self.manager.set_boundary(WallBC(BCLocation.NORTH))
        self.manager.set_boundary(WallBC(BCLocation.EAST))  # Should be periodic

        is_valid, errors = self.manager.validate()
        assert not is_valid
        assert any("paired with non-periodic" in err for err in errors)

    def test_set_all_walls(self):
        """Test setting all walls"""
        self.manager.set_all_walls()

        for location in BCLocation:
            assert self.manager.has_boundary(location)
            bc = self.manager.get_boundary(location)
            assert isinstance(bc, WallBC)

    def test_set_channel_bcs(self):
        """Test channel BC setup"""
        self.manager.set_channel_bcs(inflow_depth=5.0, inflow_velocity=2.0)

        # Check inflow
        inflow = self.manager.get_boundary(BCLocation.WEST)
        assert isinstance(inflow, InflowBC)
        assert inflow.depth == 5.0
        assert inflow.velocity_x == 2.0

        # Check outflow
        outflow = self.manager.get_boundary(BCLocation.EAST)
        assert isinstance(outflow, OutflowBC)

        # Check walls
        north = self.manager.get_boundary(BCLocation.NORTH)
        south = self.manager.get_boundary(BCLocation.SOUTH)
        assert isinstance(north, WallBC)
        assert isinstance(south, WallBC)

    def test_apply_to_mesh(self):
        """Test applying BCs to mesh"""
        self.manager.set_all_walls()

        generator = MeshGenerator(self.domain)
        mesh = generator.generate_uniform_mesh(20, 10)

        masks = self.manager.apply_to_mesh(mesh)

        assert 'west' in masks
        assert 'east' in masks
        assert 'south' in masks
        assert 'north' in masks

        # Check west mask
        assert np.all(masks['west'][0, :] == True)
        assert np.all(masks['west'][1:, :] == False)

    def test_get_boundary_cells(self):
        """Test getting boundary cell indices"""
        generator = MeshGenerator(self.domain)
        mesh = generator.generate_uniform_mesh(10, 5)

        # West boundary
        west_cells = self.manager.get_boundary_cells(mesh, BCLocation.WEST)
        assert len(west_cells) == 5  # ny cells
        assert all(cell[0] == 0 for cell in west_cells)  # i=0

        # East boundary
        east_cells = self.manager.get_boundary_cells(mesh, BCLocation.EAST)
        assert len(east_cells) == 5
        assert all(cell[0] == 9 for cell in east_cells)  # i=nx-1

    def test_export_import_json(self):
        """Test JSON export/import"""
        # Setup BCs
        self.manager.set_channel_bcs(inflow_depth=3.0, inflow_velocity=1.5)

        # Export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name

        try:
            self.manager.export_to_json(filepath)

            # Import
            manager_loaded = BoundaryConditionManager.import_from_json(filepath)

            # Verify domain
            assert manager_loaded.domain.xmin == self.domain.xmin
            assert manager_loaded.domain.xmax == self.domain.xmax

            # Verify boundaries
            for location in BCLocation:
                orig_bc = self.manager.get_boundary(location)
                loaded_bc = manager_loaded.get_boundary(location)

                assert loaded_bc is not None
                assert loaded_bc.bc_type == orig_bc.bc_type
                assert loaded_bc.location == orig_bc.location

        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_summary(self):
        """Test summary generation"""
        self.manager.set_all_walls()

        summary = self.manager.summary()

        assert "BOUNDARY CONDITIONS SUMMARY" in summary
        assert "WEST" in summary
        assert "EAST" in summary
        assert "VALID" in summary


class TestBCFactory:
    """Test BC factory function"""

    def test_create_wall_from_dict(self):
        """Test creating wall BC from dict"""
        data = {
            'type': 'wall',
            'location': 'west',
            'roughness': 0.03
        }

        bc = create_bc_from_dict(data)

        assert isinstance(bc, WallBC)
        assert bc.location == BCLocation.WEST
        assert bc.roughness == 0.03

    def test_create_inflow_from_dict(self):
        """Test creating inflow BC from dict"""
        data = {
            'type': 'inflow',
            'location': 'west',
            'depth': 5.0,
            'velocity_x': 2.0,
            'velocity_y': 0.0
        }

        bc = create_bc_from_dict(data)

        assert isinstance(bc, InflowBC)
        assert bc.depth == 5.0
        assert bc.velocity_x == 2.0

    def test_create_time_series_from_dict(self):
        """Test creating time-series BC from dict"""
        data = {
            'type': 'time_series',
            'location': 'west',
            'times': [0.0, 10.0, 20.0],
            'depths': [1.0, 2.0, 1.5],
            'velocity_x': [0.5, 1.0, 0.8]
        }

        bc = create_bc_from_dict(data)

        assert isinstance(bc, TimeSeriesBC)
        assert len(bc.times) == 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
