# -*- coding: utf-8 -*-
"""
Integration tests for complete simulation workflows

Tests end-to-end integration of all modules.
"""

import pytest
import numpy as np
import tempfile
import os
import sys
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation import SimulationConfig, create_dam_break_simulation, create_channel_flow_simulation
from preprocessing.mesh_generation import DomainParams, MeshGenerator
from preprocessing.geometry import GeometryGenerator
from preprocessing.boundary_conditions import InflowBC, OutflowBC, WallBC, BCLocation
from preprocessing.initial_conditions import UniformIC, DamBreakIC, DryBedIC


class TestSimulationConfig:
    """Test simulation configuration"""

    def test_creation(self):
        """Test simulation config creation"""
        config = SimulationConfig(name="Test", description="Test simulation")

        assert config.name == "Test"
        assert config.description == "Test simulation"
        assert config.mesh is None

    def test_set_domain_and_mesh(self):
        """Test setting domain and mesh"""
        config = SimulationConfig()
        config.set_domain_and_mesh(0, 100, 0, 50, 20, 10)

        assert config.domain is not None
        assert config.mesh is not None
        assert config.mesh.nx == 20
        assert config.mesh.ny == 10

    def test_set_flat_terrain(self):
        """Test setting flat terrain"""
        config = SimulationConfig()
        config.set_domain_and_mesh(0, 100, 0, 50, 20, 10)
        config.set_flat_terrain(5.0)

        assert config.terrain is not None
        assert config.terrain.shape == (20, 10)
        assert np.all(config.terrain == 5.0)

    def test_setup_boundary_conditions(self):
        """Test boundary condition setup"""
        config = SimulationConfig()
        config.set_domain_and_mesh(0, 100, 0, 50, 20, 10)

        bc_manager = config.setup_boundary_conditions()

        assert bc_manager is not None
        assert config.bc_manager is bc_manager

    def test_setup_initial_conditions(self):
        """Test initial condition setup"""
        config = SimulationConfig()
        config.set_domain_and_mesh(0, 100, 0, 50, 20, 10)

        ic_manager = config.setup_initial_conditions()

        assert ic_manager is not None
        assert config.ic_manager is ic_manager

    def test_validation_incomplete(self):
        """Test validation with incomplete config"""
        config = SimulationConfig()

        is_valid, errors = config.validate()

        assert not is_valid
        assert len(errors) > 0

    def test_validation_complete(self):
        """Test validation with complete config"""
        config = SimulationConfig()
        config.set_domain_and_mesh(0, 100, 0, 50, 20, 10)
        config.set_flat_terrain(0.0)

        bc_manager = config.setup_boundary_conditions()
        bc_manager.set_all_walls()

        ic_manager = config.setup_initial_conditions()
        ic_manager.set_initial_condition(UniformIC(depth=2.0))

        config.set_simulation_parameters(10.0, 1.0, 0.5)

        is_valid, errors = config.validate()

        assert is_valid
        assert len(errors) == 0

    def test_export_configuration(self):
        """Test configuration export"""
        config = SimulationConfig()
        config.set_domain_and_mesh(0, 100, 0, 50, 10, 5)
        config.set_flat_terrain(0.0)

        bc_manager = config.setup_boundary_conditions()
        bc_manager.set_all_walls()

        ic_manager = config.setup_initial_conditions()
        ic_manager.set_initial_condition(UniformIC(depth=2.0))

        config.set_simulation_parameters(10.0, 1.0, 0.5)

        # Export to temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'test_export')
            config.export_configuration(output_dir)

            # Check files exist
            assert os.path.exists(os.path.join(output_dir, 'simulation_config.json'))
            assert os.path.exists(os.path.join(output_dir, 'boundary_conditions.json'))
            assert os.path.exists(os.path.join(output_dir, 'initial_conditions.json'))
            assert os.path.exists(os.path.join(output_dir, 'mesh.vtk'))
            assert os.path.exists(os.path.join(output_dir, 'initial_fields.npz'))

    def test_get_statistics(self):
        """Test statistics computation"""
        config = SimulationConfig()
        config.set_domain_and_mesh(0, 100, 0, 50, 20, 10)
        config.set_flat_terrain(0.0)

        ic_manager = config.setup_initial_conditions()
        ic_manager.set_initial_condition(UniformIC(depth=3.0))

        config.set_simulation_parameters(50.0, 5.0, 0.5)

        stats = config.get_statistics()

        assert 'mesh' in stats
        assert 'domain' in stats
        assert 'simulation' in stats
        assert stats['mesh']['nx'] == 20
        assert stats['mesh']['ny'] == 10
        assert stats['simulation']['time'] == 50.0


class TestFactoryFunctions:
    """Test factory functions for common scenarios"""

    def test_create_dam_break_simulation(self):
        """Test dam break factory function"""
        config = create_dam_break_simulation(
            length=200.0,
            width=100.0,
            nx=50,
            ny=25,
            simulation_time=10.0
        )

        assert config.name == "Dam Break"
        assert config.mesh is not None
        assert config.mesh.nx == 50
        assert config.mesh.ny == 25

        # Check validation
        is_valid, errors = config.validate()
        assert is_valid

    def test_create_channel_flow_simulation(self):
        """Test channel flow factory function"""
        config = create_channel_flow_simulation(
            length=500.0,
            width=100.0,
            nx=50,
            ny=25,
            simulation_time=100.0
        )

        assert config.name == "Channel Flow"
        assert config.mesh is not None

        # Check validation
        is_valid, errors = config.validate()
        assert is_valid


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows"""

    def test_dam_break_workflow(self):
        """Test complete dam break workflow"""
        # Create configuration
        config = SimulationConfig(name="Dam Break Test")

        # Setup mesh
        config.set_domain_and_mesh(0, 200, 0, 100, 50, 25)

        # Setup terrain
        config.set_flat_terrain(0.0)

        # Setup boundary conditions
        bc_manager = config.setup_boundary_conditions()
        bc_manager.set_all_walls()

        # Setup initial conditions
        ic_manager = config.setup_initial_conditions()
        ic = DamBreakIC(0.5, 10.0, 0.0, 'x')
        ic_manager.set_initial_condition(ic)

        # Set simulation parameters
        config.set_simulation_parameters(10.0, 1.0, 0.5)

        # Validate
        is_valid, errors = config.validate()
        assert is_valid

        # Check statistics
        stats = config.get_statistics()
        assert stats['initial_conditions']['total_volume'] > 0

    def test_channel_flow_workflow(self):
        """Test complete channel flow workflow"""
        # Create configuration
        config = SimulationConfig(name="Channel Flow Test")

        # Setup mesh
        config.set_domain_and_mesh(0, 500, 0, 100, 50, 25)

        # Setup terrain with slope
        terrain_gen = GeometryGenerator()
        terrain = terrain_gen.inclined_plane(50, 25, slope_x=0.001, cellsize=10.0)
        config.set_terrain(terrain.data)

        # Setup boundary conditions
        bc_manager = config.setup_boundary_conditions()
        bc_manager.set_channel_bcs(5.0, 2.0)

        # Setup initial conditions
        ic_manager = config.setup_initial_conditions()
        ic_manager.set_initial_condition(DryBedIC())

        # Set simulation parameters
        config.set_simulation_parameters(100.0, 5.0, 0.5)

        # Validate
        is_valid, errors = config.validate()
        assert is_valid

    def test_custom_scenario_workflow(self):
        """Test custom scenario workflow"""
        # Create configuration
        config = SimulationConfig(name="Custom Test")

        # Setup mesh
        config.set_domain_and_mesh(0, 1000, 0, 500, 100, 50)

        # Setup complex terrain
        terrain_gen = GeometryGenerator()
        terrain = terrain_gen.composite_terrain(
            features=[
                ('inclined_plane', {'slope_x': 0.002, 'base_elevation': 0.0}),
                ('gaussian_hill', {'center_x': 500.0, 'center_y': 250.0, 'height': 20.0, 'width': 100.0})
            ],
            ncols=100,
            nrows=50,
            cellsize=10.0
        )
        config.set_terrain(terrain.data)

        # Setup boundary conditions
        bc_manager = config.setup_boundary_conditions()
        bc_manager.set_boundary(InflowBC(BCLocation.WEST, depth=5.0, velocity_x=2.0))
        bc_manager.set_boundary(OutflowBC(BCLocation.EAST, outflow_type='zero_gradient'))
        bc_manager.set_boundary(WallBC(BCLocation.NORTH))
        bc_manager.set_boundary(WallBC(BCLocation.SOUTH))

        # Setup initial conditions
        ic_manager = config.setup_initial_conditions()
        ic_manager.set_initial_condition(UniformIC(depth=3.0))

        # Set simulation parameters
        config.set_simulation_parameters(200.0, 10.0, 0.5)

        # Validate
        is_valid, errors = config.validate()
        assert is_valid

        # Export and check
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'custom_test')
            config.export_configuration(output_dir)

            # Verify all files exist
            assert os.path.exists(os.path.join(output_dir, 'simulation_config.json'))
            assert os.path.exists(os.path.join(output_dir, 'boundary_conditions.json'))
            assert os.path.exists(os.path.join(output_dir, 'initial_conditions.json'))

    def test_export_and_reimport(self):
        """Test export and reimport workflow"""
        # Create and setup configuration
        config1 = create_dam_break_simulation(
            length=200.0,
            width=100.0,
            nx=50,
            ny=25,
            simulation_time=10.0
        )

        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = os.path.join(tmpdir, 'export_test')
            config1.export_configuration(output_dir)

            # Verify files
            assert os.path.exists(os.path.join(output_dir, 'simulation_config.json'))
            assert os.path.exists(os.path.join(output_dir, 'initial_fields.npz'))

            # Load initial fields
            data = np.load(os.path.join(output_dir, 'initial_fields.npz'))
            assert 'depth' in data
            assert 'velocity_x' in data
            assert 'velocity_y' in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
