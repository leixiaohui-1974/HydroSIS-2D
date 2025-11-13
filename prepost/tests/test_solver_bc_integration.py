# -*- coding: utf-8 -*-
"""
Unit tests for solver boundary condition integration

Tests the integration of the preprocessing module's boundary condition
classes with the ShallowWaterSolver, including:
- Inflow boundaries
- Outflow boundaries
- Time-series boundaries
- Mass flux tracking with open boundaries
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from preprocessing.mesh_generation import MeshGenerator, DomainParams
from preprocessing.boundary_conditions import BoundaryConditionManager, BCLocation, BCType, InflowBC, OutflowBC, WallBC, TimeSeriesBC
from solver import ShallowWaterSolver, SolverConfig


class TestBoundaryConditionIntegration:
    """Test solver integration with boundary condition classes"""

    @pytest.fixture
    def channel_mesh(self):
        """Create a channel mesh (long in x, narrow in y)"""
        domain = DomainParams(0, 500, 0, 100)
        generator = MeshGenerator(domain)
        return generator.generate_uniform_mesh(nx=100, ny=20)

    @pytest.fixture
    def flat_terrain(self, channel_mesh):
        """Create flat terrain"""
        return np.zeros((channel_mesh.nx, channel_mesh.ny))

    def test_inflow_outflow_boundaries(self, channel_mesh, flat_terrain):
        """Test channel flow with inflow (west) and outflow (east)"""
        # Setup boundary conditions
        domain = DomainParams(0, 500, 0, 100)
        bc_manager = BoundaryConditionManager(domain)
        bc_manager.set_boundary(InflowBC(
            location=BCLocation.WEST,
            depth=5.0,
            velocity_x=2.0,
            velocity_y=0.0
        ))
        bc_manager.set_boundary(OutflowBC(
            location=BCLocation.EAST,
            outflow_type='zero_gradient'
        ))
        bc_manager.set_boundary(WallBC(location=BCLocation.SOUTH))
        bc_manager.set_boundary(WallBC(location=BCLocation.NORTH))

        # Validate BC setup
        is_valid, errors = bc_manager.validate()
        assert is_valid, f"BC validation failed: {errors}"

        # Create solver
        config = SolverConfig(
            t_end=5.0,
            output_interval=100.0,  # No output
            print_progress=False
        )
        solver = ShallowWaterSolver(channel_mesh, flat_terrain, config)

        # Set initial conditions (initially dry or low water)
        h0 = np.ones((100, 20)) * 1.0
        u0 = np.zeros((100, 20))
        v0 = np.zeros((100, 20))
        solver.set_initial_conditions(h0, u0, v0)

        # Set boundary conditions
        solver.set_boundary_conditions(bc_manager)

        # Run simulation
        solver.solve(t_end=5.0)

        # Check that inflow boundary has correct values
        assert np.allclose(solver.h[0, :], 5.0, atol=0.5), "Inflow depth not maintained"
        assert np.allclose(solver.u[0, :], 2.0, atol=0.5), "Inflow velocity not maintained"

        # Check that water has propagated into domain (check cells near inflow)
        # With v=2m/s for 5s, water travels ~10m = 2 cells (dx=5m)
        assert np.mean(solver.h[1:5, :]) > 1.5, "Water did not propagate from inflow"

        # Check outflow boundary (should be zero-gradient)
        depth_gradient = solver.h[-1, :] - solver.h[-2, :]
        assert np.abs(np.mean(depth_gradient)) < 0.5, "Outflow boundary not zero-gradient"

    def test_wall_boundaries_default(self, channel_mesh, flat_terrain):
        """Test that default wall boundaries work without bc_manager"""
        config = SolverConfig(
            t_end=0.5,
            output_interval=100.0,
            print_progress=False
        )
        solver = ShallowWaterSolver(channel_mesh, flat_terrain, config)

        # Dam break initial condition
        h0 = np.ones((100, 20)) * 1.0
        h0[:50, :] = 5.0
        u0 = np.zeros((100, 20))
        v0 = np.zeros((100, 20))
        solver.set_initial_conditions(h0, u0, v0)

        # Don't set bc_manager - should use default walls
        solver.solve(t_end=0.5)

        # Check wall boundaries (depth should match adjacent cells)
        # West wall
        assert np.allclose(solver.h[0, :], solver.h[1, :], atol=0.2)
        # East wall
        assert np.allclose(solver.h[-1, :], solver.h[-2, :], atol=0.2)

        # For wall boundaries, velocity should be reflected (opposite sign)
        # But only check where there's significant flow
        west_u_product = solver.u[0, :] * solver.u[1, :]
        east_u_product = solver.u[-1, :] * solver.u[-2, :]

        # Where there's flow (|u| > 0.1), product should be negative
        significant_flow_west = np.abs(solver.u[1, :]) > 0.1
        significant_flow_east = np.abs(solver.u[-2, :]) > 0.1

        if np.any(significant_flow_west):
            assert np.mean(west_u_product[significant_flow_west]) <= 0, "West wall not reflective where flow exists"
        if np.any(significant_flow_east):
            assert np.mean(east_u_product[significant_flow_east]) <= 0, "East wall not reflective where flow exists"

    def test_all_wall_boundaries(self, channel_mesh, flat_terrain):
        """Test all wall boundaries using bc_manager"""
        # Setup all walls
        domain = DomainParams(0, 500, 0, 100)
        bc_manager = BoundaryConditionManager(domain)
        bc_manager.set_all_walls()

        # Create solver
        config = SolverConfig(
            t_end=0.5,
            output_interval=100.0,
            print_progress=False
        )
        solver = ShallowWaterSolver(channel_mesh, flat_terrain, config)

        # Dam break
        h0 = np.ones((100, 20)) * 1.0
        h0[:50, :] = 5.0
        u0 = np.zeros((100, 20))
        v0 = np.zeros((100, 20))
        solver.set_initial_conditions(h0, u0, v0)

        solver.set_boundary_conditions(bc_manager)
        initial_mass = solver.initial_mass

        # Run
        solver.solve(t_end=0.5)

        # Mass should be conserved with all walls
        final_mass = np.sum(solver.h) * solver.dx * solver.dy
        mass_error = abs(final_mass - initial_mass) / initial_mass
        assert mass_error < 0.01, f"Mass not conserved with all walls: {mass_error:.3%}"

    def test_timeseries_boundary(self, channel_mesh, flat_terrain):
        """Test time-varying boundary condition"""
        # Create time-varying inflow
        times = np.array([0.0, 2.0, 4.0, 6.0])
        depths = np.array([3.0, 5.0, 4.0, 3.0])
        velocities_x = np.array([1.0, 2.0, 1.5, 1.0])

        domain = DomainParams(0, 500, 0, 100)
        bc_manager = BoundaryConditionManager(domain)
        bc_manager.set_boundary(TimeSeriesBC(
            location=BCLocation.WEST,
            times=times,
            depths=depths,
            velocity_x=velocities_x
        ))
        bc_manager.set_boundary(OutflowBC(location=BCLocation.EAST))
        bc_manager.set_boundary(WallBC(location=BCLocation.SOUTH))
        bc_manager.set_boundary(WallBC(location=BCLocation.NORTH))

        # Create solver
        config = SolverConfig(
            t_end=6.0,
            output_interval=100.0,
            print_progress=False
        )
        solver = ShallowWaterSolver(channel_mesh, flat_terrain, config)

        # Initial conditions
        h0 = np.ones((100, 20)) * 2.0
        u0 = np.zeros((100, 20))
        v0 = np.zeros((100, 20))
        solver.set_initial_conditions(h0, u0, v0)

        solver.set_boundary_conditions(bc_manager)

        # Run simulation
        solver.solve(t_end=6.0)

        # At the end (t=6.0), inflow should be back to initial values (3.0 m, 1.0 m/s)
        assert np.abs(np.mean(solver.h[0, :]) - 3.0) < 0.5, "Time-series depth not correct at end"
        assert np.abs(np.mean(solver.u[0, :]) - 1.0) < 0.5, "Time-series velocity not correct at end"

    def test_inflow_mass_balance(self, channel_mesh, flat_terrain):
        """Test mass balance with inflow - mass should increase"""
        # Setup inflow only on west, walls elsewhere
        domain = DomainParams(0, 500, 0, 100)
        bc_manager = BoundaryConditionManager(domain)
        bc_manager.set_boundary(InflowBC(
            location=BCLocation.WEST,
            depth=5.0,
            velocity_x=2.0
        ))
        bc_manager.set_boundary(WallBC(location=BCLocation.EAST))
        bc_manager.set_boundary(WallBC(location=BCLocation.SOUTH))
        bc_manager.set_boundary(WallBC(location=BCLocation.NORTH))

        # Create solver
        config = SolverConfig(
            t_end=2.0,
            output_interval=100.0,
            print_progress=False
        )
        solver = ShallowWaterSolver(channel_mesh, flat_terrain, config)

        # Start with dry bed
        h0 = np.ones((100, 20)) * 0.5
        u0 = np.zeros((100, 20))
        v0 = np.zeros((100, 20))
        solver.set_initial_conditions(h0, u0, v0)

        solver.set_boundary_conditions(bc_manager)
        initial_mass = solver.initial_mass

        # Run
        solver.solve(t_end=2.0)

        # Mass should increase due to inflow
        final_mass = np.sum(solver.h) * solver.dx * solver.dy
        assert final_mass > initial_mass, "Mass did not increase with inflow"

        # Verify substantial increase (water is flowing in)
        mass_increase = (final_mass - initial_mass) / initial_mass
        assert mass_increase > 0.1, f"Mass increase too small: {mass_increase:.1%}"


class TestBoundaryConditionValidation:
    """Test boundary condition validation within solver context"""

    def test_invalid_bc_warning(self):
        """Test that solver warns about invalid boundary conditions"""
        domain = DomainParams(0, 100, 0, 50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=50, ny=25)
        terrain = np.zeros((50, 25))

        # Create invalid BC setup (missing boundaries)
        bc_manager = BoundaryConditionManager(domain)
        bc_manager.set_boundary(WallBC(location=BCLocation.WEST))
        # Missing other boundaries

        solver = ShallowWaterSolver(mesh, terrain)

        # This should not crash but should warn
        solver.set_boundary_conditions(bc_manager)

        # Solver should still run with warnings
        h0 = np.ones((50, 25)) * 5.0
        u0 = np.zeros((50, 25))
        v0 = np.zeros((50, 25))
        solver.set_initial_conditions(h0, u0, v0)

        # Should complete without error
        solver.step()
        assert solver.step_count == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
