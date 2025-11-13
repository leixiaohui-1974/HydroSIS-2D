# -*- coding: utf-8 -*-
"""
Unit tests for the shallow water solver module

Tests the ShallowWaterSolver class functionality including:
- Initialization
- Time step computation
- Flux computation
- Mass conservation
- Solver convergence
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from preprocessing.mesh_generation import MeshGenerator, DomainParams
from solver import ShallowWaterSolver, SolverConfig


class TestSolverConfig:
    """Test SolverConfig class"""

    def test_default_config(self):
        """Test default configuration values"""
        config = SolverConfig()

        assert config.cfl == 0.5
        assert config.g == 9.81
        assert config.h_dry == 1e-4
        assert config.manning_n == 0.03
        assert config.flux_scheme == 'HLL'
        assert config.time_scheme == 'euler'

    def test_custom_config(self):
        """Test custom configuration"""
        config = SolverConfig(
            cfl=0.3,
            g=10.0,
            manning_n=0.05,
            t_end=20.0
        )

        assert config.cfl == 0.3
        assert config.g == 10.0
        assert config.manning_n == 0.05
        assert config.t_end == 20.0


class TestSolverInitialization:
    """Test solver initialization"""

    @pytest.fixture
    def simple_mesh(self):
        """Create a simple test mesh"""
        domain = DomainParams(0, 100, 0, 50)
        generator = MeshGenerator(domain)
        return generator.generate_uniform_mesh(nx=50, ny=25)

    @pytest.fixture
    def simple_terrain(self, simple_mesh):
        """Create flat terrain"""
        return np.zeros((simple_mesh.nx, simple_mesh.ny))

    def test_solver_initialization(self, simple_mesh, simple_terrain):
        """Test basic solver initialization"""
        config = SolverConfig()
        solver = ShallowWaterSolver(simple_mesh, simple_terrain, config)

        assert solver.nx == 50
        assert solver.ny == 25
        assert solver.dx == 2.0
        assert solver.dy == 2.0
        assert solver.t == 0.0
        assert solver.step_count == 0

    def test_solver_arrays_shape(self, simple_mesh, simple_terrain):
        """Test that solver arrays have correct shapes"""
        solver = ShallowWaterSolver(simple_mesh, simple_terrain)

        assert solver.h.shape == (50, 25)
        assert solver.u.shape == (50, 25)
        assert solver.v.shape == (50, 25)
        assert solver.hu.shape == (50, 25)
        assert solver.hv.shape == (50, 25)
        assert solver.z.shape == (50, 25)

        # Fluxes should be at cell interfaces
        assert solver.flux_x.shape == (3, 51, 25)  # nx+1 interfaces in x
        assert solver.flux_y.shape == (3, 50, 26)  # ny+1 interfaces in y

    def test_initial_conditions(self, simple_mesh, simple_terrain):
        """Test setting initial conditions"""
        solver = ShallowWaterSolver(simple_mesh, simple_terrain)

        h0 = np.ones((50, 25)) * 5.0
        u0 = np.ones((50, 25)) * 1.0
        v0 = np.zeros((50, 25))

        solver.set_initial_conditions(h0, u0, v0)

        assert np.allclose(solver.h, 5.0)
        assert np.allclose(solver.u, 1.0)
        assert np.allclose(solver.v, 0.0)
        assert np.allclose(solver.hu, 5.0)  # hu = h * u
        assert solver.initial_mass > 0


class TestTimestepComputation:
    """Test CFL time step computation"""

    @pytest.fixture
    def solver_with_ic(self):
        """Create solver with initial conditions"""
        domain = DomainParams(0, 100, 0, 50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=50, ny=25)
        terrain = np.zeros((50, 25))

        solver = ShallowWaterSolver(mesh, terrain)

        # Set uniform initial conditions
        h0 = np.ones((50, 25)) * 5.0
        u0 = np.ones((50, 25)) * 2.0
        v0 = np.zeros((50, 25))
        solver.set_initial_conditions(h0, u0, v0)

        return solver

    def test_timestep_cfl_condition(self, solver_with_ic):
        """Test that time step satisfies CFL condition"""
        solver = solver_with_ic
        dt = solver.compute_timestep()

        # Compute expected max time step
        c = np.sqrt(solver.config.g * solver.h)
        lambda_x = np.abs(solver.u) + c
        lambda_y = np.abs(solver.v) + c
        lambda_max = max(np.max(lambda_x), np.max(lambda_y))

        dt_expected = solver.config.cfl * min(solver.dx, solver.dy) / lambda_max

        assert dt <= dt_expected * 1.01  # Allow 1% tolerance
        assert dt > 0

    def test_timestep_with_still_water(self):
        """Test time step with still water (no velocity)"""
        domain = DomainParams(0, 100, 0, 50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=50, ny=25)
        terrain = np.zeros((50, 25))

        solver = ShallowWaterSolver(mesh, terrain)

        # Still water
        h0 = np.ones((50, 25)) * 5.0
        u0 = np.zeros((50, 25))
        v0 = np.zeros((50, 25))
        solver.set_initial_conditions(h0, u0, v0)

        dt = solver.compute_timestep()

        # For still water: dt = CFL * dx / sqrt(gh)
        c = np.sqrt(solver.config.g * 5.0)
        dt_expected = solver.config.cfl * min(solver.dx, solver.dy) / c

        assert abs(dt - dt_expected) < 1e-6

    def test_timestep_dry_bed(self):
        """Test time step with dry bed"""
        domain = DomainParams(0, 100, 0, 50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=50, ny=25)
        terrain = np.zeros((50, 25))

        solver = ShallowWaterSolver(mesh, terrain)

        # Dry bed
        h0 = np.zeros((50, 25))
        u0 = np.zeros((50, 25))
        v0 = np.zeros((50, 25))
        solver.set_initial_conditions(h0, u0, v0)

        dt = solver.compute_timestep()

        # Should return maximum allowed time step for dry bed
        assert dt > 0
        assert dt <= solver.config.dt_max


class TestMassConservation:
    """Test mass conservation"""

    def test_mass_conservation_still_water(self):
        """Test that mass is conserved for still water"""
        domain = DomainParams(0, 100, 0, 50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=50, ny=25)
        terrain = np.zeros((50, 25))

        config = SolverConfig(
            t_end=1.0,
            output_interval=10.0,  # No output
            print_progress=False
        )
        solver = ShallowWaterSolver(mesh, terrain, config)

        # Still water
        h0 = np.ones((50, 25)) * 5.0
        u0 = np.zeros((50, 25))
        v0 = np.zeros((50, 25))
        solver.set_initial_conditions(h0, u0, v0)

        initial_mass = solver.initial_mass

        # Run solver for 1 second
        solver.solve(t_end=1.0)

        final_mass = np.sum(solver.h) * solver.dx * solver.dy
        mass_error = abs(final_mass - initial_mass) / initial_mass

        # Mass should be conserved to high accuracy for still water
        assert mass_error < 1e-10

    def test_mass_conservation_dam_break(self):
        """Test mass conservation for dam break"""
        domain = DomainParams(0, 200, 0, 20)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=100, ny=10)
        terrain = np.zeros((100, 10))

        config = SolverConfig(
            t_end=1.0,
            output_interval=10.0,
            print_progress=False
        )
        solver = ShallowWaterSolver(mesh, terrain, config)

        # Dam break initial conditions
        h0 = np.zeros((100, 10))
        h0[:50, :] = 10.0  # Upstream
        h0[50:, :] = 1.0   # Downstream
        u0 = np.zeros((100, 10))
        v0 = np.zeros((100, 10))
        solver.set_initial_conditions(h0, u0, v0)

        initial_mass = solver.initial_mass

        # Run solver
        solver.solve(t_end=1.0)

        final_mass = np.sum(solver.h) * solver.dx * solver.dy
        mass_error = abs(final_mass - initial_mass) / initial_mass

        # Mass should be conserved to within 1% for dam break
        assert mass_error < 0.01


class TestPhysicalConstraints:
    """Test physical constraints"""

    def test_non_negative_depth(self):
        """Test that depth remains non-negative"""
        domain = DomainParams(0, 100, 0, 50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=50, ny=25)
        terrain = np.zeros((50, 25))

        config = SolverConfig(
            t_end=1.0,
            output_interval=10.0,
            print_progress=False
        )
        solver = ShallowWaterSolver(mesh, terrain, config)

        # Mixed wet/dry initial conditions
        h0 = np.random.rand(50, 25) * 5.0
        u0 = np.zeros((50, 25))
        v0 = np.zeros((50, 25))
        solver.set_initial_conditions(h0, u0, v0)

        # Run solver
        solver.solve(t_end=1.0)

        # Depth should be non-negative everywhere
        assert np.all(solver.h >= 0.0)

    def test_finite_values(self):
        """Test that all values remain finite"""
        domain = DomainParams(0, 100, 0, 50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=50, ny=25)
        terrain = np.zeros((50, 25))

        config = SolverConfig(
            t_end=0.5,
            output_interval=10.0,
            print_progress=False
        )
        solver = ShallowWaterSolver(mesh, terrain, config)

        # Dam break
        h0 = np.zeros((50, 25))
        h0[:25, :] = 5.0
        u0 = np.zeros((50, 25))
        v0 = np.zeros((50, 25))
        solver.set_initial_conditions(h0, u0, v0)

        # Run solver
        solver.solve(t_end=0.5)

        # All values should be finite
        assert np.all(np.isfinite(solver.h))
        assert np.all(np.isfinite(solver.u))
        assert np.all(np.isfinite(solver.v))


class TestSolverMethods:
    """Test individual solver methods"""

    @pytest.fixture
    def basic_solver(self):
        """Create a basic solver for testing"""
        domain = DomainParams(0, 100, 0, 50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=50, ny=25)
        terrain = np.zeros((50, 25))

        solver = ShallowWaterSolver(mesh, terrain)

        h0 = np.ones((50, 25)) * 5.0
        u0 = np.zeros((50, 25))
        v0 = np.zeros((50, 25))
        solver.set_initial_conditions(h0, u0, v0)

        return solver

    def test_compute_source_terms(self, basic_solver):
        """Test source term computation"""
        solver = basic_solver

        S_h, S_hu, S_hv = solver.compute_source_terms()

        # Should have correct shapes
        assert S_h.shape == (50, 25)
        assert S_hu.shape == (50, 25)
        assert S_hv.shape == (50, 25)

        # For flat bed with no velocity, source should be zero
        assert np.allclose(S_h, 0.0)
        assert np.allclose(S_hu, 0.0)
        assert np.allclose(S_hv, 0.0)

    def test_get_state(self, basic_solver):
        """Test get_state method"""
        solver = basic_solver

        state = solver.get_state()

        assert 'h' in state
        assert 'u' in state
        assert 'v' in state
        assert 'z' in state
        assert 't' in state
        assert 'step' in state

        assert state['h'].shape == (50, 25)
        assert state['t'] == 0.0
        assert state['step'] == 0

    def test_single_step(self, basic_solver):
        """Test single time step"""
        solver = basic_solver

        t_initial = solver.t
        step_initial = solver.step_count

        solver.step()

        # Time should advance
        assert solver.t > t_initial
        assert solver.step_count == step_initial + 1

        # Physical constraints
        assert np.all(solver.h >= 0.0)
        assert np.all(np.isfinite(solver.h))


class TestSolverIntegration:
    """Integration tests for the solver"""

    def test_simple_run(self):
        """Test a simple solver run"""
        domain = DomainParams(0, 100, 0, 50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=50, ny=25)
        terrain = np.zeros((50, 25))

        config = SolverConfig(
            t_end=0.1,
            output_interval=10.0,
            print_progress=False
        )
        solver = ShallowWaterSolver(mesh, terrain, config)

        h0 = np.ones((50, 25)) * 5.0
        u0 = np.zeros((50, 25))
        v0 = np.zeros((50, 25))
        solver.set_initial_conditions(h0, u0, v0)

        stats = solver.solve()

        assert stats['steps'] > 0
        assert stats['simulated_time'] >= 0.1
        assert 'mass_error' in stats

    def test_callback_function(self):
        """Test callback function"""
        domain = DomainParams(0, 100, 0, 50)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=50, ny=25)
        terrain = np.zeros((50, 25))

        config = SolverConfig(
            t_end=1.0,  # Run longer to get more time steps
            output_interval=10.0,
            print_progress=False
        )
        solver = ShallowWaterSolver(mesh, terrain, config)

        # Dam break to get more dynamics
        h0 = np.ones((50, 25)) * 5.0
        h0[:25, :] = 10.0  # Create some variation
        u0 = np.zeros((50, 25))
        v0 = np.zeros((50, 25))
        solver.set_initial_conditions(h0, u0, v0)

        # Track callback calls
        callback_times = []
        def callback(t, step, dt):
            callback_times.append(t)

        solver.set_callback(callback, interval=1)  # Call every step
        solver.solve()

        # Callback should have been called multiple times
        assert len(callback_times) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
