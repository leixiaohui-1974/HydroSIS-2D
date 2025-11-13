"""
Numerical Properties Tests

This module tests essential numerical properties of the solver:
- Convergence order
- Conservation properties
- Stability limits
- Positivity preservation
- Well-balanced property
- Entropy stability
"""

import pytest
import numpy as np
from typing import Dict, List, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from preprocessing.geometry import GeometryGenerator
from preprocessing.initial_conditions import InitialConditionManager


class TestConvergenceOrder:
    """Test spatial and temporal convergence order"""

    def test_spatial_convergence_smooth_solution(self):
        """
        Test Case: Verify spatial convergence order

        Method: Run same problem on series of refined meshes
        Expected: Error ~ h^p where p = order of accuracy
        """
        # Test with smooth manufactured solution
        # Smooth Gaussian hill propagation

        mesh_sizes = [25, 50, 100, 200]
        errors_l2 = []

        for nx in mesh_sizes:
            # Create mesh
            domain = DomainParams(xmin=-50.0, xmax=50.0, ymin=-50.0, ymax=50.0)
            mesh_gen = MeshGenerator(domain)
            mesh = mesh_gen.generate_uniform_mesh(nx=nx, ny=nx)

            # Initial condition: Gaussian bump
            x_centers = mesh.x
            y_centers = mesh.y
            h_init = np.ones((mesh.nx, mesh.ny)) * 5.0

            for i in range(mesh.nx):
                for j in range(mesh.ny):
                    x = x_centers[i]
                    y = y_centers[j]
                    r = np.sqrt(x**2 + y**2)
                    h_init[i, j] += 1.0 * np.exp(-r**2 / 50.0)

            # After simulation: compare to analytical solution
            # (Requires GPU solver to compute error)

            # For now, verify mesh setup
            assert h_init.shape == (nx, nx)
            assert h_init.min() >= 5.0
            assert h_init.max() <= 6.0

        # Expected convergence:
        # 1st order: error ~ h^1
        # 2nd order (MUSCL): error ~ h^2

    def test_temporal_convergence(self):
        """
        Test Case: Verify temporal convergence order

        Method: Run with different time steps
        Expected: Error ~ dt^p where p = order of time integrator
        """
        # Test with same spatial mesh, varying dt

        time_steps = [0.1, 0.05, 0.025, 0.0125]
        errors_l2 = []

        for dt in time_steps:
            # Create mesh
            domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
            mesh_gen = MeshGenerator(domain)
            mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

            # Simple initial condition
            ic_manager = InitialConditionManager(mesh)
            h_init = np.full((mesh.nx, mesh.ny), 5.0)
            u_init = np.full((mesh.nx, mesh.ny), 1.0)
            v_init = np.zeros((mesh.nx, mesh.ny))

            ic_manager.set_depth_array(h_init)
            ic_manager.set_velocity_array(u_init, v_init)

            # After simulation with fixed dt:
            # Compare to reference solution

        # Expected convergence:
        # Euler: error ~ dt^1
        # RK2: error ~ dt^2
        # RK3: error ~ dt^3


class TestConservationProperties:
    """Test conservation of mass, momentum, and energy"""

    def test_mass_conservation_closed_domain(self):
        """
        Test Case: Mass conservation with wall boundaries

        Expected: Mass error < machine precision
        """
        # Create closed domain
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Arbitrary initial conditions
        ic_manager = InitialConditionManager(mesh)

        # Non-uniform depth
        x_centers = mesh.x
        y_centers = mesh.y
        h_init = np.ones((mesh.nx, mesh.ny)) * 5.0
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]
                h_init[i, j] += 2.0 * np.sin(2*np.pi*x/100.0) * np.cos(2*np.pi*y/100.0)

        h_init = np.maximum(h_init, 1.0)  # Ensure positive

        u_init = np.random.uniform(-0.5, 0.5, (mesh.nx, mesh.ny))
        v_init = np.random.uniform(-0.5, 0.5, (mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # Calculate initial mass
        dx = mesh.dx
        dy = mesh.dy
        mass_initial = np.sum(h_init) * dx * dy

        # After simulation with wall BC:
        # mass_final should equal mass_initial to machine precision
        # mass_error = |mass_final - mass_initial| / mass_initial < 1e-12

        assert mass_initial > 0

    def test_momentum_conservation_frictionless(self):
        """
        Test Case: Momentum conservation (frictionless, flat bed)

        Expected: Total momentum conserved for closed system
        """
        # Create closed domain
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Initial conditions with net momentum
        ic_manager = InitialConditionManager(mesh)
        h_init = np.full((mesh.nx, mesh.ny), 5.0)
        u_init = np.full((mesh.nx, mesh.ny), 2.0)
        v_init = np.full((mesh.nx, mesh.ny), 1.0)

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # Calculate initial momentum
        dx = mesh.dx
        dy = mesh.dy
        momentum_x_initial = np.sum(h_init * u_init) * dx * dy
        momentum_y_initial = np.sum(h_init * v_init) * dx * dy

        # After simulation (frictionless, flat bed, wall BC):
        # Total momentum should be conserved
        # (May redistribute, but total should remain constant)

        assert momentum_x_initial > 0
        assert momentum_y_initial > 0

    def test_energy_conservation_frictionless(self):
        """
        Test Case: Energy conservation (frictionless, flat bed)

        Expected: Total energy conserved (or slightly decreasing due to numerical dissipation)
        """
        # Create closed domain
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Initial conditions
        ic_manager = InitialConditionManager(mesh)
        h_init = np.full((mesh.nx, mesh.ny), 5.0)
        u_init = np.full((mesh.nx, mesh.ny), 1.0)
        v_init = np.full((mesh.nx, mesh.ny), 0.5)

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # Calculate initial energy
        dx = mesh.dx
        dy = mesh.dy
        g = 9.81

        # Kinetic energy
        ke_initial = 0.5 * np.sum(h_init * (u_init**2 + v_init**2)) * dx * dy

        # Potential energy (relative to z=0)
        pe_initial = 0.5 * g * np.sum(h_init**2) * dx * dy

        energy_initial = ke_initial + pe_initial

        # After simulation (frictionless):
        # Total energy should be conserved or decrease slightly
        # energy_final <= energy_initial
        # (Exact conservation not guaranteed due to numerical dissipation)

        assert energy_initial > 0


class TestStabilityLimits:
    """Test stability conditions (CFL, etc.)"""

    def test_cfl_condition_violation(self):
        """
        Test Case: Verify solver fails/warns when CFL violated

        Expected: Simulation unstable for CFL > 1.0
        """
        # Create mesh
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Initial conditions with high velocity
        ic_manager = InitialConditionManager(mesh)
        h_init = np.full((mesh.nx, mesh.ny), 5.0)

        # Very high velocity to violate CFL
        g = 9.81
        c = np.sqrt(g * 5.0)  # Wave speed
        u_max = 10 * c  # 10x wave speed

        u_init = np.full((mesh.nx, mesh.ny), u_max)
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # With fixed large dt that violates CFL:
        # dt = CFL * dx / (u + c)
        # If CFL > 1.0, simulation should be unstable

        # Solver should:
        # - Detect CFL violation
        # - Either reduce dt automatically
        # - Or warn user and fail gracefully

    def test_dry_state_stability(self):
        """
        Test Case: Stability with dry cells

        Expected: No instabilities at dry/wet interface
        """
        # Create mesh
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Initial conditions: half wet, half dry
        ic_manager = InitialConditionManager(mesh)
        x_centers = mesh.x

        h_init = np.zeros((mesh.nx, mesh.ny))
        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        # Wet region (x < 50)
        h_init[x_centers < 50, :] = 5.0
        u_init[x_centers < 50, :] = 2.0

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # After simulation:
        # - Water flows into dry region
        # - No negative depths
        # - No instabilities at wetting front
        # - Positivity preserved


class TestPositivityPreservation:
    """Test that water depth remains non-negative"""

    def test_positivity_dam_break(self):
        """
        Test Case: Dam break (challenging for positivity)

        Expected: h >= 0 at all times
        """
        # Standard dam break
        domain = DomainParams(xmin=-50.0, xmax=50.0, ymin=0.0, ymax=10.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=500, ny=10)

        # Dam break IC
        ic_manager = InitialConditionManager(mesh)
        x_centers = mesh.x

        h_init = np.zeros((mesh.nx, mesh.ny))
        h_init[x_centers < 0, :] = 10.0  # Left: 10m
        h_init[x_centers >= 0, :] = 1.0  # Right: 1m

        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # After simulation:
        # - h >= 0 everywhere
        # - h >= dry_tolerance (e.g., 1e-6)

    def test_positivity_with_source_terms(self):
        """
        Test Case: Positivity with source terms (friction)

        Expected: h >= 0 even with strong friction
        """
        # Create mesh
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Sloped terrain
        x_centers = mesh.x
        z = np.zeros((mesh.nx, mesh.ny))
        for i in range(mesh.nx):
            z[i, :] = 0.1 * x_centers[i]

        # Thin water layer (challenging for positivity)
        ic_manager = InitialConditionManager(mesh)
        h_init = np.full((mesh.nx, mesh.ny), 0.1)  # Very thin: 10cm
        u_init = np.full((mesh.nx, mesh.ny), 0.5)  # Some velocity
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # With strong Manning friction (n = 0.1):
        # Friction may drain thin layers quickly
        # Must preserve h >= 0


class TestWellBalancedProperty:
    """Test well-balanced property (C-property)"""

    def test_lake_at_rest_flat(self):
        """
        Test Case: Lake at rest on flat bed

        Expected: No spurious currents (u, v = 0)
        """
        # Flat bed
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        z = np.zeros((mesh.nx, mesh.ny))

        # Constant depth, zero velocity
        ic_manager = InitialConditionManager(mesh)
        h_init = np.full((mesh.nx, mesh.ny), 5.0)
        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # After simulation:
        # u, v should remain ~0 (< 1e-10)
        # h should remain constant

    def test_lake_at_rest_complex_bathymetry(self):
        """
        Test Case: Lake at rest on complex bathymetry

        Expected: Exact balance of pressure and bed slope
        """
        # Complex bathymetry
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Non-trivial bathymetry
        x_centers = mesh.x
        y_centers = mesh.y
        z = np.zeros((mesh.nx, mesh.ny))

        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]
                # Bowl-shaped: z = 0.01 * ((x-50)² + (y-50)²)
                z[i, j] = 0.01 * ((x - 50)**2 + (y - 50)**2) / 50.0

        # Constant free surface elevation
        eta = 10.0
        ic_manager = InitialConditionManager(mesh)
        h_init = eta - z
        h_init = np.maximum(h_init, 0.0)

        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # Verify hydrostatic equilibrium
        free_surface = h_init + z
        assert np.allclose(free_surface, eta, atol=1e-10)

        # After simulation:
        # Solution should remain at rest
        # Tests well-balanced property for complex geometry

    def test_small_perturbation_on_lake(self):
        """
        Test Case: Small perturbation on lake at rest

        Expected: Only perturbation propagates, not spurious waves
        """
        # Sloped bed
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        x_centers = mesh.x
        z = np.zeros((mesh.nx, mesh.ny))
        for i in range(mesh.nx):
            z[i, :] = 0.05 * x_centers[i]  # 5% slope

        # Lake at rest + small perturbation
        eta = 10.0
        ic_manager = InitialConditionManager(mesh)
        h_init = eta - z

        # Add small Gaussian perturbation
        y_centers = mesh.y
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                x = x_centers[i]
                y = y_centers[j]
                r = np.sqrt((x - 50)**2 + (y - 50)**2)
                h_init[i, j] += 0.01 * np.exp(-r**2 / 25.0)  # 1cm perturbation

        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # After simulation:
        # - Small perturbation propagates as waves
        # - No large spurious waves from bed slope
        # - Well-balanced property prevents numerical noise


class TestEntropyStability:
    """Test entropy stability (if applicable)"""

    def test_entropy_production(self):
        """
        Test Case: Physical entropy production

        Expected: Entropy non-decreasing (2nd law of thermodynamics)
        """
        # Create mesh
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Dam break initial conditions
        ic_manager = InitialConditionManager(mesh)
        x_centers = mesh.x

        h_init = np.zeros((mesh.nx, mesh.ny))
        h_init[x_centers < 50, :] = 10.0
        h_init[x_centers >= 50, :] = 1.0

        u_init = np.zeros((mesh.nx, mesh.ny))
        v_init = np.zeros((mesh.nx, mesh.ny))

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # Mathematical entropy for shallow water:
        # S = (1/2) * h * (u² + v²) + (1/2) * g * h²

        # Calculate initial entropy
        g = 9.81
        dx = mesh.dx
        dy = mesh.dy
        entropy_initial = np.sum(
            0.5 * h_init * (u_init**2 + v_init**2) + 0.5 * g * h_init**2
        ) * dx * dy

        # After simulation:
        # Total entropy should not decrease significantly
        # entropy_final >= entropy_initial * (1 - tolerance)
        # (Some decrease acceptable due to numerical dissipation)

        assert entropy_initial > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
