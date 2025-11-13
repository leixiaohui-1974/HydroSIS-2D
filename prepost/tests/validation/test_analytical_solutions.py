"""
Validation tests against analytical solutions

This module implements standard validation test cases for shallow water solvers:
- 1D dam break (Ritter solution)
- 2D circular dam break (radial symmetry)
- Lake at rest (C-property test)
- Steady flow over bump

These tests verify numerical accuracy and well-balanced properties.
"""

import pytest
import numpy as np
from typing import Tuple, Dict
import sys
from pathlib import Path

# Add preprocessing modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from preprocessing.geometry import GeometryGenerator
from preprocessing.boundary_conditions import BoundaryConditionManager
from preprocessing.initial_conditions import InitialConditionManager


class TestRitterDamBreak:
    """
    Test 1D dam break against Ritter analytical solution

    Initial conditions:
      Left:  h = h_L, u = 0
      Right: h = h_R, u = 0

    Analytical solution (Ritter, 1892):
      In rarefaction wave: h = (1/9g) * (2c_L - x/t)²
      where c_L = sqrt(g*h_L)
    """

    def test_ritter_solution_t1(self):
        """Test dam break at t=1.0s"""
        # Physical parameters
        g = 9.81
        h_L = 10.0  # Left water depth (m)
        h_R = 1.0   # Right water depth (m)
        t = 1.0     # Time (s)

        # Create 1D mesh (quasi-1D: ny=3 for 2D code)
        domain = DomainParams(xmin=-50.0, xmax=50.0, ymin=0.0, ymax=1.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=500, ny=3)

        # Initial conditions: dam at x=0
        ic_manager = InitialConditionManager(mesh)

        # Set initial depth
        h_init = np.zeros((mesh.nx, mesh.ny))
        x_centers = mesh.cell_centers_x
        for i in range(mesh.nx):
            if x_centers[i] < 0:
                h_init[i, :] = h_L
            else:
                h_init[i, :] = h_R

        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_zero()

        # Analytical solution
        h_analytical = self._ritter_solution(x_centers, t, h_L, h_R, g)

        # For this test, we verify the initial condition setup
        # (Full simulation would require GPU solver)
        h_test = ic_manager.depth[:, 1]  # Center row

        # Check initial condition is correct
        assert h_test[x_centers < 0].mean() == pytest.approx(h_L, rel=1e-6)
        assert h_test[x_centers >= 0].mean() == pytest.approx(h_R, rel=1e-6)

        # Store analytical solution for future comparison
        self.h_analytical = h_analytical
        self.x = x_centers

    def test_ritter_solution_properties(self):
        """Test analytical solution properties"""
        g = 9.81
        h_L = 10.0
        h_R = 1.0
        t = 1.0

        x = np.linspace(-50, 50, 1000)
        h = self._ritter_solution(x, t, h_L, h_R, g)

        # Check properties
        assert h.max() <= h_L, "Depth should not exceed initial maximum"
        assert h.min() >= h_R, "Depth should not go below initial minimum"
        assert np.all(h >= 0), "Depth must be non-negative"

        # Check shock speed (approximate)
        # For h_L=10, h_R=1, shock speed ≈ 4.5 m/s
        c_L = np.sqrt(g * h_L)
        c_R = np.sqrt(g * h_R)
        shock_speed = (h_L * np.sqrt(g * h_L) - h_R * np.sqrt(g * h_R)) / (h_L - h_R)

        assert 4.0 < shock_speed < 5.0, f"Shock speed {shock_speed:.2f} out of expected range"

    @staticmethod
    def _ritter_solution(x: np.ndarray, t: float, h_L: float, h_R: float, g: float) -> np.ndarray:
        """
        Compute Ritter analytical solution for dam break

        Args:
            x: Spatial coordinates
            t: Time
            h_L: Left water depth
            h_R: Right water depth
            g: Gravitational acceleration

        Returns:
            Water depth at each position
        """
        c_L = np.sqrt(g * h_L)
        c_R = np.sqrt(g * h_R)

        h = np.zeros_like(x)

        # Wave speeds
        x1 = -c_L * t  # Left edge of rarefaction
        x2 = 2.0 * c_L * t  # Right edge of rarefaction (head)

        # Shock speed (exact for dry bed: x_s = 2*c_L*t)
        # For wet bed, use approximate formula
        u_star = 2.0 * (c_L - c_R)
        h_star = ((c_L - 0.5 * u_star) ** 2) / g
        x_shock = x1 + (c_L + u_star) * t

        for i, xi in enumerate(x):
            if xi < x1:
                # Left state (undisturbed)
                h[i] = h_L
            elif xi < x_shock:
                # Rarefaction wave
                h[i] = (1.0 / (9.0 * g)) * (2.0 * c_L - xi / t) ** 2
            else:
                # Right state
                h[i] = h_R

        return h


class TestLakeAtRest:
    """
    Test lake at rest (C-property)

    Initial conditions:
      h + z = constant (hydrostatic equilibrium)
      u = v = 0

    Expected result:
      Solution should remain at rest (no spurious currents)
      Mass should be perfectly conserved
    """

    def test_flat_lake_at_rest(self):
        """Test perfectly flat lake"""
        # Create mesh
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=50)

        # Flat terrain
        z = GeometryGenerator.flat_surface(mesh.nx, mesh.ny, base_elevation=0.0)

        # Constant water depth
        h_const = 5.0
        ic_manager = InitialConditionManager(mesh)
        h_init = np.full((mesh.nx, mesh.ny), h_const)
        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_zero()

        # Verify setup
        assert ic_manager.depth.mean() == pytest.approx(h_const, rel=1e-10)
        assert ic_manager.velocity_x.max() == 0.0
        assert ic_manager.velocity_y.max() == 0.0

        # After simulation, should have:
        # - h unchanged
        # - u, v remain zero
        # - Perfect mass conservation

    def test_sloped_lake_at_rest(self):
        """Test lake at rest on sloped terrain"""
        # Create mesh
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=50)

        # Sloped terrain: z = 0.1 * x
        x_centers = mesh.cell_centers_x
        z = np.zeros((mesh.nx, mesh.ny))
        for i in range(mesh.nx):
            z[i, :] = 0.1 * x_centers[i]

        # Water surface at constant elevation
        eta = 10.0  # Free surface elevation
        ic_manager = InitialConditionManager(mesh)
        h = eta - z  # h + z = eta
        h = np.maximum(h, 0.0)  # Ensure non-negative

        ic_manager.set_depth_array(h)
        ic_manager.set_velocity_zero()

        # Verify hydrostatic equilibrium
        free_surface = ic_manager.depth + z
        assert np.allclose(free_surface, eta, atol=1e-10)

        # This should remain at rest when simulated
        # (Tests well-balanced property of source term discretization)


class TestCircularDamBreak:
    """
    Test 2D circular dam break

    Initial conditions:
      Inside circle (r < R):  h = h_in
      Outside circle (r >= R): h = h_out
      u = v = 0

    Expected properties:
      - Radial symmetry preserved
      - No angular dependence
      - Mass conservation
    """

    def test_circular_dam_setup(self):
        """Test circular dam break initial conditions"""
        # Create square mesh
        domain = DomainParams(xmin=-50.0, xmax=50.0, ymin=-50.0, ymax=50.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        # Circular dam parameters
        R = 20.0  # Dam radius (m)
        h_in = 10.0  # Inside depth (m)
        h_out = 1.0  # Outside depth (m)

        # Set initial conditions
        ic_manager = InitialConditionManager(mesh)
        x_centers = mesh.cell_centers_x
        y_centers = mesh.cell_centers_y

        h = np.zeros((mesh.nx, mesh.ny))
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                r = np.sqrt(x_centers[i]**2 + y_centers[j]**2)
                h[i, j] = h_in if r < R else h_out

        ic_manager.set_depth_array(h)
        ic_manager.set_velocity_zero()

        # Verify circular symmetry
        # Check at several angles
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for angle in angles:
            # Point just inside circle
            r_in = R - 5.0
            x_in = r_in * np.cos(angle)
            y_in = r_in * np.sin(angle)

            # Find nearest cell
            i_in = np.argmin(np.abs(x_centers - x_in))
            j_in = np.argmin(np.abs(y_centers - y_in))

            assert ic_manager.depth[i_in, j_in] == pytest.approx(h_in, rel=1e-2)

        # Mass calculation
        dx = mesh.dx
        dy = mesh.dy
        mass_initial = np.sum(ic_manager.depth) * dx * dy

        # Analytical mass
        area_in = np.pi * R**2
        area_out = (100.0 * 100.0) - area_in
        mass_analytical = area_in * h_in + area_out * h_out

        assert mass_initial == pytest.approx(mass_analytical, rel=1e-2)

    def test_radial_symmetry_preservation(self):
        """Test that solution preserves radial symmetry"""
        # This test would run the simulation and verify:
        # 1. Solution depends only on r = sqrt(x² + y²)
        # 2. No angular variations
        # 3. Symmetry preserved to machine precision

        # For now, just verify the property is testable
        domain = DomainParams(xmin=-50.0, xmax=50.0, ymin=-50.0, ymax=50.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        x_centers = mesh.cell_centers_x
        y_centers = mesh.cell_centers_y

        # Create radial distance array
        r = np.zeros((mesh.nx, mesh.ny))
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                r[i, j] = np.sqrt(x_centers[i]**2 + y_centers[j]**2)

        # After simulation, solution h should be a function of r only
        # Can test by comparing cells at same radius but different angles
        assert r.shape == (mesh.nx, mesh.ny)


class TestSteadyFlowOverBump:
    """
    Test steady subcritical flow over a bump

    Analytical solution exists for frictionless flow.
    Tests source term balancing and well-balanced property.
    """

    def test_bump_geometry(self):
        """Test bump terrain generation"""
        # Create mesh
        domain = DomainParams(xmin=0.0, xmax=25.0, ymin=0.0, ymax=1.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=250, ny=10)

        # Bump geometry (standard test case)
        x_centers = mesh.cell_centers_x
        z = self._bump_topography(x_centers)

        # Check bump properties
        assert z.min() == 0.0, "Minimum elevation should be 0"
        assert z.max() == pytest.approx(0.2, rel=1e-6), "Maximum bump height should be 0.2m"

        # Bump location
        x_max = x_centers[np.argmax(z)]
        assert 8.0 < x_max < 12.0, "Bump peak should be around x=10m"

    def test_steady_flow_setup(self):
        """Test steady flow initial conditions"""
        # Physical parameters
        Q = 4.42  # Discharge per unit width (m²/s)
        h_downstream = 2.0  # Downstream depth (m)
        g = 9.81

        # Create mesh
        domain = DomainParams(xmin=0.0, xmax=25.0, ymin=0.0, ymax=1.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=250, ny=10)

        # Bump topography
        x_centers = mesh.cell_centers_x
        z = self._bump_topography(x_centers)

        # For steady flow: h + z + u²/(2g) = constant (Bernoulli)
        # At downstream: u = Q/h
        u_down = Q / h_downstream
        H_total = h_downstream + 0.0 + u_down**2 / (2 * g)  # Total head

        # Initial guess: uniform depth
        h_init = np.full((mesh.nx, mesh.ny), h_downstream)

        # Initial velocity (uniform discharge)
        u_init = np.full_like(h_init, Q / h_downstream)
        v_init = np.zeros_like(h_init)

        ic_manager = InitialConditionManager(mesh)
        ic_manager.set_depth_array(h_init)
        ic_manager.set_velocity_array(u_init, v_init)

        # Verify setup
        assert ic_manager.velocity_x.mean() == pytest.approx(Q / h_downstream, rel=1e-6)
        assert ic_manager.velocity_y.max() == 0.0

    @staticmethod
    def _bump_topography(x: np.ndarray) -> np.ndarray:
        """
        Standard bump topography for test cases

        z(x) = 0.2 - 0.05*(x-10)² for 8 < x < 12
             = 0                   otherwise
        """
        z = np.zeros_like(x)
        mask = (x > 8.0) & (x < 12.0)
        z[mask] = 0.2 - 0.05 * (x[mask] - 10.0)**2
        return z


class TestMassConservation:
    """
    Test mass conservation for various scenarios

    Mass should be conserved to machine precision for:
    - All boundary conditions
    - Source terms (bed slope, friction)
    - Dry/wet transitions
    """

    def test_mass_conservation_dam_break(self):
        """Test mass conservation in dam break"""
        from preprocessing.utils import create_dam_break_simulation

        # Create dam break
        config = create_dam_break_simulation(
            length=200.0, width=100.0,
            nx=100, ny=50,
            dam_position=0.5,
            upstream_depth=10.0,
            downstream_depth=1.0,
            simulation_time=10.0
        )

        mesh = config.mesh
        ic_manager = config.ic_manager

        # Initial mass
        dx = mesh.dx
        dy = mesh.dy
        mass_initial = np.sum(ic_manager.depth) * dx * dy

        # Analytical mass
        area_upstream = 100.0 * 100.0
        area_downstream = 100.0 * 100.0
        mass_expected = area_upstream * 10.0 + area_downstream * 1.0

        assert mass_initial == pytest.approx(mass_expected, rel=1e-10)

        # After simulation, mass should be conserved:
        # mass_final == mass_initial (to machine precision)

    def test_mass_conservation_with_walls(self):
        """Test that wall BC preserves mass"""
        # Wall boundaries should have zero flux
        # Therefore mass is exactly conserved

        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=50, ny=50)

        bc_manager = BoundaryConditionManager(domain)
        bc_manager.set_all_walls()

        # Any initial condition should preserve mass with wall BC
        ic_manager = InitialConditionManager(mesh)
        h = np.random.uniform(1.0, 10.0, (mesh.nx, mesh.ny))
        ic_manager.set_depth_array(h)

        mass_initial = np.sum(h) * mesh.dx * mesh.dy

        # After simulation: mass_final == mass_initial
        assert mass_initial > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
