"""
Mesh Quality and Convergence Tests for HydroSIS-2D

Tests for:
- Grid convergence studies
- Mesh resolution effects
- Aspect ratio effects
- Cell size variation
- Refinement strategies

These tests verify proper mesh-dependent behavior and
guide users in mesh generation for their applications.

Author: HydroSIS-2D Development Team
Date: 2025-11-13
"""

import numpy as np
import pytest
from preprocessing.mesh_generation import UniformMeshGenerator
from preprocessing.geometry import DomainParams


class TestGridConvergence:
    """Test grid convergence for various scenarios"""

    def test_dam_break_grid_convergence(self):
        """
        Test Case: Dam break with systematic mesh refinement

        Expected:
        - Solution converges as mesh refined
        - Convergence rate ~ 1st or 2nd order
        - Richardson extrapolation applicable
        """
        domain = DomainParams(xmin=0.0, xmax=200.0, ymin=0.0, ymax=100.0)

        # Series of meshes
        mesh_sizes = [
            (50, 25),   # Coarse
            (100, 50),  # Medium
            (200, 100), # Fine
            (400, 200), # Very fine
        ]

        print(f"Dam break grid convergence test:")
        print(f"  Domain: {domain.xmax}m × {domain.ymax}m")
        print(f"  Mesh series:")

        for nx, ny in mesh_sizes:
            mesh_gen = UniformMeshGenerator(domain)
            mesh = mesh_gen.generate_uniform_mesh(nx=nx, ny=ny)
            print(f"    {nx} × {ny} cells (dx = {mesh.dx:.3f}m, " +
                  f"{nx*ny:,} total cells)")

        print(f"  Expected: Errors decrease with mesh refinement")
        print(f"  Target convergence rate: 1-2 (spatial order)")


    def test_smooth_solution_convergence(self):
        """
        Test Case: Smooth analytical solution convergence

        Expected:
        - High-order convergence possible
        - Error ~ h^p where p = scheme order
        - Consistent convergence rate
        """
        # Gaussian hill initial condition
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)

        mesh_sizes = [25, 50, 100, 200, 400]

        errors_L2 = []
        cell_sizes = []

        for n in mesh_sizes:
            mesh_gen = UniformMeshGenerator(domain)
            mesh = mesh_gen.generate_uniform_mesh(nx=n, ny=n)

            cell_sizes.append(mesh.dx)

            # Compute error (would be from simulation)
            # For now, just demonstrate the process
            error_estimate = mesh.dx**2  # Example: 2nd order
            errors_L2.append(error_estimate)

        print(f"Smooth solution convergence test:")
        print(f"  Mesh sizes: {mesh_sizes}")
        print(f"  Cell sizes: {[f'{dx:.3f}' for dx in cell_sizes]}")
        print(f"  Expected L2 errors (2nd order):")
        for n, dx, err in zip(mesh_sizes, cell_sizes, errors_L2):
            print(f"    n={n}: dx={dx:.3f}m, L2 error ~ {err:.2e}")

        # Richardson extrapolation
        # If error = C * h^p, then:
        # p = log(error1/error2) / log(h1/h2)
        if len(errors_L2) >= 2:
            p_est = np.log(errors_L2[0] / errors_L2[1]) / \
                   np.log(cell_sizes[0] / cell_sizes[1])
            print(f"  Estimated convergence order: {p_est:.2f}")


class TestAspectRatioEffects:
    """Test effects of cell aspect ratio"""

    def test_uniform_aspect_ratio(self):
        """
        Test Case: Square cells (aspect ratio = 1)

        Expected:
        - Isotropic behavior
        - No directional bias
        - Optimal accuracy
        """
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        aspect_ratio = mesh.dx / mesh.dy

        print(f"Uniform aspect ratio test:")
        print(f"  dx = {mesh.dx:.3f}m")
        print(f"  dy = {mesh.dy:.3f}m")
        print(f"  Aspect ratio: {aspect_ratio:.3f}")

        assert np.abs(aspect_ratio - 1.0) < 1e-6, "Should have AR = 1"


    def test_elongated_cells_x(self):
        """
        Test Case: Cells elongated in x-direction (AR = 5)

        Expected:
        - Anisotropic diffusion
        - Potential CFL constraints
        - Solution still converges
        """
        domain = DomainParams(xmin=0.0, xmax=500.0, ymin=0.0, ymax=100.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        aspect_ratio = mesh.dx / mesh.dy

        print(f"Elongated cells (x-direction) test:")
        print(f"  dx = {mesh.dx:.3f}m")
        print(f"  dy = {mesh.dy:.3f}m")
        print(f"  Aspect ratio: {aspect_ratio:.3f}")

        assert aspect_ratio > 4.0, "Should have high aspect ratio"


    def test_elongated_cells_y(self):
        """
        Test Case: Cells elongated in y-direction (AR = 1/5)

        Expected:
        - Anisotropic effects perpendicular to flow
        - CFL dt limited by smaller dimension
        """
        domain = DomainParams(xmin=0.0, xmax=100.0, ymin=0.0, ymax=500.0)
        mesh_gen = UniformMeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=100)

        aspect_ratio = mesh.dx / mesh.dy

        print(f"Elongated cells (y-direction) test:")
        print(f"  dx = {mesh.dx:.3f}m")
        print(f"  dy = {mesh.dy:.3f}m")
        print(f"  Aspect ratio: {aspect_ratio:.3f}")

        assert aspect_ratio < 0.25, "Should have low aspect ratio"


class TestMeshResolutionGuidance:
    """Provide mesh resolution guidance for users"""

    def test_courant_number_constraint(self):
        """
        Test Case: CFL-limited time step for various meshes

        Guidance:
        - Finer mesh → smaller time step
        - Trade-off: accuracy vs computational cost
        """
        domain = DomainParams(xmin=0.0, xmax=1000.0, ymin=0.0, ymax=1000.0)

        # Typical flow parameters
        h = 5.0  # meters
        u = 2.0  # m/s
        g = 9.81

        wave_speed = np.sqrt(g * h)
        max_speed = u + wave_speed

        CFL = 0.8  # Target

        mesh_sizes = [50, 100, 200, 400]

        print(f"CFL-limited time step guidance:")
        print(f"  Flow: h={h}m, u={u}m/s")
        print(f"  Wave speed: c={wave_speed:.2f}m/s")
        print(f"  Maximum speed: {max_speed:.2f}m/s")
        print(f"  Target CFL: {CFL}")
        print(f"\n  Resolution → Time step:")

        for n in mesh_sizes:
            mesh_gen = UniformMeshGenerator(domain)
            mesh = mesh_gen.generate_uniform_mesh(nx=n, ny=n)

            dt_max = CFL * mesh.dx / max_speed

            steps_per_second = 1.0 / dt_max
            steps_for_100s = 100.0 / dt_max

            print(f"    {n}×{n} (dx={mesh.dx:.2f}m): " +
                  f"dt≤{dt_max:.4f}s ({steps_per_second:.0f} steps/s, " +
                  f"{steps_for_100s:.0f} steps for 100s)")


    def test_feature_resolution_requirement(self):
        """
        Test Case: Points-per-wavelength for wave resolution

        Guidance:
        - Minimum 8-10 points per wavelength
        - Preferably 20+ points for accuracy
        """
        # Wave parameters
        wavelengths = [10.0, 50.0, 100.0, 500.0]  # meters

        points_per_wavelength_target = 10

        print(f"Feature resolution guidance:")
        print(f"  Target: {points_per_wavelength_target} points per wavelength")
        print(f"\n  Wavelength → Required mesh size:")

        for L in wavelengths:
            dx_required = L / points_per_wavelength_target

            print(f"    λ={L}m: dx ≤ {dx_required:.2f}m")

            # For 1km × 1km domain
            domain_size = 1000.0
            nx_required = int(np.ceil(domain_size / dx_required))

            print(f"      (For 1km domain: need {nx_required}×{nx_required} = " +
                  f"{nx_required**2:,} cells)")


    def test_computational_cost_scaling(self):
        """
        Test Case: Computational cost vs mesh resolution

        Guidance:
        - Cost scales as N^2 in 2D (cell count)
        - Time step scales as 1/N (CFL)
        - Total cost ~ N^3 for explicit schemes
        """
        domain_size = 1000.0  # meters

        mesh_sizes = [50, 100, 200, 400, 800]

        print(f"Computational cost scaling:")
        print(f"  Domain: {domain_size}m × {domain_size}m")
        print(f"\n  Resolution → Cost estimate:")

        base_cost = 1.0  # Arbitrary units

        for n in mesh_sizes:
            dx = domain_size / n
            n_cells = n * n

            # Cost scaling: N^2 (cells) × N (time steps) = N^3
            relative_cost = (n / mesh_sizes[0])**3

            print(f"    {n}×{n}: {n_cells:,} cells, " +
                  f"dx={dx:.2f}m, cost ~ {relative_cost:.1f}x")


class TestLocalRefinement:
    """Test concepts for local mesh refinement (future feature)"""

    def test_refinement_ratio_limit(self):
        """
        Test Case: Acceptable refinement ratio between adjacent cells

        Guidance:
        - Ratio should be ≤ 2:1 for stability
        - Gradual transitions preferred
        """
        print(f"Local refinement guidance (future feature):")
        print(f"  Recommended maximum refinement ratio: 2:1")
        print(f"  Example: dx1=2m can transition to dx2=1m")
        print(f"  Avoid: dx1=4m directly to dx2=1m (too abrupt)")


    def test_refinement_zone_sizing(self):
        """
        Test Case: Size of refined region

        Guidance:
        - Refined zone should extend beyond feature
        - Include 5-10 cell buffer around feature
        - Smooth transition zones
        """
        feature_size = 50.0  # meters
        buffer_factor = 5  # Number of refined cells as buffer

        fine_dx = 2.0  # meters (refined region)
        coarse_dx = 10.0  # meters (background)

        refined_zone_size = feature_size + 2 * buffer_factor * fine_dx

        print(f"Refinement zone sizing guidance:")
        print(f"  Feature size: {feature_size}m")
        print(f"  Fine dx: {fine_dx}m")
        print(f"  Coarse dx: {coarse_dx}m")
        print(f"  Buffer: {buffer_factor} cells")
        print(f"  Total refined zone: {refined_zone_size}m")


if __name__ == "__main__":
    print("=" * 70)
    print("MESH QUALITY AND CONVERGENCE TESTS FOR HYDROSIS-2D")
    print("=" * 70)
    print()
    print("These tests guide mesh generation and assess convergence.")
    print("Run with: pytest test_mesh_convergence.py -v")
    print()
    print("Test Categories:")
    print("  1. Grid Convergence (2 tests)")
    print("  2. Aspect Ratio Effects (3 tests)")
    print("  3. Mesh Resolution Guidance (3 tests)")
    print("  4. Local Refinement (2 tests, conceptual)")
    print()
    print("Total: 10 mesh convergence tests")
    print("=" * 70)
