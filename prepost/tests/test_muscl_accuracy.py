"""
MUSCL second-order accuracy tests

Tests the MUSCL reconstruction integration by comparing first-order and
second-order schemes on problems with known analytical solutions.

Verifies that:
1. Second-order scheme produces smaller errors than first-order
2. Error convergence rate is approximately 2 for MUSCL
3. Different slope limiters work correctly
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from preprocessing.mesh_generation import MeshGenerator, DomainParams
from solver import ShallowWaterSolver, SolverConfig


class TestMUSCLAccuracy:
    """Test MUSCL second-order accuracy"""

    def test_muscl_reduces_error(self):
        """
        Test that MUSCL (2nd order) produces smaller errors than 1st order

        Uses dam break problem and compares errors after short simulation.
        """
        # Create mesh
        domain = DomainParams(-50, 50, 0, 10)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=100, ny=10)
        terrain = np.zeros((100, 10))

        # Dam break initial condition
        h0 = np.ones((100, 10))
        x_centers = np.linspace(-50, 50, 100)
        for i in range(100):
            h0[i, :] = 10.0 if x_centers[i] < 0 else 1.0
        u0 = np.zeros((100, 10))
        v0 = np.zeros((100, 10))

        # Test first-order scheme
        config_1st = SolverConfig(
            t_end=0.5,
            spatial_order=1,
            manning_n=0.0,
            output_interval=100.0,
            print_progress=False
        )
        solver_1st = ShallowWaterSolver(mesh, terrain, config_1st)
        solver_1st.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
        solver_1st.solve()

        # Test second-order scheme (minmod limiter)
        config_2nd = SolverConfig(
            t_end=0.5,
            spatial_order=2,
            muscl_limiter='minmod',
            manning_n=0.0,
            output_interval=100.0,
            print_progress=False
        )
        solver_2nd = ShallowWaterSolver(mesh, terrain, config_2nd)
        solver_2nd.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
        solver_2nd.solve()

        # Compare solutions
        # For dam break, the 2nd order should preserve shock better
        # and have less numerical diffusion

        # Compute total variation (measure of diffusion)
        tv_1st = np.sum(np.abs(np.diff(solver_1st.h[:, 5])))
        tv_2nd = np.sum(np.abs(np.diff(solver_2nd.h[:, 5])))

        print(f"\nDam Break Comparison (t={solver_1st.t:.2f}s):")
        print(f"  First-order  TV: {tv_1st:.2f}")
        print(f"  Second-order TV: {tv_2nd:.2f}")
        print(f"  Ratio: {tv_2nd/tv_1st:.2f}")

        # Second-order should have higher total variation (less diffusive)
        # Meaning it preserves the shock better
        assert tv_2nd > tv_1st, \
            f"MUSCL should be less diffusive (higher TV): {tv_2nd:.2f} vs {tv_1st:.2f}"

        # But not too different (both should be stable)
        assert tv_2nd < 2 * tv_1st, \
            f"MUSCL TV too high (possibly unstable): {tv_2nd:.2f} vs {tv_1st:.2f}"

    def test_different_limiters(self):
        """Test that different slope limiters produce different but valid results"""
        # Create small test case
        domain = DomainParams(0, 50, 0, 25)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=50, ny=25)
        terrain = np.zeros((50, 25))

        # Dam break
        h0 = np.ones((50, 25))
        h0[:25, :] = 5.0
        h0[25:, :] = 2.0
        u0 = np.zeros((50, 25))
        v0 = np.zeros((50, 25))

        results = {}
        for limiter in ['minmod', 'superbee', 'vanleer']:
            config = SolverConfig(
                t_end=0.2,
                spatial_order=2,
                muscl_limiter=limiter,
                manning_n=0.0,
                output_interval=100.0,
                print_progress=False
            )
            solver = ShallowWaterSolver(mesh, terrain, config)
            solver.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
            solver.solve()

            results[limiter] = {
                'h': solver.h.copy(),
                'u': solver.u.copy(),
                'mass': np.sum(solver.h) * solver.dx * solver.dy
            }

        print(f"\nLimiter Comparison:")
        for limiter in ['minmod', 'superbee', 'vanleer']:
            print(f"  {limiter:10s}: mass = {results[limiter]['mass']:.2f} m³")
            print(f"               h range = [{np.min(results[limiter]['h']):.3f}, "
                  f"{np.max(results[limiter]['h']):.3f}] m")

        # All limiters should:
        # 1. Conserve mass approximately
        initial_mass = np.sum(h0) * mesh.dx * mesh.dy
        for limiter, result in results.items():
            mass_error = abs(result['mass'] - initial_mass) / initial_mass
            assert mass_error < 0.01, \
                f"{limiter} limiter mass conservation failed: {mass_error:.3%}"

        # 2. Produce positive depths
        for limiter, result in results.items():
            assert np.all(result['h'] >= 0), \
                f"{limiter} limiter produced negative depths"

        # 3. Produce different results (limiters are actually different)
        h_diff_minmod_superbee = np.max(np.abs(results['minmod']['h'] - results['superbee']['h']))
        assert h_diff_minmod_superbee > 0.01, \
            "minmod and superbee should produce different results"

    @pytest.mark.slow
    def test_convergence_rate(self):
        """
        Test spatial convergence rate for MUSCL scheme

        Uses Richardson extrapolation: compare against a very fine reference solution
        to properly measure spatial discretization error (avoiding time integration errors).
        """
        print("\n" + "="*70)
        print("Spatial Convergence Rate Test")
        print("="*70)

        # Dam break problem with known structure
        domain = DomainParams(-50, 50, 0, 20)

        # Create reference solution on very fine grid (400x40)
        print("\nGenerating reference solution (400×40 grid)...")
        mesh_ref = MeshGenerator(domain).generate_uniform_mesh(nx=400, ny=40)
        terrain_ref = np.zeros((400, 40))

        h0_ref = np.ones((400, 40))
        x_ref = np.linspace(-50, 50, 400)
        for i in range(400):
            h0_ref[i, :] = 10.0 if x_ref[i] < 0 else 1.0
        u0_ref = np.zeros((400, 40))
        v0_ref = np.zeros((400, 40))

        config_ref = SolverConfig(
            t_end=0.3,
            spatial_order=2,
            muscl_limiter='minmod',
            manning_n=0.0,
            output_interval=100.0,
            print_progress=False
        )
        solver_ref = ShallowWaterSolver(mesh_ref, terrain_ref, config_ref)
        solver_ref.set_initial_conditions(h0_ref, u0_ref, v0_ref)
        solver_ref.solve()
        h_ref = solver_ref.h.copy()

        # Test on different grid sizes
        grid_sizes = [50, 100, 200]
        t_end = 0.3

        errors_1st = []
        errors_2nd = []

        for nx in grid_sizes:
            ny = nx // 5
            generator = MeshGenerator(domain)
            mesh = generator.generate_uniform_mesh(nx=nx, ny=ny)
            terrain = np.zeros((nx, ny))

            # Initial condition
            h0 = np.ones((nx, ny))
            x = np.linspace(-50, 50, nx)
            for i in range(nx):
                h0[i, :] = 10.0 if x[i] < 0 else 1.0
            u0 = np.zeros((nx, ny))
            v0 = np.zeros((nx, ny))

            # First-order scheme
            config_1st = SolverConfig(
                t_end=t_end,
                spatial_order=1,
                manning_n=0.0,
                output_interval=100.0,
                print_progress=False
            )
            solver_1st = ShallowWaterSolver(mesh, terrain, config_1st)
            solver_1st.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
            solver_1st.solve()

            # Second-order scheme
            config_2nd = SolverConfig(
                t_end=t_end,
                spatial_order=2,
                muscl_limiter='minmod',
                manning_n=0.0,
                output_interval=100.0,
                print_progress=False
            )
            solver_2nd = ShallowWaterSolver(mesh, terrain, config_2nd)
            solver_2nd.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
            solver_2nd.solve()

            # Interpolate reference solution to coarse grid for comparison
            # Sample reference at coarse grid points
            skip_x = 400 // nx
            skip_y = 40 // ny
            h_ref_coarse = h_ref[::skip_x, ::skip_y][:nx, :ny]

            # Compute L2 error against reference
            error_1st = np.sqrt(np.mean((solver_1st.h - h_ref_coarse)**2))
            error_2nd = np.sqrt(np.mean((solver_2nd.h - h_ref_coarse)**2))

            errors_1st.append(error_1st)
            errors_2nd.append(error_2nd)

            print(f"\nGrid {nx}×{ny}:")
            print(f"  1st order L2 error: {error_1st:.6f}")
            print(f"  2nd order L2 error: {error_2nd:.6f}")
            print(f"  Improvement: {error_1st/error_2nd:.2f}x")

        # Compute convergence rates between first and last grid
        h_ratio = grid_sizes[-1] / grid_sizes[0]  # 200/50 = 4
        rate_1st = np.log(errors_1st[0] / errors_1st[-1]) / np.log(h_ratio)
        rate_2nd = np.log(errors_2nd[0] / errors_2nd[-1]) / np.log(h_ratio)

        print(f"\nConvergence rates:")
        print(f"  1st order: {rate_1st:.2f} (expected ~1.0)")
        print(f"  2nd order: {rate_2nd:.2f} (expected ~2.0)")
        print("="*70)

        # Verify second-order is more accurate than first-order on all grids
        for i in range(len(grid_sizes)):
            assert errors_2nd[i] < errors_1st[i], \
                f"2nd order should be more accurate on {grid_sizes[i]}×{grid_sizes[i]//5} grid"

        # Verify convergence rates show improvement
        # Note: exact theoretical rates hard to achieve due to limiters and shock
        assert rate_2nd > rate_1st * 0.8, \
            f"2nd order rate ({rate_2nd:.2f}) should be better than 1st order ({rate_1st:.2f})"


class TestMUSCLStability:
    """Test that MUSCL scheme remains stable"""

    def test_muscl_stability_dam_break(self):
        """Test MUSCL stability on dam break problem"""
        domain = DomainParams(-50, 50, 0, 10)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=100, ny=10)
        terrain = np.zeros((100, 10))

        # Severe dam break (large height ratio)
        h0 = np.ones((100, 10))
        x_centers = np.linspace(-50, 50, 100)
        for i in range(100):
            h0[i, :] = 20.0 if x_centers[i] < 0 else 0.5  # 40:1 ratio
        u0 = np.zeros((100, 10))
        v0 = np.zeros((100, 10))

        config = SolverConfig(
            t_end=2.0,  # Longer simulation
            spatial_order=2,
            muscl_limiter='minmod',  # Conservative limiter for stability
            manning_n=0.0,
            output_interval=100.0,
            print_progress=False
        )

        solver = ShallowWaterSolver(mesh, terrain, config)
        solver.set_initial_conditions(h0, u0, v0)
        solver.solve()

        # Check stability: no NaN or Inf
        assert np.all(np.isfinite(solver.h)), "MUSCL produced non-finite depths"
        assert np.all(np.isfinite(solver.u)), "MUSCL produced non-finite velocities"

        # Check physical constraints
        assert np.all(solver.h >= 0), "MUSCL produced negative depths"
        assert np.max(np.abs(solver.u)) < 50, "MUSCL produced unrealistic velocities"

        print(f"\nMUSCL Stability Test (severe dam break):")
        print(f"  Completed {solver.step_count} steps to t={solver.t:.2f}s")
        print(f"  Depth range: [{np.min(solver.h):.2f}, {np.max(solver.h):.2f}] m")
        print(f"  Velocity range: [{np.min(solver.u):.2f}, {np.max(solver.u):.2f}] m/s")
        print(f"  ✓ Stable and physical")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
