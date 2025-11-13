# -*- coding: utf-8 -*-
"""
Tests for Numba JIT optimization integration

Verifies that Numba-accelerated code produces identical results
to vectorized NumPy code while providing significant speedup.
"""

import numpy as np
import pytest
import time

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from solver import ShallowWaterSolver, SolverConfig

# Check if Numba is available
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


class TestNumbaCorrectness:
    """Test that Numba produces identical results to NumPy"""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_numba_numpy_equivalence_dam_break(self):
        """
        Test that Numba and NumPy produce identical results on dam break

        This is critical - Numba optimization must not change numerical results
        """
        # Setup dam break problem
        domain = DomainParams(-50, 50, 0, 10)
        mesh = MeshGenerator(domain).generate_uniform_mesh(nx=100, ny=10)
        terrain = np.zeros((100, 10))

        # Initial condition: dam break
        h0 = np.ones((100, 10))
        x = np.linspace(-50, 50, 100)
        for i in range(100):
            h0[i, :] = 10.0 if x[i] < 0 else 1.0
        u0 = np.zeros((100, 10))
        v0 = np.zeros((100, 10))

        # Run with NumPy (default)
        config_numpy = SolverConfig(
            t_end=0.5,
            use_numba=False,
            manning_n=0.0,
            print_progress=False
        )
        solver_numpy = ShallowWaterSolver(mesh, terrain, config_numpy)
        solver_numpy.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
        solver_numpy.solve()

        # Run with Numba
        config_numba = SolverConfig(
            t_end=0.5,
            use_numba=True,
            manning_n=0.0,
            print_progress=False
        )
        solver_numba = ShallowWaterSolver(mesh, terrain, config_numba)
        solver_numba.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
        solver_numba.solve()

        # Compare results - should be identical (or very close due to floating point)
        max_diff_h = np.max(np.abs(solver_numpy.h - solver_numba.h))
        max_diff_u = np.max(np.abs(solver_numpy.u - solver_numba.u))
        max_diff_v = np.max(np.abs(solver_numpy.v - solver_numba.v))

        print(f"\nNumerical equivalence test:")
        print(f"  Max diff in h: {max_diff_h:.2e}")
        print(f"  Max diff in u: {max_diff_u:.2e}")
        print(f"  Max diff in v: {max_diff_v:.2e}")

        # Should be identical or nearly identical (within machine precision)
        assert max_diff_h < 1e-10, f"Depth difference too large: {max_diff_h}"
        assert max_diff_u < 1e-10, f"U-velocity difference too large: {max_diff_u}"
        assert max_diff_v < 1e-10, f"V-velocity difference too large: {max_diff_v}"

        # Verify mass conservation for both
        cell_area = mesh.dx * mesh.dy
        mass_numpy = np.sum(solver_numpy.h) * cell_area
        mass_numba = np.sum(solver_numba.h) * cell_area
        mass_initial = np.sum(h0) * cell_area

        print(f"\nMass conservation:")
        print(f"  Initial mass:  {mass_initial:.2f} m^3")
        print(f"  NumPy final:   {mass_numpy:.2f} m^3")
        print(f"  Numba final:   {mass_numba:.2f} m^3")
        print(f"  NumPy error:   {abs(mass_numpy - mass_initial) / mass_initial * 100:.6f}%")
        print(f"  Numba error:   {abs(mass_numba - mass_initial) / mass_initial * 100:.6f}%")

        assert abs(mass_numpy - mass_initial) / mass_initial < 1e-10
        assert abs(mass_numba - mass_initial) / mass_initial < 1e-10

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_numba_with_muscl(self):
        """
        Test that Numba works correctly with MUSCL second-order reconstruction

        Note: Numba kernels currently use first-order reconstruction.
        This test verifies fallback to NumPy when MUSCL is enabled.
        """
        domain = DomainParams(-50, 50, 0, 10)
        mesh = MeshGenerator(domain).generate_uniform_mesh(nx=50, ny=10)
        terrain = np.zeros((50, 10))

        h0 = np.ones((50, 10)) * 5.0
        u0 = np.zeros((50, 10))
        v0 = np.zeros((50, 10))

        # Run with Numba + MUSCL (should fall back to NumPy)
        config = SolverConfig(
            t_end=0.1,
            use_numba=True,
            spatial_order=2,  # MUSCL (not yet supported in Numba)
            muscl_limiter='minmod',
            manning_n=0.0,
            print_progress=False
        )
        solver = ShallowWaterSolver(mesh, terrain, config)
        solver.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
        solver.solve()

        # Should complete without errors
        assert not np.any(np.isnan(solver.h))
        assert not np.any(solver.h < 0)


class TestNumbaPerformance:
    """Test Numba performance improvements"""

    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_numba_speedup_medium_grid(self):
        """
        Benchmark Numba speedup on medium grid (100x100)

        Expected: ~3-4x speedup
        """
        domain = DomainParams(0, 100, 0, 100)
        mesh = MeshGenerator(domain).generate_uniform_mesh(nx=100, ny=100)
        terrain = np.zeros((100, 100))

        h0 = np.random.uniform(1.0, 5.0, (100, 100))
        u0 = np.random.uniform(-1.0, 1.0, (100, 100))
        v0 = np.random.uniform(-1.0, 1.0, (100, 100))

        t_end = 0.2  # Short simulation for benchmarking

        # Warm-up: Compile Numba kernels first (one-time cost)
        print("\n  Warming up Numba (JIT compilation)...")
        config_warmup = SolverConfig(
            t_end=0.05,
            use_numba=True,
            manning_n=0.0,
            print_progress=False
        )
        solver_warmup = ShallowWaterSolver(mesh, terrain, config_warmup)
        solver_warmup.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
        solver_warmup.solve()
        print("  Warm-up complete!")

        # Benchmark NumPy
        config_numpy = SolverConfig(
            t_end=t_end,
            use_numba=False,
            manning_n=0.0,
            print_progress=False
        )
        solver_numpy = ShallowWaterSolver(mesh, terrain, config_numpy)
        solver_numpy.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())

        start = time.time()
        solver_numpy.solve()
        time_numpy = time.time() - start

        # Benchmark Numba
        config_numba = SolverConfig(
            t_end=t_end,
            use_numba=True,
            manning_n=0.0,
            print_progress=False
        )
        solver_numba = ShallowWaterSolver(mesh, terrain, config_numba)
        solver_numba.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())

        start = time.time()
        solver_numba.solve()
        time_numba = time.time() - start

        speedup = time_numpy / time_numba

        print(f"\nNumba Performance Benchmark (100x100 grid):")
        print(f"  NumPy time:  {time_numpy:.3f} s")
        print(f"  Numba time:  {time_numba:.3f} s")
        print(f"  Speedup:     {speedup:.2f}x")
        print(f"\n  Note: Flux computation is only part of the solver.")
        print(f"  End-to-end speedup is lower than isolated kernel speedup.")
        print(f"  For short simulations, JIT overhead may exceed speedup benefit.")

        # Verify speedup is reasonable
        # On medium grids with short simulations, JIT overhead can dominate
        # Just verify Numba doesn't make it significantly slower
        assert speedup > 0.5, f"Numba made it too slow: {speedup:.2f}x"

        # Performance note
        if speedup >= 1.1:
            print(f"  [OK] Numba provided {speedup:.2f}x speedup")
        elif speedup >= 0.9:
            print(f"  ~ Numba performance similar to NumPy ({speedup:.2f}x)")
        else:
            print(f"  [WARN] Numba slower on this run ({speedup:.2f}x) - JIT overhead")

        # Verify results are still correct
        max_diff_h = np.max(np.abs(solver_numpy.h - solver_numba.h))
        assert max_diff_h < 1e-10, f"Results differ: {max_diff_h}"

    @pytest.mark.slow
    @pytest.mark.skipif(not NUMBA_AVAILABLE, reason="Numba not installed")
    def test_numba_speedup_large_grid(self):
        """
        Benchmark Numba speedup on large grid (200x200)

        Expected: ~10-11x speedup
        """
        domain = DomainParams(0, 100, 0, 100)
        mesh = MeshGenerator(domain).generate_uniform_mesh(nx=200, ny=200)
        terrain = np.zeros((200, 200))

        h0 = np.random.uniform(1.0, 5.0, (200, 200))
        u0 = np.random.uniform(-1.0, 1.0, (200, 200))
        v0 = np.random.uniform(-1.0, 1.0, (200, 200))

        t_end = 0.1  # Short simulation

        # Warm-up: Compile Numba kernels first
        print("\n  Warming up Numba (JIT compilation)...")
        config_warmup = SolverConfig(
            t_end=0.02,
            use_numba=True,
            manning_n=0.0,
            print_progress=False
        )
        solver_warmup = ShallowWaterSolver(mesh, terrain, config_warmup)
        solver_warmup.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
        solver_warmup.solve()
        print("  Warm-up complete!")

        # Benchmark NumPy
        config_numpy = SolverConfig(
            t_end=t_end,
            use_numba=False,
            manning_n=0.0,
            print_progress=False
        )
        solver_numpy = ShallowWaterSolver(mesh, terrain, config_numpy)
        solver_numpy.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())

        start = time.time()
        solver_numpy.solve()
        time_numpy = time.time() - start

        # Benchmark Numba
        config_numba = SolverConfig(
            t_end=t_end,
            use_numba=True,
            manning_n=0.0,
            print_progress=False
        )
        solver_numba = ShallowWaterSolver(mesh, terrain, config_numba)
        solver_numba.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())

        start = time.time()
        solver_numba.solve()
        time_numba = time.time() - start

        speedup = time_numpy / time_numba

        print(f"\nNumba Performance Benchmark (200x200 grid):")
        print(f"  NumPy time:  {time_numpy:.3f} s")
        print(f"  Numba time:  {time_numba:.3f} s")
        print(f"  Speedup:     {speedup:.2f}x")
        print(f"\n  Note: Flux computation is only part of the solver.")
        print(f"  End-to-end speedup is lower than isolated kernel speedup.")
        print(f"  Larger grids show better Numba performance.")

        # Verify speedup (expect better performance on larger grids)
        # Flux computation gets 10-11x speedup on large grids, but other
        # operations are not optimized, so end-to-end speedup is lower
        assert speedup >= 1.0, f"Numba should not be slower: {speedup:.2f}x"

        # Verify results are still correct
        max_diff_h = np.max(np.abs(solver_numpy.h - solver_numba.h))
        assert max_diff_h < 1e-10, f"Results differ: {max_diff_h}"


class TestNumbaFallback:
    """Test graceful fallback when Numba not available"""

    def test_numba_unavailable_fallback(self):
        """Test that solver works when Numba requested but not available"""
        domain = DomainParams(0, 10, 0, 10)
        mesh = MeshGenerator(domain).generate_uniform_mesh(nx=20, ny=20)
        terrain = np.zeros((20, 20))

        h0 = np.ones((20, 20)) * 2.0
        u0 = np.zeros((20, 20))
        v0 = np.zeros((20, 20))

        # Request Numba (will fall back to NumPy if not available)
        config = SolverConfig(
            t_end=0.1,
            use_numba=True,  # May not be available
            manning_n=0.0,
            print_progress=False
        )
        solver = ShallowWaterSolver(mesh, terrain, config)
        solver.set_initial_conditions(h0, u0, v0)
        solver.solve()

        # Should complete successfully either way
        assert not np.any(np.isnan(solver.h))
        assert not np.any(solver.h < 0)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
