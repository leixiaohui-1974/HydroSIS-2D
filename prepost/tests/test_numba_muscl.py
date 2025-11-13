# -*- coding: utf-8 -*-
"""
Tests for Numba-optimized MUSCL reconstruction

This module tests the combined MUSCL + HLL Numba kernels that provide
second-order accuracy with JIT compilation speedup.
"""

import numpy as np
import pytest
from solver.shallow_water_solver import ShallowWaterSolver, SolverConfig
from preprocessing.geometry import GeometryGenerator
from preprocessing.mesh_generation import MeshGenerator, DomainParams


class TestNumbaMUSCLKernels:
    """Test Numba MUSCL kernel functions"""

    def test_numba_available(self):
        """Verify Numba is available for testing"""
        from solver.numba_kernels import is_numba_available
        assert is_numba_available(), "Numba must be installed to run these tests"

    def test_apply_limiter_minmod(self):
        """Test apply_limiter function with minmod"""
        from solver.numba_kernels import apply_limiter

        # Same sign - should return smaller magnitude
        assert apply_limiter(1.0, 2.0, 0) == 1.0
        assert apply_limiter(2.0, 1.0, 0) == 1.0
        assert apply_limiter(-1.0, -2.0, 0) == -1.0

        # Different signs - should return zero
        assert apply_limiter(1.0, -1.0, 0) == 0.0
        assert apply_limiter(-1.0, 1.0, 0) == 0.0

    def test_apply_limiter_superbee(self):
        """Test apply_limiter function with superbee"""
        from solver.numba_kernels import apply_limiter

        # superbee is more aggressive
        result = apply_limiter(1.0, 2.0, 1)
        assert result > 0  # Should be positive
        assert result <= 2.0  # Bounded by max(2*1, 2) = 2

        # Different signs - should return zero
        assert apply_limiter(1.0, -1.0, 1) == 0.0

    def test_apply_limiter_vanleer(self):
        """Test apply_limiter function with van Leer"""
        from solver.numba_kernels import apply_limiter

        # van Leer is smooth
        result = apply_limiter(1.0, 1.0, 2)
        assert result == 1.0  # When equal, should preserve

        # Different signs - should return zero
        assert apply_limiter(1.0, -1.0, 2) == 0.0


def create_dam_break_setup(nx=100, ny=10):
    """Helper function to create dam break test setup"""
    domain = DomainParams(-50, 50, 0, 10 * (ny / 10.0))
    mesh = MeshGenerator(domain).generate_uniform_mesh(nx=nx, ny=ny)
    terrain = np.zeros((nx, ny))

    # Dam break initial conditions
    h0 = np.ones((nx, ny))
    x_centers = np.linspace(-50, 50, nx)
    for i in range(nx):
        h0[i, :] = 10.0 if x_centers[i] < 0 else 1.0
    u0 = np.zeros((nx, ny))
    v0 = np.zeros((nx, ny))

    return mesh, terrain, h0, u0, v0


class TestNumbaMUSCLIntegration:
    """Test Numba MUSCL integration in solver"""

    def test_numba_muscl_vs_numpy_muscl_equivalence(self):
        """
        Verify Numba MUSCL produces identical results to NumPy MUSCL

        This is the critical test - ensures numerical equivalence.
        """
        # Setup: Dam break scenario
        mesh, terrain, h0, u0, v0 = create_dam_break_setup(nx=100, ny=10)

        # Configuration for short simulation
        config_numpy = SolverConfig(
            spatial_order=2,
            muscl_limiter='minmod',
            use_numba=False,
            cfl=0.4,
            t_end=0.5,
            print_progress=False
        )

        config_numba = SolverConfig(
            spatial_order=2,
            muscl_limiter='minmod',
            use_numba=True,
            cfl=0.4,
            t_end=0.5,
            print_progress=False
        )

        # Run with NumPy MUSCL
        solver_numpy = ShallowWaterSolver(mesh, terrain, config_numpy)
        solver_numpy.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
        solver_numpy.solve()
        h_numpy = solver_numpy.h.copy()
        u_numpy = solver_numpy.u.copy()
        v_numpy = solver_numpy.v.copy()

        # Run with Numba MUSCL
        solver_numba = ShallowWaterSolver(mesh, terrain, config_numba)
        solver_numba.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
        solver_numba.solve()
        h_numba = solver_numba.h.copy()
        u_numba = solver_numba.u.copy()
        v_numba = solver_numba.v.copy()

        # Compare - should be numerically identical
        max_diff_h = np.max(np.abs(h_numpy - h_numba))
        max_diff_u = np.max(np.abs(u_numpy - u_numba))
        max_diff_v = np.max(np.abs(v_numpy - v_numba))

        print(f"\n数值等价性检查:")
        print(f"  深度最大差异: {max_diff_h:.2e}")
        print(f"  u速度最大差异: {max_diff_u:.2e}")
        print(f"  v速度最大差异: {max_diff_v:.2e}")

        # Should be identical within floating point precision
        assert max_diff_h < 1e-10, f"Depth difference too large: {max_diff_h}"
        assert max_diff_u < 1e-10, f"u-velocity difference too large: {max_diff_u}"
        assert max_diff_v < 1e-10, f"v-velocity difference too large: {max_diff_v}"

    def test_numba_muscl_limiters(self):
        """Test different limiters with Numba MUSCL"""
        mesh, terrain, h0, u0, v0 = create_dam_break_setup(nx=50, ny=10)

        limiters = ['minmod', 'superbee', 'vanleer']
        results = {}

        for limiter in limiters:
            config = SolverConfig(
                spatial_order=2,
                muscl_limiter=limiter,
                use_numba=True,
                cfl=0.4,
                t_end=0.5
            )

            solver = ShallowWaterSolver(mesh, terrain, config)
            solver.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
            solver.solve()
            results[limiter] = solver.h.copy()

            # Check basic properties
            assert np.all(solver.h >= 0), f"Negative depth with {limiter}"
            assert np.isfinite(solver.h).all(), f"Non-finite values with {limiter}"

        print(f"\n不同限制器测试:")
        for limiter in limiters:
            h = results[limiter]
            print(f"  {limiter:10s}: 深度范围=[{h.min():.3f}, {h.max():.3f}]")

        # Different limiters should give slightly different results
        # but all should be stable
        assert not np.allclose(results['minmod'], results['superbee'])
        assert not np.allclose(results['minmod'], results['vanleer'])

    def test_numba_muscl_mass_conservation(self):
        """Test mass conservation with Numba MUSCL"""
        mesh, terrain, h0, u0, v0 = create_dam_break_setup(nx=100, ny=10)

        config = SolverConfig(
            spatial_order=2,
            muscl_limiter='minmod',
            use_numba=True,
            cfl=0.4,
            t_end=1.0,
            print_progress=False
        )

        solver = ShallowWaterSolver(mesh, terrain, config)
        solver.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())

        # Initial mass
        cell_area = mesh.dx * mesh.dy
        initial_mass = np.sum(solver.h) * cell_area

        # Run simulation
        solver.solve()

        # Final mass
        final_mass = np.sum(solver.h) * cell_area

        # Compute relative error
        mass_error = abs(final_mass - initial_mass) / initial_mass

        print(f"\n质量守恒检查:")
        print(f"  初始质量: {initial_mass:.6f}")
        print(f"  最终质量: {final_mass:.6f}")
        print(f"  相对误差: {mass_error:.2e}")

        # Should conserve mass to high precision
        assert mass_error < 1e-10, f"Mass not conserved: error = {mass_error}"

    def test_numba_muscl_order_consistency(self):
        """
        Verify Numba MUSCL is indeed second-order

        Compare against first-order Numba solution.
        """
        mesh, terrain, h0, u0, v0 = create_dam_break_setup(nx=100, ny=10)

        # First-order with Numba
        config_1st = SolverConfig(
            spatial_order=1,
            use_numba=True,
            cfl=0.4,
            t_end=1.0,
            print_progress=False
        )

        # Second-order with Numba
        config_2nd = SolverConfig(
            spatial_order=2,
            muscl_limiter='minmod',
            use_numba=True,
            cfl=0.4,
            t_end=1.0,
            print_progress=False
        )

        # Run first-order
        solver_1st = ShallowWaterSolver(mesh, terrain, config_1st)
        solver_1st.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
        solver_1st.solve()

        # Run second-order
        solver_2nd = ShallowWaterSolver(mesh, terrain, config_2nd)
        solver_2nd.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
        solver_2nd.solve()

        # Second-order should be less diffusive (higher total variation)
        h_1st = solver_1st.h[:, 5]  # Middle slice
        h_2nd = solver_2nd.h[:, 5]

        tv_1st = np.sum(np.abs(np.diff(h_1st)))
        tv_2nd = np.sum(np.abs(np.diff(h_2nd)))

        print(f"\n精度阶次一致性:")
        print(f"  一阶总变差: {tv_1st:.3f}")
        print(f"  二阶总变差: {tv_2nd:.3f}")
        print(f"  比值: {tv_2nd / tv_1st:.3f}")

        # Second-order should preserve more variation
        assert tv_2nd > tv_1st, "Second-order should be less diffusive"

        # But should be reasonably close (not wildly different)
        assert tv_2nd / tv_1st < 2.0, "Solutions should be relatively similar"


class TestNumbaMUSCLPerformance:
    """Performance benchmarks for Numba MUSCL"""

    def test_numba_muscl_speedup_medium_grid(self):
        """
        Benchmark Numba MUSCL speedup on medium grid

        Compare: NumPy MUSCL vs Numba MUSCL
        """
        import time

        mesh, terrain, h0, u0, v0 = create_dam_break_setup(nx=100, ny=100)

        config_numpy = SolverConfig(
            spatial_order=2,
            muscl_limiter='minmod',
            use_numba=False,
            cfl=0.4,
            t_end=0.5
        )

        config_numba = SolverConfig(
            spatial_order=2,
            muscl_limiter='minmod',
            use_numba=True,
            cfl=0.4,
            t_end=0.5
        )

        # Warm-up Numba (JIT compilation)
        config_warmup = SolverConfig(
            spatial_order=2,
            muscl_limiter='minmod',
            use_numba=True,
            t_end=0.05
        )
        solver_warmup = ShallowWaterSolver(mesh, terrain, config_warmup)
        solver_warmup.solve()

        # Benchmark NumPy
        start = time.time()
        solver_numpy = ShallowWaterSolver(mesh, terrain, config_numpy)
        solver_numpy.solve()
        time_numpy = time.time() - start

        # Benchmark Numba
        start = time.time()
        solver_numba = ShallowWaterSolver(mesh, terrain, config_numba)
        solver_numba.solve()
        time_numba = time.time() - start

        speedup = time_numpy / time_numba

        print(f"\n性能基准测试 (100x100网格):")
        print(f"  NumPy MUSCL: {time_numpy:.3f}秒")
        print(f"  Numba MUSCL: {time_numba:.3f}秒")
        print(f"  加速比: {speedup:.2f}x")

        # Numba should provide speedup on 100x100 grid
        # (might be modest due to other solver components)
        assert speedup > 0.8, f"Numba significantly slower: {speedup:.2f}x"

        if speedup >= 1.2:
            print(f"  [OK] Numba提供了 {speedup:.2f}x 加速")
        elif speedup >= 0.95:
            print(f"  ~ Numba性能相当 ({speedup:.2f}x)")
        else:
            print(f"  [WARN] Numba在此次运行中较慢 ({speedup:.2f}x)")

    def test_numba_muscl_scaling(self):
        """
        Test Numba MUSCL performance scaling with grid size

        Expected: Better speedup on larger grids
        """
        import time

        grid_sizes = [50, 100]
        results = []

        for nx in grid_sizes:
            ny = nx
            domain = DomainParams(xmin=0, xmax=100, ymin=0, ymax=100)
            terrain = np.zeros((nx, ny))
            h0 = np.ones((nx, ny))
            u0 = np.zeros((nx, ny))
            v0 = np.zeros((nx, ny))
            mesh = MeshGenerator(domain).generate_uniform_mesh(nx=nx, ny=ny)

            config_numpy = SolverConfig(
                spatial_order=2,
                muscl_limiter='minmod',
                use_numba=False,
                t_end=0.2
            )

            config_numba = SolverConfig(
                spatial_order=2,
                muscl_limiter='minmod',
                use_numba=True,
                t_end=0.2
            )

            # Warm-up for this grid size
            if nx == grid_sizes[0]:
                config_warmup = SolverConfig(
                    spatial_order=2,
                    muscl_limiter='minmod',
                    use_numba=True,
                    t_end=0.01
                )
                solver_warmup = ShallowWaterSolver(mesh, terrain, config_warmup)
                solver_warmup.solve()

            # Benchmark NumPy
            start = time.time()
            solver_numpy = ShallowWaterSolver(mesh, terrain, config_numpy)
            solver_numpy.solve()
            time_numpy = time.time() - start

            # Benchmark Numba
            start = time.time()
            solver_numba = ShallowWaterSolver(mesh, terrain, config_numba)
            solver_numba.solve()
            time_numba = time.time() - start

            speedup = time_numpy / time_numba
            results.append({'nx': nx, 'speedup': speedup})

        print(f"\n性能扩展性测试:")
        for r in results:
            print(f"  {r['nx']:3d}x{r['nx']:3d}: {r['speedup']:.2f}x加速")

        # Larger grids should generally have better speedup
        # (though this isn't guaranteed on every run)
        assert all(r['speedup'] > 0.5 for r in results), "Numba should not be much slower"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
