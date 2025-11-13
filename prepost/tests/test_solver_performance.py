# -*- coding: utf-8 -*-
"""
Performance benchmarks for shallow water solver

Measures performance of key solver components, particularly comparing
vectorized vs loop-based implementations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from preprocessing.mesh_generation import MeshGenerator, DomainParams
from solver import ShallowWaterSolver, SolverConfig


class TestFluxComputationPerformance:
    """Benchmark flux computation methods"""

    def benchmark_flux_computation(self, nx, ny, num_iterations=100):
        """
        Benchmark flux computation for given grid size

        Args:
            nx, ny: Grid dimensions
            num_iterations: Number of flux computations to perform

        Returns:
            dict: Timing results
        """
        # Create mesh
        domain = DomainParams(0, nx*2, 0, ny*2)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=nx, ny=ny)
        terrain = np.zeros((nx, ny))

        # Create solver
        config = SolverConfig(output_interval=1000.0, print_progress=False)
        solver = ShallowWaterSolver(mesh, terrain, config)

        # Set initial conditions (dam break for non-trivial flow)
        h0 = np.ones((nx, ny)) * 5.0
        h0[:nx//2, :] = 10.0
        u0 = np.zeros((nx, ny))
        v0 = np.zeros((nx, ny))
        solver.set_initial_conditions(h0, u0, v0)

        # Warm-up
        solver.compute_fluxes_hll()
        solver.compute_fluxes_hll_loop()

        # Benchmark vectorized version
        start = time.time()
        for _ in range(num_iterations):
            solver.compute_fluxes_hll()
        time_vectorized = time.time() - start

        # Benchmark loop version
        start = time.time()
        for _ in range(num_iterations):
            solver.compute_fluxes_hll_loop()
        time_loop = time.time() - start

        speedup = time_loop / time_vectorized

        return {
            'grid_size': (nx, ny),
            'num_cells': nx * ny,
            'num_iterations': num_iterations,
            'time_vectorized': time_vectorized,
            'time_loop': time_loop,
            'time_per_call_vectorized': time_vectorized / num_iterations * 1000,  # ms
            'time_per_call_loop': time_loop / num_iterations * 1000,  # ms
            'speedup': speedup
        }

    def test_flux_performance_small_grid(self):
        """Benchmark on small grid (50x50)"""
        results = self.benchmark_flux_computation(50, 50, num_iterations=100)

        print(f"\n{'='*70}")
        print(f"Flux Computation Performance - Small Grid")
        print(f"{'='*70}")
        print(f"Grid size: {results['grid_size'][0]} x {results['grid_size'][1]} = {results['num_cells']} cells")
        print(f"Iterations: {results['num_iterations']}")
        print(f"\nVectorized version:")
        print(f"  Total time: {results['time_vectorized']:.3f} s")
        print(f"  Per call: {results['time_per_call_vectorized']:.3f} ms")
        print(f"\nLoop version:")
        print(f"  Total time: {results['time_loop']:.3f} s")
        print(f"  Per call: {results['time_per_call_loop']:.3f} ms")
        print(f"\nSpeedup: {results['speedup']:.2f}x")
        print(f"{'='*70}")

        # Vectorized should be faster
        assert results['speedup'] > 1.0, f"Vectorized not faster: {results['speedup']}x"
        # Expect at least 5x speedup on small grid
        assert results['speedup'] > 3.0, f"Speedup too low: {results['speedup']:.1f}x (expected > 3x)"

    def test_flux_performance_medium_grid(self):
        """Benchmark on medium grid (100x100)"""
        results = self.benchmark_flux_computation(100, 100, num_iterations=50)

        print(f"\n{'='*70}")
        print(f"Flux Computation Performance - Medium Grid")
        print(f"{'='*70}")
        print(f"Grid size: {results['grid_size'][0]} x {results['grid_size'][1]} = {results['num_cells']} cells")
        print(f"Iterations: {results['num_iterations']}")
        print(f"\nVectorized version:")
        print(f"  Total time: {results['time_vectorized']:.3f} s")
        print(f"  Per call: {results['time_per_call_vectorized']:.3f} ms")
        print(f"\nLoop version:")
        print(f"  Total time: {results['time_loop']:.3f} s")
        print(f"  Per call: {results['time_per_call_loop']:.3f} ms")
        print(f"\nSpeedup: {results['speedup']:.2f}x")
        print(f"{'='*70}")

        # Expect better speedup on larger grid
        assert results['speedup'] > 5.0, f"Speedup too low: {results['speedup']:.1f}x (expected > 5x)"

    def test_flux_performance_large_grid(self):
        """Benchmark on large grid (200x200)"""
        results = self.benchmark_flux_computation(200, 200, num_iterations=20)

        print(f"\n{'='*70}")
        print(f"Flux Computation Performance - Large Grid")
        print(f"{'='*70}")
        print(f"Grid size: {results['grid_size'][0]} x {results['grid_size'][1]} = {results['num_cells']} cells")
        print(f"Iterations: {results['num_iterations']}")
        print(f"\nVectorized version:")
        print(f"  Total time: {results['time_vectorized']:.3f} s")
        print(f"  Per call: {results['time_per_call_vectorized']:.3f} ms")
        print(f"\nLoop version:")
        print(f"  Total time: {results['time_loop']:.3f} s")
        print(f"  Per call: {results['time_per_call_loop']:.3f} ms")
        print(f"\nSpeedup: {results['speedup']:.2f}x")
        print(f"{'='*70}")

        # Larger grids should show even better speedup
        assert results['speedup'] > 8.0, f"Speedup too low: {results['speedup']:.1f}x (expected > 8x)"


class TestSolverEndToEndPerformance:
    """Benchmark end-to-end solver performance"""

    def benchmark_solver(self, nx, ny, t_end=1.0):
        """
        Benchmark full solver run

        Args:
            nx, ny: Grid dimensions
            t_end: Simulation time

        Returns:
            dict: Performance metrics
        """
        # Create mesh
        domain = DomainParams(0, nx, 0, ny)
        generator = MeshGenerator(domain)
        mesh = generator.generate_uniform_mesh(nx=nx, ny=ny)
        terrain = np.zeros((nx, ny))

        # Create solver
        config = SolverConfig(
            t_end=t_end,
            output_interval=1000.0,
            print_progress=False,
            check_mass_conservation=True
        )
        solver = ShallowWaterSolver(mesh, terrain, config)

        # Dam break initial condition
        h0 = np.ones((nx, ny)) * 1.0
        h0[:nx//2, :] = 10.0
        u0 = np.zeros((nx, ny))
        v0 = np.zeros((nx, ny))
        solver.set_initial_conditions(h0, u0, v0)

        # Run and measure
        start = time.time()
        solver.solve(t_end=t_end)
        wall_time = time.time() - start

        # Calculate metrics
        cells_per_second = (nx * ny * solver.step_count) / wall_time
        speedup_vs_realtime = solver.t / wall_time

        return {
            'grid_size': (nx, ny),
            'num_cells': nx * ny,
            'simulated_time': solver.t,
            'wall_time': wall_time,
            'num_steps': solver.step_count,
            'cells_per_second': cells_per_second,
            'speedup_vs_realtime': speedup_vs_realtime,
            'time_per_step': wall_time / solver.step_count * 1000  # ms
        }

    def test_end_to_end_performance(self):
        """Benchmark end-to-end solver performance on various grid sizes"""

        test_cases = [
            (50, 50, 1.0),
            (100, 100, 1.0),
            (200, 200, 1.0)
        ]

        print(f"\n{'='*80}")
        print(f"End-to-End Solver Performance")
        print(f"{'='*80}\n")

        for nx, ny, t_end in test_cases:
            results = self.benchmark_solver(nx, ny, t_end)

            print(f"Grid: {results['grid_size'][0]}x{results['grid_size'][1]} = {results['num_cells']:,} cells")
            print(f"  Simulated time: {results['simulated_time']:.2f} s")
            print(f"  Wall time: {results['wall_time']:.2f} s")
            print(f"  Time steps: {results['num_steps']}")
            print(f"  Time per step: {results['time_per_step']:.2f} ms")
            print(f"  Throughput: {results['cells_per_second']/1e6:.2f} M cells/s")
            print(f"  Speedup vs realtime: {results['speedup_vs_realtime']:.1f}x")
            print()

        print(f"{'='*80}")

        # Basic sanity check: should complete faster than realtime for small grids
        results_small = self.benchmark_solver(50, 50, 1.0)
        assert results_small['speedup_vs_realtime'] > 0.5, \
            "Solver too slow for small grid"


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v', '-s'])
