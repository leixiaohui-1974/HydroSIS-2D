#!/usr/bin/env python3
"""
Performance Benchmark Script

This script benchmarks different solver configurations to help users
choose the optimal settings for their simulations.

Tested configurations:
1. First-order + NumPy (baseline)
2. Second-order MUSCL + NumPy
3. First-order + Numba
4. Second-order MUSCL + Numba (falls back to NumPy currently)

Usage:
    python benchmark_performance.py

Output:
    - Performance comparison table
    - Recommendations based on grid size
    - Speedup analysis
"""

import numpy as np
import time
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from solver import ShallowWaterSolver, SolverConfig

# Check Numba availability
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def run_benchmark(grid_sizes=[50, 100, 200], t_end=1.0, num_runs=3):
    """
    Run performance benchmarks on different grid sizes

    Parameters
    ----------
    grid_sizes : list
        List of grid sizes to test (nx dimension)
    t_end : float
        Simulation time for each benchmark
    num_runs : int
        Number of runs to average (for stability)
    """
    print("="*80)
    print("PERFORMANCE BENCHMARK")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Grid sizes: {grid_sizes}")
    print(f"  Simulation time: {t_end} s")
    print(f"  Runs per config: {num_runs}")
    print(f"  Numba available: {NUMBA_AVAILABLE}")
    print("="*80)

    results = {}

    for nx in grid_sizes:
        print(f"\n{'='*80}")
        print(f"GRID SIZE: {nx}×{nx//2} cells")
        print(f"{'='*80}")

        ny = nx // 2
        domain = DomainParams(0, 100, 0, 50)
        mesh = MeshGenerator(domain).generate_uniform_mesh(nx=nx, ny=ny)
        terrain = np.zeros((nx, ny))

        # Random initial conditions for realistic test
        np.random.seed(42)  # Reproducible
        h0 = np.random.uniform(2.0, 5.0, (nx, ny))
        u0 = np.random.uniform(-0.5, 0.5, (nx, ny))
        v0 = np.random.uniform(-0.5, 0.5, (nx, ny))

        grid_results = {}

        # ======================================================================
        # Configuration 1: First-order + NumPy (BASELINE)
        # ======================================================================
        config_name = "1st-order + NumPy"
        print(f"\n[1/4] Testing: {config_name}")

        config = SolverConfig(
            t_end=t_end,
            spatial_order=1,
            use_numba=False,
            manning_n=0.0,
            print_progress=False
        )

        times = []
        for run in range(num_runs):
            solver = ShallowWaterSolver(mesh, terrain, config)
            solver.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())

            start = time.time()
            solver.solve()
            elapsed = time.time() - start
            times.append(elapsed)

            print(f"  Run {run+1}/{num_runs}: {elapsed:.3f} s", end='\r')

        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"  Average: {avg_time:.3f} ± {std_time:.3f} s")

        grid_results[config_name] = {
            'time': avg_time,
            'std': std_time,
            'speedup': 1.0  # Baseline
        }

        # ======================================================================
        # Configuration 2: Second-order MUSCL + NumPy
        # ======================================================================
        config_name = "2nd-order + NumPy"
        print(f"\n[2/4] Testing: {config_name}")

        config = SolverConfig(
            t_end=t_end,
            spatial_order=2,
            muscl_limiter='minmod',
            use_numba=False,
            manning_n=0.0,
            print_progress=False
        )

        times = []
        for run in range(num_runs):
            solver = ShallowWaterSolver(mesh, terrain, config)
            solver.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())

            start = time.time()
            solver.solve()
            elapsed = time.time() - start
            times.append(elapsed)

            print(f"  Run {run+1}/{num_runs}: {elapsed:.3f} s", end='\r')

        avg_time = np.mean(times)
        std_time = np.std(times)
        baseline_time = grid_results["1st-order + NumPy"]['time']
        speedup = baseline_time / avg_time

        print(f"  Average: {avg_time:.3f} ± {std_time:.3f} s (slowdown: {1/speedup:.2f}x)")

        grid_results[config_name] = {
            'time': avg_time,
            'std': std_time,
            'speedup': speedup
        }

        # ======================================================================
        # Configuration 3: First-order + Numba
        # ======================================================================
        if NUMBA_AVAILABLE:
            config_name = "1st-order + Numba"
            print(f"\n[3/4] Testing: {config_name}")

            # Warm-up Numba (JIT compilation)
            print("  Warming up Numba JIT...", end='')
            config_warmup = SolverConfig(
                t_end=0.1,
                spatial_order=1,
                use_numba=True,
                manning_n=0.0,
                print_progress=False
            )
            solver_warmup = ShallowWaterSolver(mesh, terrain, config_warmup)
            solver_warmup.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())
            solver_warmup.solve()
            print(" Done!")

            config = SolverConfig(
                t_end=t_end,
                spatial_order=1,
                use_numba=True,
                manning_n=0.0,
                print_progress=False
            )

            times = []
            for run in range(num_runs):
                solver = ShallowWaterSolver(mesh, terrain, config)
                solver.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())

                start = time.time()
                solver.solve()
                elapsed = time.time() - start
                times.append(elapsed)

                print(f"  Run {run+1}/{num_runs}: {elapsed:.3f} s", end='\r')

            avg_time = np.mean(times)
            std_time = np.std(times)
            speedup = baseline_time / avg_time

            print(f"  Average: {avg_time:.3f} ± {std_time:.3f} s (speedup: {speedup:.2f}x)")

            grid_results[config_name] = {
                'time': avg_time,
                'std': std_time,
                'speedup': speedup
            }
        else:
            print(f"\n[3/4] Skipped: 1st-order + Numba (Numba not available)")

        # ======================================================================
        # Configuration 4: Second-order + Numba (currently falls back to NumPy)
        # ======================================================================
        if NUMBA_AVAILABLE:
            config_name = "2nd-order + NumPy*"
            print(f"\n[4/4] Testing: {config_name} (*Numba fallback)")
            print("  Note: MUSCL currently uses NumPy even with use_numba=True")

            config = SolverConfig(
                t_end=t_end,
                spatial_order=2,
                muscl_limiter='minmod',
                use_numba=True,  # Will fall back to NumPy for MUSCL
                manning_n=0.0,
                print_progress=False
            )

            times = []
            for run in range(num_runs):
                solver = ShallowWaterSolver(mesh, terrain, config)
                solver.set_initial_conditions(h0.copy(), u0.copy(), v0.copy())

                start = time.time()
                solver.solve()
                elapsed = time.time() - start
                times.append(elapsed)

                print(f"  Run {run+1}/{num_runs}: {elapsed:.3f} s", end='\r')

            avg_time = np.mean(times)
            std_time = np.std(times)
            speedup = baseline_time / avg_time

            print(f"  Average: {avg_time:.3f} ± {std_time:.3f} s")

            grid_results[config_name] = {
                'time': avg_time,
                'std': std_time,
                'speedup': speedup
            }
        else:
            print(f"\n[4/4] Skipped: 2nd-order + Numba (Numba not available)")

        results[f"{nx}×{ny}"] = grid_results

    # ==========================================================================
    # Print comparison table
    # ==========================================================================
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*80)

    for grid_size, grid_results in results.items():
        print(f"\n{grid_size} cells:")
        print(f"  {'Configuration':<20} {'Time [s]':<12} {'Speedup':<10} {'vs Baseline'}")
        print(f"  {'-'*60}")
        for config_name, data in grid_results.items():
            speedup_str = f"{data['speedup']:.2f}x"
            if data['speedup'] > 1.0:
                speedup_str = f"\033[92m{speedup_str}\033[0m"  # Green for speedup
            elif data['speedup'] < 1.0:
                speedup_str = f"\033[91m{speedup_str}\033[0m"  # Red for slowdown

            print(f"  {config_name:<20} {data['time']:.3f} ± {data['std']:.3f}  {speedup_str}")

    # ==========================================================================
    # Recommendations
    # ==========================================================================
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("\nFor SMALL grids (< 100×100):")
    print("  → Use: spatial_order=1, use_numba=False")
    print("  → Reason: Simple is fast, Numba overhead not worth it")

    print("\nFor MEDIUM grids (100×100 to 200×200):")
    if NUMBA_AVAILABLE:
        print("  → Use: spatial_order=1, use_numba=True")
        print("  → Reason: Numba provides modest speedup")
    else:
        print("  → Use: spatial_order=1, use_numba=False")
        print("  → Reason: NumPy vectorization is efficient")

    print("\nFor LARGE grids (> 200×200):")
    if NUMBA_AVAILABLE:
        print("  → Use: spatial_order=1, use_numba=True")
        print("  → Reason: Numba speedup increases with grid size")
    else:
        print("  → Use: spatial_order=1, use_numba=False")
        print("  → Note: Consider installing Numba for better performance")

    print("\nFor HIGH ACCURACY requirements:")
    print("  → Use: spatial_order=2, muscl_limiter='minmod'")
    print("  → Reason: Reduced numerical diffusion, sharper shocks")
    print("  → Trade-off: ~80% computational overhead")

    print("\nFuture optimizations:")
    print("  → Numba-optimized MUSCL: Would combine speed + accuracy")
    print("  → Parallel execution: Would provide multi-core speedup")
    print("  → GPU acceleration: Would provide 100-1000x speedup")

    print("="*80)


if __name__ == '__main__':
    # Run benchmark with default parameters
    run_benchmark(grid_sizes=[50, 100, 200], t_end=0.5, num_runs=3)
