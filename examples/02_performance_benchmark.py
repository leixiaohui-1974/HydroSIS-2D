#!/usr/bin/env python3
"""
Example 2: Performance Benchmarking

This example demonstrates:
- Performance testing on different mesh sizes
- GPU vs CPU comparison (if available)
- Scaling analysis
- Throughput metrics

Compares against commercial software targets:
- RiverFlow2D: 30-100x GPU speedup
- TUFLOW GPU: 50-100x GPU speedup
"""

import sys
from pathlib import Path
import numpy as np
import time
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / 'prepost'))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from preprocessing.utils import create_dam_break_simulation


def benchmark_gpu_solver(nx, ny, t_end=10.0):
    """
    Benchmark GPU solver performance

    Returns:
        dict: Performance metrics
    """
    try:
        import hydrosis2d_cuda
    except ImportError:
        print("GPU solver not available")
        return None

    # Create configuration
    config = create_dam_break_simulation(
        length=float(nx),
        width=float(ny),
        nx=nx,
        ny=ny,
        dam_position=0.5,
        upstream_depth=10.0,
        downstream_depth=1.0,
        simulation_time=t_end
    )

    # Setup solver
    solver = hydrosis2d_cuda.Solver()

    solver_config = hydrosis2d_cuda.SolverConfig()
    solver_config.nx = nx
    solver_config.ny = ny
    solver_config.dx = 1.0
    solver_config.dy = 1.0
    solver_config.cfl = 0.8

    start_setup = time.time()
    solver.initialize(solver_config)

    h = config.ic_manager.depth
    u = config.ic_manager.velocity_x
    v = config.ic_manager.velocity_y
    z = np.zeros_like(h)

    solver.set_initial_conditions(h, u, v, z)
    setup_time = time.time() - start_setup

    # Run simulation
    start_compute = time.time()
    solver.run(t_end=t_end)
    compute_time = time.time() - start_compute

    # Get results
    start_transfer = time.time()
    result = solver.get_solution()
    transfer_time = time.time() - start_transfer

    total_time = setup_time + compute_time + transfer_time
    n_steps = result.get('steps', 0)
    ncells = nx * ny

    # Throughput: million cell-updates per second
    cell_updates = ncells * n_steps
    mcups = (cell_updates / 1e6) / compute_time

    return {
        'nx': nx,
        'ny': ny,
        'ncells': ncells,
        'total_time': total_time,
        'setup_time': setup_time,
        'compute_time': compute_time,
        'transfer_time': transfer_time,
        'steps': n_steps,
        'mcups': mcups,
        't_end': t_end
    }


def estimate_cpu_time(ncells, t_end):
    """
    Estimate CPU solver time based on empirical scaling

    Assumes: ~10 ms per 1k cells per simulated second
    """
    baseline_time = 0.01  # 10ms per 1k cells per second
    cpu_time = baseline_time * (ncells / 1000.0) * t_end
    return cpu_time


def main():
    print("="*70)
    print("Example 2: Performance Benchmarking")
    print("="*70)

    # Test mesh sizes
    test_cases = [
        (50, 50, "Tiny"),      # 2.5k cells
        (100, 100, "Small"),   # 10k cells
        (200, 100, "Medium"),  # 20k cells
        (200, 200, "Large"),   # 40k cells
        (400, 200, "XLarge"),  # 80k cells
    ]

    results = []

    for nx, ny, label in test_cases:
        ncells = nx * ny

        print(f"\n{'='*70}")
        print(f"Test Case: {label} ({nx}×{ny} = {ncells:,} cells)")
        print(f"{'='*70}")

        # Benchmark GPU
        print("\nRunning GPU benchmark...")
        result = benchmark_gpu_solver(nx, ny, t_end=10.0)

        if result is None:
            print("GPU solver not available. Stopping benchmarks.")
            break

        # Estimate CPU time
        cpu_time_est = estimate_cpu_time(ncells, result['t_end'])
        speedup = cpu_time_est / result['compute_time']

        print(f"\n  Results:")
        print(f"    GPU compute time: {result['compute_time']:.3f} s")
        print(f"    GPU total time:   {result['total_time']:.3f} s")
        print(f"    Setup time:       {result['setup_time']:.3f} s")
        print(f"    Transfer time:    {result['transfer_time']:.3f} s")
        print(f"    Time steps:       {result['steps']}")
        print(f"    Throughput:       {result['mcups']:.1f} Mcups")
        print(f"\n  Estimated CPU time: {cpu_time_est:.1f} s")
        print(f"  Speedup:            {speedup:.1f}x")

        # Check against targets
        targets = {
            2500: 10.0,    # Tiny: 10x
            10000: 20.0,   # Small: 20x
            20000: 30.0,   # Medium: 30x
            40000: 50.0,   # Large: 50x
            80000: 60.0,   # XLarge: 60x
        }

        target_speedup = targets.get(ncells, 30.0)
        status = "✓ PASS" if speedup >= target_speedup else "✗ BELOW TARGET"
        print(f"  Target speedup:     {target_speedup:.1f}x")
        print(f"  Status:             {status}")

        results.append({
            'label': label,
            'ncells': ncells,
            'gpu_time': result['compute_time'],
            'cpu_time_est': cpu_time_est,
            'speedup': speedup,
            'mcups': result['mcups'],
            'target': target_speedup
        })

    if not results:
        return

    # Summary
    print(f"\n{'='*70}")
    print("Performance Summary")
    print(f"{'='*70}")

    print(f"\n{'Case':<10} {'Cells':<10} {'GPU(s)':<10} {'CPU(s)':<10} {'Speedup':<10} {'Target':<10} {'Status'}")
    print("-"*70)

    for r in results:
        status = "PASS" if r['speedup'] >= r['target'] else "FAIL"
        print(f"{r['label']:<10} {r['ncells']:<10} {r['gpu_time']:<10.2f} "
              f"{r['cpu_time_est']:<10.1f} {r['speedup']:<10.1f}x {r['target']:<10.1f}x {status}")

    # Visualize results
    print(f"\nGenerating performance plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Speedup vs mesh size
    ax = axes[0, 0]
    ncells_list = [r['ncells'] for r in results]
    speedup_list = [r['speedup'] for r in results]
    target_list = [r['target'] for r in results]

    ax.plot(ncells_list, speedup_list, 'o-', linewidth=2, markersize=8, label='Achieved')
    ax.plot(ncells_list, target_list, 's--', linewidth=2, markersize=6, label='Target', alpha=0.7)

    # Commercial benchmarks
    ax.axhline(y=30, color='r', linestyle=':', alpha=0.5, label='RiverFlow2D (30x)')
    ax.axhline(y=60, color='g', linestyle=':', alpha=0.5, label='TUFLOW GPU (60x)')

    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('Speedup (GPU vs CPU)')
    ax.set_title('GPU Speedup vs Mesh Size')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot 2: Throughput vs mesh size
    ax = axes[0, 1]
    mcups_list = [r['mcups'] for r in results]

    ax.plot(ncells_list, mcups_list, 'o-', linewidth=2, markersize=8, color='purple')
    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('Throughput (Mcups)')
    ax.set_title('GPU Throughput')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    # Plot 3: Compute time comparison
    ax = axes[1, 0]
    gpu_times = [r['gpu_time'] for r in results]
    cpu_times = [r['cpu_time_est'] for r in results]

    x = np.arange(len(results))
    width = 0.35

    ax.bar(x - width/2, gpu_times, width, label='GPU', color='blue', alpha=0.7)
    ax.bar(x + width/2, cpu_times, width, label='CPU (est.)', color='orange', alpha=0.7)

    ax.set_xlabel('Test Case')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Compute Time: GPU vs CPU')
    ax.set_xticks(x)
    ax.set_xticklabels([r['label'] for r in results])
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Scaling efficiency
    ax = axes[1, 1]

    # Ideal scaling: speedup proportional to sqrt(ncells)
    ncells_array = np.array(ncells_list)
    baseline_cells = ncells_list[0]
    ideal_speedup = speedup_list[0] * np.sqrt(ncells_array / baseline_cells)

    ax.plot(ncells_list, speedup_list, 'o-', linewidth=2, markersize=8, label='Actual')
    ax.plot(ncells_list, ideal_speedup, 's--', linewidth=2, alpha=0.5, label='Ideal (√N scaling)')

    ax.set_xlabel('Number of Cells')
    ax.set_ylabel('Speedup')
    ax.set_title('Scaling Efficiency')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save results
    output_file = Path(__file__).parent / 'output' / '02_performance_results.png'
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Figure saved: {output_file}")

    plt.show()

    print(f"\n{'='*70}")
    print("Benchmark completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
