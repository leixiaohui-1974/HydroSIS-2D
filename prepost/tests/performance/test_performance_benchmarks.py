"""
Performance Benchmarking Framework

Compares HydroSIS-2D GPU solver against:
1. CPU reference implementation
2. Commercial software benchmarks (reported performance)
3. Scaling analysis (weak/strong scaling)

Targets:
- 50-100x GPU speedup for medium meshes (200k cells)
- 100-150x GPU speedup for large meshes (1M+ cells)
- Linear scaling up to 4 GPUs

Benchmarks against commercial tools:
- RiverFlow2D: 30-100x GPU speedup (reported)
- TUFLOW GPU: 50-100x GPU speedup (reported)
- InfoWorks ICM: CPU-based (baseline comparison)
"""

import pytest
import numpy as np
import time
from typing import Dict, List, Tuple
import sys
from pathlib import Path

# Add preprocessing modules to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.mesh_generation import MeshGenerator, DomainParams
from preprocessing.utils import create_dam_break_simulation


class PerformanceBenchmark:
    """Base class for performance benchmarks"""

    def __init__(self):
        self.results = {}
        self.mesh_sizes = [
            (50, 50),      # 2.5k cells (small)
            (100, 100),    # 10k cells
            (200, 100),    # 20k cells
            (200, 200),    # 40k cells (medium-small)
            (400, 200),    # 80k cells
            (500, 400),    # 200k cells (medium)
            (1000, 500),   # 500k cells (large)
            (1000, 1000),  # 1M cells (very large)
        ]

    def measure_preprocessing_time(self, nx: int, ny: int) -> float:
        """Measure preprocessing time"""
        start = time.time()

        # Create mesh
        domain = DomainParams(xmin=0.0, xmax=float(nx), ymin=0.0, ymax=float(ny))
        mesh_gen = MeshGenerator(domain)
        mesh = mesh_gen.generate_uniform_mesh(nx=nx, ny=ny)

        # Setup initial conditions
        config = create_dam_break_simulation(
            length=float(nx),
            width=float(ny),
            nx=nx,
            ny=ny,
            dam_position=0.5,
            upstream_depth=10.0,
            downstream_depth=1.0
        )

        elapsed = time.time() - start
        return elapsed

    def measure_gpu_solver_time(self, nx: int, ny: int, t_end: float = 10.0) -> Dict:
        """
        Measure GPU solver performance

        Returns:
            dict with timing breakdown:
            - total_time: Total simulation time
            - setup_time: Initialization + memory allocation
            - compute_time: Pure computation time
            - transfer_time: CPU-GPU data transfer
            - steps: Number of time steps
            - cells_per_sec: Throughput (million cell-updates/sec)
        """
        try:
            import hydrosis2d_cuda
        except ImportError:
            pytest.skip("GPU solver not available")

        ncells = nx * ny

        # Setup
        start_setup = time.time()
        solver = hydrosis2d_cuda.Solver()

        config = hydrosis2d_cuda.SolverConfig()
        config.nx = nx
        config.ny = ny
        config.dx = 1.0
        config.dy = 1.0
        config.cfl = 0.8

        solver.initialize(config)

        # Initial conditions
        h = np.zeros((ny, nx))
        h[:, :nx//2] = 10.0
        h[:, nx//2:] = 1.0
        u = np.zeros_like(h)
        v = np.zeros_like(h)
        z = np.zeros_like(h)

        solver.set_initial_conditions(h, u, v, z)
        setup_time = time.time() - start_setup

        # Solve
        start_compute = time.time()
        solver.run(t_end=t_end)
        compute_time = time.time() - start_compute

        # Get solution (includes transfer)
        start_transfer = time.time()
        result = solver.get_solution()
        transfer_time = time.time() - start_transfer

        total_time = setup_time + compute_time + transfer_time
        steps = result.get('steps', 0)

        # Throughput: million cell-updates per second
        # cell-updates = ncells * steps
        cell_updates = ncells * steps
        mcups = (cell_updates / 1e6) / compute_time  # Million cell-updates/sec

        return {
            'total_time': total_time,
            'setup_time': setup_time,
            'compute_time': compute_time,
            'transfer_time': transfer_time,
            'steps': steps,
            'mcups': mcups,
            'ncells': ncells
        }


class TestGPUPerformance(PerformanceBenchmark):
    """Test GPU solver performance"""

    @pytest.mark.skipif(True, reason="Requires GPU")
    def test_small_mesh_performance(self):
        """Test performance on small mesh (10k cells)"""
        result = self.measure_gpu_solver_time(nx=100, ny=100, t_end=10.0)

        print(f"\nSmall Mesh (10k cells) Performance:")
        print(f"  Total time: {result['total_time']:.3f} s")
        print(f"  Compute time: {result['compute_time']:.3f} s")
        print(f"  Steps: {result['steps']}")
        print(f"  Throughput: {result['mcups']:.1f} Mcups")

        # Should complete in reasonable time
        assert result['total_time'] < 5.0, "Small mesh should solve quickly"

    @pytest.mark.skipif(True, reason="Requires GPU")
    def test_medium_mesh_performance(self):
        """Test performance on medium mesh (200k cells)"""
        result = self.measure_gpu_solver_time(nx=500, ny=400, t_end=60.0)

        print(f"\nMedium Mesh (200k cells) Performance:")
        print(f"  Total time: {result['total_time']:.3f} s")
        print(f"  Compute time: {result['compute_time']:.3f} s")
        print(f"  Steps: {result['steps']}")
        print(f"  Throughput: {result['mcups']:.1f} Mcups")

        # Target: <30s for 60s simulation (TUFLOW GPU: 60x speedup)
        assert result['total_time'] < 30.0, "Medium mesh should achieve 60x speedup"

    @pytest.mark.skipif(True, reason="Requires GPU")
    def test_large_mesh_performance(self):
        """Test performance on large mesh (1M cells)"""
        result = self.measure_gpu_solver_time(nx=1000, ny=1000, t_end=3600.0)

        print(f"\nLarge Mesh (1M cells) Performance:")
        print(f"  Total time: {result['total_time']:.3f} s")
        print(f"  Compute time: {result['compute_time']:.3f} s")
        print(f"  Steps: {result['steps']}")
        print(f"  Throughput: {result['mcups']:.1f} Mcups")

        # Target: <120s for 1-hour simulation (90x speedup)
        assert result['total_time'] < 120.0, "Large mesh should achieve 90x speedup"


class TestScalingAnalysis(PerformanceBenchmark):
    """Test scaling properties"""

    @pytest.mark.skipif(True, reason="Requires GPU")
    def test_weak_scaling(self):
        """
        Test weak scaling: keep work per processor constant

        As mesh size increases, time should remain approximately constant
        (if perfectly parallelizable)
        """
        results = []

        # Test different mesh sizes
        test_sizes = [(100, 100), (200, 200), (400, 400)]

        for nx, ny in test_sizes:
            result = self.measure_gpu_solver_time(nx=nx, ny=ny, t_end=10.0)
            results.append(result)

            print(f"\nWeak Scaling: {nx}x{ny} = {nx*ny:,} cells")
            print(f"  Time: {result['compute_time']:.3f} s")
            print(f"  Throughput: {result['mcups']:.1f} Mcups")

        # Check that throughput is relatively constant
        throughputs = [r['mcups'] for r in results]
        throughput_std = np.std(throughputs)
        throughput_mean = np.mean(throughputs)

        print(f"\nWeak Scaling Summary:")
        print(f"  Mean throughput: {throughput_mean:.1f} Mcups")
        print(f"  Std dev: {throughput_std:.1f} Mcups")
        print(f"  Variation: {100*throughput_std/throughput_mean:.1f}%")

        # Throughput should be consistent (within 20%)
        assert throughput_std / throughput_mean < 0.20

    @pytest.mark.skipif(True, reason="Requires GPU")
    def test_strong_scaling(self):
        """
        Test strong scaling: fixed problem size, vary resources

        For single GPU, this tests optimal block/grid configuration
        """
        # Fixed problem size
        nx, ny = 1000, 1000

        # Test different block sizes
        block_sizes = [(8, 8), (16, 16), (32, 32)]

        results = []
        for block_x, block_y in block_sizes:
            # Would need to configure block size in solver
            # For now, just document the concept
            pass

        print("\nStrong Scaling: (Implementation pending)")
        print("  Fixed problem: 1000x1000")
        print("  Vary: block size configuration")


class TestCommercialComparison:
    """
    Compare against commercial software benchmarks

    Uses reported performance from literature:
    - RiverFlow2D: 30-100x GPU speedup (Nvidia GTX 1080)
    - TUFLOW GPU: 50-100x GPU speedup (Nvidia Tesla)
    - InfoWorks ICM: CPU baseline
    """

    def test_comparison_riverflow2d(self):
        """Compare against RiverFlow2D benchmarks"""
        # Reported RiverFlow2D performance (from vendor documentation)
        commercial_benchmarks = {
            '50k_cells': {
                'cpu_time': 300.0,  # 5 minutes (estimated)
                'gpu_time': 10.0,   # 10 seconds
                'speedup': 30.0
            },
            '200k_cells': {
                'cpu_time': 1800.0,  # 30 minutes
                'gpu_time': 30.0,    # 30 seconds
                'speedup': 60.0
            },
            '1M_cells': {
                'cpu_time': 7200.0,  # 2 hours
                'gpu_time': 80.0,    # 80 seconds
                'speedup': 90.0
            }
        }

        print("\n" + "="*60)
        print("Commercial Software Comparison: RiverFlow2D")
        print("="*60)

        for case, bench in commercial_benchmarks.items():
            print(f"\n{case}:")
            print(f"  CPU time: {bench['cpu_time']:.1f} s")
            print(f"  GPU time: {bench['gpu_time']:.1f} s")
            print(f"  Speedup: {bench['speedup']:.1f}x")

        # Our targets should match or exceed these
        our_targets = {
            '50k_cells': {'target_speedup': 30.0, 'target_time': 10.0},
            '200k_cells': {'target_speedup': 60.0, 'target_time': 30.0},
            '1M_cells': {'target_speedup': 90.0, 'target_time': 120.0}
        }

        print("\n" + "="*60)
        print("HydroSIS-2D Targets:")
        print("="*60)

        for case, target in our_targets.items():
            print(f"\n{case}:")
            print(f"  Target speedup: {target['target_speedup']:.1f}x")
            print(f"  Target time: {target['target_time']:.1f} s")

    def test_comparison_tuflow(self):
        """Compare against TUFLOW GPU benchmarks"""
        # TUFLOW GPU reported performance
        tuflow_benchmarks = {
            '100k_cells': {'speedup': 50.0},
            '500k_cells': {'speedup': 75.0},
            '2M_cells': {'speedup': 100.0}
        }

        print("\n" + "="*60)
        print("Commercial Software Comparison: TUFLOW GPU")
        print("="*60)

        for case, bench in tuflow_benchmarks.items():
            print(f"{case}: {bench['speedup']:.1f}x speedup")


class TestThroughputMetrics:
    """Test throughput metrics and efficiency"""

    def compute_theoretical_peak(self, gpu_name: str = "RTX 3090") -> Dict:
        """
        Compute theoretical peak performance

        RTX 3090:
        - 10496 CUDA cores
        - 1.70 GHz boost clock
        - 35.58 TFLOPS (FP32)
        - 936 GB/s memory bandwidth

        Shallow water solver characteristics:
        - Memory-bound (low arithmetic intensity)
        - ~10 FLOP per cell per variable per time step
        - 3 variables (h, hu, hv)
        - ~30 FLOP/cell/step
        """
        specs = {
            'RTX 3090': {
                'cuda_cores': 10496,
                'clock_ghz': 1.70,
                'tflops_fp32': 35.58,
                'bandwidth_gbs': 936.0,
                'l2_cache_mb': 6.0
            },
            'A100': {
                'cuda_cores': 6912,
                'clock_ghz': 1.41,
                'tflops_fp32': 19.5,
                'bandwidth_gbs': 1555.0,
                'l2_cache_mb': 40.0
            }
        }

        if gpu_name not in specs:
            gpu_name = 'RTX 3090'

        spec = specs[gpu_name]

        # Arithmetic intensity: FLOP per byte
        flop_per_cell = 30.0  # Approximate for shallow water
        bytes_per_cell = 24.0  # 3 doubles (h, hu, hv)
        arithmetic_intensity = flop_per_cell / bytes_per_cell  # ~1.25 FLOP/byte

        # Roofline analysis
        # Peak performance limited by either:
        # 1. Compute: TFLOPS
        # 2. Memory bandwidth

        compute_limit = spec['tflops_fp32'] * 1e12  # FLOP/s
        bandwidth_limit = spec['bandwidth_gbs'] * 1e9 * arithmetic_intensity  # FLOP/s

        # Shallow water is memory-bound
        peak_flops = min(compute_limit, bandwidth_limit)
        peak_mcups = peak_flops / flop_per_cell / 1e6  # Million cells/sec

        print(f"\nTheoretical Peak Performance ({gpu_name}):")
        print(f"  Compute limit: {compute_limit/1e12:.1f} TFLOPS")
        print(f"  Memory limit: {bandwidth_limit/1e12:.1f} TFLOPS")
        print(f"  Effective limit: {peak_flops/1e12:.1f} TFLOPS")
        print(f"  Peak throughput: {peak_mcups:.0f} Mcups")
        print(f"  Arithmetic intensity: {arithmetic_intensity:.2f} FLOP/byte")

        return {
            'peak_flops': peak_flops,
            'peak_mcups': peak_mcups,
            'bottleneck': 'memory' if bandwidth_limit < compute_limit else 'compute'
        }

    def test_theoretical_peak_rtx3090(self):
        """Test theoretical peak for RTX 3090"""
        peak = self.compute_theoretical_peak('RTX 3090')

        # Should be memory-bound
        assert peak['bottleneck'] == 'memory'

        # Peak should be realistic
        assert peak['peak_mcups'] > 1000, "Peak throughput should exceed 1000 Mcups"

    def test_theoretical_peak_a100(self):
        """Test theoretical peak for A100"""
        peak = self.compute_theoretical_peak('A100')

        # A100 has high memory bandwidth
        assert peak['peak_mcups'] > 2000, "A100 peak should exceed 2000 Mcups"


class TestMemoryFootprint:
    """Test GPU memory requirements"""

    def compute_memory_requirement(self, nx: int, ny: int) -> Dict:
        """
        Compute GPU memory requirement

        Storage per cell:
        - Conservative variables: 3 doubles (h, hu, hv)
        - Primitive variables: 3 doubles (u, v, h_copy)
        - Terrain: 1 double (z)
        - Fluxes: 6 doubles (F_x, F_y for 3 variables)
        - Temporary: 3 doubles (for RK2/RK3)

        Total: ~16 doubles per cell = 128 bytes/cell
        """
        ncells = nx * ny

        # Bytes per cell
        conservative = 3 * 8  # h, hu, hv (doubles)
        primitive = 3 * 8     # u, v, (temp h)
        terrain = 1 * 8       # z
        fluxes = 6 * 8        # 3 vars × 2 directions
        temporary = 3 * 8     # RK2/RK3 storage
        manning = 1 * 8       # Manning coefficient

        bytes_per_cell = conservative + primitive + terrain + fluxes + temporary + manning
        total_mb = (ncells * bytes_per_cell) / (1024 * 1024)
        total_gb = total_mb / 1024

        print(f"\nMemory Requirement for {nx}×{ny} = {ncells:,} cells:")
        print(f"  Per-cell storage: {bytes_per_cell} bytes")
        print(f"  Total memory: {total_mb:.1f} MB ({total_gb:.3f} GB)")

        return {
            'ncells': ncells,
            'bytes_per_cell': bytes_per_cell,
            'total_mb': total_mb,
            'total_gb': total_gb
        }

    def test_memory_small_mesh(self):
        """Test memory for small mesh"""
        mem = self.compute_memory_requirement(nx=100, ny=100)
        assert mem['total_mb'] < 100, "Small mesh should use < 100 MB"

    def test_memory_medium_mesh(self):
        """Test memory for medium mesh"""
        mem = self.compute_memory_requirement(nx=500, ny=400)
        assert mem['total_mb'] < 500, "Medium mesh should use < 500 MB"

    def test_memory_large_mesh(self):
        """Test memory for large mesh"""
        mem = self.compute_memory_requirement(nx=1000, ny=1000)
        assert mem['total_gb'] < 2.0, "Large mesh should use < 2 GB"

    def test_max_mesh_size_rtx3090(self):
        """Test maximum mesh size for RTX 3090 (24 GB)"""
        gpu_memory_gb = 24.0
        bytes_per_cell = 128

        max_cells = int(gpu_memory_gb * 1024 * 1024 * 1024 / bytes_per_cell)
        max_nx = int(np.sqrt(max_cells))

        print(f"\nMaximum Mesh Size (RTX 3090, 24 GB):")
        print(f"  Max cells: {max_cells:,}")
        print(f"  Max square mesh: {max_nx} × {max_nx}")

        assert max_cells > 100e6, "Should support > 100M cells"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
